"""Train and evaluate all predictive-maintenance models.

    python train.py                 # generate data (if missing) and train everything
    python train.py --regenerate    # force new synthetic data
    python train.py --skip-dl       # classical ML models only (fast)
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from pdm import data_generator
from pdm.config import (ARTIFACTS_DIR, DL_MODELS, ML_MODELS, MODELS_DIR, PLOTS_DIR,
                        PRIMARY_MODEL, RAW_DATA_PATH, SENSORS)
from pdm.explain import SensorExplainer
from pdm.health import fit_baseline
from pdm.models_dl import build_dl_model, predict_proba_dl, train_dl_model
from pdm.models_ml import build_ml_model
from pdm.preprocessing import (TARGET, clean, engineer_features, feature_columns,
                               fit_sequence_scaler, make_sequences, split_by_vehicle)


def evaluate(y, p, threshold=0.5) -> dict:
    pred = (p >= threshold).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--vehicles", type=int, default=300)
    ap.add_argument("--skip-dl", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    for d in (MODELS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Data
    if args.regenerate or not RAW_DATA_PATH.exists():
        print("[1/6] Generating synthetic fleet telemetry")
        data_generator.main(args.vehicles)
    df = pd.read_csv(RAW_DATA_PATH)

    # 2. Preprocessing
    print("[2/6] Preprocessing and feature engineering")
    feats = engineer_features(clean(df))
    feat_cols = feature_columns(feats)
    train, val, test = split_by_vehicle(feats)
    print(f"      train/val/test vehicles: {train.vehicle_id.nunique()}/"
          f"{val.vehicle_id.nunique()}/{test.vehicle_id.nunique()}  "
          f"({len(feat_cols)} features)")

    seq_scaler = fit_sequence_scaler(train)
    Xs_tr, ys_tr, _ = make_sequences(train, seq_scaler, stride=2)
    Xs_val, ys_val, _ = make_sequences(val, seq_scaler)
    Xs_te, ys_te, te_idx = make_sequences(test, seq_scaler)
    # Evaluate every model on the same test rows (those with a full sequence window).
    test_eval = test.loc[te_idx]
    y_te = test_eval[TARGET].values

    results, probs = {}, {}

    # 3. Classical ML
    print("[3/6] Training ML models")
    for name in ML_MODELS:
        t0 = time.time()
        model = build_ml_model(name, train[TARGET].values)
        model.fit(train[feat_cols], train[TARGET])
        probs[name] = model.predict_proba(test_eval[feat_cols])[:, 1]
        results[name] = evaluate(y_te, probs[name]) | {"train_sec": time.time() - t0}
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"      {name:20s} ROC-AUC={results[name]['roc_auc']:.4f} "
              f"PR-AUC={results[name]['pr_auc']:.4f} F1={results[name]['f1']:.4f}")

    # 4. Deep learning
    trained_dl = []
    if not args.skip_dl:
        print(f"[4/6] Training DL sequence models on {torch.__version__} "
              f"({len(Xs_tr):,} windows)")
        for name in DL_MODELS:
            t0 = time.time()
            print(f"    {name}")
            model = train_dl_model(build_dl_model(name, len(SENSORS)), Xs_tr, ys_tr,
                                   Xs_val, ys_val, epochs=args.epochs)
            probs[name] = predict_proba_dl(model, Xs_te)
            results[name] = evaluate(ys_te, probs[name]) | {"train_sec": time.time() - t0}
            torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
            trained_dl.append(name)
            print(f"      {name:20s} ROC-AUC={results[name]['roc_auc']:.4f} "
                  f"PR-AUC={results[name]['pr_auc']:.4f} F1={results[name]['f1']:.4f}")
    else:
        print("[4/6] Skipping DL models")

    metrics = pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)
    metrics.to_csv(ARTIFACTS_DIR / "metrics.csv", float_format="%.4f")
    print("\nTest-set comparison (sorted by PR-AUC):")
    print(metrics.round(4).to_string())

    # 5. Save metadata needed at inference time
    print("\n[5/6] Saving artifacts")
    joblib.dump(seq_scaler, MODELS_DIR / "sequence_scaler.joblib")
    meta = {"feature_columns": feat_cols, "baseline": fit_baseline(train),
            "models": ML_MODELS + trained_dl, "primary_model": PRIMARY_MODEL,
            "test_vehicles": sorted(test.vehicle_id.unique().tolist())}
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    # 6. Plots + XAI
    print("[6/6] Plotting evaluation and SHAP explanations")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        prec, rec, _ = precision_recall_curve(y_te, p)
        axes[0].plot(fpr, tpr, label=f"{name} ({results[name]['roc_auc']:.3f})")
        axes[1].plot(rec, prec, label=f"{name} ({results[name]['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_pr_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_te, (probs[PRIMARY_MODEL] >= 0.5).astype(int),
                                            display_labels=["No failure", "Failure ≤48h"],
                                            cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix – {PRIMARY_MODEL}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    xgb = joblib.load(MODELS_DIR / "xgboost.joblib")
    sample = test_eval[feat_cols].sample(min(3000, len(test_eval)), random_state=0)
    explainer = SensorExplainer(xgb, feat_cols)
    sv = explainer.feature_shap(sample)
    plt.figure()
    shap.summary_plot(sv, sample, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=130)
    plt.close()

    sensor_imp = explainer.sensor_contributions(sample).abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sensor_imp.plot.barh(ax=ax, color="#3b82f6")
    ax.set(title="Global sensor importance (mean |SHAP|, XGBoost)", xlabel="mean |SHAP| (log-odds)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sensor_importance.png", dpi=130)
    plt.close(fig)

    print(f"Done. Models -> {MODELS_DIR}\n      Plots  -> {PLOTS_DIR}")


if __name__ == "__main__":
    main()

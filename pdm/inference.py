"""End-to-end inference: sensor history -> failure probability -> health -> alert."""
from __future__ import annotations

import json

import joblib
import pandas as pd
import torch

from .config import (DL_MODELS, MAINTENANCE_ACTIONS, MODELS_DIR, PRIMARY_MODEL, SENSOR_LABELS,
                     SENSORS)
from .explain import SensorExplainer
from .health import CONDITION_SENSORS, anomaly_index, describe_factor, health_score, sensor_deviation, status_from
from .models_dl import build_dl_model, predict_proba_dl
from .preprocessing import clean, engineer_features, last_window


# Minimum SHAP contribution (log-odds) for a sensor to be reported as a risk factor.
MIN_FACTOR_SHAP = 0.25


class VehicleHealthMonitor:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        meta = json.loads((models_dir / "meta.json").read_text())
        self.feature_cols = meta["feature_columns"]
        self.baseline = meta["baseline"]
        self.available = meta["models"]
        self.seq_scaler = joblib.load(models_dir / "sequence_scaler.joblib")
        self._models = {}
        self.explainer = SensorExplainer(self._load("xgboost"), self.feature_cols)

    def _load(self, name: str):
        if name not in self._models:
            if name in DL_MODELS:
                model = build_dl_model(name, len(SENSORS))
                model.load_state_dict(torch.load(self.models_dir / f"{name}.pt", map_location="cpu"))
                model.eval()
            else:
                model = joblib.load(self.models_dir / f"{name}.joblib")
            self._models[name] = model
        return self._models[name]

    def failure_probability(self, history: pd.DataFrame, feat_row: pd.DataFrame, model: str) -> float:
        m = self._load(model)
        if model in DL_MODELS:
            return float(predict_proba_dl(m, last_window(history, self.seq_scaler))[0])
        return float(m.predict_proba(feat_row[self.feature_cols])[0, 1])

    def probability_timeline(self, history: pd.DataFrame, model: str = PRIMARY_MODEL) -> pd.Series:
        """Failure probability at every hour of a vehicle's history (tabular models only)."""
        if model in DL_MODELS:
            raise ValueError("Timeline is computed with tabular models")
        feats = engineer_features(clean(history.copy()))
        p = self._load(model).predict_proba(feats[self.feature_cols])[:, 1]
        return pd.Series(p, index=feats["timestamp"].values)

    def assess(self, history: pd.DataFrame, model: str = PRIMARY_MODEL, top_k: int = 3) -> dict:
        """Assess one vehicle given its chronological sensor history (latest reading last)."""
        history = clean(history.copy())
        feats = engineer_features(history)
        row = feats.iloc[[-1]]

        prob = self.failure_probability(history, row, model)
        dev = sensor_deviation(row.iloc[0], self.baseline)
        anomaly = anomaly_index(dev)
        health = health_score(prob, anomaly)
        status = status_from(prob, health)

        contrib = self.explainer.sensor_contributions(row).iloc[0]
        # Report component-condition sensors only; speed / hours are usage context.
        contrib = contrib[CONDITION_SENSORS]
        factors = []
        ranked = contrib.sort_values(ascending=False).items() if status != "NORMAL" else []
        for sensor, value in ranked:
            if value < MIN_FACTOR_SHAP or len(factors) == top_k:
                break
            factors.append({
                "sensor": sensor,
                "description": describe_factor(sensor, dev, SENSOR_LABELS[sensor]),
                "shap": round(float(value), 3),
                "z_score": round(float(dev[sensor]["z"]), 2),
                "action": MAINTENANCE_ACTIONS[sensor],
            })

        latest = history.iloc[-1]
        return {
            "vehicle_id": str(latest.get("vehicle_id", "UNKNOWN")),
            "timestamp": int(latest.get("timestamp", len(history) - 1)),
            "model": model,
            "failure_probability": prob,
            "health_score": health,
            "anomaly_index": anomaly,
            "status": status,
            "factors": factors,
            "readings": {s: float(latest[s]) for s in SENSORS},
            "sensor_contributions": {s: float(v) for s, v in contrib.items()},
            "alert": maintenance_alert(status, factors),
        }


def maintenance_alert(status: str, factors: list[dict]) -> str | None:
    if status == "NORMAL":
        return None
    urgency = "IMMEDIATE service required - remove from duty" if status == "CRITICAL" \
        else "Schedule maintenance within the next 48 operating hours"
    actions = "; ".join(f["action"] for f in factors) or "Run full diagnostic check"
    return f"{urgency}. Recommended: {actions}."


def format_report(r: dict) -> str:
    lines = [
        f"Vehicle: {r['vehicle_id']}  (hour {r['timestamp']}, model: {r['model']})",
        f"Vehicle Health: {r['health_score']:.0f}%",
        f"Failure Probability: {r['failure_probability'] * 100:.0f}%",
        f"Status: {r['status']}",
    ]
    if r["factors"]:
        lines += ["", "Major contributing factors:"]
        lines += [f"{i}. {f['description'][0].upper() + f['description'][1:]}"
                  for i, f in enumerate(r["factors"], 1)]
    if r["alert"]:
        lines += ["", f"Maintenance Alert: {r['alert']}"]
    return "\n".join(lines)

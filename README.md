# AI/ML-Based Predictive Maintenance and Vehicle Health Monitoring System

Module 3 (Connectivity) – Project 1. The system predicts vehicle/component failure **within the next 48 operating hours** from on-board sensor data, converts that risk into a **Vehicle Health Score**, explains the prediction with **XAI (SHAP)**, and raises a **maintenance alert**.

```
Vehicle Sensors ─► RPM | Temperature | Pressure | Vibration | Battery | Fuel | Speed | Hours
      ▼
Data Preprocessing   (cleaning, interpolation, rolling mean/std/trend, ratios, scaling, windows)
      ▼
ML/DL Model          (Logistic Regression, Decision Tree, Random Forest, XGBoost, LSTM, GRU, Transformer)
      ▼
Failure Probability  (P[failure within 48 h])
      ▼
Vehicle Health Score (risk + deviation from the learned normal operating envelope)
      ▼
Maintenance Alert    (status, SHAP-ranked contributing factors, recommended actions)
```

## Sample output

```
Vehicle: V0005  (hour 497, model: xgboost)
Vehicle Health: 45%
Failure Probability: 52%
Status: WARNING

Major contributing factors:
1. High vibration

Maintenance Alert: Schedule maintenance within the next 48 operating hours. Recommended: Inspect wheel/engine bearings, mounts and wheel balance.
```

## Quick start

```bash
./setup.sh                     # Python 3.12 venv + all dependencies (see requirements.txt)
source .venv/bin/activate

python train.py                # generate data, train & evaluate all 7 models, save plots
python predict.py              # demo: healthy vehicle + one vehicle 150/60/20 h before failure
streamlit run app.py           # interactive dashboard
```

More prediction options:

```bash
python predict.py --vehicle V0012 --hour 400 --model lstm   # any model: logistic_regression,
                                                            # decision_tree, random_forest, xgboost,
                                                            # lstm, gru, transformer
python predict.py --fleet                                   # status table for the test fleet
python predict.py --csv my_log.csv                          # your own data (9 sensor columns)
python predict.py --vehicle V0012 --json                    # machine-readable result
python train.py --skip-dl                                   # classical ML only (~30 s)
python train.py --regenerate --vehicles 500                 # new synthetic fleet
```

## Project structure

| Path | Purpose |
|---|---|
| `pdm/config.py` | Sensors, thresholds, horizons, maintenance actions |
| `pdm/data_generator.py` | Synthetic fleet telemetry with 4 failure modes (bearing wear, overheating, oil system, electrical) |
| `pdm/preprocessing.py` | Cleaning, feature engineering, vehicle-level train/val/test split, sequence windows |
| `pdm/models_ml.py` | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| `pdm/models_dl.py` | LSTM, GRU, Transformer encoder (PyTorch; uses Apple MPS GPU if available) |
| `pdm/health.py` | Normal-envelope baseline, anomaly index, health score, status, factor wording |
| `pdm/explain.py` | SHAP TreeExplainer, feature → sensor aggregation |
| `pdm/inference.py` | `VehicleHealthMonitor`: history → probability → health → alert |
| `train.py` / `predict.py` / `app.py` | Training pipeline, CLI, Streamlit dashboard |
| `artifacts/` | Trained models, `metrics.csv`, evaluation and SHAP plots |

## Data

No public dataset is required: `pdm/data_generator.py` simulates 300 vehicles (~165k hourly readings) with all nine inputs from the brief (engine temperature, RPM, oil pressure, vibration, battery voltage, coolant temperature, fuel consumption, vehicle speed, operating hours). About 55 % of vehicles develop a failure mode whose sensors drift progressively for 70–220 h before breakdown; the rest run healthy but with noise, transient spikes and slow ageing. The target is `failure_within_horizon` = 1 when the vehicle fails within 48 h.

To use real data, supply a CSV with `vehicle_id`, `timestamp`, the nine sensor columns and a `failure_within_horizon` label (plus `time_to_failure`, NaN for healthy vehicles), save it as `data/vehicle_sensor_data.csv`, and run `python train.py`.

## Method

- **Preprocessing** – per-vehicle interpolation and clipping; 12 h rolling mean, std and trend for each sensor; physics ratios (engine–coolant gap, oil pressure per 1000 RPM, fuel per speed). The data is split **by vehicle** (70/15/15), so no vehicle's readings appear in both training and test sets.
- **Models** – tabular models use 36 engineered features; sequence models use 24 h windows of the 9 standardised raw sensors. Class imbalance (~5 % positives) is handled with class weights / `scale_pos_weight` / weighted BCE. DL models use early stopping on validation PR-AUC.
- **Health score** – `100 × (1 − 0.35·P(failure) − 0.65·anomaly_index)`, where the anomaly index measures how far the smoothed sensor levels sit outside the normal envelope learned from healthy training data. Health therefore reflects current condition, and failure probability reflects predicted risk.
- **Status** – `CRITICAL` if P ≥ 80 % or health < 40; `WARNING` if P ≥ 40 % or health < 70; otherwise `NORMAL`.
- **XAI** – SHAP values from XGBoost are summed per physical sensor. The top positive contributors become the "major contributing factors", worded from the sensor's z-score and trend (High / Low / Rising / Falling / Abnormal), and each factor is mapped to a maintenance action.

## Results (held-out test fleet, 45 vehicles)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.992 | 0.929 | 0.621 | 0.965 | 0.756 |
| GRU | 0.988 | 0.925 | 0.609 | 0.969 | 0.748 |
| Transformer | 0.987 | 0.923 | 0.651 | 0.950 | 0.773 |
| LSTM | 0.986 | 0.917 | 0.620 | 0.975 | 0.758 |
| XGBoost | 0.988 | 0.905 | 0.746 | 0.895 | 0.814 |
| Random Forest | 0.991 | 0.903 | 0.706 | 0.929 | 0.802 |
| Decision Tree | 0.945 | 0.760 | 0.578 | 0.934 | 0.714 |

Precision, recall and F1 use a 0.5 threshold. See `artifacts/plots/` for ROC/PR curves, the confusion matrix and SHAP plots. Because the data is synthetic, these scores show the pipeline works; they are not a benchmark for real vehicles.

## macOS note

XGBoost links Homebrew's `libomp`, while PyTorch wheels ship their own copy. Loading both in one process crashes. `setup.sh` fixes this inside `.venv` only, by symlinking PyTorch's `libomp.dylib` to Homebrew's (the original is kept as `libomp.dylib.bundled`).

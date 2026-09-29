"""Vehicle Health Score, status classification and maintenance alerts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CRITICAL_THRESHOLD, WARNING_THRESHOLD
from .preprocessing import DYNAMIC_SENSORS, TARGET

# Sensors that reflect component condition (speed is a usage input, not a fault signal).
CONDITION_SENSORS = [s for s in DYNAMIC_SENSORS if s != "vehicle_speed"]


def fit_baseline(train_feat: pd.DataFrame) -> dict:
    """Normal operating envelope learned from healthy training readings.

    Uses readings that are far from any failure so the envelope is not
    contaminated by degradation.
    """
    healthy = train_feat[(train_feat[TARGET] == 0) &
                         (train_feat["time_to_failure"].isna() | (train_feat["time_to_failure"] > 250))]
    base = {}
    for s in DYNAMIC_SENSORS:
        base[s] = {
            "mean": float(healthy[f"{s}_mean"].mean()),
            "std": float(healthy[f"{s}_mean"].std() + 1e-6),
            "trend_std": float(healthy[f"{s}_trend"].std() + 1e-6),
            "p01": float(healthy[s].quantile(0.01)),
            "p99": float(healthy[s].quantile(0.99)),
        }
    return base


def sensor_deviation(row: pd.Series, baseline: dict) -> dict:
    """z-score of the smoothed level and of the recent trend for each sensor."""
    return {s: {"z": (row[f"{s}_mean"] - b["mean"]) / b["std"],
                "trend_z": row[f"{s}_trend"] / b["trend_std"]}
            for s, b in baseline.items()}


def anomaly_index(dev: dict) -> float:
    """0 (inside normal envelope) .. 1 (far outside) summarising current condition."""
    sev = np.array([np.clip((abs(dev[s]["z"]) - 1.5) / 4.0, 0, 1) for s in CONDITION_SENSORS])
    return float(0.6 * sev.max() + 0.4 * sev.mean())


def health_score(failure_prob: float, anomaly: float) -> float:
    """Health (0-100 %) blends predicted risk with the current sensor condition."""
    score = 100 * (1 - (0.35 * failure_prob + 0.65 * anomaly))
    return float(np.clip(score, 0, 100))


def status_from(failure_prob: float, health: float) -> str:
    if failure_prob >= CRITICAL_THRESHOLD or health < 40:
        return "CRITICAL"
    if failure_prob >= WARNING_THRESHOLD or health < 70:
        return "WARNING"
    return "NORMAL"


def describe_factor(sensor: str, dev: dict, label: str) -> str:
    z, tz = dev[sensor]["z"], dev[sensor]["trend_z"]
    if abs(z) >= 2.5:
        return f"{'High' if z > 0 else 'Low'} {label}"
    if abs(tz) >= 2.0:
        return f"{'Rising' if tz > 0 else 'Falling'} {label}"
    return f"Abnormal {label}"

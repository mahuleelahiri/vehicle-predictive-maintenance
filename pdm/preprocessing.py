"""Data preprocessing: cleaning, feature engineering, splitting and windowing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .config import ROLL, SEED, SENSORS, WINDOW

TARGET = "failure_within_horizon"
# Sensors whose rolling statistics are informative (operating hours is monotonic).
DYNAMIC_SENSORS = [s for s in SENSORS if s != "operating_hours"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, interpolate missing values per vehicle and clip physically impossible values."""
    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)
    df[SENSORS] = df.groupby("vehicle_id")[SENSORS].transform(
        lambda s: s.interpolate(limit_direction="both"))
    df["rpm"] = df["rpm"].clip(0, 8000)
    df["vehicle_speed"] = df["vehicle_speed"].clip(0, 250)
    df["oil_pressure"] = df["oil_pressure"].clip(0, 120)
    df["battery_voltage"] = df["battery_voltage"].clip(0, 20)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean, rolling std and trend features per vehicle.

    Trend = change of the rolling mean over the last ROLL hours, capturing
    "rising temperature" / "falling oil pressure" type degradation.
    """
    g = df.groupby("vehicle_id")
    out = df.copy()
    for s in DYNAMIC_SENSORS:
        mean = g[s].transform(lambda x: x.rolling(ROLL, min_periods=1).mean())
        out[f"{s}_mean"] = mean
        out[f"{s}_std"] = g[s].transform(lambda x: x.rolling(ROLL, min_periods=2).std()).fillna(0)
        out[f"{s}_trend"] = (mean - mean.groupby(df["vehicle_id"]).shift(ROLL)).fillna(0)
    # Physics-inspired ratios.
    out["temp_gap"] = out["engine_temp"] - out["coolant_temp"]
    out["oil_per_krpm"] = out["oil_pressure"] / (out["rpm"] / 1000)
    out["fuel_per_speed"] = out["fuel_consumption"] / (out["vehicle_speed"] + 5)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    extra = [c for c in df.columns if c.endswith(("_mean", "_std", "_trend"))]
    return SENSORS + extra + ["temp_gap", "oil_per_krpm", "fuel_per_speed"]


def split_by_vehicle(df: pd.DataFrame, test_size=0.15, val_size=0.15, seed=SEED):
    """Split by vehicle so no vehicle's readings leak across train/val/test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trval_idx, test_idx = next(gss.split(df, groups=df["vehicle_id"]))
    trval, test = df.iloc[trval_idx], df.iloc[test_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    tr_idx, val_idx = next(gss2.split(trval, groups=trval["vehicle_id"]))
    return trval.iloc[tr_idx].copy(), trval.iloc[val_idx].copy(), test.copy()


def fit_sequence_scaler(train: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(train[SENSORS].values)


def make_sequences(df: pd.DataFrame, scaler: StandardScaler, window: int = WINDOW, stride: int = 1):
    """Build (N, window, n_sensors) sliding windows; label = label at the window end."""
    X_list, y_list, idx_list = [], [], []
    for _, grp in df.groupby("vehicle_id", sort=False):
        if len(grp) < window:
            continue
        values = scaler.transform(grp[SENSORS].values).astype(np.float32)
        wins = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)  # (n-w+1, F, w)
        wins = wins.transpose(0, 2, 1)[::stride]
        X_list.append(wins)
        y_list.append(grp[TARGET].values[window - 1:][::stride])
        idx_list.append(grp.index.values[window - 1:][::stride])
    return (np.concatenate(X_list).astype(np.float32),
            np.concatenate(y_list).astype(np.float32),
            np.concatenate(idx_list))


def last_window(vehicle_df: pd.DataFrame, scaler: StandardScaler, window: int = WINDOW) -> np.ndarray:
    """Most recent window for a single vehicle, left-padded if history is short."""
    values = scaler.transform(vehicle_df[SENSORS].values).astype(np.float32)
    if len(values) < window:
        values = np.vstack([np.repeat(values[:1], window - len(values), axis=0), values])
    return values[-window:][None, ...]

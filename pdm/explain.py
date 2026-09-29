"""Explainable AI (XAI): SHAP attributions aggregated back to physical sensors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .config import SENSORS

# Derived features attributed to the sensor they mostly describe.
DERIVED_TO_SENSOR = {
    "temp_gap": "engine_temp",
    "oil_per_krpm": "oil_pressure",
    "fuel_per_speed": "fuel_consumption",
}


def feature_to_sensor(feature: str) -> str:
    if feature in DERIVED_TO_SENSOR:
        return DERIVED_TO_SENSOR[feature]
    for suffix in ("_mean", "_std", "_trend"):
        if feature.endswith(suffix):
            return feature[: -len(suffix)]
    return feature


class SensorExplainer:
    """Wraps a SHAP TreeExplainer and reports per-sensor contributions (log-odds)."""

    def __init__(self, tree_model, feature_names: list[str]):
        self.explainer = shap.TreeExplainer(tree_model)
        self.feature_names = feature_names
        self.sensor_of = np.array([feature_to_sensor(f) for f in feature_names])

    def feature_shap(self, X: pd.DataFrame) -> np.ndarray:
        values = self.explainer.shap_values(X[self.feature_names])
        if isinstance(values, list):  # older sklearn-style binary output
            values = values[1]
        if values.ndim == 3:
            values = values[..., 1]
        return values

    def sensor_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        sv = self.feature_shap(X)
        return pd.DataFrame({s: sv[:, self.sensor_of == s].sum(axis=1) for s in SENSORS},
                            index=X.index)

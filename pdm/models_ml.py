"""Classical ML models: Logistic Regression, Decision Tree, Random Forest, XGBoost."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import SEED


def build_ml_model(name: str, y_train: np.ndarray):
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                      class_weight="balanced", random_state=SEED)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=10,
                                      class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=SEED)
    if name == "xgboost":
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=pos_weight, eval_metric="aucpr",
                             tree_method="hist", n_jobs=-1, random_state=SEED)
    raise ValueError(f"Unknown ML model: {name}")

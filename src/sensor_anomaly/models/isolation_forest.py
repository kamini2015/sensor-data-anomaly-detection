"""Isolation Forest wrapper for training and scoring anomalies."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from sensor_anomaly.config import ModelConfig


class IsolationForestAnomalyDetector:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = IsolationForest(
            contamination=config.contamination,
            n_estimators=config.n_estimators,
            random_state=config.random_state,
        )

    def fit(self, x: pd.DataFrame) -> None:
        self.model.fit(x)

    def score(self, x: pd.DataFrame) -> pd.Series:
        # Higher score means more abnormal.
        raw = self.model.decision_function(x)
        return pd.Series(-raw, index=x.index, name="anomaly_score")

    def predict(self, x: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(x)
        return pd.Series((preds == -1).astype(int), index=x.index, name="is_anomaly")

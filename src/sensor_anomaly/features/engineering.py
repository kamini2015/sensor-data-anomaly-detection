"""Feature engineering for sensor anomaly detection."""

from __future__ import annotations

import pandas as pd

from sensor_anomaly.config import FeatureConfig
from sensor_anomaly.data.generate_synthetic import FEATURE_COLUMNS


def build_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    features = df.copy()

    for col in FEATURE_COLUMNS:
        features[f"{col}_diff_1"] = features[col].diff().fillna(0.0)
        features[f"{col}_roll_mean"] = (
            features[col].rolling(config.rolling_window, min_periods=1).mean()
        )
        features[f"{col}_roll_std"] = (
            features[col].rolling(config.rolling_window, min_periods=1).std().fillna(0.0)
        )

    return features


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    excluded = {"timestamp", "is_injected_anomaly", "anomaly_score", "is_anomaly"}
    cols = [c for c in df.columns if c not in excluded]
    return df[cols]

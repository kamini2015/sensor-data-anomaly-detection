"""Preprocessing utilities for sensor data."""

from __future__ import annotations

import pandas as pd

from sensor_anomaly.config import PreprocessingConfig
from sensor_anomaly.data.generate_synthetic import FEATURE_COLUMNS


def preprocess_sensor_data(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    required = {"timestamp", *FEATURE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = df.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"])
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)

    for col in FEATURE_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned[col] = cleaned[col].interpolate(method="linear").ffill().bfill()

        q_low = cleaned[col].quantile(config.clip_lower_quantile)
        q_high = cleaned[col].quantile(config.clip_upper_quantile)
        cleaned[col] = cleaned[col].clip(lower=q_low, upper=q_high)

    return cleaned

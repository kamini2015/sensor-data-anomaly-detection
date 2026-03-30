"""Synthetic time-series dataset generator for sensor telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sensor_anomaly.config import SyntheticDataConfig


FEATURE_COLUMNS = ["pressure", "temperature", "flow"]


def generate_synthetic_sensor_data(config: SyntheticDataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)
    ts = pd.date_range("2025-01-01", periods=config.n_samples, freq=config.freq)

    t = np.arange(config.n_samples)
    pressure = 100 + 8 * np.sin(2 * np.pi * t / 150) + rng.normal(0, 1.2, config.n_samples)
    temperature = 60 + 5 * np.sin(2 * np.pi * t / 250 + 0.5) + rng.normal(0, 0.8, config.n_samples)
    flow = 220 + 18 * np.cos(2 * np.pi * t / 180) + rng.normal(0, 2.0, config.n_samples)

    data = pd.DataFrame(
        {
            "timestamp": ts,
            "pressure": pressure,
            "temperature": temperature,
            "flow": flow,
            "is_injected_anomaly": 0,
        }
    )

    n_anomalies = max(1, int(config.n_samples * config.anomaly_fraction))
    anomaly_idx = rng.choice(config.n_samples, size=n_anomalies, replace=False)

    data.loc[anomaly_idx, "pressure"] += rng.normal(35, 6, size=n_anomalies)
    data.loc[anomaly_idx, "temperature"] -= rng.normal(18, 4, size=n_anomalies)
    data.loc[anomaly_idx, "flow"] += rng.normal(45, 8, size=n_anomalies)
    data.loc[anomaly_idx, "is_injected_anomaly"] = 1

    return data

"""Configuration objects for the anomaly detection pipeline."""

from dataclasses import dataclass


@dataclass(slots=True)
class SyntheticDataConfig:
    n_samples: int = 1500
    freq: str = "min"
    random_state: int = 42
    anomaly_fraction: float = 0.03


@dataclass(slots=True)
class PreprocessingConfig:
    clip_lower_quantile: float = 0.005
    clip_upper_quantile: float = 0.995


@dataclass(slots=True)
class FeatureConfig:
    rolling_window: int = 10


@dataclass(slots=True)
class ModelConfig:
    contamination: float = 0.03
    n_estimators: int = 200
    random_state: int = 42

"""End-to-end orchestration of anomaly detection workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sensor_anomaly.config import FeatureConfig, ModelConfig, PreprocessingConfig, SyntheticDataConfig
from sensor_anomaly.data.generate_synthetic import generate_synthetic_sensor_data
from sensor_anomaly.data.preprocessing import preprocess_sensor_data
from sensor_anomaly.features.engineering import build_features, feature_matrix
from sensor_anomaly.models.isolation_forest import IsolationForestAnomalyDetector

logger = logging.getLogger(__name__)


def run_pipeline(
    synthetic_cfg: SyntheticDataConfig,
    preprocessing_cfg: PreprocessingConfig,
    feature_cfg: FeatureConfig,
    model_cfg: ModelConfig,
) -> pd.DataFrame:
    logger.info("Generating synthetic sensor dataset")
    df = generate_synthetic_sensor_data(synthetic_cfg)

    logger.info("Preprocessing sensor dataset")
    clean = preprocess_sensor_data(df, preprocessing_cfg)

    logger.info("Engineering features")
    featured = build_features(clean, feature_cfg)

    logger.info("Training Isolation Forest and inferring anomalies")
    x = feature_matrix(featured)
    detector = IsolationForestAnomalyDetector(model_cfg)
    detector.fit(x)

    featured["anomaly_score"] = detector.score(x)
    featured["is_anomaly"] = detector.predict(x)

    return featured


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Saved file: %s", output)
    return output

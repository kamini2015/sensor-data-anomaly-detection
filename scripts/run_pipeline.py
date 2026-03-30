"""CLI script to execute the full anomaly detection pipeline."""

from __future__ import annotations

import argparse

from sensor_anomaly.config import FeatureConfig, ModelConfig, PreprocessingConfig, SyntheticDataConfig
from sensor_anomaly.pipeline import run_pipeline, save_dataframe
from sensor_anomaly.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sensor anomaly detection pipeline")
    parser.add_argument(
        "--output",
        default="data/sample/synthetic_sensor_data_with_anomalies.csv",
        help="Destination CSV for scored data",
    )
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--anomaly-fraction", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    synthetic_cfg = SyntheticDataConfig(
        n_samples=args.n_samples,
        anomaly_fraction=args.anomaly_fraction,
    )

    result = run_pipeline(
        synthetic_cfg=synthetic_cfg,
        preprocessing_cfg=PreprocessingConfig(),
        feature_cfg=FeatureConfig(),
        model_cfg=ModelConfig(contamination=args.anomaly_fraction),
    )
    save_dataframe(result, args.output)


if __name__ == "__main__":
    main()

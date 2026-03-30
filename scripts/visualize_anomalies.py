"""CLI script for plotting anomaly overlays on sensor time-series data."""

from __future__ import annotations

import argparse

import pandas as pd

from sensor_anomaly.utils.logging_config import configure_logging
from sensor_anomaly.visualization.plot import plot_time_series_anomalies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize time-series anomalies")
    parser.add_argument(
        "--input",
        default="data/sample/synthetic_sensor_data_with_anomalies.csv",
        help="Input CSV containing is_anomaly column",
    )
    parser.add_argument(
        "--output",
        default="data/sample/anomaly_plot.png",
        help="Output PNG path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    plot_time_series_anomalies(df, args.output)


if __name__ == "__main__":
    main()

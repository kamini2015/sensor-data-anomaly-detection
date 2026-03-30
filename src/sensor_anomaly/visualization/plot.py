"""Plotting utilities for anomaly overlays on sensor time-series."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_time_series_anomalies(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    sensors = ["pressure", "temperature", "flow"]

    for ax, sensor in zip(axes, sensors):
        ax.plot(df["timestamp"], df[sensor], label=sensor, linewidth=1.2)
        anomalies = df[df["is_anomaly"] == 1]
        ax.scatter(
            anomalies["timestamp"],
            anomalies[sensor],
            color="red",
            s=18,
            label="detected anomaly",
        )
        ax.set_ylabel(sensor)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("timestamp")
    fig.suptitle("Sensor Time-Series with Isolation Forest Anomalies")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

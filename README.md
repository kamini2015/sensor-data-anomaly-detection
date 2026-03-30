# Sensor Data Anomaly Detection Pipeline

A modular, production-style Python project for detecting anomalies in sensor time-series data.

## Features
- Synthetic dataset generation for pressure, temperature, and flow sensors
- Data preprocessing (sorting, de-duplication, missing value imputation, clipping)
- Feature engineering (rolling statistics + rate-of-change)
- Isolation Forest model training and scoring
- Time-series anomaly visualization

## Project Structure

```text
.
├── pyproject.toml
├── README.md
├── scripts
│   ├── run_pipeline.py
│   └── visualize_anomalies.py
└── src/sensor_anomaly
    ├── config.py
    ├── pipeline.py
    ├── data
    │   ├── generate_synthetic.py
    │   └── preprocessing.py
    ├── features
    │   └── engineering.py
    ├── models
    │   └── isolation_forest.py
    ├── utils
    │   └── logging_config.py
    └── visualization
        └── plot.py
```

## Quick Start

```bash
python -m pip install -e .
python scripts/run_pipeline.py --output data/sample/synthetic_sensor_data.csv
python scripts/visualize_anomalies.py --input data/sample/synthetic_sensor_data_with_anomalies.csv
```

## Notes
- `run_pipeline.py` creates synthetic data, preprocesses it, engineers features, trains/scoring with Isolation Forest, and exports results.
- `visualize_anomalies.py` reads scored output and generates a PNG plot.

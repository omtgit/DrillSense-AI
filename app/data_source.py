"""Where the app gets its sensor data from.

DATA_SOURCE=local (default) reads CSV/parquet from LOCAL_DATA_PATH, or data/sample/ when it is unset.
DATA_SOURCE=bigquery reads drillsense.sensor_data; it needs google-cloud credentials and is optional.
Clients are created inside functions, so importing this module never needs credentials.
"""
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_PATH = REPO_ROOT / "data" / "sample"

COLUMNS = [
    "timestamp", "well_id", "pressure_psi", "temperature_c", "flow_rate_bpd",
    "vibration", "gas_ratio", "pump_rpm", "torque", "rop", "anomaly_flag",
    "severity", "recommended_response", "health_score", "predicted_anomaly",
    "predicted_risk_score",
]


def data_source_name():
    name = os.getenv("DATA_SOURCE", "local").strip().lower() or "local"
    if name not in ("local", "bigquery"):
        raise ValueError(f"DATA_SOURCE must be 'local' or 'bigquery', got {name!r}")
    return name


def _read_file(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_local(path=None):
    path = Path(path or os.getenv("LOCAL_DATA_PATH") or DEFAULT_LOCAL_PATH).expanduser()
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix in (".csv", ".parquet"))
        if not files:
            raise FileNotFoundError(f"No .csv or .parquet files in {path}")
        df = pd.concat([_read_file(p) for p in files], ignore_index=True)
    elif path.is_file():
        df = _read_file(path)
    else:
        raise FileNotFoundError(f"LOCAL_DATA_PATH does not exist: {path}")
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df[COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_bigquery():
    from google.cloud import bigquery  # imported here so local mode needs no GCP setup

    client = bigquery.Client()
    table = os.getenv("BIGQUERY_TABLE") or f"{client.project}.drillsense.sensor_data"
    query = f"SELECT {', '.join(COLUMNS)} FROM `{table}`"
    return client.query(query).to_dataframe()


def load_sensor_data():
    if data_source_name() == "bigquery":
        return load_bigquery()
    return load_local()

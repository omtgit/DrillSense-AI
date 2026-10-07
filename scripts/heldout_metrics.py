"""
Compute detection precision/recall for the XGBoost classifier on WELLS IT NEVER SAW.

Run AFTER notebooks/01 -> 02 -> 03:

    python scripts/heldout_metrics.py

Writes docs/heldout_metrics.json, which the dashboard displays. The dashboard's own
`predicted_anomaly` column is produced by a model trained on a random 80/20 row split and
applied to every row, so it must not be used to score the model; this script is the honest
number. Split: 10 of 50 wells held out (seeded shuffle, same as scripts/audit_baseline.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "drillsense_processed_data.csv"
OUT = ROOT / "docs" / "heldout_metrics.json"

FEATS = [
    "pressure_psi", "temperature_c", "flow_rate_bpd", "vibration", "gas_ratio",
    "mud_density", "pump_rpm", "torque", "rop",
    "pressure_change", "temperature_change", "flow_change", "vibration_change",
    "rolling_pressure_mean", "rolling_temperature_mean",
    "rolling_flow_mean", "rolling_vibration_mean",
]
LABELS = {"Normal": 0, "Leak Risk": 1, "Kick Risk": 2,
          "Pump Failure": 3, "Sensor Drift": 4, "High Vibration": 5}


def main():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    X = df[FEATS].fillna(0)
    y = df["anomaly_type"].map(LABELS)

    wells = sorted(df["well_id"].unique())
    np.random.RandomState(0).shuffle(wells)
    held_out = sorted(wells[:10])
    te = df["well_id"].isin(held_out)

    model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                          objective="multi:softmax", num_class=6, random_state=42)
    model.fit(X[~te], y[~te])
    pred = model.predict(X[te])

    true_flag = (y[te] != 0).astype(int)
    pred_flag = (pred != 0).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        true_flag, pred_flag, average="binary", zero_division=0)

    per_type = {}
    for name, code in LABELS.items():
        if code == 0:
            continue
        m = (y[te] == code).values
        per_type[name] = {
            "rows": int(m.sum()),
            "recall": round(float((pred[m] != 0).mean()), 4) if m.any() else None,
        }

    out = {
        "split": "held-out wells",
        "train_wells": int(len(wells) - len(held_out)),
        "test_wells": held_out,
        "test_rows": int(te.sum()),
        "test_anomaly_rows": int(true_flag.sum()),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "per_type_recall": per_type,
        "note": "Synthetic data with very large injected shifts; not evidence of field performance.",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""
DrillSense AI - evaluation audit.

Run AFTER notebooks/01 -> 02 -> 03 have produced data/processed/drillsense_processed_data.csv.

Purpose: show how much of the headline accuracy is real by comparing the XGBoost
classifier against (a) stricter splits, (b) a no-ML baseline, (c) the Isolation Forest.
Re-run after every change to the generator or models and keep the numbers in docs/EVAL.md.

    python scripts/audit_baseline.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "drillsense_processed_data.csv"

FEATS = [
    "pressure_psi", "temperature_c", "flow_rate_bpd", "vibration", "gas_ratio",
    "mud_density", "pump_rpm", "torque", "rop",
    "pressure_change", "temperature_change", "flow_change", "vibration_change",
    "rolling_pressure_mean", "rolling_temperature_mean",
    "rolling_flow_mean", "rolling_vibration_mean",
]
RAW = ["pressure_psi", "temperature_c", "flow_rate_bpd", "vibration",
       "gas_ratio", "pump_rpm", "torque"]
LABELS = {"Normal": 0, "Leak Risk": 1, "Kick Risk": 2,
          "Pump Failure": 3, "Sensor Drift": 4, "High Vibration": 5}


def xgb():
    return XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                         objective="multi:softmax", num_class=6, random_state=42)


def anomaly_recall(y_true, y_pred):
    return precision_recall_fscore_support(
        y_true, y_pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)[1]


def main():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    X = df[FEATS].fillna(0)
    y = df["anomaly_type"].map(LABELS)
    flag = df["anomaly_flag"].values

    print("== XGBoost under different splits ==")
    # Held-out wells: the model never sees the test wells.
    wells = sorted(df.well_id.unique())
    np.random.RandomState(0).shuffle(wells)
    te = df.well_id.isin(set(wells[:10]))
    m = xgb().fit(X[~te], y[~te])
    p = m.predict(X[te])
    print(f"held-out WELLS : acc={accuracy_score(y[te], p):.4f}  anomaly macro-recall={anomaly_recall(y[te], p):.3f}")

    # Held-out time: train on the first 80% of the timeline, test on the last 20%.
    tt = df.timestamp > df.timestamp.quantile(0.8)
    m = xgb().fit(X[~tt], y[~tt])
    p = m.predict(X[tt])
    print(f"held-out TIME  : acc={accuracy_score(y[tt], p):.4f}  anomaly macro-recall={anomaly_recall(y[tt], p):.3f}")

    print("\n== No-ML baseline: per-well robust z-score on raw sensors ==")
    z = df.groupby("well_id")[RAW].transform(
        lambda s: (s - s.median()) / (1.4826 * (s - s.median()).abs().median() + 1e-9)).abs()
    score = z.max(axis=1)
    for thr in (4, 5, 6):
        pred = (score > thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(flag, pred, average="binary", zero_division=0)
        print(f"z > {thr}: precision={pr:.3f} recall={rc:.3f} f1={f1:.3f}")
    print(f"AUROC={roc_auc_score(flag, score):.4f}  PR-AUC={average_precision_score(flag, score):.4f}")

    print("\n== Isolation Forest vs injected labels ==")
    iso = IsolationForest(n_estimators=100, contamination=0.015, random_state=42).fit(X)
    pred = (iso.predict(X) == -1).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(flag, pred, average="binary", zero_division=0)
    auc = roc_auc_score(flag, -iso.score_samples(X))
    print(f"precision={pr:.3f} recall={rc:.3f} f1={f1:.3f} AUROC={auc:.4f}")

    print("\n== How big are the injected shifts? (mean shift in noise-sigma units) ==")
    base = df[df.anomaly_flag == 0].groupby("well_id")[RAW].agg(["mean", "std"])
    for t in [k for k in LABELS if k != "Normal"]:
        sub = df[df.anomaly_type == t]
        out = {}
        for c in RAW:
            mu = base[(c, "mean")].reindex(sub.well_id).values
            sd = base[(c, "std")].reindex(sub.well_id).values
            out[c] = float(np.mean((sub[c].values - mu) / sd))
        top = sorted(out.items(), key=lambda kv: -abs(kv[1]))[:2]
        print(f"{t:15s}", ", ".join(f"{k}: {v:+.1f} sigma" for k, v in top))


if __name__ == "__main__":
    main()

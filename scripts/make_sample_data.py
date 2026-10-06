"""
Cut a small, committable sample from the full processed dataset so the repo can be
explored without running notebooks 01-03 or having cloud access.

Run AFTER notebooks/01 -> 02 -> 03:

    python scripts/make_sample_data.py

Output: data/sample/drillsense_sample.csv (3 wells x first 3 days, 5-min rows).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "drillsense_processed_data.csv"
OUT = ROOT / "data" / "sample" / "drillsense_sample.csv"

N_WELLS = 3
N_DAYS = 3


def main():
    df = pd.read_csv(SRC, parse_dates=["timestamp"])
    wells = sorted(df["well_id"].unique())[:N_WELLS]
    end = df["timestamp"].min() + pd.Timedelta(days=N_DAYS)
    sample = df[df["well_id"].isin(wells) & (df["timestamp"] < end)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT, index=False)
    print(f"{len(sample)} rows, {sample['anomaly_flag'].sum()} injected anomaly rows -> {OUT}")


if __name__ == "__main__":
    main()

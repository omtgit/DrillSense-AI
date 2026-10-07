"""Generate seeded synthetic drilling telemetry (1-minute rows) and its event table.

    python scripts/generate_drilling_data.py                 # data/raw/drilling/ (gitignored)
    python scripts/generate_drilling_data.py --sample        # data/sample/drilling/ (committed)

Writes drilling_data.csv (one row per minute per well) and drilling_events.csv (the injected
events: type, start row, ramp/hold/release minutes, scale, drifting channel).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generators.drilling import generate_dataset  # noqa: E402

# Sample: small, but with all six anomaly types and at least one overlap (checked in the tests).
SAMPLE = dict(wells=3, days=1.0, seed=11, events_per_day=4.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wells", type=int, default=10)
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--events-per-day", type=float, default=0.8)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "drilling")
    ap.add_argument("--sample", action="store_true", help="write the small committed sample")
    a = ap.parse_args()
    if a.sample:
        a.wells, a.days, a.seed, a.events_per_day = (SAMPLE[k] for k in ("wells", "days", "seed", "events_per_day"))
        a.out = ROOT / "data" / "sample" / "drilling"
    df, events = generate_dataset(a.wells, a.days, a.seed, a.events_per_day)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "drilling_data.csv", index=False)
    events.to_csv(a.out / "drilling_events.csv", index=False)
    print(f"{len(df):,} rows, {len(events)} events, {df['anomaly_flag'].mean():.1%} rows inside an event -> {a.out}")


if __name__ == "__main__":
    main()

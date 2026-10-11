"""Headless runner for DrillSense AI streaming engine.

Usage:
    python -m engine.run --source replay --wells 3 --speed 5.0 --max-ticks 100
    python -m engine.run --source csv --file data/sample/drilling/drilling_data.csv --mapping config/csv_mapping.yaml

Strictly READ-ONLY with respect to data sources.
Outputs alert state machine changes directly to SQLite (data/runtime/drill_runtime.db).
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.detector import train_detector
from app.engine.engine import LiveEngine
from app.engine.quality import DataQualityMonitor
from app.engine.storage import RuntimeStorage
from app.ingest.sources import CsvSource, ReplaySource
from generators.drilling import generate_dataset
from profiles import get_profile


def run_headless(args):
    print("=" * 60)
    print("DrillSense AI - Headless Live Engine Runner (READ-ONLY)")
    print("=" * 60)

    profile = get_profile("drilling")
    print("Initializing detector on background baseline wells...")
    det = train_detector(profile, rounds=50)

    storage = RuntimeStorage()
    quality = DataQualityMonitor()
    engine = LiveEngine(profile_name="drilling", detector_instance=det, storage=storage, quality_monitor=quality)

    if args.source == "replay":
        print(f"Generating synthetic telemetry stream (wells={args.wells}, days=1.0)...")
        df_raw, _ = generate_dataset(n_wells=args.wells, days=1.0, seed=42)
        source = ReplaySource(df_raw, speed_multiplier=args.speed, loop=False)
    elif args.source == "csv":
        print(f"Loading external CSV source from {args.file}...")
        source = CsvSource(args.file, args.mapping)
    else:
        raise ValueError(f"Unknown source type: {args.source}")

    print(f"Streaming telemetry rows (wells={source.well_ids}). Press Ctrl+C to terminate.")
    tick_count = 0
    t0 = time.time()

    try:
        for row in source.stream_rows():
            res = engine.process_row(row)
            tick_count += 1

            if res.is_alarm:
                print(f"[{res.timestamp}] [ALARM] Well {res.well_id} -> {res.predicted_anomaly} ({res.predicted_severity}), Risk: {res.risk_score:.1f}%")
            elif tick_count % 50 == 0:
                print(f"[{res.timestamp}] [OK] Ticks: {tick_count}, Well: {res.well_id}, State: {res.rig_state}, Risk: {res.risk_score:.1f}%")

            if args.max_ticks and tick_count >= args.max_ticks:
                print(f"Reached max ticks limit ({args.max_ticks}). Stopping.")
                break
    except KeyboardInterrupt:
        print("\nHeadless runner interrupted by operator.")

    elapsed = time.time() - t0
    print(f"Finished processing {tick_count} rows in {elapsed:.2f}s ({tick_count / max(0.001, elapsed):.1f} rows/s).")
    print(f"Runtime database: {storage.db_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["replay", "csv"], default="replay")
    parser.add_argument("--wells", type=int, default=2)
    parser.add_argument("--speed", type=float, default=0.0, help="Speed multiplier (0 = max throughput)")
    parser.add_argument("--file", type=str, default="", help="Path to CSV file (if source=csv)")
    parser.add_argument("--mapping", type=str, default="config/csv_mapping.yaml", help="Path to YAML mapping file")
    parser.add_argument("--max-ticks", type=int, default=0, help="Max ticks to process (0 = indefinite)")
    args = parser.parse_args()
    run_headless(args)


if __name__ == "__main__":
    main()

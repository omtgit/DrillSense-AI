"""
CPU vs GPU benchmark: read the DrillSense CSV with pandas and with RAPIDS cuDF.

Requires an NVIDIA GPU and RAPIDS (`cudf`); the Streamlit app does not need either.
Original published numbers (Colab, Tesla T4) are in docs/BENCHMARKS.md.

    python benchmarks/gpu_benchmark.py                      # CSV read only
    python benchmarks/gpu_benchmark.py --xgboost            # also XGBoost CPU vs device="cuda"
    python benchmarks/gpu_benchmark.py --csv path/to.csv --repeats 5
"""
import argparse
import statistics
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "raw" / "drillsense_sensor_data.csv"


def timed(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def report(name, times):
    print(f"{name:22s} median {statistics.median(times):.4f}s  "
          f"min {min(times):.4f}s  max {max(times):.4f}s  (n={len(times)})")
    return statistics.median(times)


def bench_csv(path, repeats):
    print(f"\n== CSV read: {path} ({path.stat().st_size / 1e6:.0f} MB) ==")
    cpu = report("pandas (CPU)", timed(lambda: pd.read_csv(path), repeats))
    try:
        import cudf
    except ImportError:
        print("cudf not installed; skipping GPU read.")
        return
    cudf.read_csv(path)  # warm-up: CUDA context and JIT, excluded from timing
    gpu = report("cuDF (GPU)", timed(lambda: cudf.read_csv(path), repeats))
    print(f"speed-up: {cpu / gpu:.2f}x")


def bench_xgboost(path, repeats):
    from xgboost import XGBClassifier

    print("\n== XGBoost training (150 trees, depth 6) ==")
    df = pd.read_csv(path)
    labels = {"Normal": 0, "Leak Risk": 1, "Kick Risk": 2,
              "Pump Failure": 3, "Sensor Drift": 4, "High Vibration": 5}
    feats = ["pressure_psi", "temperature_c", "flow_rate_bpd", "vibration",
             "gas_ratio", "mud_density", "pump_rpm", "torque", "rop"]
    X, y = df[feats], df["anomaly_type"].map(labels)

    def fit(device):
        XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                      tree_method="hist", device=device, random_state=42).fit(X, y)

    cpu = report("XGBoost cpu", timed(lambda: fit("cpu"), repeats))
    try:
        fit("cuda")  # warm-up
    except Exception as exc:
        print(f"CUDA XGBoost unavailable ({type(exc).__name__}); skipping.")
        return
    gpu = report("XGBoost cuda", timed(lambda: fit("cuda"), repeats))
    print(f"speed-up: {cpu / gpu:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--xgboost", action="store_true")
    args = ap.parse_args()
    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found; run notebooks/01_generate_synthetic_data.py first.")
    bench_csv(args.csv, args.repeats)
    if args.xgboost:
        bench_xgboost(args.csv, args.repeats)


if __name__ == "__main__":
    main()

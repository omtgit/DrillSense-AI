# CPU vs GPU benchmark

## Original result (Google Colab, NVIDIA Tesla T4)

Recorded once for the original competition entry, which asked for a CPU-vs-GPU comparison.

| Item | Value |
|---|---|
| What was measured | Time to **read the CSV file** only |
| File | 432,000 rows, 107 MB (the synthetic DrillSense dataset) |
| CPU | pandas, 2.0563 s |
| GPU | RAPIDS cuDF on Tesla T4, 0.7807 s |
| Speed-up | 2.63x |

Caveats, stated plainly:
- It is a single run. No repeats, variance or warm-up handling were recorded.
- It measures file I/O and parsing, not feature engineering, model training or inference.
- The Streamlit app runs on CPU and never uses the GPU.
- The original notebook is not in the repo, so the figures cannot be re-derived from it.

## Reproducing

`benchmarks/gpu_benchmark.py` repeats the measurement (median of N runs, GPU warm-up excluded)
and can optionally compare XGBoost training on CPU vs `device="cuda"`. It needs an NVIDIA GPU
and RAPIDS cuDF; it has not been run in this repo's CI or cloud sessions, which have no GPU.
Add your numbers below when you run it.

```
python benchmarks/gpu_benchmark.py --xgboost
```

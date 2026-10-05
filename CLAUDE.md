# DrillSense AI - project context for Claude Code

Owner: Om (B.Tech Petroleum Engineering, RGIPT). Public repo, MIT licence. Goal: turn this from a
competition submission into a credible, free-to-run portfolio project with honest evaluation.

## Hard rules
- Budget is **free**. No paid services, no new cloud resources. GCP credits are ending.
- Never read, print, edit or commit `.env`, API keys or anything under `models/` or `data/`.
- Keep Streamlit. Do not migrate the frontend to another framework.
- Ask before deleting files or changing the public README claims. Small commits, one concern each.
- After any change to the generator or models, run `python scripts/audit_baseline.py` and record the numbers in `docs/EVAL.md`.
- Prefer honest reporting over impressive numbers. If a result is weak, say so in the docs.

## How the pipeline works (verified by reading and running it)
1. `notebooks/01_generate_synthetic_data.py`: 50 wells x 30 days x 5-min = 432,000 rows. Each well is a constant baseline + a daily sine + Gaussian noise. 1.5% of points are injected as 6-row anomaly windows (5 types). Severity, priority and `recommended_response` are looked up from the anomaly type.
2. `notebooks/02_feature_engineering.py`: diffs and 12-step rolling means per well (past-only, no look-ahead). `health_score` is derived from the TRUE severity.
3. `notebooks/03_anomaly_detection.py`: Isolation Forest (saved, never used downstream) and a 6-class XGBoost trained with a random 80/20 row split. Predictions are then made on ALL rows, including the training rows.
4. Data is uploaded to BigQuery (`drillsense.sensor_data`) by hand; there is no upload script.
5. `app/main.py` (Streamlit) reads BigQuery at import time. `app/gemini_utils.py` calls Gemini at `gemini-2.5-flash`.

## Audit findings (reproduced locally with scripts/audit_baseline.py)
- XGBoost accuracy: 99.96% (random row split), 98.2% (held-out wells), 99.95% (held-out time).
- A no-ML per-well robust z-score (threshold 4) gets precision 0.98 / recall 0.98, AUROC 0.9998. The ML adds almost nothing on this data.
- Injected shifts are huge versus noise: kick gas +40 sigma, pump rpm -25 sigma, vibration +29 sigma. Sensor Drift is the weakest at about +6.5 sigma. The benchmark is too easy to prove anything.
- Isolation Forest vs injected labels: precision 0.50, recall 0.52, AUROC 0.91, even with `contamination` set to the true rate. It is not used by the app.
- Random row split leaks: neighbouring rows of the same 6-row window land in both train and test.

## App honesty problems to fix
- "Detected Anomalies" on the dashboard sums the ground-truth `anomaly_flag`, not model output. `severity`, `health_score` and `recommended_response` are also ground-truth lookups.
- The AI Decision Center shows "Google Gemini successfully analyzed wells." unconditionally.
- GPU Performance page is hard-coded numbers from a one-off Colab T4 CSV-read benchmark (2.06 s vs 0.78 s), done because the original competition asked for a CPU-vs-GPU comparison. It is legitimate evidence, but the app itself runs on CPU and the page does not say what was measured. DECISION: keep the page, relabel it honestly (see Phase 1).
- "Highest Risk Well" takes `iloc[0]` after sorting, but 1,277 rows tie at the max score, so the pick is arbitrary.
- Domain framing is mixed: `flow_rate_bpd`, `gas_ratio`, `pump_rpm` are production-style variables; `torque`, `rop`, `mud_density` are drilling. README says drilling anomalies; the Gemini prompt says "production engineer". There is no depth, WOB, standpipe pressure, pit volume, or flow-in/flow-out. The kick signature (pressure up + gas up) should be checked against a drilling source; kicks are normally caught via pit gain and flow-out increase.

## Engineering problems
- `main.py` creates a BigQuery client at import, so the app cannot run without GCP credentials. `app/data_loader.py` is unused, duplicates the query and hard-codes the project id.
- `gemini_utils.py` builds the Gemini client at import (likely fails without a key) and hard-codes the model. Gemini 2.5 Flash shutdown dates are inconsistent across Google pages, so make the model an env var.
- `requirements.txt` is UTF-16 with CRLF and lists `google-cloud-bigquery` twice (pinned and unpinned). Convert to UTF-8.
- No tests, no run instructions, no sample data, and `data/` + `models/` are gitignored, so nobody can reproduce the app from the repo.
- Good: no secret patterns found in the 8-commit git history; `.env` is gitignored.

## Task plan (do in order; stop after each phase and summarise)
**Phase 0 - hygiene (no behaviour change):** UTF-8 requirements without duplicates; README run steps; commit a small `data/sample/` (a few wells, a few days) so the app runs without cloud access.

**Phase 1 - honest dashboard:** label ground truth as "Injected anomalies (ground truth)" and show model detections separately; show detection precision/recall computed on a held-out split; remove the unconditional Gemini success message; keep the GPU page but retitle it "CPU vs GPU benchmark (Colab, Tesla T4, offline)", state clearly that it measures CSV read time of the 432,000-row file, change the subtitle from "GPU-Accelerated" to something accurate such as "GPU-benchmarked", and add the benchmark as a reproducible `benchmarks/gpu_benchmark.py` (pandas vs cuDF; optionally XGBoost `device="cuda"` vs CPU) with the original numbers kept in `docs/BENCHMARKS.md`. Do not run GPU code in the app; fix the tie-break for highest-risk well (score, then most recent, then well).

**Phase 2 - data source abstraction:** `DATA_SOURCE=local|bigquery` (default local, reading parquet/CSV). Create clients lazily inside functions. Delete the duplicate loader after confirming with the owner.

**Phase 3 - report generator:** deterministic template report built from the structured finding (always works, no API). Optional "bring your own key" box in the sidebar, key kept only in `st.session_state`, never logged or persisted. Model name from env var, default to a current Flash model. The LLM may only rephrase the structured facts it is given.

**Phase 4a - domain profiles:** introduce `profiles/drilling.py` and `profiles/production.py`, each defining channels, anomaly types, physics features, severity rules and report vocabulary. The engine (loader, features, detectors, evaluation, report) stays generic. Build the DRILLING profile fully first (depth, WOB, RPM, torque, ROP, standpipe pressure, flow in/out, pit volume, mud weight, gas). The production profile can stay a stub that maps the current variables until the 3W adapter is built.

**Phase 4 - credible evaluation:** harder generator (gradual ramps, correlated channels, baseline drift between wells, overlapping and lower-magnitude events, drilling-realistic channels); well-held-out and time-held-out splits; event-level metrics (events caught, false alarms per day, detection lead time); no-ML and Isolation Forest baselines in the same table; SHAP for XGBoost. Write everything to `docs/EVAL.md`.

**Phase 5 - real data:** adapter for Volve-derived or DataDrill drilling data into the drilling profile, with required attribution (Equinor Open Data Licence for Volve). Run the downloads locally (see cloud-session notes), then commit only a small sample.

## Decisions made by the owner (October 2026)
1. Scope: DRILLING is the primary profile (matches the project name, the owner's degree, and the open Volve/DataDrill data). Production monitoring stays supported through the profile layer, built later.
2. BigQuery: optional connector only. The default data source must work with no cloud account. (The GCP free trial ends and resources stop unless the owner upgrades to a paid account, so the public demo must not depend on it.)
3. GPU benchmark: keep, relabelled and made reproducible (Phase 1).

## Working in Claude Code cloud sessions
- Cloud sessions start from a fresh clone of the GitHub repo, so this file and `scripts/audit_baseline.py` must be committed and pushed before the session starts.
- Default network access is limited to package registries and common dev domains. `pip install` from PyPI works; downloading Volve, DataDrill or other datasets may be blocked. Do Phase 5 data downloads locally, or ask the owner to widen the environment's network access.
- There are no GCP credentials in the cloud session and there must not be. Everything in Phases 0-4 must run on local/generated data.
- Work on a branch per phase and open a pull request; do not push to `main` directly.
- Credits are limited: do one phase per session, avoid printing large dataframes, and do not regenerate the 432,000-row dataset more than needed (it is seeded and deterministic).

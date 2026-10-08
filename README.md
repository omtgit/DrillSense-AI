# <img src="assets/logo.png" width="42"> DrillSense AI

### GPU-Accelerated Decision Intelligence for Oilfield Monitoring

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Cloud_Run-4285F4)
![BigQuery](https://img.shields.io/badge/BigQuery-Data_Warehouse-669DF6)
![Gemini](https://img.shields.io/badge/Gemini-Flash_Lite-8E75B2)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![RAPIDS](https://img.shields.io/badge/RAPIDS-cuDF-success)

---

## Dashboard

<p align="center">
<img src="assets/dashboard.png" width="900">
</p>

---

## Overview

DrillSense AI is a cloud-native, AI-powered oilfield monitoring platform that combines machine learning, GPU acceleration, Google Cloud, and Gemini Flash to detect drilling anomalies, assess operational risk, and generate engineering recommendations. 
The platform demonstrates how modern AI can support production engineers through intelligent monitoring, predictive analytics, and cloud-scale decision intelligenc

---

## Key Features

- Executive Dashboard for operational monitoring
- Interactive Well Explorer with sensor trend visualization
- Gemini-powered AI Engineering Reports
- Machine Learning anomaly detection using Isolation Forest
- XGBoost-based operational risk prediction
- GPU benchmarking with NVIDIA Tesla T4 and RAPIDS cuDF
- Cloud-native architecture using Google Cloud Run
- BigQuery-backed analytics for scalable data processing

---

## Technology Stack

| Layer | Technology |
|------|------------|
| Programming Language | Python |
| Frontend | Streamlit |
| Data Visualization | Plotly |
| Machine Learning | XGBoost, Isolation Forest |
| Generative AI | Gemini Flash |
| Cloud Platform | Google Cloud Platform (GCP) |
| Data Warehouse | BigQuery |
| Cloud Deployment | Cloud Run |
| Containerization | Docker |
| GPU Analytics | NVIDIA Tesla T4 + RAPIDS cuDF |
| Version Control | Git & GitHub |

---

## Workflow

```text
Sensor Data
      │
      ▼
Feature Engineering
      │
      ▼
Anomaly Detection
      │
      ▼
Risk Prediction
      │
      ▼
AI Decision Intelligence
      │
      ▼
Cloud Deployment
```

---

## Repository Structure

```
DrillSense-AI/
│
├── app/
│   ├── main.py
│   ├── data_source.py
│   ├── detector.py        (drilling detector, trained at start-up)
│   ├── drilling_pages.py
│   ├── report.py / report_ui.py
│   └── gemini_utils.py
│
├── assets/
│   ├── logo.png
│   └── dashboard.png
│
├── notebooks/
├── scripts/
├── data/sample/
│
├── Dockerfile
├── cloudbuild.yaml
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── .dockerignore
```

---

## Run locally

Requires **Python 3.12+** (the pinned `xgboost==3.3.0` does not install on 3.11).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Regenerate the synthetic data and train the models (seeded, deterministic)
python notebooks/01_generate_synthetic_data.py   # -> data/raw/        (432,000 rows)
python notebooks/02_feature_engineering.py       # -> data/processed/
python notebooks/03_anomaly_detection.py         # -> models/ + predictions

# Optional: check how much of the headline accuracy is real
python scripts/audit_baseline.py
```

A small committed sample (3 wells x 3 days, 2,592 rows) is in [`data/sample/`](data/sample/)
so you can inspect the data without running the pipeline.

### Run the app

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

No cloud account or API key is needed. By default the app reads the committed sample in
`data/sample/`.

| Variable | Default | Meaning |
|---|---|---|
| `DATA_SOURCE` | `local` | `local` reads CSV/parquet files; `bigquery` reads `<project>.drillsense.sensor_data` (optional, needs Google Cloud credentials) |
| `LOCAL_DATA_PATH` | `data/sample` | Production profile: a CSV/parquet file, or a directory of them, with the columns in `app/data_source.py` |
| `DRILLING_DATA_PATH` | `data/sample/drilling` | Drilling profile: a `drilling_data.csv` (or its directory), e.g. from `scripts/generate_drilling_data.py` |
| `BIGQUERY_TABLE` | `<project>.drillsense.sensor_data` | Override the BigQuery table (only with `DATA_SOURCE=bigquery`) |
| `GEMINI_API_KEY` | unset | Optional. Only used by "Rephrase with Gemini"; a key pasted in the sidebar takes precedence. The template report works without it |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Default model for rephrasing; also editable in the sidebar. Not verified against a live key; a 404 shows a "try another model name" message |

Run the tests with `pip install pytest && python -m pytest tests`.

To run on the full dataset after the pipeline above, set
`LOCAL_DATA_PATH=data/processed/drillsense_processed_data.csv` (written by notebook 03).

### Domain profiles and the drilling generator

Everything domain-specific (channels, anomaly types, physics features, severity rules, report
wording) lives in `profiles/`. `profiles/production.py` describes the original production-style
variables (unchanged); `profiles/drilling.py` describes rig drilling channels: depth, WOB, RPM,
torque, ROP, standpipe pressure, flow in/out, pit volume, mud weight and gas. The report and
feature code read the profile, so a new domain is a new profile module.

A seeded drilling generator (`generators/drilling.py`) simulates connections, formation changes
and six gradual anomalies (kick, lost circulation, stuck pipe, washout, pack-off, sensor drift)
with variable, sometimes small, magnitudes and some overlap:

```bash
python scripts/generate_drilling_data.py            # -> data/raw/drilling/ (10 wells x 3 days, gitignored)
python scripts/generate_drilling_data.py --sample   # -> data/sample/drilling/ (committed, 4,320 rows)
```

Every signature and assumption, with sources and **NEEDS REVIEW** flags, is in
[`docs/PHYSICS.md`](docs/PHYSICS.md). The physics is a plausible model written by the project
author with AI assistance, not field data, and has not been reviewed by a drilling engineer.

#### How the app uses the drilling profile

The sidebar has a **Domain profile** selector: *Drilling* (default) or *Production* (the original
dataset). For Drilling, the Executive Dashboard, Well Explorer and AI Decision Center read the
committed sample in `data/sample/drilling/` and show two things separately: **injected anomalies
(ground truth)** and **model detections**. Severity, recommended response and report wording for a
detection come from the profile's rules for the *predicted* class, not from ground-truth labels, and
reports say the rule table has not been reviewed by a drilling engineer.

**The detector is trained at app start-up, not shipped as a model file.** On first load the app
generates 8 training wells and 4 validation wells (seeds 1001 and 2002; the committed sample uses
seed 11, so the sample wells are never trained on), fits a 7-class XGBoost model, and caches it for
the life of the server process (`st.cache_resource`). Measured on a 4-core cloud container this takes
about 10 s including the first page render, well under the 30 s target. I chose this over a
committed model file because a pickled booster is an unreviewable binary tied to one `xgboost`
version, `models/` is gitignored on purpose, and the seeded run is deterministic. The cost is a
one-off wait on each server start. The dashboard's precision/recall come from the 4 validation
wells at a fixed rule (flag when P(Normal) < 0.5); recall is low (~0.16 row-level, ~57% of events
caught) because every event's ramp start is labelled anomalous while barely visible. The full
event-level comparison, with baselines, is on the Model Evaluation page and in
[`docs/EVAL.md`](docs/EVAL.md).

Only the *Drilling* profile is wired to the detector. *Production* still shows the original
precomputed predictions. `DATA_SOURCE=bigquery` applies to the Production profile only; Drilling
always reads local files (override the path with `DRILLING_DATA_PATH`).

---

## Future Enhancements

- Live streaming data
- Predictive maintenance scheduling
- Automated alert notifications
- Multi-well fleet monitoring
- Enhanced engineering dashboard
- Time-series forecasting

---

## License

This project is licensed under the MIT License.

---
Om Tripathi
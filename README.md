# <img src="assets/logo.png" width="42"> DrillSense AI

### GPU-Accelerated Decision Intelligence for Oilfield Monitoring

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Cloud_Run-4285F4)
![BigQuery](https://img.shields.io/badge/BigQuery-Data_Warehouse-669DF6)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-8E75B2)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![RAPIDS](https://img.shields.io/badge/RAPIDS-cuDF-success)

---

## Dashboard

<p align="center">
<img src="assets/dashboard.png" width="900">
</p>

---

## Overview

DrillSense AI is a cloud-native, AI-powered oilfield monitoring platform that combines machine learning, GPU acceleration, Google Cloud, and Gemini 2.5 Flash to detect drilling anomalies, assess operational risk, and generate engineering recommendations. 
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
| Generative AI | Gemini 2.5 Flash |
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
│   ├── gemini_utils.py
│   └── data_loader.py
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

> **Note:** the Streamlit app (`streamlit run app/main.py`) currently still reads from
> BigQuery and Gemini, so it needs Google Cloud credentials (`GEMINI_API_KEY` in `.env` for
> Gemini). A local data source is planned for a later phase.

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
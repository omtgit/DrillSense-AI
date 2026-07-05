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
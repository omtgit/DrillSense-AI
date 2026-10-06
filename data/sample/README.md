# Sample data

`drillsense_sample.csv` is a tiny slice of the **synthetic** DrillSense dataset: wells
`WELL-001` to `WELL-003`, first 3 days (2,592 rows at 5-minute spacing, 60 injected anomaly rows).

Regenerate it with `python scripts/make_sample_data.py` after running notebooks 01-03.

Caveats:
- Fully synthetic (a constant baseline per well + daily sine + noise, with injected anomaly windows). It is not real field data.
- `anomaly_type`, `anomaly_flag`, `severity`, `priority_score`, `recommended_response` and `health_score` are **ground truth** from the generator, not model output.
- `iso_prediction`, `predicted_anomaly` and `predicted_risk_score` come from models trained on a random 80/20 row split of the full dataset, so these sample rows are mostly training rows. Do not use them to judge model quality.

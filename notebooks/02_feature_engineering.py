import pandas as pd
from pathlib import Path

print("="*60)
print("DrillSense AI - Feature Engineering")
print("="*60)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "drillsense_sensor_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "drillsense_processed_data.csv"

df = pd.read_csv(INPUT_FILE)

print(f"Rows Loaded : {len(df):,}")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(["well_id", "timestamp"])

grouped = df.groupby("well_id")

# -------------------------
# Change Features
# -------------------------

df["pressure_change"] = grouped["pressure_psi"].diff()

df["temperature_change"] = grouped["temperature_c"].diff()

df["flow_change"] = grouped["flow_rate_bpd"].diff()

df["vibration_change"] = grouped["vibration"].diff()

# -------------------------
# Rolling Features
# -------------------------

df["rolling_pressure_mean"] = grouped["pressure_psi"].transform(
    lambda x: x.rolling(12, min_periods=1).mean()
)

df["rolling_temperature_mean"] = grouped["temperature_c"].transform(
    lambda x: x.rolling(12, min_periods=1).mean()
)

df["rolling_flow_mean"] = grouped["flow_rate_bpd"].transform(
    lambda x: x.rolling(12, min_periods=1).mean()
)

df["rolling_vibration_mean"] = grouped["vibration"].transform(
    lambda x: x.rolling(12, min_periods=1).mean()
)

# -------------------------
# Health Score
# -------------------------

df["health_score"] = 100

df.loc[df["severity"] == "Low", "health_score"] = 80

df.loc[df["severity"] == "Medium", "health_score"] = 60

df.loc[df["severity"] == "High", "health_score"] = 35

df.loc[df["severity"] == "Critical", "health_score"] = 10

df = df.fillna(0)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature Engineering Complete\n")

print(f"Output File : {OUTPUT_FILE}")

print(f"Rows : {len(df):,}")

print(f"Columns : {len(df.columns)}")

print(df.head())
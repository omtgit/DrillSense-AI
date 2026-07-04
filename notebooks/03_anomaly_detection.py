import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

print("=" * 60)
print("DrillSense AI - Anomaly Detection")
print("=" * 60)

# ----------------------------------------------------
# Paths
# ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "drillsense_processed_data.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ISO_MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_classifier.pkl"

# ----------------------------------------------------
# Load data
# ----------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"Rows Loaded : {len(df):,}")

# ----------------------------------------------------
# Features
# ----------------------------------------------------

feature_columns = [
    "pressure_psi",
    "temperature_c",
    "flow_rate_bpd",
    "vibration",
    "gas_ratio",
    "mud_density",
    "pump_rpm",
    "torque",
    "rop",
    "pressure_change",
    "temperature_change",
    "flow_change",
    "vibration_change",
    "rolling_pressure_mean",
    "rolling_temperature_mean",
    "rolling_flow_mean",
    "rolling_vibration_mean"
]

X = df[feature_columns].fillna(0)

# ----------------------------------------------------
# Isolation Forest
# ----------------------------------------------------

print("\nTraining Isolation Forest...")

iso = IsolationForest(
    n_estimators=100,
    contamination=0.015,
    random_state=42
)

iso.fit(X)

df["iso_prediction"] = iso.predict(X)

joblib.dump(iso, ISO_MODEL_PATH)

print("Isolation Forest Saved")

# ----------------------------------------------------
# XGBoost Classifier
# ----------------------------------------------------

print("\nTraining XGBoost...")

label_mapping = {
    "Normal": 0,
    "Leak Risk": 1,
    "Kick Risk": 2,
    "Pump Failure": 3,
    "Sensor Drift": 4,
    "High Vibration": 5
}

reverse_mapping = {v: k for k, v in label_mapping.items()}

y = df["anomaly_type"].map(label_mapping)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

xgb = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=6,
    random_state=42,
    eval_metric="mlogloss"
)

xgb.fit(X_train, y_train)

predictions = xgb.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy")

print(f"{accuracy * 100:.2f}%")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Normal",
            "Leak Risk",
            "Kick Risk",
            "Pump Failure",
            "Sensor Drift",
            "High Vibration"
        ]
    )
)

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, predictions))

joblib.dump(xgb, XGB_MODEL_PATH)

# ----------------------------------------------------
# Generate predictions for the full dataset
# ----------------------------------------------------

full_predictions = xgb.predict(X)

df["predicted_anomaly"] = [
    reverse_mapping[int(i)] for i in full_predictions
]

risk_mapping = {
    "Normal": 5,
    "Sensor Drift": 30,
    "Pump Failure": 60,
    "High Vibration": 75,
    "Leak Risk": 90,
    "Kick Risk": 100
}

df["predicted_risk_score"] = (
    df["predicted_anomaly"]
    .map(risk_mapping)
    .astype(int)
)

OUTPUT_DATA = PROJECT_ROOT / "data" / "processed" / "drillsense_processed_data.csv"

df.to_csv(OUTPUT_DATA, index=False)

print("\nModels Saved Successfully")
print(f"\nIsolation Forest : {ISO_MODEL_PATH}")
print(f"XGBoost          : {XGB_MODEL_PATH}")
print(f"\nUpdated Dataset  : {OUTPUT_DATA}")
print("\nPrediction columns added successfully.")
print("\nDone.")
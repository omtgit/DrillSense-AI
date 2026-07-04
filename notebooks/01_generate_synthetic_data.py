import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# DrillSense AI - Synthetic Oilfield Telemetry Generator
# ============================================================

np.random.seed(42)

# ---------------- CONFIG ---------------- #

NUM_WELLS = 50
DAYS = 30
INTERVAL_MIN = 5

ANOMALY_PERCENT = 0.015      # 1.5% anomaly windows
WINDOW_SIZE = 6              # 6 readings = 30 minutes

# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "drillsense_sensor_data.csv"

timestamps = pd.date_range(
    start="2026-05-01",
    periods=int((24*60/INTERVAL_MIN)*DAYS),
    freq=f"{INTERVAL_MIN}min"
)

rows = []

for well in range(1, NUM_WELLS + 1):

    pressure_base = np.random.uniform(3000, 3600)
    temperature_base = np.random.uniform(75, 95)
    flow_base = np.random.uniform(500, 900)
    vibration_base = np.random.uniform(1.5, 3.0)
    gas_base = np.random.uniform(15, 30)
    mud_base = np.random.uniform(9.8, 11.2)
    rpm_base = np.random.uniform(110, 160)
    torque_base = np.random.uniform(3500, 6500)
    rop_base = np.random.uniform(18, 35)

    total_points = len(timestamps)

    number_of_windows = max(
        1,
        int(total_points * ANOMALY_PERCENT / WINDOW_SIZE)
    )

    starts = np.random.choice(
        total_points - WINDOW_SIZE,
        number_of_windows,
        replace=False
    )

    anomaly_lookup = {}

    anomaly_types = [
        "Leak Risk",
        "Kick Risk",
        "Pump Failure",
        "Sensor Drift",
        "High Vibration"
    ]

    for s in starts:

        anomaly = np.random.choice(anomaly_types)

        for j in range(WINDOW_SIZE):
            anomaly_lookup[s + j] = anomaly

    for i, ts in enumerate(timestamps):

        hour = ts.hour

        pressure = (
            pressure_base
            + 30*np.sin(2*np.pi*hour/24)
            + np.random.normal(0,18)
        )

        temperature = (
            temperature_base
            + 2*np.sin(2*np.pi*hour/24)
            + np.random.normal(0,1)
        )

        flow = (
            flow_base
            + 15*np.cos(2*np.pi*hour/24)
            + np.random.normal(0,8)
        )

        vibration = vibration_base + np.random.normal(0,0.12)

        gas = gas_base + np.random.normal(0,0.35)

        mud = mud_base + np.random.normal(0,0.05)

        rpm = rpm_base + np.random.normal(0,2)

        torque = torque_base + np.random.normal(0,100)

        rop = rop_base + np.random.normal(0,0.4)

        anomaly_flag = 0
        anomaly_type = "Normal"

        if i in anomaly_lookup:

            anomaly_flag = 1
            anomaly_type = anomaly_lookup[i]

            if anomaly_type == "Leak Risk":

                pressure -= np.random.uniform(250,450)
                flow -= np.random.uniform(150,250)

            elif anomaly_type == "Kick Risk":

                pressure += np.random.uniform(350,600)
                gas += np.random.uniform(10,18)

            elif anomaly_type == "Pump Failure":

                rpm -= np.random.uniform(40,60)
                vibration += np.random.uniform(1.5,2.5)

            elif anomaly_type == "Sensor Drift":

                temperature += np.random.uniform(8,15)

            elif anomaly_type == "High Vibration":

                vibration += np.random.uniform(2.5,4.5)
                torque += np.random.uniform(900,1800)

                # -----------------------------
        # Decision Intelligence Fields
        # -----------------------------

        severity = "None"
        priority_score = 0
        recommended_response = "No Action Required"

        if anomaly_type == "Kick Risk":
            severity = "Critical"
            priority_score = 100
            recommended_response = "Immediate inspection and pressure control check"

        elif anomaly_type == "Leak Risk":
            severity = "High"
            priority_score = 90
            recommended_response = "Inspect within 2 hours"

        elif anomaly_type == "High Vibration":
            severity = "High"
            priority_score = 85
            recommended_response = "Inspect rotating equipment within 4 hours"

        elif anomaly_type == "Pump Failure":
            severity = "Medium"
            priority_score = 70
            recommended_response = "Maintenance within 6 hours"

        elif anomaly_type == "Sensor Drift":
            severity = "Low"
            priority_score = 40
            recommended_response = "Calibrate sensor during next maintenance"

        rows.append({

            "timestamp": ts,
            "well_id": f"WELL-{well:03}",

            "pressure_psi": round(pressure,2),
            "temperature_c": round(temperature,2),
            "flow_rate_bpd": round(flow,2),
            "vibration": round(vibration,2),
            "gas_ratio": round(gas,2),
            "mud_density": round(mud,2),
            "pump_rpm": round(rpm,2),
            "torque": round(torque,2),
            "rop": round(rop,2),

            "anomaly_type": anomaly_type,
            "anomaly_flag": anomaly_flag,

            "severity": severity,
            "priority_score": priority_score,
            "recommended_response": recommended_response
        })

df = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("="*60)
print(" DrillSense AI Dataset Generated Successfully")
print("="*60)
print(f"Rows              : {len(df):,}")
print(f"Wells             : {NUM_WELLS}")
print(f"Days              : {DAYS}")
print(f"Time Interval     : {INTERVAL_MIN} minutes")
print(f"Anomalies         : {df['anomaly_flag'].sum():,}")
print(f"Output            : {OUTPUT_PATH}")
print("="*60)

print("\nSample Data:\n")

print(df.head())
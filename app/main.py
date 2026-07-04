import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from gemini_utils import generate_report

st.set_page_config(
    page_title="DrillSense AI",
    page_icon="⛽",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "drillsense_processed_data.csv"

df = pd.read_csv(DATA_FILE)

st.title("⛽ DrillSense AI")
st.subheader("GPU-Accelerated Decision Intelligence for Oilfield Monitoring")

# ------------------------
# Sidebar
# ------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Executive Dashboard",
        "Well Explorer",
        "AI Decision Center",
        "GPU Performance",
        "About"
    ]
)

# ===================================================
# Executive Dashboard
# ===================================================

if page == "Executive Dashboard":

    total_wells = df["well_id"].nunique()

    total_records = len(df)

    anomalies = int(df["anomaly_flag"].sum())

    avg_health = round(df["health_score"].mean(),2)

    highest = df.sort_values(
        "predicted_risk_score",
        ascending=False
    ).iloc[0]

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Total Wells",total_wells)

    c2.metric("Records",f"{total_records:,}")

    c3.metric("Detected Anomalies",anomalies)

    c4.metric("Average Health",f"{avg_health}%")

    st.divider()

    st.subheader("Highest Risk Well")

    st.write(f"**Well :** {highest['well_id']}")

    st.write(f"**Predicted Issue :** {highest['predicted_anomaly']}")

    st.write(f"**Risk Score :** {highest['predicted_risk_score']}")

    st.write(f"**Severity :** {highest['severity']}")

    st.write(f"**Recommended Response :** {highest['recommended_response']}")

    st.divider()

    chart = (
        df.groupby("predicted_anomaly")
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        chart,
        x="predicted_anomaly",
        y="Count",
        title="Predicted Anomaly Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)

# ===================================================
# Well Explorer
# ===================================================

elif page == "Well Explorer":

    wells = sorted(df["well_id"].unique())

    selected = st.selectbox(
        "Select Well",
        wells
    )

    temp = df[df["well_id"]==selected]

    fig1 = px.line(
        temp,
        x="timestamp",
        y="pressure_psi",
        title="Pressure"
    )

    st.plotly_chart(fig1,use_container_width=True)

    fig2 = px.line(
        temp,
        x="timestamp",
        y="flow_rate_bpd",
        title="Flow Rate"
    )

    st.plotly_chart(fig2,use_container_width=True)

    fig3 = px.line(
        temp,
        x="timestamp",
        y="temperature_c",
        title="Temperature"
    )

    st.plotly_chart(fig3,use_container_width=True)

    fig4 = px.line(
        temp,
        x="timestamp",
        y="vibration",
        title="Vibration"
    )

    st.plotly_chart(fig4,use_container_width=True)

# ===================================================
# AI Decision Center
# ===================================================

elif page == "AI Decision Center":

    st.header("AI Decision Center")

    top_risk = (
        df.sort_values(
            "predicted_risk_score",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        top_risk[
            [
                "well_id",
                "predicted_anomaly",
                "predicted_risk_score",
                "severity",
                "recommended_response"
            ]
        ],
        use_container_width=True
    )

    selected = st.selectbox(
        "Select a Well",
        top_risk["well_id"].unique()
    )

    row = top_risk[top_risk["well_id"] == selected].iloc[0]

    st.subheader("Incident Summary")

    st.write(f"**Well** : {row['well_id']}")

    st.write(f"**Predicted Issue** : {row['predicted_anomaly']}")

    st.write(f"**Severity** : {row['severity']}")

    st.write(f"**Risk Score** : {row['predicted_risk_score']}")

    st.write(f"**Recommended Response** : {row['recommended_response']}")
    

    if st.button("Generate AI Engineering Report"):

      with st.spinner("Generating report..."):

        report = generate_report(row)

      st.subheader("AI Engineering Report")

      st.write(report)

# ===================================================
# GPU Performance
# ===================================================

elif page == "GPU Performance":

    st.header("GPU Performance")

    gpu = {
        "Metric": [
            "Dataset Size",
            "Records",
            "CPU Library",
            "GPU Library",
            "GPU Used",
            "Estimated Speed-up"
        ],
        "Value": [
            "107 MB",
            "432,000",
            "Pandas",
            "cuDF (RAPIDS)",
            "NVIDIA T4",
            "To be measured in Colab"
        ]
    }

    st.table(pd.DataFrame(gpu))

    st.info(
        "GPU benchmark will be generated after running the RAPIDS benchmark in Google Colab."
    )

# ===================================================
# About
# ===================================================

elif page == "About":

    st.header("About DrillSense AI")

    st.markdown("""

### Technologies

- Google Cloud Storage
- BigQuery
- Gemini
- Streamlit
- XGBoost
- Isolation Forest
- RAPIDS cuDF
- Plotly

### Workflow

Sensor Data

↓

Feature Engineering

↓

Anomaly Detection

↓

Risk Prediction

↓

AI Decision Intelligence

↓

Cloud Deployment

""")
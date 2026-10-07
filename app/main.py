import json
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from data_source import data_source_name, load_sensor_data
from gemini_utils import generate_report

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
HELDOUT_METRICS = Path(__file__).resolve().parent.parent / "docs" / "heldout_metrics.json"

st.set_page_config(
    page_title="DrillSense AI",
    page_icon=str(ASSETS_DIR / "logo.png"),
    layout="wide"
)

@st.cache_data
def load_data():
    return load_sensor_data()

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load data: {type(exc).__name__}: {exc}")
    st.stop()


def rank_by_risk(frame):
    """Highest predicted risk first; ties go to the most recent row, then the lowest well id.

    Many rows share the maximum score (the score is a lookup on predicted class), so
    sorting on the score alone gives an arbitrary pick.
    """
    return frame.sort_values(
        ["predicted_risk_score", "timestamp", "well_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )


@st.cache_data
def load_heldout_metrics():
    try:
        return json.loads(HELDOUT_METRICS.read_text())
    except (OSError, ValueError):
        return None


st.title("⛽ DrillSense AI")
st.subheader("GPU-Benchmarked Decision Intelligence for Oilfield Monitoring")

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
        "CPU vs GPU Benchmark",
        "About"
    ]
)

st.sidebar.caption(f"Data source: {data_source_name()}")

# ===================================================
# Executive Dashboard
# ===================================================

if page == "Executive Dashboard":

    total_wells = df["well_id"].nunique()

    total_records = len(df)

    injected = int(df["anomaly_flag"].sum())

    detected = int((df["predicted_anomaly"] != "Normal").sum())

    avg_health = round(df["health_score"].mean(),2)

    ranked = rank_by_risk(df)

    highest = ranked.iloc[0]

    n_tied = int(
        (df["predicted_risk_score"] == highest["predicted_risk_score"]).sum()
    )

    c1,c2,c3 = st.columns(3)

    c1.metric("Total Wells",total_wells)

    c2.metric("Records",f"{total_records:,}")

    c3.metric(
        "Average Health (from ground-truth severity)",
        f"{avg_health}%"
    )

    c4,c5 = st.columns(2)

    c4.metric("Injected anomalies (ground truth)",f"{injected:,}")

    c5.metric("Model detections (predicted \u2260 Normal)",f"{detected:,}")

    st.caption(
        "Data is synthetic. 'Injected' counts rows the generator labelled as anomalous. "
        "'Model detections' counts rows the XGBoost classifier flagged. The model was "
        "trained on a random 80% of these rows and then applied to all of them, so this "
        "count flatters the model; see the held-out result below."
    )

    st.subheader("Detection quality on held-out wells")

    hm = load_heldout_metrics()

    if hm:
        h1,h2,h3 = st.columns(3)
        h1.metric("Precision",f"{hm['precision']:.2f}")
        h2.metric("Recall",f"{hm['recall']:.2f}")
        h3.metric("F1",f"{hm['f1']:.2f}")
        st.caption(
            f"Anomaly vs normal, scored on {len(hm['test_wells'])} wells "
            f"({hm['test_rows']:,} rows, {hm['test_anomaly_rows']:,} injected anomaly rows) "
            f"that the model was not trained on; trained on the other {hm['train_wells']}. "
            "Reproduce with `python scripts/heldout_metrics.py`. "
            "Synthetic data with very large injected shifts: this is not evidence of field "
            "performance, and a simple per-well z-score baseline does as well or better."
        )
    else:
        st.info(
            "Held-out metrics not found. Run `python scripts/heldout_metrics.py` "
            "to generate docs/heldout_metrics.json."
        )

    st.divider()

    st.subheader("Highest Risk Well")

    st.error(
       f"""
    Well ID: {highest['well_id']}

    Timestamp: {highest['timestamp']}

    Predicted Issue: {highest['predicted_anomaly']}

    Risk Score: {highest['predicted_risk_score']}

    Severity (ground truth): {highest['severity']}
    """
    )

    st.caption(
        f"{n_tied:,} rows tie at this risk score. Ties are broken by most recent "
        "timestamp, then well id."
    )

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
        title="Model-predicted anomaly distribution (all rows, including training rows)"
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

    st.caption(
        "The table ranks rows by the XGBoost risk score (ties: most recent first). "
        "Severity and recommended response are looked up from the injected ground-truth "
        "anomaly type, not predicted. Gemini is only called when you press the "
        "report button below."
    )

    top_risk = rank_by_risk(df).head(10)

    st.dataframe(
        top_risk[
            [
                "well_id",
                "timestamp",
                "predicted_anomaly",
                "predicted_risk_score",
                "severity",
                "recommended_response"
            ]
        ].rename(columns={
            "severity": "severity (ground truth)",
            "recommended_response": "recommended_response (ground truth)",
        }),
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

    st.write(f"**Severity (ground truth)** : {row['severity']}")

    st.write(f"**Risk Score** : {row['predicted_risk_score']}")

    st.write(f"**Recommended Response (ground truth)** : {row['recommended_response']}")
    

    if st.button("Generate AI Engineering Report"):

      try:
        with st.spinner("Generating report..."):
          report = generate_report(row)
      except Exception as exc:
        st.error(f"Gemini report failed: {type(exc).__name__}: {exc}")
      else:
        st.subheader("AI Engineering Report")
        st.caption("Generated by Gemini from the incident summary above.")
        st.write(report)

# ===================================================
# GPU Performance
# ===================================================

elif page == "CPU vs GPU Benchmark":

    st.header("CPU vs GPU benchmark (Colab, Tesla T4, offline)")

    st.warning(
        "This is a one-off offline benchmark, not part of this app. It measured only "
        "the time to **read the 432,000-row, 107 MB CSV file** with pandas (CPU) versus "
        "RAPIDS cuDF (GPU) on Google Colab. This app runs on CPU and does no GPU "
        "work; model training and inference were not benchmarked."
    )

    gpu = {
        "Metric": [
            "Task measured",
            "Dataset Size",
            "Records",
            "CPU Library",
            "GPU Library",
            "GPU Hardware",
            "CPU Read Time",
            "GPU Read Time",
            "Measured Speed-up"
        ],
        "Value": [
            "CSV read time only",
            "107 MB",
            "432,000",
            "Pandas",
            "RAPIDS cuDF",
            "NVIDIA Tesla T4 (Google Colab)",
            "2.0563 sec",
            "0.7807 sec",
            "2.63x"
        ]
    }

    st.table(pd.DataFrame(gpu))

    st.caption(
        "Single run, numbers recorded at the time (see docs/BENCHMARKS.md); no repeats "
        "or variance were kept. Re-run on your own GPU with "
        "`python benchmarks/gpu_benchmark.py`."
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
- RAPIDS cuDF (offline CSV-read benchmark only)
- Plotly

### Workflow

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

""")
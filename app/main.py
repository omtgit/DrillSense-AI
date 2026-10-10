import json
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import detector
import drilling_pages
from data_source import data_source_name, load_drilling, load_sensor_data
from gemini_utils import model_name
from profiles import get_profile
from drilling_pages import HELP_F1, HELP_PRECISION, HELP_RECALL
from ui import (DOCS_URL, REPO_URL, apply_plotly_theme, footer,
                inject_design_system, severity_chip_html, show_error)
from report_ui import rank_by_risk, report_section

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
HELDOUT_METRICS = Path(__file__).resolve().parent.parent / "docs" / "heldout_metrics.json"

st.set_page_config(
    page_title="DrillSense AI | Drilling anomaly detection demo",
    page_icon=str(ASSETS_DIR / "logo.png") if (ASSETS_DIR / "logo.png").is_file() else "⛽",
    layout="wide"
)
inject_design_system()


@st.cache_data(show_spinner="Loading the production demo wells...")
def load_production_data():
    return load_sensor_data()


@st.cache_data(show_spinner=False)  # covered by the startup status in ensure_ready()
def load_drilling_scored():
    """Sample telemetry with the detector's predictions. The detector never saw these wells."""
    return get_detector().predict(detector.prepare(load_drilling(), get_profile("drilling")))


@st.cache_resource(show_spinner=False)  # covered by the startup status in ensure_ready()
def get_detector():
    return detector.train_detector(get_profile("drilling"))


@st.cache_resource(show_spinner=False)
def _startup_state():
    """Remembers, per server process, whether the one-off set-up has finished."""
    return {"ready": False}


def ensure_ready():
    """Explain the slow first start once, in plain words, instead of showing a bare spinner."""
    state = _startup_state()
    if state["ready"]:
        return
    with st.status("Starting the demo (about 15 s, first visit only)", expanded=True) as status:
        st.write("Training on simulated wells...")
        get_detector()
        load_drilling_scored()
        status.update(label="Demo ready", state="complete", expanded=False)
    state["ready"] = True


@st.cache_data(show_spinner="Loading the saved held-out results...")
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

profile_label = st.sidebar.selectbox(
    "Domain profile",
    ["Drilling", "Production"],
    help="Drilling: rig channels (WOB, SPP, flow in/out, pit volume...) from the built-in drilling "
         "simulator. Production: the original production-style synthetic dataset.",
)
profile_name = profile_label.lower()

page = st.sidebar.radio(
    "Select Page",
    [
        "Executive Dashboard",
        "Well Explorer",
        "AI Decision Center",
        "Model Evaluation",
        "CPU vs GPU Benchmark",
        "About"
    ]
)

if profile_name == "drilling" or data_source_name() == "local":
    st.sidebar.caption("Data: built-in simulated demo wells")
else:
    st.sidebar.caption("Data: simulated wells loaded from a connected BigQuery table")

with st.sidebar.expander("AI assistant (optional)"):
    st.caption(
        "The full report works without a key. Add your own free Google AI Studio key to get the same facts "
        "written for a specific reader, in English or Hindi, and to ask questions about this event. "
        "Detection is done by the model; the assistant only helps you communicate it. It cannot invent "
        "numbers: everything is checked against the report. Your key stays in this browser session and is "
        "never stored or logged."
    )
    st.markdown("[Get a free key from Google AI Studio](https://aistudio.google.com/apikey)")
    st.text_input("Google AI Studio key", type="password", key="gemini_api_key")
    st.text_input("Model name (advanced)", value=model_name(), key="gemini_model")

# ===================================================
# Data for the three profile-aware pages
# ===================================================

PROFILE_PAGES = ("Executive Dashboard", "Well Explorer", "AI Decision Center")

if page in PROFILE_PAGES and profile_name == "drilling":
    try:
        ensure_ready()
        scored = load_drilling_scored()
        det = get_detector()
    except Exception as exc:
        show_error("the demo wells could not be loaded or scored", exc)
        st.stop()
    if page == "Executive Dashboard":
        drilling_pages.render_dashboard(scored, det)
    elif page == "Well Explorer":
        drilling_pages.render_explorer(scored, det.profile)
    else:
        drilling_pages.render_decision_center(scored, det)
    footer()
    st.stop()

if page in PROFILE_PAGES:
    try:
        df = load_production_data()
    except Exception as exc:
        show_error("the demo wells could not be loaded", exc)
        st.stop()

# ===================================================
# Executive Dashboard (production profile)
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

    st.markdown(
        """
        <div class="ds-hero">
            <div class="ds-hero-title">📊 Executive Dashboard</div>
            <p class="ds-hero-subtitle">Continuous oilfield surveillance, anomaly detection, and operational risk assessment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status strip: wells normal / watch / alarm
    well_max_risk = df.groupby("well_id")["predicted_risk_score"].max()
    n_alarm = int((well_max_risk >= 70).sum())
    n_watch = int(((well_max_risk >= 30) & (well_max_risk < 70)).sum())
    n_normal = int((well_max_risk < 30).sum())

    st.markdown(
        f"""
        <div class="ds-status-strip">
            <div class="ds-status-item" style="border-left: 4px solid var(--ds-success);">
                <span class="ds-status-item-label">Normal Wells</span>
                <span class="ds-status-item-val" style="color: #2ed573;">{n_normal}</span>
            </div>
            <div class="ds-status-item" style="border-left: 4px solid var(--ds-warning);">
                <span class="ds-status-item-label">Watch List</span>
                <span class="ds-status-item-val" style="color: #ffa502;">{n_watch}</span>
            </div>
            <div class="ds-status-item" style="border-left: 4px solid var(--ds-danger);">
                <span class="ds-status-item-label">Active Alarms</span>
                <span class="ds-status-item-val" style="color: #ff6b6b;">{n_alarm}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1,c2,c3 = st.columns(3)

    c1.container(border=True).metric("Total Wells",total_wells)

    c2.container(border=True).metric("Records",f"{total_records:,}")

    c3.container(border=True).metric(
        "Average Health (based on the planted severity)",
        f"{avg_health}%"
    )

    c4,c5 = st.columns(2)

    c4.container(border=True).metric("Injected anomalies (ground truth)",f"{injected:,}")

    c5.container(border=True).metric("Model detections (predicted \u2260 Normal)",f"{detected:,}")

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
        h1.container(border=True).metric("Precision",f"{hm['precision']:.2f}",help=HELP_PRECISION)
        h2.container(border=True).metric("Recall",f"{hm['recall']:.2f}",help=HELP_RECALL)
        h3.container(border=True).metric("F1",f"{hm['f1']:.2f}",help=HELP_F1)
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

    with st.container(border=True):
        chip = severity_chip_html(highest['severity'])
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:600; color:var(--ds-text); font-size:1.05rem;">Critical Telemetry Alert</span>
                <div>{chip}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.error(
           f"""
        Well ID: {highest['well_id']}

        Timestamp: {highest['timestamp']}

        Predicted Issue: {highest['predicted_anomaly']}

        Risk Score (0 to 100): {highest['predicted_risk_score']}

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
        title="Model-predicted anomaly distribution (all rows, including training rows)",
        labels={"predicted_anomaly": "Predicted issue", "Count": "Number of rows"}
    )
    apply_plotly_theme(fig)

    st.plotly_chart(fig,width="stretch")

# ===================================================
# Well Explorer
# ===================================================

elif page == "Well Explorer":

    wells = sorted(df["well_id"].unique())

    PROD_LABELS = {"timestamp": "Time", **get_profile("production").channel_labels}

    selected = st.selectbox(
        "Select Well",
        wells
    )

    temp = df[df["well_id"]==selected]

    tab_signals, tab_events, tab_details = st.tabs(["Signals", "Events", "Details"])

    with tab_signals:
        prod_sigs = [
            ("pressure_psi", "Pressure (psi)"),
            ("flow_rate_bpd", "Flow Rate (bpd)"),
            ("temperature_c", "Temperature (°C)"),
            ("vibration", "Vibration (g)"),
        ]
        for col, col_title in prod_sigs:
            if col in temp.columns:
                med = float(temp[col].median())
                std = float(temp[col].std())
                fig = px.line(
                    temp,
                    x="timestamp",
                    y=col,
                    title=f"{col_title} — Dynamic Operating Envelope",
                    labels=PROD_LABELS
                )
                if std > 0:
                    fig.add_hrect(
                        y0=med - 2 * std,
                        y1=med + 2 * std,
                        fillcolor="rgba(16, 172, 132, 0.08)",
                        line_width=0,
                        annotation_text="Nominal Envelope (±2σ)",
                        annotation_position="top left",
                        annotation_font_size=10,
                        annotation_font_color="#8b949e",
                    )
                fig.update_traces(
                    fill="tozeroy",
                    fillcolor="rgba(0, 210, 190, 0.04)",
                    line=dict(color="#00d2be", width=2),
                    hovertemplate="<b>%{x}</b><br>" + col_title + ": %{y:.2f}<extra></extra>",
                )
                apply_plotly_theme(fig)
                fig.update_layout(yaxis_title=col_title, xaxis_title="Time", hovermode="x unified")
                st.plotly_chart(fig, width="stretch")

    with tab_events:
        st.subheader("Ground truth vs model")
        c_e1, c_e2, c_e3 = st.columns(3)
        c_e1.container(border=True).metric("Well Readings", f"{len(temp):,}")
        c_e2.container(border=True).metric("Injected Anomaly (rows)", int(temp["anomaly_flag"].sum()))
        c_e3.container(border=True).metric("Model Detections (rows)", int((temp["predicted_anomaly"] != "Normal").sum()))

        anomaly_col = "anomaly_type" if "anomaly_type" in temp.columns else "severity"
        strip = temp[temp["anomaly_flag"] == 1][["timestamp", anomaly_col]].rename(columns={anomaly_col: "type"})
        strip["source"] = "Injected (ground truth)"
        pred = temp[temp["predicted_anomaly"] != "Normal"][["timestamp", "predicted_anomaly"]].rename(columns={"predicted_anomaly": "type"})
        pred["source"] = "Model prediction"
        both = pd.concat([strip, pred])
        if both.empty:
            st.info("No injected or predicted anomalies for this well.")
        else:
            fig = px.scatter(
                both, x="timestamp", y="source", color="type",
                symbol="source",
                symbol_map={"Model prediction": "circle", "Injected (ground truth)": "diamond"},
                category_orders={"source": ["Model prediction", "Injected (ground truth)"]},
                labels={"timestamp": "Time", "source": "Event Source", "type": "Anomaly Type"}
            )
            fig.update_traces(marker=dict(size=8))
            apply_plotly_theme(fig)
            fig.update_layout(yaxis_title="Source", xaxis_title="Time", height=280)
            st.plotly_chart(fig, width="stretch")
            st.caption("Each marker represents one observation. Diamonds: injected ground truth; Circles: model predictions.")

    with tab_details:
        st.subheader(f"Well {selected} Sensor Summary")
        prod_channels = ["pressure_psi", "flow_rate_bpd", "temperature_c", "vibration"]
        stats_rows = []
        for ch in prod_channels:
            if ch in temp.columns:
                s = temp[ch]
                stats_rows.append({
                    "Sensor": PROD_LABELS.get(ch, ch),
                    "Min": f"{s.min():.2f}",
                    "Mean": f"{s.mean():.2f}",
                    "Max": f"{s.max():.2f}",
                    "Std Dev": f"{s.std():.2f}"
                })
        st.dataframe(pd.DataFrame(stats_rows), hide_index=True, width="stretch")

# ===================================================
# AI Decision Center
# ===================================================

elif page == "AI Decision Center":

    st.header("AI Decision Center")

    st.caption(
        "The table ranks rows by the XGBoost risk score (ties: most recent first). "
        "Severity and recommended response are looked up from the injected ground-truth "
        "anomaly type, not predicted. The optional AI assistant below is only called when you press a button."
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
            "well_id": "Well", "timestamp": "Time", "predicted_anomaly": "Predicted issue",
            "predicted_risk_score": "Risk score",
            "severity": "Severity (ground truth)",
            "recommended_response": "Recommended response (ground truth)",
        }),
        hide_index=True,
        width="stretch"
    )

    selected = st.selectbox(
        "Select a Well",
        top_risk["well_id"].unique()
    )

    row = top_risk[top_risk["well_id"] == selected].iloc[0]

    st.subheader("Incident Summary")
    with st.container(border=True):
        col_is1, col_is2 = st.columns([3, 1])
        with col_is1:
            st.write(f"**Well** : {row['well_id']}")
            st.write(f"**Predicted Issue** : {row['predicted_anomaly']}")
            st.write(f"**Severity (ground truth)** : {row['severity']}")
            st.write(f"**Risk Score** : {row['predicted_risk_score']}")
            st.write(f"**Recommended Response (ground truth)** : {row['recommended_response']}")
        with col_is2:
            st.markdown(f'<div style="text-align:right;">{severity_chip_html(row["severity"])}</div>', unsafe_allow_html=True)
    

    report_section(df, row, get_profile("production"))

# ===================================================
# Model Evaluation (reads docs/eval_results.json; no training here)
# ===================================================

elif page == "Model Evaluation":

    import eval_page
    eval_page.render(st)


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

    b1, b2, b3 = st.columns(3)
    b1.container(border=True).metric("CPU Read Time", "2.056 s", help="Pandas CSV read time on Google Colab")
    b2.container(border=True).metric("GPU Read Time", "0.781 s", help="RAPIDS cuDF CSV read time on NVIDIA Tesla T4")
    b3.container(border=True).metric("Measured Speed-up", "2.63x", help="Relative acceleration factor")

    with st.container(border=True):
        st.dataframe(pd.DataFrame(gpu), hide_index=True, width="stretch")

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

    with st.container(border=True):
        st.markdown("""

### Learn more

- [Source code on GitHub]({repo})
- [Physics of the simulated rig signals (PHYSICS.md)]({docs}/PHYSICS.md)
- [How the models were evaluated (EVAL.md)]({docs}/EVAL.md)
- [CPU vs GPU benchmark notes (BENCHMARKS.md)]({docs}/BENCHMARKS.md)

It is an educational project, not a tool for operational decisions.

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
```

""".replace("{repo}", REPO_URL).replace("{docs}", DOCS_URL))

footer()

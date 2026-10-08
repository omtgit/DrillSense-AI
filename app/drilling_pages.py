"""Dashboard, Well Explorer and Decision Center for the drilling profile.

`scored` is the sample telemetry after detector.predict(): ground-truth columns (anomaly_type,
anomaly_flag) sit next to model columns (predicted_*). The two are always labelled separately.
Severity and response here come from the profile's rules for the PREDICTED class.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from report_ui import rank_by_risk, report_section

HELP_PRECISION = "Of the minutes the model flagged, the share that really were inside a simulated incident."
HELP_RECALL = "Of the minutes inside a simulated incident, the share the model flagged."
HELP_F1 = "One score that balances precision and recall (1.0 is perfect)."
EXPLORER_DEFAULT = ["flow_out_lpm", "pit_volume_m3", "spp_bar", "rop_mph"]


def render_dashboard(scored, det):
    profile = det.profile
    injected = int(scored["anomaly_flag"].sum())
    detected = int((scored["predicted_anomaly"] != "Normal").sum())
    ranked = rank_by_risk(scored)
    top = ranked.iloc[0]
    n_tied = int((scored["predicted_risk_score"] == top["predicted_risk_score"]).sum())

    c1, c2 = st.columns(2)
    c1.metric("Wells", scored["well_id"].nunique())
    c2.metric("Records (1-minute)", f"{len(scored):,}")

    c4, c5 = st.columns(2)
    c4.metric("Injected anomaly rows (ground truth)", f"{injected:,}")
    c5.metric("Model detections (rows predicted ≠ Normal)", f"{detected:,}")
    st.caption(
        "Data is synthetic. 'Injected' counts rows the generator labelled as inside an anomaly, "
        "including the faint first minutes of each ramp. 'Model detections' counts rows flagged by "
        "the XGBoost detector, which was never trained on these wells."
    )

    i = det.info
    st.subheader("Detection quality on held-out generated wells")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Row precision", f"{i['precision']:.2f}", help=HELP_PRECISION)
    m2.metric("Row recall", f"{i['recall']:.2f}", help=HELP_RECALL)
    m3.metric("Row F1", f"{i['f1']:.2f}", help=HELP_F1)
    m4.metric("Events detected", f"{i['event_recall']:.0%}",
              help="The share of simulated incidents where the model raised at least one alarm during or just after it.")
    st.caption(
        f"Trained on {i['train_wells']} generated wells ({i['train_rows']:,} rows, seed {i['train_seed']}); "
        f"scored on {i['valid_wells']} other generated wells ({i['valid_rows']:,} rows, "
        f"{i['valid_anomaly_rows']:,} injected anomaly rows, {i['valid_events']} events, seed {i['valid_seed']}). "
        "A row is flagged when P(Normal) < 0.5; no threshold was tuned. 'Events detected' = events with at "
        "least one flagged row inside the event or the 15 minutes after it. Recall is low because the first "
        "minutes of every ramp are labelled anomalous but barely visible. Event-level results with false "
        "alarms and baselines are on the Model Evaluation page."
    )

    st.divider()
    st.subheader("Highest Risk Well")
    st.error(
        f"""
Well ID: {top['well_id']}

Timestamp: {top['timestamp']}

Predicted issue: {top['predicted_anomaly']}

Risk score (0 to 100, higher means the model is more sure something is wrong): {top['predicted_risk_score']}

Severity (from the drilling rules, for the predicted issue): {top['predicted_severity']}

Ground truth at this row: {top['anomaly_type']}
"""
    )
    st.caption(
        f"{n_tied:,} rows share this risk score. Ties are broken by most recent timestamp, then well id."
    )

    st.divider()
    labels = ["Normal"] + list(profile.anomaly_types)
    both = (
        scored.assign(truth=scored["anomaly_type"], model=scored["predicted_anomaly"])
        .melt(value_vars=["truth", "model"], var_name="source", value_name="class")
        .groupby(["class", "source"]).size().reset_index(name="rows")
    )
    both["source"] = both["source"].map({"truth": "Injected (ground truth)", "model": "Model prediction"})
    fig = px.bar(
        both[both["class"] != "Normal"], x="class", y="rows", color="source", barmode="group",
        category_orders={"class": labels[1:]},
        title="Anomaly rows by type: injected vs predicted (Normal rows omitted)",
        labels={"class": "Anomaly type", "rows": "Number of one-minute rows", "source": "Source"},
    )
    st.plotly_chart(fig, width="stretch")


def _runs_frame(scored, column, label):
    sub = scored[scored[column] != "Normal"][["timestamp", column]].rename(columns={column: "class"})
    sub["source"] = label
    return sub


def render_explorer(scored, profile):
    wells = sorted(scored["well_id"].unique())
    selected = st.selectbox("Select Well", wells)
    temp = scored[scored["well_id"] == selected]

    channels = list(profile.channels)
    choice = st.multiselect(
        "Channels", channels, default=[c for c in EXPLORER_DEFAULT if c in channels],
        format_func=lambda c: profile.channels[c].report_label,
    )
    for ch in choice:
        st.plotly_chart(
            px.line(temp, x="timestamp", y=ch, title=profile.channels[ch].report_label,
                    labels={"timestamp": "Time", ch: profile.channels[ch].report_label}),
            width="stretch",
        )

    st.subheader("Ground truth vs model")
    strip = _runs_frame(temp, "anomaly_type", "Injected (ground truth)")
    strip = strip.rename(columns={"class": "type"})
    pred = _runs_frame(temp, "predicted_anomaly", "Model prediction").rename(columns={"class": "type"})
    both = pd.concat([strip[["timestamp", "type", "source"]], pred[["timestamp", "type", "source"]]])
    if both.empty:
        st.info("No injected or predicted anomalies for this well.")
        return
    fig = px.scatter(both, x="timestamp", y="source", color="type",
                     category_orders={"source": ["Model prediction", "Injected (ground truth)"]})
    fig.update_traces(marker=dict(size=7, symbol="square"))
    fig.update_layout(yaxis_title=None, height=260)
    st.plotly_chart(fig, width="stretch")
    st.caption("Each marker is one minute. Where the rows differ, the model missed it or raised a false alarm.")


def render_decision_center(scored, det):
    profile = det.profile
    st.header("AI Decision Center")
    st.caption(
        "One row per well: its highest-risk minute by the XGBoost risk score (ties: most recent first). "
        "Severity and response come from the drilling rules for the PREDICTED issue, not from ground truth. "
        f"{profile.review_note} The optional AI assistant below is only called when you press a button."
    )
    best = rank_by_risk(scored).drop_duplicates("well_id").head(10)
    st.dataframe(
        best[["well_id", "timestamp", "predicted_anomaly", "predicted_risk_score",
              "predicted_severity", "predicted_response", "anomaly_type"]].rename(columns={
            "well_id": "Well", "timestamp": "Time", "predicted_anomaly": "Predicted issue",
            "predicted_risk_score": "Risk score",
            "predicted_severity": "Severity (drilling rule)",
            "predicted_response": "Recommended response (drilling rule)",
            "anomaly_type": "Ground truth at this row",
        }),
        hide_index=True,
        width="stretch",
    )

    selected = st.selectbox("Select a Well", best["well_id"].unique())
    row = best[best["well_id"] == selected].iloc[0]

    st.subheader("Incident Summary")
    st.write(f"**Well** : {row['well_id']}")
    st.write(f"**Predicted Issue** : {row['predicted_anomaly']}")
    st.write(f"**Severity (drilling rule)** : {row['predicted_severity']}")
    st.write(f"**Risk Score** : {row['predicted_risk_score']}")
    st.write(f"**Recommended Response (drilling rule)** : {row['predicted_response']}")
    st.write(f"**Ground truth at this row** : {row['anomaly_type']}")

    report_section(scored, row, profile, rules_from_profile=True)

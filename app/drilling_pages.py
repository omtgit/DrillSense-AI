"""Dashboard, Well Explorer and Decision Center for the drilling profile.

`scored` is the sample telemetry after detector.predict(): ground-truth columns (anomaly_type,
anomaly_flag) sit next to model columns (predicted_*). The two are always labelled separately.
Severity and response here come from the profile's rules for the PREDICTED class.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from report_ui import rank_by_risk, report_section
from ui import apply_plotly_theme, severity_chip_html, page_header, kv_grid

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

    from app.home_page import render_home_hero
    render_home_hero(active_profile=profile.name)

    page_header(
        "Executive Dashboard",
        subtitle="Continuous rig telemetry surveillance, anomaly detection, and operational risk assessment across active wells.",
        icon="📊"
    )

    # Status strip: wells in normal, watch, and alarm states
    well_max_risk = scored.groupby("well_id")["predicted_risk_score"].max()
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

    # Operational KPI cards
    c1, c2, c4, c5 = st.columns(4)
    c1.container(border=True).metric("Wells", scored["well_id"].nunique())
    c2.container(border=True).metric("Records (1-minute)", f"{len(scored):,}")
    c4.container(border=True).metric("Injected anomaly rows (ground truth)", f"{injected:,}")
    c5.container(border=True).metric("Model detections (rows predicted ≠ Normal)", f"{detected:,}")
    st.caption(
        "Data is synthetic. 'Injected' counts rows the generator labelled as inside an anomaly, "
        "including the faint first minutes of each ramp. 'Model detections' counts rows flagged by "
        "the XGBoost detector, which was never trained on these wells."
    )

    i = det.info
    st.subheader("Detection quality on held-out generated wells")
    m1, m2, m3, m4 = st.columns(4)
    m1.container(border=True).metric("Row precision", f"{i['precision']:.2f}", help=HELP_PRECISION)
    m2.container(border=True).metric("Row recall", f"{i['recall']:.2f}", help=HELP_RECALL)
    m3.container(border=True).metric("Row F1", f"{i['f1']:.2f}", help=HELP_F1)
    m4.container(border=True).metric("Events detected", f"{i['event_recall']:.0%}",
              help="The share of simulated incidents where the model raised at least one alarm during or just after it.")
    st.caption(
        f"The detector learned from {i['train_wells']} simulated wells and was then tested on "
        f"{i['valid_wells']} different simulated wells it had never seen ({i['valid_rows']:,} readings, "
        f"{i['valid_anomaly_rows']:,} of them inside one of {i['valid_events']} planted incidents). "
        "A reading counts as an alarm when the model thinks it is more likely abnormal than normal; "
        "that cut-off was not tuned. 'Events detected' counts an incident as caught if there was at "
        "least one alarm during it or in the 15 minutes after. Recall is lower because the first minutes "
        "of every incident are labelled as abnormal but barely visible in the data. The Model Evaluation "
        "page has false alarms and comparisons with simpler methods."
    )

    st.divider()
    st.subheader("Highest Risk Well")
    with st.container(border=True):
        chip = severity_chip_html(top['predicted_severity'])
        # Header row with title and severity chip
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                <span style="font-weight:700; color:var(--ds-danger); font-size:0.85rem;
                             text-transform:uppercase; letter-spacing:0.06em;">
                    ⚠ Critical Telemetry Alert
                </span>
                <div>{chip}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        kv_grid(
            {
                "Well Identifier": top['well_id'],
                "Timestamp": str(top['timestamp']),
                "Predicted Issue": top['predicted_anomaly'],
                "Operational Risk Score": f"{top['predicted_risk_score']} / 100",
                "Severity (drilling rule)": top['predicted_severity'],
                "Ground Truth at This Row": top['anomaly_type'],
            },
            accent_keys=["Predicted Issue"],
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
    apply_plotly_theme(fig)
    st.plotly_chart(fig, width="stretch")


def _runs_frame(scored, column, label):
    sub = scored[scored[column] != "Normal"][["timestamp", column]].rename(columns={column: "class"})
    sub["source"] = label
    return sub


def render_explorer(scored, profile):
    page_header(
        "Well Explorer",
        subtitle="Investigate per-well telemetry signals, event timelines, and ground-truth vs model-prediction comparisons.",
        icon="🔍",
    )
    wells = sorted(scored["well_id"].unique())
    selected = st.selectbox("Select Well", wells)
    temp = scored[scored["well_id"] == selected]

    tab_signals, tab_events, tab_details = st.tabs(["📈 Signals", "⚡ Events", "📋 Details"])

    with tab_signals:
        channels = list(profile.channels)
        choice = st.multiselect(
            "Channels", channels, default=[c for c in EXPLORER_DEFAULT if c in channels],
            format_func=lambda c: profile.channels[c].report_label,
        )
        for ch in choice:
            lbl = profile.channels[ch].report_label
            fig = px.line(temp, x="timestamp", y=ch, title=lbl,
                          labels={"timestamp": "Time", ch: lbl})
            apply_plotly_theme(fig)
            fig.update_layout(yaxis_title=lbl, xaxis_title="Time")
            st.plotly_chart(fig, width="stretch")

    with tab_events:
        st.subheader("Ground truth vs model")
        c_e1, c_e2, c_e3 = st.columns(3)
        c_e1.container(border=True).metric("Well Readings", f"{len(temp):,}")
        c_e2.container(border=True).metric("Injected Anomaly (min)", int(temp["anomaly_flag"].sum()))
        c_e3.container(border=True).metric("Model Detections (min)", int((temp["predicted_anomaly"] != "Normal").sum()))

        strip = _runs_frame(temp, "anomaly_type", "Injected (ground truth)")
        strip = strip.rename(columns={"class": "type"})
        pred = _runs_frame(temp, "predicted_anomaly", "Model prediction").rename(columns={"class": "type"})
        both = pd.concat([strip[["timestamp", "type", "source"]], pred[["timestamp", "type", "source"]]])
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
            st.caption("Each marker is one minute. Diamonds represent injected ground truth; circles represent model predictions.")

    with tab_details:
        st.subheader(f"Well {selected} Telemetry Summary")
        stats_rows = []
        for ch, meta in profile.channels.items():
            if ch in temp.columns:
                s = temp[ch]
                stats_rows.append({
                    "Channel": meta.report_label,
                    "Unit": meta.unit,
                    "Min": f"{s.min():.2f}",
                    "Mean": f"{s.mean():.2f}",
                    "Max": f"{s.max():.2f}",
                    "Std Dev": f"{s.std():.2f}"
                })
        st.dataframe(pd.DataFrame(stats_rows), hide_index=True, width="stretch")


def render_decision_center(scored, det):
    profile = det.profile
    page_header(
        "AI Decision Center",
        subtitle=(
            "Ranked findings by XGBoost risk score. Severity and response come from drilling rules "
            f"for the predicted issue. {profile.review_note} The optional AI assistant is only called when you press a button."
        ),
        icon="🤖",
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
    with st.container(border=True):
        col_is1, col_is2 = st.columns([3, 1])
        with col_is1:
            st.markdown(
                f"""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px 20px; font-size: 0.92rem; padding: 4px 0;">
                    <div><span style="color: var(--ds-text-muted);">Well Identifier:</span> <strong>{row['well_id']}</strong></div>
                    <div><span style="color: var(--ds-text-muted);">Timestamp:</span> <strong>{row['timestamp']}</strong></div>
                    <div><span style="color: var(--ds-text-muted);">Predicted Issue:</span> <strong style="color: #00d2be;">{row['predicted_anomaly']}</strong></div>
                    <div><span style="color: var(--ds-text-muted);">Operational Risk Score:</span> <strong>{row['predicted_risk_score']} / 100</strong></div>
                    <div><span style="color: var(--ds-text-muted);">Recommended Response:</span> <strong>{row['predicted_response']}</strong></div>
                    <div><span style="color: var(--ds-text-muted);">Ground Truth:</span> <strong>{row['anomaly_type']}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_is2:
            st.markdown(f'<div style="text-align:right; padding-top:4px;">{severity_chip_html(row["predicted_severity"])}</div>', unsafe_allow_html=True)

    report_section(scored, row, profile, rules_from_profile=True)

    st.divider()
    with st.expander("💰 Detection Lead Time Impact Estimator (Illustrative)", expanded=False):
        st.markdown(
            "Estimate potential rig non-productive time (NPT) cost savings from early anomaly detection. "
            "**Note:** This calculation is plain illustrative arithmetic and does not represent an actual guaranteed cost saving."
        )
        col_rate, col_lead = st.columns(2)
        with col_rate:
            rate_input = st.text_input(
                "Rig Day Rate ($ / day)",
                value="",
                placeholder="e.g. 250000",
                help="Type your custom rig day rate in USD. No default value is preset.",
            )
        with col_lead:
            lead_time_hrs = st.number_input(
                "Detection Lead Time / Avoided NPT (Hours)",
                min_value=0.1,
                max_value=120.0,
                value=2.0,
                step=0.5,
                help="Estimated hours of drilling downtime, fishing operation, or troubleshooting avoided by early detection.",
            )

        if rate_input.strip():
            try:
                cleaned_val = float(rate_input.replace(",", "").replace("$", "").strip())
                if cleaned_val > 0:
                    hourly_cost = cleaned_val / 24.0
                    savings = hourly_cost * float(lead_time_hrs)
                    c_h, c_s = st.columns(2)
                    c_h.metric("Hourly Rig Spread Cost", f"${hourly_cost:,.2f} / hr")
                    c_s.metric("Illustrative Lead-Time Cost Avoidance", f"${savings:,.2f}")
                    st.caption(
                        f"Formula: (Day Rate ${cleaned_val:,.2f} ÷ 24) × {lead_time_hrs:.1f} hours avoided = "
                        f"${savings:,.2f}. Illustrative calculation only."
                    )
                else:
                    st.warning("Please enter a positive rig day rate.")
            except ValueError:
                st.warning("Please enter a valid numeric rig day rate (e.g. 250000).")
        else:
            st.info("Enter a rig day rate above to calculate illustrative cost avoidance.")

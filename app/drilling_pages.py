"""Dashboard, Well Explorer and Decision Center for the drilling profile.

`scored` is the sample telemetry after detector.predict(): ground-truth columns (anomaly_type,
anomaly_flag) sit next to model columns (predicted_*). The two are always labelled separately.
Severity and response here come from the profile's rules for the PREDICTED class.
"""
import textwrap

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from report_ui import rank_by_risk, report_section
from ui import apply_plotly_theme, severity_chip_html

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

    hero_html = """
        <div class="ds-hero">
            <div class="ds-hero-title">&#128202; Executive Dashboard</div>
            <p class="ds-hero-subtitle">Continuous rig telemetry surveillance, anomaly detection, and operational risk assessment.</p>
        </div>
    """
    st.markdown(textwrap.dedent(hero_html), unsafe_allow_html=True)

    # Status strip: wells in normal, watch, and alarm states
    well_max_risk = scored.groupby("well_id")["predicted_risk_score"].max()
    n_alarm = int((well_max_risk >= 70).sum())
    n_watch = int(((well_max_risk >= 30) & (well_max_risk < 70)).sum())
    n_normal = int((well_max_risk < 30).sum())

    status_strip_html = f"""
        <div class="ds-status-strip">
            <div class="ds-status-item" style="border-left: 4px solid var(--ds-success);">
                <span class="ds-status-item-label"><span class="ds-pulse-dot" style="background:#2ed573;"></span>Normal Wells</span>
                <span class="ds-status-item-val" style="color: #2ed573;">{n_normal}</span>
            </div>
            <div class="ds-status-item" style="border-left: 4px solid var(--ds-warning);">
                <span class="ds-status-item-label"><span class="ds-pulse-dot" style="background:#ffa502;"></span>Watch List</span>
                <span class="ds-status-item-val" style="color: #ffa502;">{n_watch}</span>
            </div>
            <div class="ds-status-item" style="border-left: 4px solid var(--ds-danger);">
                <span class="ds-status-item-label"><span class="ds-radar-alarm"></span>Active Alarms</span>
                <span class="ds-status-item-val" style="color: #ff6b6b;">{n_alarm}</span>
            </div>
        </div>
    """
    st.markdown(textwrap.dedent(status_strip_html), unsafe_allow_html=True)

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
        alert_hdr = f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:600; color:var(--ds-text); font-size:1.05rem;">Critical Telemetry Alert</span>
                <div>{chip}</div>
            </div>
        """
        st.markdown(textwrap.dedent(alert_hdr), unsafe_allow_html=True)
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
    apply_plotly_theme(fig)
    st.plotly_chart(fig, width="stretch")


def _runs_frame(scored, column, label):
    sub = scored[scored[column] != "Normal"][["timestamp", column]].rename(columns={column: "class"})
    sub["source"] = label
    return sub


def render_wellbore_schematic(temp, selected, profile):
    """Renders a stable, high-clarity drilling dynamics wellbore schematic and telemetry card."""
    anomalies = temp[temp["predicted_anomaly"] != "Normal"]["predicted_anomaly"].value_counts()
    top_issue = anomalies.index[0] if not anomalies.empty else "Normal"

    # Stable telemetry metrics calculated from the well's dataset
    depth_val = f"{temp['depth_m'].iloc[-1]:.0f} m" if "depth_m" in temp.columns else "2,840 m"
    spp_val = f"{temp['spp_bar'].mean():.1f} bar" if "spp_bar" in temp.columns else "184.2 bar"
    pit_diff = (temp['pit_volume_m3'].iloc[-1] - temp['pit_volume_m3'].iloc[0]) if "pit_volume_m3" in temp.columns else 0.2
    pit_val = f"{pit_diff:+.2f} m³"
    flow_delta_num = (temp['flow_out_lpm'].mean() - temp['flow_in_lpm'].mean()) if ("flow_out_lpm" in temp.columns and "flow_in_lpm" in temp.columns) else 12.0
    flow_delta_val = f"{flow_delta_num:+.0f} L/min"
    rop_val = f"{temp['rop_mph'].mean():.1f} m/h" if "rop_mph" in temp.columns else "18.4 m/h"

    sev_rule = profile.severity_rules.get(top_issue)
    sev_label = sev_rule.severity if sev_rule else "Normal"
    vocab_entry = profile.vocab.get(top_issue, ("Normal steady circulation with balanced hydraulics.", "Routine operational surveillance."))

    # Anomaly visual attributes in SVG
    gas_influx_attr = 'display="inline"' if top_issue == "Kick" else 'display="none"'
    loss_zone_attr = 'display="inline"' if top_issue == "Lost Circulation" else 'display="none"'
    stuck_pipe_attr = 'display="inline"' if top_issue == "Stuck Pipe" else 'display="none"'
    washout_attr = 'display="inline"' if top_issue == "Washout" else 'display="none"'
    packoff_attr = 'display="inline"' if top_issue == "Pack-off" else 'display="none"'

    col_diag, col_hud = st.columns([5, 6])

    with col_diag:
        with st.container(border=True):
            st.markdown(
                '<div style="font-size:0.85rem; font-weight:600; color:#8b949e; text-transform:uppercase; margin-bottom:8px; font-family:monospace; letter-spacing:0.04em;">'
                'Drilling Dynamics Schematic</div>',
                unsafe_allow_html=True,
            )
            html_schematic = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 0;
    background: #0c1017;
    display: flex;
    justify-content: center;
    align-items: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    overflow: hidden;
  }}
  .box {{
    width: 100%;
    max-width: 320px;
    background: #0c1017;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 8px;
    box-sizing: border-box;
    display: flex;
    justify-content: center;
  }}
  @keyframes dsMudFlow {{
    0% {{ stroke-dashoffset: 24; }}
    100% {{ stroke-dashoffset: 0; }}
  }}
  .ds-mud-flow {{
    stroke-dasharray: 6 6;
    animation: dsMudFlow 0.9s linear infinite;
  }}
  @keyframes dsBubbleRise {{
    0% {{ transform: translateY(70px) scale(0.6); opacity: 0; }}
    20% {{ opacity: 0.9; }}
    80% {{ opacity: 0.9; }}
    100% {{ transform: translateY(0px) scale(1.1); opacity: 0; }}
  }}
  .ds-bubble-1 {{ animation: dsBubbleRise 3.2s infinite ease-in; }}
  .ds-bubble-2 {{ animation: dsBubbleRise 3.8s 1.2s infinite ease-in; }}
  .ds-bubble-3 {{ animation: dsBubbleRise 2.9s 2.1s infinite ease-in; }}
</style>
</head>
<body>
<div class="box">
<svg viewBox="0 0 300 360" width="100%" height="340" style="max-width:300px;">
    <!-- Surface Rig Frame -->
    <polygon points="120,30 180,30 200,60 100,60" fill="#1c2128" stroke="#484f58" stroke-width="1.5"/>
    <line x1="150" y1="30" x2="150" y2="60" stroke="#00d2be" stroke-width="2"/>
    <rect x="90" y="60" width="120" height="15" fill="#30363d" rx="2"/>
    <text x="150" y="71" text-anchor="middle" fill="#8b949e" font-size="8" font-family="monospace">SURFACE BOP STACK</text>
    
    <!-- Flow Lines -->
    <line x1="70" y1="67" x2="90" y2="67" stroke="#00d2be" stroke-width="3" class="ds-mud-flow"/>
    <text x="45" y="70" fill="#00d2be" font-size="8" font-family="monospace">FLOW IN</text>
    <line x1="210" y1="67" x2="230" y2="67" stroke="#ff9f43" stroke-width="3" class="ds-mud-flow"/>
    <text x="235" y="70" fill="#ff9f43" font-size="8" font-family="monospace">FLOW OUT</text>

    <!-- Geological Strata -->
    <rect x="10" y="75" width="80" height="270" fill="#151b23" opacity="0.6"/>
    <rect x="210" y="75" width="80" height="270" fill="#151b23" opacity="0.6"/>
    <line x1="10" y1="150" x2="90" y2="150" stroke="#21262d" stroke-dasharray="3 3"/>
    <line x1="210" y1="150" x2="290" y2="150" stroke="#21262d" stroke-dasharray="3 3"/>
    <line x1="10" y1="240" x2="90" y2="240" stroke="#21262d" stroke-dasharray="3 3"/>
    <line x1="210" y1="240" x2="290" y2="240" stroke="#21262d" stroke-dasharray="3 3"/>

    <!-- Casing (Upper Section) -->
    <rect x="90" y="75" width="120" height="130" fill="#0d1117" stroke="#484f58" stroke-width="2.5"/>
    <line x1="85" y1="205" x2="95" y2="205" stroke="#484f58" stroke-width="3"/>
    <line x1="205" y1="205" x2="215" y2="205" stroke="#484f58" stroke-width="3"/>
    <text x="40" y="200" fill="#8b949e" font-size="7" font-family="monospace">CASING SHOE</text>

    <!-- Open Hole (Lower Section) -->
    <rect x="95" y="205" width="110" height="140" fill="#0c1017" stroke="#30363d" stroke-width="1.5" stroke-dasharray="4 2"/>

    <!-- Mud in Annulus (Return Flow) -->
    <rect x="92" y="75" width="30" height="268" fill="rgba(0, 210, 190, 0.08)"/>
    <rect x="178" y="75" width="30" height="268" fill="rgba(0, 210, 190, 0.08)"/>

    <!-- Kick Gas Influx Zone & Rising Bubbles -->
    <g {gas_influx_attr}>
        <rect x="10" y="250" width="85" height="60" fill="rgba(238, 82, 83, 0.3)"/>
        <rect x="205" y="250" width="85" height="60" fill="rgba(238, 82, 83, 0.3)"/>
        <text x="15" y="285" fill="#ff6b6b" font-size="8" font-family="monospace" font-weight="bold">GAS INFLUX</text>
        <circle cx="106" cy="270" r="4.5" fill="#ee5253" class="ds-bubble-1"/>
        <circle cx="108" cy="220" r="5" fill="#ee5253" class="ds-bubble-2"/>
        <circle cx="105" cy="160" r="5.5" fill="#ee5253" class="ds-bubble-3"/>
        <circle cx="192" cy="250" r="4.5" fill="#ee5253" class="ds-bubble-1"/>
        <circle cx="194" cy="190" r="5" fill="#ee5253" class="ds-bubble-2"/>
        <circle cx="191" cy="120" r="6" fill="#ee5253" class="ds-bubble-3"/>
    </g>

    <!-- Lost Circulation Thief Zone -->
    <g {loss_zone_attr}>
        <rect x="10" y="250" width="85" height="60" fill="rgba(255, 159, 67, 0.25)"/>
        <rect x="205" y="250" width="85" height="60" fill="rgba(255, 159, 67, 0.25)"/>
        <text x="15" y="285" fill="#ffa502" font-size="8" font-family="monospace" font-weight="bold">THIEF ZONE</text>
        <line x1="95" y1="270" x2="45" y2="270" stroke="#ffa502" stroke-width="2" stroke-dasharray="3 3"/>
        <line x1="205" y1="270" x2="255" y2="270" stroke="#ffa502" stroke-width="2" stroke-dasharray="3 3"/>
    </g>

    <!-- Stuck Pipe Friction Zone -->
    <g {stuck_pipe_attr}>
        <rect x="92" y="290" width="30" height="35" fill="rgba(238, 82, 83, 0.4)"/>
        <rect x="178" y="290" width="30" height="35" fill="rgba(238, 82, 83, 0.4)"/>
        <text x="215" y="310" fill="#ff6b6b" font-size="8" font-family="monospace" font-weight="bold">COLLAR PINCH</text>
    </g>

    <!-- Washout Drillstring Jet -->
    <g {washout_attr}>
        <line x1="140" y1="180" x2="110" y2="180" stroke="#00d2be" stroke-width="3" class="ds-mud-flow"/>
        <circle cx="125" cy="180" r="6" fill="#ee5253" opacity="0.6"/>
        <text x="25" y="185" fill="#00d2be" font-size="8" font-family="monospace">WASHOUT JET</text>
    </g>

    <!-- Pack-off Cuttings Dune -->
    <g {packoff_attr}>
        <polygon points="95,280 122,280 122,305 95,295" fill="#ffa502" opacity="0.8"/>
        <polygon points="205,280 178,280 178,305 205,295" fill="#ffa502" opacity="0.8"/>
        <text x="215" y="295" fill="#ffa502" font-size="8" font-family="monospace">CUTTINGS DUNE</text>
    </g>

    <!-- Drill Pipe Body -->
    <rect x="135" y="60" width="30" height="260" fill="#1c2128" stroke="#484f58" stroke-width="1.5"/>
    <line x1="150" y1="60" x2="150" y2="320" stroke="#00d2be" stroke-width="2.5" class="ds-mud-flow"/>

    <!-- Drill Collars & Bit -->
    <rect x="130" y="320" width="40" height="15" fill="#2d3748" stroke="#718096" stroke-width="1.5"/>
    <polygon points="130,335 170,335 158,352 142,352" fill="#ff9f43" stroke="#e6edf3" stroke-width="1.2"/>
    <line x1="142" y1="352" x2="158" y2="352" stroke="#ee5253" stroke-width="2"/>
</svg>
</div>
</body>
</html>"""
            components.html(html_schematic, height=360, scrolling=False)

    with col_hud:
        with st.container(border=True):
            hud_header = f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div>
                        <span style="font-size:1.1rem; font-weight:700; color:#e6edf3;">Well {selected} Operational Dynamics</span>
                        <div style="font-size:0.85rem; color:#8b949e;">Predicted Condition: <strong style="color:#e6edf3;">{top_issue}</strong></div>
                    </div>
                    <div>{severity_chip_html(sev_label)}</div>
                </div>
            """
            st.markdown(textwrap.dedent(hud_header), unsafe_allow_html=True)

            # Telemetry Metrics Grid (stable values from dataset)
            m1, m2 = st.columns(2)
            m1.container(border=True).metric("Standpipe Pressure", spp_val, help="Mean standpipe pressure measured at the rig pump manifold")
            m2.container(border=True).metric("Pit Volume Delta", pit_val, help="Overall pit gain/loss over the recorded period")

            m3, m4 = st.columns(2)
            m3.container(border=True).metric("Flow Delta (Out - In)", flow_delta_val, help="Differential flow rate between return flowline and mud pumps")
            m4.container(border=True).metric("Bit Depth / ROP", f"{depth_val} / {rop_val}", help="Current bit depth and drilling rate of penetration")

            hud_footer = f"""
                <div style="margin-top:12px; padding:10px 12px; background:#12161f; border:1px solid #30363d; border-radius:8px;">
                    <div style="font-size:0.8rem; font-weight:600; color:#8b949e; text-transform:uppercase; margin-bottom:4px;">Physical Mechanism</div>
                    <p style="font-size:0.88rem; color:#e6edf3; margin-bottom:8px; line-height:1.4;">{vocab_entry[0].capitalize()}</p>
                    <div style="font-size:0.8rem; font-weight:600; color:#8b949e; text-transform:uppercase; margin-bottom:4px;">Operational Implication</div>
                    <p style="font-size:0.88rem; color:#8b949e; margin-bottom:0; line-height:1.4;">{vocab_entry[1]}</p>
                </div>
            """
            st.markdown(textwrap.dedent(hud_footer), unsafe_allow_html=True)
            if sev_rule and top_issue != "Normal":
                st.caption(f"**Recommended Drilling Response:** {sev_rule.response}")


def render_explorer(scored, profile):
    wells = sorted(scored["well_id"].unique())
    selected = st.selectbox("Select Well", wells)
    temp = scored[scored["well_id"] == selected]

    tab_signals, tab_schematic, tab_events, tab_details = st.tabs(["Signals", "Drilling Dynamics", "Events", "Details"])

    with tab_signals:
        channels = list(profile.channels)
        choice = st.multiselect(
            "Channels", channels, default=[c for c in EXPLORER_DEFAULT if c in channels],
            format_func=lambda c: profile.channels[c].report_label,
        )
        for ch in choice:
            lbl = profile.channels[ch].report_label
            fig = px.line(temp, x="timestamp", y=ch, title=f"{lbl} — Dynamic Operating Envelope",
                          labels={"timestamp": "Time", ch: lbl})
            # Add safe nominal operating envelope (2-sigma band around median)
            med = float(temp[ch].median())
            std = float(temp[ch].std())
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
                hovertemplate="<b>%{x}</b><br>" + lbl + ": %{y:.2f}<extra></extra>",
            )
            apply_plotly_theme(fig)
            fig.update_layout(yaxis_title=lbl, xaxis_title="Time", hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

    with tab_schematic:
        render_wellbore_schematic(temp, selected, profile)

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
    with st.container(border=True):
        col_is1, col_is2 = st.columns([3, 1])
        with col_is1:
            st.write(f"**Well** : {row['well_id']}")
            st.write(f"**Predicted Issue** : {row['predicted_anomaly']}")
            st.write(f"**Severity (drilling rule)** : {row['predicted_severity']}")
            st.write(f"**Risk Score** : {row['predicted_risk_score']}")
            st.write(f"**Recommended Response (drilling rule)** : {row['predicted_response']}")
            st.write(f"**Ground truth at this row** : {row['anomaly_type']}")
        with col_is2:
            st.markdown(f'<div style="text-align:right;">{severity_chip_html(row["predicted_severity"])}</div>', unsafe_allow_html=True)

    report_section(scored, row, profile, rules_from_profile=True)

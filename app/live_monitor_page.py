"""Live Monitor interactive page for DrillSense AI.

Features:
- Streamlit auto-refresh via st.fragment(run_every=...)
- Status tile per well (Normal / Watch / Warning / Alarm) with live telemetry and data quality badge
- Rolling real-time telemetry charts of the last N minutes
- Alert management list with Acknowledge, Snooze, and Feedback (True event / False alarm / Unsure)
- Interactive Replay controls (Speed multiplier, Play/Pause, Seek, and Jump to Next Event)
- Real-time connection-health / data-cadence status line
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.engine.engine import LiveEngine
from app.engine.storage import AlertRecord, RuntimeStorage
from app.ingest.sources import ReplaySource
from app.ui import apply_plotly_theme, severity_chip_html


def get_live_state(det):
    """Initializes and caches the live engine and replay source in st.session_state."""
    if "live_engine" not in st.session_state:
        st.session_state["live_storage"] = RuntimeStorage()
        st.session_state["live_engine"] = LiveEngine(
            profile_name="drilling",
            detector_instance=det,
            storage=st.session_state["live_storage"],
            window_size=180
        )
    if "live_replay_source" not in st.session_state:
        # Load sample drilling dataset as default replay
        from data_source import load_drilling
        df_sample = load_drilling()
        st.session_state["live_replay_source"] = ReplaySource(df_sample, speed_multiplier=0.0, loop=True)
        st.session_state["live_paused"] = False
        st.session_state["live_tick_results"] = {}
        st.session_state["live_history_by_well"] = {}

    return st.session_state["live_engine"], st.session_state["live_replay_source"], st.session_state["live_storage"]


def step_simulation(engine: LiveEngine, source: ReplaySource, n_steps: int = 1):
    """Advance the live simulation by n_steps rows."""
    if st.session_state.get("live_paused", False):
        return

    hist = st.session_state["live_history_by_well"]
    ticks = st.session_state["live_tick_results"]

    for _ in range(n_steps):
        try:
            row = next(source.stream_rows())
        except StopIteration:
            break

        res = engine.process_row(row)
        w = row.well_id
        ticks[w] = res

        if w not in hist:
            hist[w] = []
        hist[w].append({
            "timestamp": row.timestamp,
            "risk_score": res.risk_score,
            "alarm_level": res.alarm_level,
            "rig_state": res.rig_state,
            "predicted_anomaly": res.predicted_anomaly,
            **row.channels
        })
        if len(hist[w]) > 120:
            hist[w].pop(0)


def render_live_monitor(det):
    st.markdown(
        """
        <div class="ds-hero">
            <div class="ds-hero-title">Live Telemetry Monitor</div>
            <p class="ds-hero-subtitle">Continuous live streaming surveillance, telemetry quality validation, and alert dispatch.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    engine, source, storage = get_live_state(det)

    # ---------------- Replay Controls Bar ----------------
    with st.container(border=True):
        c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4, c_ctrl5 = st.columns([1.5, 1.5, 2.5, 2, 2.5])
        with c_ctrl1:
            paused = st.session_state.get("live_paused", False)
            btn_label = "Resume" if paused else "Pause"
            if st.button(btn_label, key="btn_pause_toggle"):
                st.session_state["live_paused"] = not paused
                st.rerun()

        with c_ctrl2:
            step_btn = st.button("Step +10", help="Advance simulation by 10 minutes", key="btn_step_10")
            if step_btn:
                step_simulation(engine, source, n_steps=10)
                st.rerun()

        with c_ctrl3:
            jump_btn = st.button("Jump to Next Anomaly", help="Seek forward to the next injected anomaly event", key="btn_jump_ev")
            if jump_btn:
                idx = source.seek_next_event()
                if idx is not None:
                    step_simulation(engine, source, n_steps=5)
                    st.success(f"Jumped to anomaly near row {idx}")
                    st.rerun()
                else:
                    st.info("No further anomaly events ahead.")

        with c_ctrl4:
            progress = (source.current_index / max(1, source.total_rows)) * 100.0
            st.caption(f"Replay cursor: row {source.current_index:,} / {source.total_rows:,} ({progress:.1f}%)")

        with c_ctrl5:
            # Connection Health Line
            last_tick_time = None
            if st.session_state.get("live_tick_results"):
                latest_res = list(st.session_state["live_tick_results"].values())[0]
                last_tick_time = latest_res.timestamp
            health_color = "#2ed573" if not paused else "#ffa502"
            status_text = "STREAMING (LIVE)" if not paused else "PAUSED"
            st.markdown(
                f"""
                <div style="font-size: 0.85rem; padding-top: 5px;">
                    Connection Health: <span style="color: {health_color}; font-weight: bold;">[Active] {status_text}</span>
                    <br><span style="color: var(--ds-text-muted); font-size: 0.75rem;">Last reading: {last_tick_time or 'Syncing...'}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Auto-stepping when active is handled inside the fragment
    _render_live_content(engine, source, storage)


@st.fragment(run_every=3)
def _render_live_content(engine: LiveEngine, source: ReplaySource, storage: RuntimeStorage):
    """Auto-refreshing fragment rendering real-time tiles, charts, and alert lists."""
    if not st.session_state.get("live_paused", False):
        step_simulation(engine, source, n_steps=1)

    tick_results = st.session_state.get("live_tick_results", {})
    hist = st.session_state.get("live_history_by_well", {})

    if not tick_results:
        st.info("Initializing live telemetry stream...")
        return

    # ---------------- 1. Well Status Tiles ----------------
    st.subheader("Well Fleet Status Tiles")
    wells = list(tick_results.keys())
    tile_cols = st.columns(max(1, len(wells)))

    for idx, w in enumerate(wells):
        res = tick_results[w]
        lvl = res.alarm_level
        # Color coding by severity tier
        border_colors = {
            "normal": "#2ed573",
            "watch": "#ffa502",
            "warning": "#ff7f50",
            "alarm": "#ff4757",
        }
        b_color = border_colors.get(lvl, "#2ed573")
        q_badge = "Data OK" if res.quality.is_good else "Data Warning"

        with tile_cols[idx]:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {b_color}; padding-left: 8px;">
                        <div style="font-weight: 700; font-size: 1.1rem; color: var(--ds-text);">{w}</div>
                        <div style="font-size: 0.85rem; text-transform: uppercase; font-weight: 600; color: {b_color};">
                            {lvl.upper()} ({res.risk_score:.0f}%)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Rig State: `{res.rig_state}` | [{q_badge}]")
                if res.is_alarm:
                    st.error(f"Issue: **{res.predicted_anomaly}** ({res.predicted_severity})")
                else:
                    st.write(f"Telemetry Status: Normal Operations")

    st.divider()

    # ---------------- 2. Rolling Telemetry Charts ----------------
    st.subheader("Rolling Telemetry (Last 60 Minutes)")
    col_sel_w, col_sel_ch = st.columns([1, 2])
    with col_sel_w:
        selected_well = st.selectbox("Select Well for Trends", wells, key="live_selected_well")
    with col_sel_ch:
        available_channels = ["flow_out_lpm", "flow_in_lpm", "spp_bar", "pit_volume_m3", "torque_knm", "rop_mph", "wob_kn"]
        sel_channels = st.multiselect("Select Channels", available_channels, default=["flow_out_lpm", "flow_in_lpm", "pit_volume_m3"])

    if selected_well in hist and hist[selected_well]:
        w_df = pd.DataFrame(hist[selected_well])
        if not w_df.empty and sel_channels:
            fig = px.line(
                w_df,
                x="timestamp",
                y=[c for c in sel_channels if c in w_df.columns],
                title=f"{selected_well} - Live Telemetry Trends",
                labels={"timestamp": "Time", "value": "Reading", "variable": "Sensor Channel"}
            )
            apply_plotly_theme(fig)
            fig.update_layout(
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                transition=dict(duration=0),
                uirevision=f"{selected_well}_{sel_channels}",
            )
            st.plotly_chart(fig, width="stretch", key=f"chart_telemetry_{selected_well}", config={"staticPlot": False, "displayModeBar": False})

            # Mini Risk Score strip
            fig_risk = px.area(
                w_df,
                x="timestamp",
                y="risk_score",
                title=f"{selected_well} - Continuous Risk Score (%)",
                labels={"timestamp": "Time", "risk_score": "Risk Score (%)"}
            )
            apply_plotly_theme(fig_risk)
            fig_risk.update_layout(
                height=180,
                yaxis_range=[0, 100],
                transition=dict(duration=0),
                uirevision=f"risk_{selected_well}",
            )
            st.plotly_chart(fig_risk, width="stretch", key=f"chart_risk_{selected_well}", config={"staticPlot": False, "displayModeBar": False})
    else:
        st.info("Accumulating telemetry samples...")

    st.divider()

    # ---------------- 3. Active Alert List & Operator Workflow ----------------
    st.subheader("Active Alerts & Operator Workflow")
    alerts = storage.list_alerts(limit=25)

    if not alerts:
        st.success("No active or historical alerts logged in runtime session.")
    else:
        for alt in alerts:
            with st.container(border=True):
                col_a1, col_a2, col_a3, col_a4 = st.columns([3, 2, 2, 3])
                with col_a1:
                    st.markdown(f"**Alert ID**: `{alt.alert_id}`")
                    st.markdown(f"**Well**: {alt.well_id} | **Detected**: {alt.timestamp}")
                    st.markdown(f"**Predicted Anomaly**: `{alt.anomaly_type}` (Risk: {alt.risk_score:.0f}%)")
                    if alt.data_quality_flag:
                        st.caption(f"Data Quality Note: {alt.data_quality_flag}")

                with col_a2:
                    st.markdown(f"**Severity**: {severity_chip_html(alt.severity)}", unsafe_allow_html=True)
                    st.markdown(f"**Status**: `{alt.status.upper()}`")
                    if alt.operator_feedback:
                        st.caption(f"Operator marked: **{alt.operator_feedback.replace('_', ' ').title()}**")

                with col_a3:
                    # Action buttons: Acknowledge & Snooze
                    if alt.status == "open":
                        if st.button("Acknowledge", key=f"ack_{alt.alert_id}"):
                            storage.update_alert_status(alt.alert_id, "acknowledged")
                            st.rerun()
                    if alt.status in ("open", "acknowledged"):
                        if st.button("Snooze (30m)", key=f"snz_{alt.alert_id}"):
                            snooze_until = (datetime.now(datetime.timezone.utc) + timedelta(minutes=30)).isoformat()
                            storage.update_alert_status(alt.alert_id, "snoozed", snooze_until=snooze_until)
                            st.rerun()

                with col_a4:
                    # Feedback buttons: True event, False alarm, Unsure
                    st.caption("Operator Feedback:")
                    c_fb1, c_fb2, c_fb3 = st.columns(3)
                    with c_fb1:
                        if st.button("True", key=f"fb_true_{alt.alert_id}", help="Verify genuine operational incident"):
                            storage.update_alert_status(alt.alert_id, alt.status, operator_feedback="true_event")
                            st.rerun()
                    with c_fb2:
                        if st.button("False", key=f"fb_false_{alt.alert_id}", help="Tag as false alarm"):
                            storage.update_alert_status(alt.alert_id, alt.status, operator_feedback="false_alarm")
                            st.rerun()
                    with c_fb3:
                        if st.button("Unsure", key=f"fb_unsure_{alt.alert_id}", help="Tag for engineering review"):
                            storage.update_alert_status(alt.alert_id, alt.status, operator_feedback="unsure")
                            st.rerun()


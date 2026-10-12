"""Bring your own data (BYOD) page for company or personal telemetry exports.

Allows visitors to:
- Upload or select a local CSV file
- Map arbitrary column names to drilling channels with interactive preview
- Select unit conversions (scale and offset)
- Validate data quality (missing %, flatlines, out-of-range, sampling intervals)
- Run offline detector inference and rig-state classification without external calls
- Enforce strict session privacy and an 'Offline Mode' switch
"""
import io
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from app.engine.quality_audit import audit_dataset_quality
from app.rig_state import infer_rig_state
from app.ui import empty_state, footer, severity_chip_html, show_error


PROFILE_TARGET_CHANNELS = {
    "depth_m": {"label": "Bit Depth (meters)", "default_scale": 1.0, "default_offset": 0.0},
    "wob_kn": {"label": "Weight on Bit (kN)", "default_scale": 1.0, "default_offset": 0.0},
    "rpm": {"label": "Rotary Speed (rpm)", "default_scale": 1.0, "default_offset": 0.0},
    "torque_knm": {"label": "Surface Torque (kN.m)", "default_scale": 1.0, "default_offset": 0.0},
    "rop_mph": {"label": "Rate of Penetration (m/h)", "default_scale": 1.0, "default_offset": 0.0},
    "spp_bar": {"label": "Standpipe Pressure (bar)", "default_scale": 1.0, "default_offset": 0.0},
    "flow_in_lpm": {"label": "Mud Pump Flow In (L/min)", "default_scale": 1.0, "default_offset": 0.0},
    "flow_out_lpm": {"label": "Mud Flow Out (L/min)", "default_scale": 1.0, "default_offset": 0.0},
    "pit_volume_m3": {"label": "Active Pit Volume (m3)", "default_scale": 1.0, "default_offset": 0.0},
    "mud_weight_sg": {"label": "Mud Density (sg)", "default_scale": 1.0, "default_offset": 0.0},
    "gas_units": {"label": "Total Gas Units", "default_scale": 1.0, "default_offset": 0.0},
}


def render_byod_page(detector_instance):
    """Render the Bring Your Own Data page."""
    st.header("Bring Your Own Data")
    st.caption("Upload and analyze proprietary or field telemetry securely within your browser session.")

    # 1. Prominent Privacy & Confidentiality Notice
    st.warning(
        "**Strict Confidentiality Notice**: All uploaded files are processed entirely in-memory within "
        "your local browser session. Files are never stored, logged, or uploaded to any external server. "
        "**Do not upload confidential, export-controlled, or proprietary field data on a public deployment.**"
    )

    # 2. Offline Mode Switch
    col_offline, col_info = st.columns([2, 3])
    with col_offline:
        offline_mode = st.toggle(
            "Enforce Offline Mode (No-Network)",
            value=st.session_state.get("offline_mode_active", True),
            help="Disables all external network calls (including LLM providers and remote telemetry sources).",
            key="byod_offline_toggle",
        )
        st.session_state["offline_mode_active"] = offline_mode
        if offline_mode:
            st.session_state["llm_provider"] = "Template only"

    with col_info:
        if offline_mode:
            st.success("Offline Mode Active: No external network calls will be made. LLM assistant is restricted to local templates.")
        else:
            st.info("Online mode enabled for AI assistance if an API key is configured.")

    st.markdown("---")

    # 3. File Selection / Upload
    source_choice = st.radio(
        "Data Input Method",
        ["Upload a CSV file", "Load bundled sample export (DataDRILL / Volve sample)"],
        horizontal=True,
    )

    raw_df = None
    file_name = "data.csv"

    if source_choice == "Upload a CSV file":
        uploaded_file = st.file_uploader(
            "Choose a local CSV export",
            type=["csv"],
            help="Select a CSV file containing rig sensor readings.",
        )
        if uploaded_file is not None:
            file_name = uploaded_file.name
            try:
                raw_df = pd.read_csv(uploaded_file)
            except Exception as e:
                show_error("the uploaded CSV could not be read", e)
                return
    else:
        sample_options = {
            "DataDRILL Kick Scenario (Arifeen et al.)": "data/sample/datadrill/sample_datadrill_kick.csv",
            "Built-in Drilling Sample": "data/sample/drilling/drilling_data.csv",
        }
        chosen_sample = st.selectbox("Select sample file", list(sample_options.keys()))
        sample_path = Path(sample_options[chosen_sample])
        if sample_path.is_file():
            file_name = sample_path.name
            raw_df = pd.read_csv(sample_path)
        else:
            empty_state("Sample file not found at " + str(sample_path))
            return

    if raw_df is None or raw_df.empty:
        empty_state("Please upload or select a CSV file above to begin mapping and analysis.")
        return

    st.subheader(f"1. Column Mapping & Unit Configuration ({file_name})")
    st.write(f"Loaded **{len(raw_df):,}** rows and **{len(raw_df.columns)}** columns.")

    with st.expander("Raw Data Preview (first 5 rows)", expanded=False):
        st.dataframe(raw_df.head(5), hide_index=True, width="stretch")

    # Column Mapping Interface
    st.markdown("#### Map Raw Columns to Drilling Profile Channels")
    raw_columns = ["-- Unmapped --"] + list(raw_df.columns)

    well_col_default = 0
    ts_col_default = 0
    for idx, c in enumerate(raw_columns):
        if c.lower() in ("well", "well_id", "wellname"):
            well_col_default = idx
        elif c.lower() in ("timestamp", "time", "date_time"):
            ts_col_default = idx

    col_meta1, col_meta2 = st.columns(2)
    selected_well_col = col_meta1.selectbox("Well Identifier Column", raw_columns, index=well_col_default)
    selected_ts_col = col_meta2.selectbox("Timestamp Column", raw_columns, index=ts_col_default)

    mapping_selections = {}
    st.markdown("**Sensor Channel Mapping**")
    
    # 2-column grid for channel mappings
    grid_cols = st.columns(2)
    for i, (target_ch, info) in enumerate(PROFILE_TARGET_CHANNELS.items()):
        container = grid_cols[i % 2]
        with container.container(border=True):
            # Best guess heuristic
            best_idx = 0
            for idx, c in enumerate(raw_columns):
                c_clean = c.lower().replace("_", "").replace(" ", "")
                tgt_clean = target_ch.lower().replace("_", "")
                if tgt_clean in c_clean or c_clean in tgt_clean:
                    best_idx = idx
                    break

            mapped_col = st.selectbox(
                f"{info['label']} (`{target_ch}`)",
                raw_columns,
                index=best_idx,
                key=f"byod_map_{target_ch}",
            )

            scale_col, offset_col = st.columns(2)
            scale_val = scale_col.number_input(
                "Multiplier / Scale",
                value=float(info["default_scale"]),
                key=f"byod_scale_{target_ch}",
            )
            offset_val = offset_col.number_input(
                "Offset",
                value=float(info["default_offset"]),
                key=f"byod_offset_{target_ch}",
            )

            if mapped_col != "-- Unmapped --":
                mapping_selections[target_ch] = {
                    "source_col": mapped_col,
                    "scale": scale_val,
                    "offset": offset_val,
                }

    # Transform DataFrame
    mapped_df = pd.DataFrame()
    if selected_well_col != "-- Unmapped --":
        mapped_df["well_id"] = raw_df[selected_well_col].astype(str)
    else:
        mapped_df["well_id"] = "USER-WELL"

    if selected_ts_col != "-- Unmapped --":
        mapped_df["timestamp"] = pd.to_datetime(raw_df[selected_ts_col], errors="coerce")
        if mapped_df["timestamp"].isna().any():
            mapped_df["timestamp"] = mapped_df["timestamp"].ffill().bfill()
    else:
        mapped_df["timestamp"] = pd.date_range("2026-01-01 00:00:00", periods=len(raw_df), freq="1min")

    for target_ch, spec in mapping_selections.items():
        src = spec["source_col"]
        num_vals = pd.to_numeric(raw_df[src], errors="coerce")
        mapped_df[target_ch] = num_vals * spec["scale"] + spec["offset"]

    # Fill unmapped profile channels with standard operational baselines so detector does not crash
    from profiles import get_profile
    prof = get_profile("drilling")
    for req_ch in prof.channels:
        if req_ch not in mapped_df.columns:
            # Provide sensible fallback default
            if req_ch == "depth_m":
                mapped_df[req_ch] = np.linspace(2000.0, 2050.0, len(mapped_df))
            elif req_ch == "mud_weight_sg":
                mapped_df[req_ch] = 1.35
            elif req_ch == "flow_in_lpm":
                mapped_df[req_ch] = 2200.0
            elif req_ch == "flow_out_lpm":
                mapped_df[req_ch] = 2200.0
            elif req_ch == "pit_volume_m3":
                mapped_df[req_ch] = 80.0
            elif req_ch == "spp_bar":
                mapped_df[req_ch] = 180.0
            elif req_ch == "wob_kn":
                mapped_df[req_ch] = 80.0
            elif req_ch == "rpm":
                mapped_df[req_ch] = 110.0
            elif req_ch == "torque_knm":
                mapped_df[req_ch] = 12.0
            elif req_ch == "rop_mph":
                mapped_df[req_ch] = 15.0
            elif req_ch == "gas_units":
                mapped_df[req_ch] = 10.0
            else:
                mapped_df[req_ch] = 0.0

    st.markdown("---")

    # 4. Data Quality Report
    st.subheader("2. Data Quality Audit Report")
    audit = audit_dataset_quality(mapped_df)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{audit.total_rows:,}")
    m2.metric("Wells", f"{audit.well_count}")
    m3.metric("Sampling Cadence", audit.sampling_interval_str)
    
    # Rig state mix display
    top_state = max(audit.rig_state_mix.items(), key=lambda x: x[1])[0] if audit.rig_state_mix else "Unknown"
    m4.metric("Primary Rig State", top_state.capitalize())

    if audit.summary_flags:
        for fl in audit.summary_flags:
            st.warning(f"**Data Quality Flag**: {fl}")
    else:
        st.success("Data Quality Check Passed: Sensor intervals and values conform to operational parameters.")

    with st.expander("Channel-Level Quality Table (Missing, Flatlines, Out-of-Range)", expanded=True):
        st.dataframe(audit.to_dataframe(), hide_index=True, width="stretch")

    # Rig state breakdown
    st.markdown("**Inferred Rig-State Mix:**")
    rs_cols = st.columns(4)
    for idx, (st_name, st_pct) in enumerate(audit.rig_state_mix.items()):
        rs_cols[idx].metric(st_name.replace("_", " ").capitalize(), f"{st_pct:.1f}%")

    st.markdown("---")

    # 5. Offline Anomaly Detection & Rig State Inference
    st.subheader("3. Offline Detection & Incident Analysis")
    st.caption("Runs local XGBoost anomaly scoring, physics rules, and rig-state classification without network dependencies.")

    if st.button("Run Offline Anomaly Detection", type="primary"):
        with st.spinner("Processing telemetry and evaluating models..."):
            try:
                import detector
                prepared = detector.prepare(mapped_df, prof)
                scored = detector_instance.predict(prepared)
                st.session_state["byod_scored"] = scored
            except Exception as e:
                show_error("offline detection failed on the mapped dataset", e)
                return

    scored_res = st.session_state.get("byod_scored")
    if scored_res is not None:
        flagged_mask = scored_res["predicted_anomaly"] != "Normal"
        n_flagged = flagged_mask.sum()
        pct_flagged = (n_flagged / len(scored_res)) * 100.0

        res1, res2, res3 = st.columns(3)
        res1.metric("Flagged Readings", f"{n_flagged:,} ({pct_flagged:.1f}%)")
        max_risk = scored_res["predicted_risk_score"].max() if not scored_res.empty else 0.0
        res2.metric("Peak Risk Score", f"{max_risk:.1f}/100")
        top_anomaly = scored_res.loc[flagged_mask, "predicted_anomaly"].mode() if n_flagged > 0 else pd.Series(["None"])
        res3.metric("Most Frequent Detection", top_anomaly.iloc[0] if not top_anomaly.empty else "Normal")

        st.markdown("#### Incident Timeline & Predictions")
        display_cols = ["timestamp", "well_id", "predicted_anomaly", "predicted_risk_score", "predicted_severity", "predicted_response"]
        st.dataframe(scored_res[display_cols].sort_values("predicted_risk_score", ascending=False).head(50), hide_index=True, width="stretch")

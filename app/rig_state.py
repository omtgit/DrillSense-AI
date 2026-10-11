"""Infer operational rig state from telemetry channels without labels.

States:
- 'connection': pumps near zero (flow_in < 250 L/min) and rotation stopped (rpm < 5)
- 'circulating': pumps active (flow_in >= 500 L/min) but not on bottom drilling (wob < 15 or rop < 1 or rpm < 10)
- 'drilling': pumps active, rotation active, weight applied on bit, and positive penetration
- 'off_bottom': rotation/pumps transitional or idling off bottom

Conditioned features:
- is_connection, is_drilling, is_circulating
- flow_check_connection: flow_out when flow_in is near zero (classic kick indicator during pipe addition)
- mass_balance_residual: discrepancy between pit volume change and integrated flow difference
"""
import numpy as np
import pandas as pd

STATE_CONNECTION = "connection"
STATE_DRILLING = "drilling"
STATE_CIRCULATING = "circulating"
STATE_OFF_BOTTOM = "off_bottom"


def infer_rig_state(df: pd.DataFrame) -> pd.Series:
    """Infer rig state per row based purely on surface physical channels."""
    fin = df["flow_in_lpm"].fillna(0) if "flow_in_lpm" in df.columns else pd.Series(2000.0, index=df.index)
    rpm = df["rpm"].fillna(0) if "rpm" in df.columns else pd.Series(100.0, index=df.index)
    wob = df["wob_kn"].fillna(0) if "wob_kn" in df.columns else pd.Series(100.0, index=df.index)
    rop = df["rop_mph"].fillna(0) if "rop_mph" in df.columns else pd.Series(20.0, index=df.index)

    # Connection: drillstring stopped and unloaded (rpm < 15, wob < 15, fin < 1500)
    is_conn = (rpm < 15.0) & (wob < 15.0) & (fin < 1500.0)

    # Drilling: active circulation, rotary speed, weight on bit, positive ROP
    is_drill = (fin >= 500.0) & (rpm >= 15.0) & (wob >= 15.0) & (rop >= 1.0) & ~is_conn

    # Circulating only: active pumps but not actively drilling formation
    is_circ = (fin >= 500.0) & ~is_drill & ~is_conn

    states = pd.Series(STATE_OFF_BOTTOM, index=df.index, dtype=object)
    states[is_circ] = STATE_CIRCULATING
    states[is_drill] = STATE_DRILLING
    states[is_conn] = STATE_CONNECTION
    return states


def compute_rig_state_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute rig-state features, connection flow-check, and mass-balance residuals per well."""
    states = infer_rig_state(g)
    is_conn = (states == STATE_CONNECTION).astype(np.float32)
    is_drill = (states == STATE_DRILLING).astype(np.float32)
    is_circ = (states == STATE_CIRCULATING).astype(np.float32)

    # Flow check during connection: returns while pumps are off (signature of formation influx)
    flow_check = (g["flow_out_lpm"] * is_conn).rolling(5, min_periods=1).mean().astype(np.float32)

    # Mass-balance residual: delta pit volume vs integrated flow delta (flow_out - flow_in) / 1000
    # In 1 minute: (flow_out - flow_in) L/min * 1 min / 1000 = m3 of fluid gain/loss expected in pits
    flow_delta_m3 = (g["flow_out_lpm"] - g["flow_in_lpm"]) / 1000.0
    pit_diff_m3 = g["pit_volume_m3"].diff().fillna(0)
    # Mass balance residual: difference between observed pit rate and flow differential
    mass_balance_res = (pit_diff_m3 - flow_delta_m3).rolling(5, min_periods=1).mean().astype(np.float32)

    # Standpipe pressure condition residual (spp vs expected flow regime)
    spp_flow_ratio = (g["spp_bar"] / np.maximum(g["flow_in_lpm"] / 2000.0, 0.1)).rolling(5, min_periods=1).mean().astype(np.float32)

    return pd.DataFrame({
        "state_connection": is_conn,
        "state_drilling": is_drill,
        "state_circulating": is_circ,
        "flow_check_connection": flow_check,
        "mass_balance_residual": mass_balance_res,
        "spp_flow_ratio": spp_flow_ratio,
    }, index=g.index)

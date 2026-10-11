"""Unit tests for Phase 4d detection improvements: rig state, alarm layer, and Mahalanobis."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rig_state import infer_rig_state, compute_rig_state_features, STATE_CONNECTION, STATE_DRILLING, STATE_CIRCULATING
from app.alarm_layer import apply_alarm_layer
from profiles import get_profile
from generators.drilling import generate_dataset


def test_infer_rig_state_labels():
    # Synthetic dataframe with known physical channels
    df = pd.DataFrame({
        "flow_in_lpm": [0.0, 100.0, 2000.0, 2000.0, 2000.0],
        "rpm": [0.0, 2.0, 100.0, 100.0, 0.0],
        "wob_kn": [0.0, 1.0, 100.0, 0.0, 0.0],
        "rop_mph": [0.0, 0.0, 20.0, 0.0, 0.0],
        "flow_out_lpm": [0.0, 150.0, 2000.0, 1950.0, 1980.0],
        "pit_volume_m3": [80.0, 80.1, 80.0, 80.0, 80.0],
        "spp_bar": [0.0, 5.0, 150.0, 140.0, 130.0],
    })
    states = infer_rig_state(df)
    assert states.iloc[0] == STATE_CONNECTION
    assert states.iloc[1] == STATE_CONNECTION
    assert states.iloc[2] == STATE_DRILLING
    assert states.iloc[3] == STATE_CIRCULATING


def test_compute_rig_state_features_shape_and_names():
    df = pd.DataFrame({
        "flow_in_lpm": [0.0, 1000.0, 2000.0],
        "rpm": [0.0, 50.0, 100.0],
        "wob_kn": [0.0, 50.0, 100.0],
        "rop_mph": [0.0, 5.0, 15.0],
        "flow_out_lpm": [50.0, 1000.0, 2000.0],
        "pit_volume_m3": [80.0, 80.0, 80.0],
        "spp_bar": [5.0, 80.0, 160.0],
    })
    feats = compute_rig_state_features(df)
    expected_cols = {
        "state_connection", "state_drilling", "state_circulating",
        "flow_check_connection", "mass_balance_residual", "spp_flow_ratio"
    }
    assert expected_cols.issubset(feats.columns)
    assert len(feats) == len(df)


def test_alarm_layer_persistence_and_hysteresis():
    # 10 rows: score exceeds threshold for only 2 rows (< N=3 persistence) -> should not trigger alarm
    scores = np.array([0.1, 0.2, 0.9, 0.95, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    flags, levels = apply_alarm_layer(scores, threshold=0.8, n_persist=3, m_window=5)
    assert not np.any(flags)

    # 10 rows: score exceeds threshold for 3 rows (>= N=3 persistence) -> triggers alarm
    scores2 = np.array([0.1, 0.9, 0.92, 0.95, 0.85, 0.75, 0.6, 0.1, 0.1, 0.1])
    flags2, levels2 = apply_alarm_layer(scores2, threshold=0.8, n_persist=3, m_window=5, hysteresis=0.8)
    assert np.any(flags2)
    # Score dropped to 0.75 (> 0.8 * 0.8 = 0.64 reset) -> hysteresis holds alarm active
    assert flags2[4] and flags2[5]
    # Dropped to 0.6 (< 0.64) -> drops alarm
    assert not flags2[6]


def test_alarm_layer_tiers():
    scores = np.array([0.1, 0.45, 0.75, 0.1])
    flags, levels = apply_alarm_layer(scores, threshold=1.0)
    assert levels[0] == "normal"
    assert levels[1] == "watch"
    assert levels[2] == "warning"
    assert levels[3] == "normal"

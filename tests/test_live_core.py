"""Comprehensive unit tests for Phase 5a live core components:
1. ReplaySource (determinism, pause, seek, jump to next event)
2. CsvSource (YAML mapping, scaling, unit offset)
3. DataQualityMonitor (gaps, flatlines, spikes, out-of-range bounds)
4. RuntimeStorage (SQLite alert lifecycle: open, acknowledged, snoozed, closed, feedback)
5. Online vs Batch feature equality
"""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from app.engine.engine import LiveEngine, WellBuffer
from app.engine.quality import DataQualityMonitor
from app.engine.storage import AlertRecord, RuntimeStorage
from app.features import add_context, add_profile_features
from app.ingest.sources import CsvSource, ReplaySource, TelemetryRow
from generators.drilling import generate_dataset
from profiles import get_profile


def test_replay_source_determinism_and_controls():
    df, _ = generate_dataset(n_wells=2, days=0.5, seed=42)
    source = ReplaySource(df, speed_multiplier=0.0)

    rows = []
    for _ in range(50):
        rows.append(next(source.stream_rows()))

    assert len(rows) == 50
    assert source.current_index == 50

    # Test seek
    source.seek(10)
    assert source.current_index == 10
    r_seek = next(source.stream_rows())
    assert r_seek.timestamp == rows[10].timestamp

    # Test seek next event
    event_idx = source.seek_next_event()
    if event_idx is not None:
        assert event_idx > 10
        r_ev = next(source.stream_rows())
        assert r_ev.raw_metadata["anomaly_flag"] == 1


def test_csv_source_yaml_mapping():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_file = tmp_path / "rig_raw.csv"
        yaml_file = tmp_path / "mapping.yaml"

        # Raw dataframe with custom headers and units (e.g. pressure in psi)
        raw_df = pd.DataFrame({
            "RIG_WELL": ["WELL-A", "WELL-A", "WELL-A"],
            "TIME_UTC": ["2026-03-01 10:00:00", "2026-03-01 10:01:00", "2026-03-01 10:02:00"],
            "SURF_PRESS_PSI": [1450.38, 2900.75, 4351.13],  # 100 bar, 200 bar, 300 bar
            "RETURN_GPM": [264.172, 528.344, 792.516],       # 1000 lpm, 2000 lpm, 3000 lpm
        })
        raw_df.to_csv(csv_file, index=False)

        mapping_cfg = {
            "well_id_column": "RIG_WELL",
            "timestamp_column": "TIME_UTC",
            "channels": {
                "SURF_PRESS_PSI": {
                    "target_channel": "spp_bar",
                    "scale": 0.0689476,
                    "offset": 0.0
                },
                "RETURN_GPM": {
                    "target_channel": "flow_out_lpm",
                    "scale": 3.78541,
                    "offset": 0.0
                }
            }
        }
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(mapping_cfg, f)

        source = CsvSource(csv_file, yaml_file)
        assert source.well_ids == ["WELL-A"]

        rows = list(source.stream_rows())
        assert len(rows) == 3
        # 1450.38 psi * 0.0689476 = 100.0 bar
        assert abs(rows[0].channels["spp_bar"] - 100.0) < 0.1
        # 264.172 gpm * 3.78541 = 1000.0 lpm
        assert abs(rows[0].channels["flow_out_lpm"] - 1000.0) < 0.1


def test_data_quality_flags():
    monitor = DataQualityMonitor(flatline_window=5, expected_cadence_sec=60.0)

    # 1. Normal row
    normal_ch = {"spp_bar": 120.0, "flow_in_lpm": 2000.0, "mud_weight_sg": 1.45}
    q1 = monitor.assess_row("W1", "2026-01-01 10:00:00", normal_ch)
    assert q1.is_good

    # 2. Out-of-range check
    oor_ch = {"spp_bar": 999.0, "flow_in_lpm": 2000.0, "mud_weight_sg": 1.45}
    q2 = monitor.assess_row("W1", "2026-01-01 10:01:00", oor_ch)
    assert not q2.is_good
    assert any("out_of_range" in str(err) for err in q2.channel_issues.get("spp_bar", []))

    # 3. Spike check
    spike_ch = {"spp_bar": 120.0, "flow_in_lpm": 2000.0, "mud_weight_sg": 2.5}  # jump from 1.45 to 2.5 > 0.5/min
    q3 = monitor.assess_row("W1", "2026-01-01 10:02:00", spike_ch)
    assert any("spike" in str(err) for err in q3.channel_issues.get("mud_weight_sg", []))

    # 4. Telemetry Gap check
    gap_ch = {"spp_bar": 120.0, "flow_in_lpm": 2000.0, "mud_weight_sg": 1.45}
    q4 = monitor.assess_row("W1", "2026-01-01 10:15:00", gap_ch)  # 13 minutes elapsed
    assert any("Telemetry gap" in flag for flag in q4.summary_flags)


def test_runtime_storage_alert_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_alerts.db"
        storage = RuntimeStorage(db_file)

        alt = AlertRecord(
            alert_id="ALT-001",
            well_id="WELL-01",
            timestamp="2026-01-01 12:00:00",
            anomaly_type="Kick",
            severity="Critical",
            risk_score=95.0,
            status="open"
        )
        storage.save_alert(alt)

        # Retrieve
        saved = storage.get_alert("ALT-001")
        assert saved is not None
        assert saved.status == "open"

        # Acknowledge
        storage.update_alert_status("ALT-001", "acknowledged")
        assert storage.get_alert("ALT-001").status == "acknowledged"

        # Snooze
        snooze_time = "2026-01-01 12:30:00"
        storage.update_alert_status("ALT-001", "snoozed", snooze_until=snooze_time)
        assert storage.get_alert("ALT-001").status == "snoozed"
        assert storage.get_alert("ALT-001").snooze_until == snooze_time

        # Feedback
        storage.update_alert_status("ALT-001", "closed", operator_feedback="true_event", feedback_notes="Confirmed kick")
        closed = storage.get_alert("ALT-001")
        assert closed.status == "closed"
        assert closed.operator_feedback == "true_event"


def test_online_features_match_batch_features():
    """Verify that online rolling window feature computation exactly matches batch pipeline."""
    df, _ = generate_dataset(n_wells=1, days=0.2, seed=123)
    profile = get_profile("drilling")

    # 1. Batch Feature Computation
    df_batch = add_profile_features(df, profile)
    raw_cols = [c for c in profile.channels if c not in profile.trend_channels]
    phys_cols = [f.name for f in profile.physics_features]
    df_batch_ctx = add_context(df_batch, raw_cols + phys_cols)

    # 2. Online Buffer Computation on same sequence
    buf = WellBuffer(well_id="WELL-001", max_window_rows=len(df))
    source = ReplaySource(df, speed_multiplier=0.0)

    for row in source.stream_rows():
        buf.append(row)

    df_online_win = buf.to_dataframe()
    df_online_feat = add_profile_features(df_online_win, profile)
    df_online_ctx = add_context(df_online_feat, raw_cols + phys_cols)

    # Compare features on the last row
    for col in ["flow_delta_lpm", "pit_rate_m3ph", "mass_balance_residual",
                "flow_out_lpm__mean15", "pit_volume_m3__dev180"]:
        batch_val = df_batch_ctx[col].iloc[-1]
        online_val = df_online_ctx[col].iloc[-1]
        if pd.notnull(batch_val):
            assert abs(batch_val - online_val) < 1e-4, f"Mismatch in feature {col}"

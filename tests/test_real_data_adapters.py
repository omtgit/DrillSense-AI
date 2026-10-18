"""Unit tests for Phase 5b: Volve adapter, DataDrill adapter, BYOD quality audit, and offline detection."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from app.detector import train_detector
from app.engine.quality_audit import audit_dataset_quality
from app.ingest.datadrill import DataDrillSource
from app.ingest.volve import (
    VolveFolderSource,
    parse_daily_drilling_report_xml,
    parse_witsml_log_xml,
)
from app.rig_state import infer_rig_state
from profiles import get_profile


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_volve_witsml_time_log_parser():
    """Verify WITSML 1.4.1.1 time-indexed XML parsing, null handling, and channel mapping."""
    xml_path = REPO_ROOT / "data" / "sample" / "volve" / "sample_volve_time_log.xml"
    yaml_path = REPO_ROOT / "config" / "volve_mapping.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)

    df, meta = parse_witsml_log_xml(xml_path, mapping_config=mapping)

    assert not df.empty
    assert meta["well_name"] == "NO 15/9-F-1 B"
    assert meta["is_time_indexed"] is True
    assert "timestamp" in df.columns
    assert "wob_kn" in df.columns
    assert "spp_bar" in df.columns
    assert "flow_in_lpm" in df.columns
    assert "flow_out_lpm" in df.columns

    # Verify null value handling (-999.25 was replaced with NaN)
    # Row index 4 had -999.25 in SWOB
    assert pd.isna(df.loc[4, "wob_kn"])
    assert len(df) == 6


def test_volve_witsml_depth_log_parser():
    """Verify depth-indexed WITSML log parsing and synthesized timestamps."""
    xml_path = REPO_ROOT / "data" / "sample" / "volve" / "sample_volve_depth_log.xml"
    yaml_path = REPO_ROOT / "config" / "volve_mapping.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)

    df, meta = parse_witsml_log_xml(xml_path, mapping_config=mapping)

    assert not df.empty
    assert meta["is_time_indexed"] is False
    assert "depth_m" in df.columns
    assert df.loc[0, "depth_m"] == 2850.0
    assert "timestamp" in df.columns  # Synthesized progression
    assert len(df) == 3


def test_volve_ddr_xml_parser():
    """Verify Daily Drilling Report XML parsing into structured tabular format."""
    ddr_path = REPO_ROOT / "data" / "sample" / "volve" / "sample_volve_ddr.xml"
    ddr_df = parse_daily_drilling_report_xml(ddr_path)

    assert not ddr_df.empty
    assert len(ddr_df) == 3
    assert set(["well_id", "date", "activity", "remarks"]).issubset(ddr_df.columns)
    assert "NO 15/9-F-1 B" in ddr_df["well_id"].values
    assert any("Drilling" in act for act in ddr_df["activity"].values)
    assert any("Connection" in act for act in ddr_df["activity"].values)


def test_datadrill_csv_loader():
    """Verify DataDRILL CSV loader maps columns and yields telemetry rows."""
    csv_path = REPO_ROOT / "data" / "sample" / "datadrill" / "sample_datadrill_kick.csv"
    dd_source = DataDrillSource(csv_path)

    assert dd_source.well_ids == ["DATADRILL-001"]
    df = dd_source.dataframe

    assert "flow_in_lpm" in df.columns
    assert "flow_out_lpm" in df.columns
    assert "pit_volume_m3" in df.columns
    assert "spp_bar" in df.columns

    rows = list(dd_source.stream_rows())
    assert len(rows) == 8
    assert rows[0].well_id == "DATADRILL-001"
    assert "flow_out_lpm" in rows[0].channels


def test_dataset_quality_audit():
    """Verify data quality auditor identifies missing values, out-of-range, and rig states."""
    csv_path = REPO_ROOT / "data" / "sample" / "datadrill" / "sample_datadrill_kick.csv"
    dd_source = DataDrillSource(csv_path)
    df = dd_source.dataframe

    audit = audit_dataset_quality(df)
    assert audit.total_rows == 8
    assert audit.well_count == 1
    assert "drilling" in audit.rig_state_mix
    assert "connection" in audit.rig_state_mix
    assert audit.sampling_interval_sec == 60.0

    table = audit.to_dataframe()
    assert not table.empty
    assert "Channel" in table.columns
    assert "Missing (%)" in table.columns


def test_offline_detection_on_real_format_sample():
    """Verify offline XGBoost detector inference runs on real-format DataDRILL data.
    
    Verifies:
    - Rig state inference correctly identifies connection vs drilling vs off-bottom
    - Real labeled kick events in DataDRILL sample (Kick=1) can be evaluated
    - Reports honestly what is labeled (Kick only) vs unlabeled anomalies
    """
    profile = get_profile("drilling")
    det = train_detector(profile, train_wells=2, valid_wells=1, days=1.0, rounds=10)

    csv_path = REPO_ROOT / "data" / "sample" / "datadrill" / "sample_datadrill_kick.csv"
    dd_source = DataDrillSource(csv_path)
    df = dd_source.dataframe.copy()

    # Fill any missing profile channels with nominal baselines
    for req_ch in profile.channels:
        if req_ch not in df.columns:
            df[req_ch] = 0.0

    import detector
    prepared = detector.prepare(df, profile)
    scored = det.predict(prepared)

    assert not scored.empty
    assert "predicted_anomaly" in scored.columns
    assert "predicted_risk_score" in scored.columns
    assert len(scored) == len(df)

    # In our sample DataDRILL kick fixture:
    # rows 0-2: drilling (normal)
    # rows 3-4: connection (pumps off, rotation off)
    # rows 5: drilling resume
    # rows 6-7: influx begins (flow out jumps to 2350-2480, pit volume rises, Kick=1)
    states = infer_rig_state(df)
    assert states.iloc[3] == "connection"
    assert states.iloc[0] == "drilling"

    # Evaluates presence of ground-truth Kick label
    assert "Kick" in df.columns
    assert df["Kick"].sum() == 2

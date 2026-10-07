import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from report import build_finding, numbers_are_grounded, render_report  # noqa: E402


def make_df():
    rng = np.random.default_rng(0)
    n = 60
    ts = pd.date_range("2026-05-01", periods=n, freq="5min")
    d = pd.DataFrame({
        "timestamp": ts, "well_id": "WELL-001",
        "pressure_psi": 3000 + rng.normal(0, 5, n), "temperature_c": 90 + rng.normal(0, 0.5, n),
        "flow_rate_bpd": 800 + rng.normal(0, 5, n), "vibration": 2 + rng.normal(0, 0.1, n),
        "gas_ratio": 17 + rng.normal(0, 0.5, n), "pump_rpm": 115 + rng.normal(0, 2, n),
        "torque": 6000 + rng.normal(0, 50, n), "rop": 28 + rng.normal(0, 0.5, n),
        "severity": "0", "recommended_response": "No Action Required",
        "predicted_anomaly": "Normal", "predicted_risk_score": 5,
    })
    ev = slice(30, 35)
    d.loc[ev, "pump_rpm"] -= 60
    d.loc[ev, "vibration"] += 3
    d.loc[ev, ["predicted_anomaly", "severity", "recommended_response", "predicted_risk_score"]] = [
        "Pump Failure", "High", "Inspect pump", 80]
    return d


def test_finding_finds_event_and_top_channels():
    d = make_df()
    f = build_finding(d, d.iloc[33])
    assert f["detected"] and f["n_rows"] == 6
    assert {c["channel"] for c in f["contributors"][:2]} == {"pump_rpm", "vibration"}
    rpm = next(c for c in f["contributors"] if c["channel"] == "pump_rpm")
    assert rpm["z"] < 0


def test_report_is_deterministic_and_has_sections():
    d = make_df()
    text = render_report(build_finding(d, d.iloc[33]))
    assert text == render_report(build_finding(d, d.iloc[33]))
    for h in ["Operational Summary", "Key Signals", "Possible Root Cause",
              "Operational Risk", "Immediate Actions", "Longer-Term Recommendation"]:
        assert f"## {h}" in text
    assert "Pump Failure" in text and "WELL-001" in text and "below normal" in text


def test_normal_row_reports_no_anomaly():
    d = make_df()
    text = render_report(build_finding(d, d.iloc[5]))
    assert "no anomaly" in text and "Key Signals" not in text


def test_unknown_class_uses_generic_wording():
    d = make_df()
    d.loc[30:35, "predicted_anomaly"] = "Mystery"
    assert "cannot attribute" in render_report(build_finding(d, d.iloc[33]))


def test_number_guard():
    src = "Pressure 3,232.59 psi over 6 readings"
    assert numbers_are_grounded("About 3232.59 psi across 6 readings", src)
    assert not numbers_are_grounded("Pressure 3,500 psi over 6 readings", src)

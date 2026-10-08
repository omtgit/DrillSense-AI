"""Tests for the start-up drilling detector (app/detector.py) and the profile-driven report."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import detector as D  # noqa: E402
import generate_drilling_data as G  # noqa: E402
from data_source import load_drilling  # noqa: E402
from generators.drilling import generate_dataset  # noqa: E402
from profiles import get_profile  # noqa: E402
from report import build_finding, render_report  # noqa: E402
from report_ui import rank_by_risk  # noqa: E402

PROFILE = get_profile("drilling")


@pytest.fixture(scope="module")
def det():
    # small run: speed matters more than quality here
    return D.train_detector(PROFILE, train_wells=3, valid_wells=2, days=1.0, rounds=30)


@pytest.fixture(scope="module")
def scored(det):
    return det.predict(D.prepare(load_drilling(), PROFILE))


def test_training_data_is_not_the_committed_sample():
    assert D.TRAIN_SEED != G.SAMPLE["seed"] and D.VALID_SEED != G.SAMPLE["seed"]
    assert D.TRAIN_SEED != D.VALID_SEED
    sample = load_drilling()
    train, _ = generate_dataset(3, 1.0, D.TRAIN_SEED, D.EVENTS_PER_DAY)
    cols = ["depth_m", "wob_kn", "spp_bar", "flow_out_lpm"]
    merged = sample[cols].round(4).merge(train[cols].round(4), how="inner")
    assert merged.empty                      # no identical rows


def test_detector_never_sees_ground_truth_columns(det):
    banned = {"anomaly_type", "anomaly_flag", "severity", "active_anomalies", "event_scale",
              "priority_score", "recommended_response", "rig_state"}
    assert not banned & set(det.columns)
    assert "depth_m" not in det.columns      # absolute depth only trends with time


def test_predictions_have_expected_columns_and_use_profile_rules(scored):
    for c in ("predicted_anomaly", "predicted_risk_score", "predicted_severity", "predicted_response"):
        assert c in scored
    assert set(scored["predicted_anomaly"]) <= {"Normal", *PROFILE.anomaly_types}
    assert scored["predicted_risk_score"].between(0, 100).all()
    for cls, g in scored[scored["predicted_anomaly"] != "Normal"].groupby("predicted_anomaly"):
        rule = PROFILE.severity_rules[cls]
        assert (g["predicted_severity"] == rule.severity).all()
        assert (g["predicted_response"] == rule.response).all()
    normal = scored[scored["predicted_anomaly"] == "Normal"]
    assert (normal["predicted_severity"] == "None").all()


def test_validation_info_is_populated(det):
    i = det.info
    assert i["valid_wells"] == 2 and i["train_wells"] == 3
    assert 0 <= i["precision"] <= 1 and 0 <= i["recall"] <= 1 and 0 <= i["event_recall"] <= 1


def test_training_is_deterministic():
    sample = D.prepare(load_drilling(), PROFILE)
    kw = dict(train_wells=2, valid_wells=1, days=0.5, rounds=10)
    a = D.train_detector(PROFILE, **kw).predict(sample)
    b = D.train_detector(PROFILE, **kw).predict(sample)
    assert a["predicted_risk_score"].equals(b["predicted_risk_score"])


def test_score_rows_and_event_recall():
    df = pd.DataFrame({"well_id": "W", "anomaly_flag": [1, 1, 0, 0],
                       "predicted_anomaly": ["Kick", "Normal", "Kick", "Normal"]})
    p, r, f1 = D.score_rows(df)
    assert (p, r) == (0.5, 0.5) and f1 == 0.5
    ev = pd.DataFrame([{"well_id": "W", "start": 0, "end": 2}, {"well_id": "W", "start": 100, "end": 120}])
    assert D.event_recall(df, ev) == 0.5


def test_rank_by_risk_breaks_ties_by_recency_then_well():
    f = pd.DataFrame({
        "well_id": ["B", "A", "A", "C"],
        "timestamp": pd.to_datetime(["2026-05-01 10:00", "2026-05-01 10:00", "2026-05-01 09:00", "2026-05-01 11:00"]),
        "predicted_risk_score": [90, 90, 90, 50],
    })
    r = rank_by_risk(f)
    assert list(r["well_id"][:2]) == ["A", "B"] and r.iloc[0]["timestamp"] == pd.Timestamp("2026-05-01 10:00")


def test_drilling_report_uses_profile_vocabulary_and_rules():
    n = 120
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"timestamp": pd.date_range("2026-05-01", periods=n, freq="min"), "well_id": "WELL-001"})
    for c in PROFILE.channels:
        d[c] = 100.0 + rng.normal(0, 1, n)
    d["depth_m"] = np.linspace(1000, 1100, n)          # trends; must not be reported as a signal
    d.loc[60:80, "spp_bar"] += 40
    d["predicted_anomaly"] = "Normal"
    d.loc[60:80, "predicted_anomaly"] = "Pack-off"
    d["predicted_risk_score"] = 10.0
    d.loc[60:80, "predicted_risk_score"] = 95.0
    row = d.iloc[70]                                    # no severity / recommended_response columns at all
    f = build_finding(d, row, profile=PROFILE, rules_from_profile=True)
    assert f["detected"] and f["severity"] == "High"
    assert f["recommended_response"] == PROFILE.severity_rules["Pack-off"].response
    assert "depth_m" not in {c["channel"] for c in f["contributors"]}
    assert f["contributors"][0]["channel"] == "spp_bar"
    text = render_report(f, PROFILE)
    assert "standpipe pressure (bar)" in text and "restricting the annulus" in text
    assert "profile rule for the predicted class" in text and "have not been reviewed" in text
    assert "ground-truth" not in text


def test_production_report_still_reads_row_labels():
    from test_report import make_df
    d = make_df()
    text = render_report(build_finding(d, d.iloc[33]))
    assert "ground-truth lookup in this dataset" in text and "have not been reviewed" not in text


def test_load_drilling_sample_columns():
    df = load_drilling()
    assert {"anomaly_type", "anomaly_flag", "flow_out_lpm"} <= set(df.columns)
    assert df["well_id"].nunique() == 3

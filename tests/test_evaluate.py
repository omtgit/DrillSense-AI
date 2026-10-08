"""Unit tests for the event-level metrics and threshold choice in scripts/evaluate.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate as E  # noqa: E402


def _rows(n, alarms):
    wells = np.array(["W"] * n)
    t = np.arange(n)
    alarm = np.zeros(n, bool)
    alarm[list(alarms)] = True
    return wells, t, alarm


def _events(*spans):
    return pd.DataFrame([{"well_id": "W", "start": s, "end": e} for s, e in spans])


def test_detection_delay_and_false_alarm():
    wells, t, alarm = _rows(1000, [120, 121, 122, 700, 701])
    r = E.alarm_eval(wells, t, alarm, _events((100, 200), (400, 450)))
    assert list(r["detected"]) == [True, False]
    assert r["delay"][0] == 20 and np.isnan(r["delay"][1])
    assert r["fa"] == 1                      # the burst at 700 matches no event


def test_close_alarm_rows_merge_into_one_episode():
    wells, t, alarm = _rows(1000, [700, 705, 709])     # gaps <= MERGE_GAP -> one false alarm
    assert E.alarm_eval(wells, t, alarm, _events((100, 200)))["fa"] == 1
    wells, t, alarm = _rows(1000, [700, 730])          # gap > MERGE_GAP -> two
    assert E.alarm_eval(wells, t, alarm, _events((100, 200)))["fa"] == 2


def test_grace_window_after_event_end():
    wells, t, alarm = _rows(1000, [200 + E.GRACE - 1])
    r = E.alarm_eval(wells, t, alarm, _events((100, 200)))
    assert r["detected"][0] and r["fa"] == 0
    wells, t, alarm = _rows(1000, [200 + E.GRACE + 5])
    r = E.alarm_eval(wells, t, alarm, _events((100, 200)))
    assert not r["detected"][0] and r["fa"] == 1


def test_alarm_before_event_is_false_alarm_unless_it_runs_into_it():
    wells, t, alarm = _rows(1000, [50])
    assert E.alarm_eval(wells, t, alarm, _events((100, 200)))["fa"] == 1
    wells, t, alarm = _rows(1000, list(range(95, 106)))   # one episode spanning the event start
    r = E.alarm_eval(wells, t, alarm, _events((100, 200)))
    assert r["fa"] == 0 and r["detected"][0] and r["delay"][0] == 0


def test_threshold_uses_only_the_scores_it_is_given_and_meets_budget():
    rng = np.random.default_rng(0)
    n = 20000
    wells, t = np.array(["W"] * n), np.arange(n)
    scores = rng.normal(size=n)
    scores[1000:1100] += 8                     # one real event
    ev = _events((1000, 1100))
    thr = E.pick_thresholds(scores, wells, t, ev, budgets=(1.0, 0.25))
    assert thr[0.25] >= thr[1.0]               # a tighter budget never lowers the threshold
    assert E.fa_per_day(wells, t, scores >= thr[1.0], ev) <= 1.0
    assert E.fa_per_day(wells, t, scores >= thr[0.25], ev) <= 0.25


def test_physics_rule_needs_agreeing_signs():
    tr = pd.DataFrame({"flow_delta_lpm": np.random.default_rng(1).normal(0, 1, 500),
                       "pit_rate_m3ph": np.random.default_rng(2).normal(0, 1, 500)})
    ap = pd.DataFrame({"flow_delta_lpm": [10.0, 10.0, -10.0], "pit_rate_m3ph": [10.0, -10.0, -10.0]})
    s = E.score_rule(ap, tr)
    assert s[0] > 5 and s[1] == 0 and s[2] > 5

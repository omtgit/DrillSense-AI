import numpy as np
import pandas as pd
import pytest

from app.features import add_profile_features
from generators.drilling import Event, generate_dataset, generate_well
from profiles import get_profile

DRILLING = get_profile("drilling")
CHANNELS = ["depth_m", "wob_kn", "rpm", "torque_knm", "rop_mph", "spp_bar", "flow_in_lpm",
            "flow_out_lpm", "pit_volume_m3", "mud_weight_sg", "gas_units"]
SEED = 11


def run(events, **kw):
    df, _ = generate_well(SEED, days=0.5, events=events, **kw)
    return df


def ev(typ, **kw):
    base = dict(event_id=0, type=typ, start=200, ramp=30, hold=120, release=10, scale=1.0)
    base.update(kw)
    return Event(**base)


def steady(df):
    """Rows with the pumps at full rate for 6+ minutes (outside connections and ramps)."""
    return df["flow_in_lpm"].rolling(6).min() > 1000


def dmean(hit, base, col, a=260, b=340):
    """Mean of (hit - base) for `col` over rows steady in both runs. A strong event changes depth
    progress and so the connection timing; this keeps the comparison on like-for-like rows."""
    m = (steady(hit) & steady(base)).iloc[a:b].to_numpy()
    return float((hit[col].iloc[a:b].to_numpy() - base[col].iloc[a:b].to_numpy())[m].mean())


# ---------------- profiles ----------------

def test_profiles_registered_and_drilling_has_required_channels():
    assert list(DRILLING.channels) == CHANNELS
    assert set(DRILLING.anomaly_types) == {"Kick", "Lost Circulation", "Stuck Pipe", "Washout",
                                           "Pack-off", "Sensor Drift"}
    with pytest.raises(ValueError):
        get_profile("nope")


def test_production_profile_unchanged():
    p = get_profile("production")
    assert p.channel_labels["pressure_psi"] == "pressure (psi)"
    assert set(p.channels) == {"pressure_psi", "temperature_c", "flow_rate_bpd", "vibration",
                               "gas_ratio", "pump_rpm", "torque", "rop"}
    assert p.severity_rules["Kick Risk"].priority == 100


# ---------------- determinism ----------------

def test_same_seed_identical_different_seed_differs():
    a, ea = generate_dataset(2, 0.5, seed=3, events_per_day=4)
    b, eb = generate_dataset(2, 0.5, seed=3, events_per_day=4)
    c, _ = generate_dataset(2, 0.5, seed=4, events_per_day=4)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ea, eb)
    assert not a[CHANNELS].equals(c[CHANNELS])


# ---------------- normal behaviour ----------------

def test_normal_data_is_sane():
    df = run([])
    assert df[CHANNELS].notna().all().all()
    assert (df["anomaly_flag"] == 0).all() and (df["anomaly_type"] == "Normal").all()
    assert (df["depth_m"].diff().dropna() > -0.5).all()           # bit never goes up while drilling
    assert df["depth_m"].iloc[-1] - df["depth_m"].iloc[0] > 50      # it makes progress
    assert df["spp_bar"].between(0, 400).all() and df["mud_weight_sg"].between(1.0, 2.2).all()


def test_connections_stop_pumps_rotation_and_progress():
    df = run([])
    conn = df[df["rig_state"] == "connection"]
    drill = df[df["rig_state"] == "drilling"]
    assert len(conn) > 10 and len(drill) > 10
    assert conn["rop_mph"].max() < 2.0 and conn["wob_kn"].max() < 4.0
    assert conn["flow_in_lpm"].min() < 20                            # pumps fully off at some point
    assert conn["flow_in_lpm"].mean() < 0.3 * drill["flow_in_lpm"].mean()


def test_formation_changes_move_rop():
    df = run([])
    d = df[df["rig_state"] == "drilling"]
    # ROP varies with formation drillability, so it is not a constant level with noise only
    assert d["rop_mph"].rolling(60).mean().dropna().std() > 0.3


def test_exact_mass_balance_without_noise_transfers_or_pit_management():
    df = run([], noise_scale=0, pit_transfers_per_day=0, pit_restore_tau_min=None)
    d_pit = df["pit_volume_m3"].diff().dropna().to_numpy()
    d_exp = ((df["flow_out_lpm"] - df["flow_in_lpm"]) / 1000.0).iloc[1:].to_numpy()
    assert np.allclose(d_pit, d_exp, atol=2e-3)


def test_connection_flowback_is_a_normal_pit_gain():
    df = run([], noise_scale=0, pit_transfers_per_day=0)
    assert df["pit_volume_m3"].max() - df["pit_volume_m3"].min() < 3.0   # small, within normal ops


# ---------------- anomaly signatures (full-size event vs identical run without it) ----------------

def test_kick_signature():
    base, hit = run([]), run([ev("Kick")])
    assert dmean(hit, base, "flow_out_lpm") > 200            # 250 L/min influx at full size
    assert dmean(hit, base, "pit_volume_m3") > 1.0
    assert dmean(hit, base, "spp_bar") < 0
    assert hit["gas_units"].iloc[300:400].max() > base["gas_units"].iloc[300:400].max() + 40


def test_kick_is_gradual_not_a_step():
    base, hit = run([]), run([ev("Kick", ramp=40)])
    inflow = (hit["flow_out_lpm"] - base["flow_out_lpm"]).rolling(5).mean().iloc[205:240]
    assert inflow.diff().dropna().abs().max() < 40          # no single-step jump
    assert inflow.iloc[-1] > inflow.iloc[0] + 50            # but it does build up


def test_lost_circulation_signature():
    base, hit = run([]), run([ev("Lost Circulation")])
    assert dmean(hit, base, "flow_out_lpm") < -400
    assert dmean(hit, base, "pit_volume_m3") < -2.0
    assert dmean(hit, base, "spp_bar") < 0


def test_stuck_pipe_signature():
    base, hit = run([]), run([ev("Stuck Pipe")])
    w = slice(260, 340)
    d = hit["rig_state"].iloc[w].eq("drilling") & base["rig_state"].iloc[w].eq("drilling")
    assert hit["torque_knm"].iloc[w][d].mean() > 1.3 * base["torque_knm"].iloc[w][d].mean()
    assert hit["rop_mph"].iloc[w][d].mean() < 0.3 * base["rop_mph"].iloc[w][d].mean()
    gained = hit["depth_m"].iloc[339] - hit["depth_m"].iloc[259]
    assert gained < 0.4 * (base["depth_m"].iloc[339] - base["depth_m"].iloc[259])


def test_washout_signature_pressure_drops_flow_unchanged():
    base, hit = run([]), run([ev("Washout", ramp=60, hold=100)])
    assert dmean(hit, base, "spp_bar", 300, 340) < -0.12 * base["spp_bar"].iloc[300:340].mean()
    assert abs(dmean(hit, base, "flow_in_lpm", 300, 340)) < 0.01 * 2000
    assert abs(dmean(hit, base, "flow_out_lpm", 300, 340)) < 0.03 * 2000


def test_pack_off_signature_pressure_up_returns_down():
    base, hit = run([]), run([ev("Pack-off")])
    assert dmean(hit, base, "spp_bar") > 0.2 * base["spp_bar"].iloc[260:340].mean()
    assert dmean(hit, base, "flow_out_lpm") < -0.1 * 2000


def test_sensor_drift_changes_only_its_channel_and_grows_linearly():
    base = run([])
    hit = run([ev("Sensor Drift", channel="pit_volume_m3", sign=1, ramp=200, hold=0, release=2)])
    others = [c for c in CHANNELS if c != "pit_volume_m3"]
    for c in others:
        assert np.allclose(base[c], hit[c]), c           # physics and other sensors untouched
    bias = (hit["pit_volume_m3"] - base["pit_volume_m3"]).to_numpy()
    assert np.allclose(bias[:200], 0)                    # nothing before onset
    seg = bias[200:400]
    assert np.all(np.diff(seg) > 0)                      # monotone ramp, no step
    assert np.allclose(seg, 3.0 * np.arange(1, 201) / 201, atol=2e-4)   # linear up to 3 m3
    assert np.allclose(bias[410:], 0)                    # recalibrated after the event


def test_small_events_are_small():
    base = run([])
    hit = run([ev("Kick", scale=0.15)])
    big = run([ev("Kick", scale=1.0)])
    small_d, big_d = dmean(hit, base, "flow_out_lpm"), dmean(big, base, "flow_out_lpm")
    assert 0 < small_d < 0.3 * big_d


# ---------------- schedule, overlap, labels ----------------

def test_random_schedule_has_variable_magnitudes_all_types_and_overlaps():
    df, events = generate_dataset(12, 3, seed=7, events_per_day=3)
    assert set(events["type"]) == set(DRILLING.anomaly_types)
    assert events["scale"].min() < 0.3 and events["scale"].max() > 0.8
    assert df["active_anomalies"].str.contains(r"\|").any()
    assert (events["ramp"] >= 5).all()                  # every event ramps; none is a one-row step


def test_labels_consistent_with_profile():
    df, _ = generate_dataset(4, 2, seed=2, events_per_day=3)
    assert ((df["anomaly_type"] != "Normal") == (df["anomaly_flag"] == 1)).all()
    flagged = df[df["anomaly_flag"] == 1]
    assert flagged["anomaly_type"].isin(DRILLING.anomaly_types).all()
    for t, r in DRILLING.severity_rules.items():
        sub = flagged[flagged["anomaly_type"] == t]
        assert (sub["severity"] == r.severity).all() and (sub["priority_score"] == r.priority).all()
    assert (df.loc[df["anomaly_flag"] == 0, "event_scale"] == 0).all()


def test_normal_behaviour_independent_of_event_schedule_when_no_overlap_with_physics():
    # A drift event only adds sensor bias, so every other channel is bit-identical to the clean run.
    a = run([])
    b = run([ev("Sensor Drift", channel="spp_bar")])
    assert np.allclose(a["torque_knm"], b["torque_knm"]) and np.allclose(a["depth_m"], b["depth_m"])


# ---------------- features ----------------

def test_physics_features_exist_and_are_past_only():
    df, _ = generate_dataset(2, 0.5, seed=5)
    f_full = add_profile_features(df, DRILLING)
    for feat in DRILLING.physics_features:
        assert feat.name in f_full.columns
    cut = df[df["timestamp"] < df["timestamp"].min() + pd.Timedelta(minutes=300)]
    f_cut = add_profile_features(cut, DRILLING)
    names = [f.name for f in DRILLING.physics_features]
    a = f_full.merge(f_cut[["well_id", "timestamp"]], on=["well_id", "timestamp"])
    pd.testing.assert_frame_equal(
        a[names].reset_index(drop=True), f_cut[names].reset_index(drop=True))


def test_kick_shows_up_in_flow_delta_feature():
    df = add_profile_features(run([ev("Kick")]).assign(), DRILLING)
    assert df["flow_delta_lpm"].iloc[260:340].mean() > 100

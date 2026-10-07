"""Synthetic drilling telemetry with physically-motivated normal behaviour and gradual anomalies.

One row per minute per well. Normal behaviour: depth progress by stands, formation changes,
connections (pumps and rotation stop), noise. Anomalies are ramps (not steps) with variable
magnitude and can overlap. Every assumption is listed in docs/PHYSICS.md (items marked
NEEDS REVIEW are unchecked by a drilling engineer).

Everything is seeded. Normal behaviour, anomaly schedule and sensor noise use independent random
streams drawn up front, so adding or removing an anomaly never changes the normal behaviour
(except through the physics, e.g. a stuck string drills less depth).
"""
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from profiles import get_profile

PROFILE = get_profile("drilling")
STAND_M = 28.5            # three ~9.5 m joints
NOMINAL_FLOW_REF = 2000.0  # L/min, reference for the standpipe-pressure relation
START_TIME = "2026-05-01"

# Relative frequency of anomaly types in the random schedule.
TYPE_WEIGHTS = {"Kick": 1.0, "Lost Circulation": 1.5, "Stuck Pipe": 1.0,
                "Washout": 1.0, "Pack-off": 1.2, "Sensor Drift": 1.5}

# (ramp, hold, release) duration ranges in minutes.
DURATIONS = {
    "Kick": ((10, 45), (10, 40), (5, 20)),
    "Lost Circulation": ((5, 40), (20, 120), (10, 40)),
    "Stuck Pipe": ((10, 60), (20, 90), (3, 10)),
    "Washout": ((60, 240), (30, 180), (1, 3)),
    "Pack-off": ((5, 30), (10, 60), (5, 20)),
    "Sensor Drift": ((120, 480), (0, 60), (1, 3)),
}

# Sensor-drift candidates: channel -> bias reached at scale 1.0 (units of that channel).
DRIFT_SPAN = {"flow_out_lpm": 150.0, "flow_in_lpm": 150.0, "pit_volume_m3": 3.0,
              "spp_bar": 15.0, "wob_kn": 15.0, "mud_weight_sg": 0.04}

SCALE_RANGE = (0.15, 1.0)   # log-uniform: many small, a few full-size events
OVERLAP_PROB = 0.3          # chance a new event starts inside an earlier one


@dataclass
class Event:
    event_id: int
    type: str
    start: int      # row index
    ramp: int
    hold: int
    release: int
    scale: float    # 0.15..1; 1.0 is the full-size effect listed in PHYSICS.md
    channel: Optional[str] = None  # Sensor Drift only
    sign: int = 1                  # Sensor Drift only

    @property
    def end(self):
        return self.start + self.ramp + self.hold + self.release

    def envelope(self, n):
        """0 outside the event, >0 inside (linear ramp up, hold at 1, linear ramp down)."""
        x = np.arange(n) - self.start
        env = np.zeros(n)
        up = (x >= 0) & (x < self.ramp)
        env[up] = (x[up] + 1) / (self.ramp + 1)
        h = (x >= self.ramp) & (x < self.ramp + self.hold)
        env[h] = 1.0
        r0 = self.ramp + self.hold
        dn = (x >= r0) & (x < r0 + self.release)
        env[dn] = 1.0 - (x[dn] - r0 + 1) / (self.release + 1)
        return env


def schedule_events(rng, n, events_per_day, interval_min=1.0, first_id=0):
    """Random event list; some events start inside earlier ones (overlap)."""
    days = n * interval_min / 1440.0
    count = int(rng.poisson(events_per_day * days))
    names = list(TYPE_WEIGHTS)
    probs = np.array([TYPE_WEIGHTS[k] for k in names])
    probs = probs / probs.sum()
    events: List[Event] = []
    for k in range(count):
        typ = names[rng.choice(len(names), p=probs)]
        start = int(rng.integers(0, n))
        if events and rng.random() < OVERLAP_PROB:
            host = events[int(rng.integers(0, len(events)))]
            start = int(rng.integers(host.start, max(host.start + 1, host.start + host.ramp + host.hold)))
        (r0, r1), (h0, h1), (l0, l1) = DURATIONS[typ]
        ev = Event(
            event_id=first_id + k, type=typ, start=min(start, n - 1),
            ramp=int(rng.integers(r0, r1 + 1)), hold=int(rng.integers(h0, h1 + 1)),
            release=int(rng.integers(l0, l1 + 1)),
            scale=float(np.exp(rng.uniform(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1])))),
        )
        if typ == "Sensor Drift":
            ev.channel = str(rng.choice(sorted(DRIFT_SPAN)))
            ev.sign = int(rng.choice([-1, 1]))
        events.append(ev)
    return events


def _ar1(rng, n, rho, scale):
    """Unit-variance-ish AR(1) noise times scale."""
    e = rng.normal(0.0, 1.0, n)
    out = np.empty(n)
    out[0] = e[0]
    s = np.sqrt(1 - rho * rho)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + s * e[i]
    return out * scale


def _shift(a, lag):
    """Delay a by `lag` samples (holds the first value at the start)."""
    if lag <= 0:
        return a.copy()
    return np.concatenate([np.full(lag, a[0]), a[:-lag]])[: len(a)]


def generate_well(seed, well_id="WELL-001", days=1.0, events=None, events_per_day=0.8,
                  noise_scale=1.0, pit_transfers_per_day=2.0, pit_restore_tau_min=360.0):
    """Simulate one well. Returns (DataFrame, list[Event]).

    events: explicit list of Event (replaces the random schedule; used by tests).
    noise_scale=0 removes sensor and process noise and per-well sensor offsets.
    pit_restore_tau_min: crews top up / dump mud, pulling pit volume back to its start value with
    this time constant (None = no pit management, exact mass balance).
    """
    n = int(round(days * 1440))
    root = np.random.SeedSequence(seed)
    s_cfg, s_ev, s_noise, s_stand = root.spawn(4)
    cfg, rev, rn, rst = (np.random.default_rng(s) for s in (s_cfg, s_ev, s_noise, s_stand))

    if events is None:
        events = schedule_events(rev, n, events_per_day)
    ns = noise_scale

    # ---- per-well configuration (drawn first, independent of events) ----
    depth0 = float(cfg.uniform(800, 3000))
    flow_nom = float(cfg.uniform(1800, 3000))
    wob_target = float(cfg.uniform(60, 160))
    rpm_base = float(cfg.uniform(80, 140))
    rop_k = float(cfg.uniform(18, 32))
    mw_offset = float(cfg.uniform(-0.05, 0.05))
    tau_out = float(cfg.uniform(1.5, 3.0))
    flow_out_offset = float(cfg.uniform(-0.01, 0.01)) * ns
    pit0 = float(cfg.uniform(60, 120))
    lag_min = int(np.clip(0.017 * depth0, 5, 60))   # surface lag of gas, ~annulus transit time
    # formations: layers below depth0
    tops, drill, gas_bg, tq = [], [], [], []
    d = depth0 - 100
    while d < depth0 + 6000:
        tops.append(d)
        drill.append(float(np.clip(np.exp(cfg.normal(0, 0.3)), 0.5, 1.8)))
        gas_bg.append(float(cfg.uniform(3, 40)))
        tq.append(float(cfg.uniform(0.9, 1.3)))
        d += float(cfg.uniform(80, 400))
    tops, drill, gas_bg, tq = map(np.array, (tops, drill, gas_bg, tq))

    # ---- pre-drawn per-stand randomness (so counterfactual runs stay aligned) ----
    conn_dur = rst.integers(4, 10, n)
    rpm_off = rst.normal(0, 6, n)
    conn_gas_amp = rst.uniform(5, 25, n)
    stand_start = float(rst.uniform(0.2, 1.0))

    # ---- pre-drawn noise (independent of events) ----
    nz = {k: rn.normal(0, 1, n) for k in
          ("fin", "fout", "pit", "spp", "wob", "rpm", "torque", "rop", "mw", "gas", "depth", "spike")}
    wob_ar = _ar1(rn, n, 0.9, 0.06 * ns)
    tq_ar = _ar1(rn, n, 0.9, 0.04 * ns)
    rop_ln = _ar1(rn, n, 0.8, 0.05 * ns)
    flow_ar = _ar1(rn, n, 0.9, 0.003 * ns)
    gas_ar = _ar1(rn, n, 0.95, 0.15 * ns)

    # ---- pit transfers: normal operations that move pit volume without any flow imbalance ----
    transfer = np.zeros(n)
    n_tr = int(rn.poisson(pit_transfers_per_day * days)) if pit_transfers_per_day > 0 else 0
    for _ in range(n_tr):
        t0 = int(rn.integers(0, n))
        amp = float(rn.choice([-1, 1]) * rn.uniform(0.5, 3.0))
        for j in range(3):  # spread over 3 minutes
            if t0 + j < n:
                transfer[t0 + j] += amp / 3.0

    # ---- anomaly envelopes ----
    a = {t: np.zeros(n) for t in PROFILE.anomaly_types}
    drift_bias = {c: np.zeros(n) for c in DRIFT_SPAN}
    for ev in events:
        contrib = ev.scale * ev.envelope(n)
        a[ev.type] += contrib
        if ev.type == "Sensor Drift":
            drift_bias[ev.channel] += ev.sign * DRIFT_SPAN[ev.channel] * contrib
    for t in a:
        a[t] = np.minimum(a[t], 1.5)  # cap overlapping events of the same type

    # ---- simulation loop ----
    depth = depth0
    stand_end = depth0 + STAND_M * stand_start
    conn_left = conn_total = since_resume = 0
    stand_i = 0
    rpm_set = rpm_base
    mw_stand = round(1.05 + 0.00018 * depth0 + mw_offset, 2)
    fo_state = flow_nom   # lagged return flow (L/min)
    spp_state = 0.0
    pit = pit0

    out = {k: np.zeros(n) for k in
           ("depth", "wob", "rpm", "torque", "rop", "spp", "fin", "fout", "pit", "mw", "gas_src")}
    rig_state = np.empty(n, dtype=object)

    for i in range(n):
        if conn_left == 0 and depth >= stand_end:
            conn_total = conn_left = int(conn_dur[stand_i])
            stand_i = min(stand_i + 1, n - 1)
            stand_end += STAND_M
            rpm_set = rpm_base + float(rpm_off[stand_i])
            mw_stand = round(1.05 + 0.00018 * depth + mw_offset, 2)
        f_idx = int(np.clip(np.searchsorted(tops, depth, side="right") - 1, 0, len(tops) - 1))
        in_conn = conn_left > 0
        gas_src = gas_bg[f_idx] * (1 + gas_ar[i])
        if in_conn:
            k = conn_total - conn_left
            pump = 0.4 if k == 0 else (0.5 if k == conn_total - 1 else 0.0)
            conn_left -= 1
            since_resume = 0
            wob = rpm = torque = rop = 0.0
            rig_state[i] = "connection"
            if k >= 1:
                gas_src += conn_gas_amp[stand_i]
        else:
            pump = 1.0
            since_resume += 1
            ramp = min(1.0, since_resume / 3.0)
            wob = wob_target * (1 + wob_ar[i]) * ramp
            rpm = rpm_set * ramp * (1 - 0.4 * a["Stuck Pipe"][i])
            rop = (rop_k * drill[f_idx] * (max(wob, 0) / 100.0) ** 0.8 * (max(rpm, 0) / 100.0) ** 0.4
                   * np.exp(rop_ln[i]))
            rop *= (1 - 0.95 * min(a["Stuck Pipe"][i], 1.0)) * (1 + 0.2 * a["Kick"][i]) \
                * (1 - 0.4 * min(a["Pack-off"][i], 1.0)) * (1 - 0.15 * min(a["Washout"][i], 1.0))
            rop = max(rop, 0.0)
            torque = (1.0 + 0.0022 * depth + 0.06 * wob * tq[f_idx]) * (1 + tq_ar[i])
            torque *= 1 + 0.8 * a["Stuck Pipe"][i] + 0.3 * a["Pack-off"][i]
            torque *= 1 + 0.1 * (a["Stuck Pipe"][i] + a["Pack-off"][i]) * ns * nz["spike"][i]
            rig_state[i] = "drilling"
            depth += rop / 60.0

        flow_in = flow_nom * pump * (1 + flow_ar[i])
        fo_state += (flow_in - fo_state) / tau_out
        loss = 600.0 * a["Lost Circulation"][i] * (flow_in / flow_nom)
        influx = 250.0 * a["Kick"][i]
        flow_out = fo_state * (1 - 0.20 * min(a["Pack-off"][i], 1.0)) + influx - loss
        flow_out = max(flow_out, 0.0)
        pit += (flow_out - flow_in) / 1000.0 + transfer[i]
        if pit_restore_tau_min:
            pit += (pit0 - pit) / pit_restore_tau_min

        spp_target = (flow_in / NOMINAL_FLOW_REF) ** 1.8 * (mw_stand / 1.5) * (40 + 0.03 * depth)
        spp_target *= (1 - 0.06 * a["Kick"][i]) * (1 - 0.10 * a["Lost Circulation"][i]) \
            * (1 - 0.25 * a["Washout"][i]) * (1 + 0.35 * a["Pack-off"][i])
        spp_state += (spp_target - spp_state) / 1.5

        out["depth"][i], out["wob"][i], out["rpm"][i] = depth, wob, rpm
        out["torque"][i], out["rop"][i], out["spp"][i] = torque, rop, spp_state
        out["fin"][i], out["fout"][i], out["pit"][i] = flow_in, flow_out, pit
        out["mw"][i], out["gas_src"][i] = mw_stand, gas_src

    # ---- sensors: true value + noise + drift ----
    gas = _shift(out["gas_src"], lag_min) + 120.0 * _shift(a["Kick"], lag_min) + 0.3 * ns * nz["gas"]
    meas = {
        "depth_m": out["depth"] + 0.02 * ns * nz["depth"],
        "wob_kn": np.maximum(out["wob"] + 0.8 * ns * nz["wob"], 0),
        "rpm": np.maximum(out["rpm"] + 0.7 * ns * nz["rpm"], 0),
        "torque_knm": np.maximum(out["torque"] + 0.12 * ns * nz["torque"], 0),
        "rop_mph": np.maximum(out["rop"] + 0.4 * ns * nz["rop"], 0),
        "spp_bar": np.maximum(out["spp"] + 0.4 * ns * nz["spp"], 0),
        "flow_in_lpm": np.maximum(out["fin"] * (1 + 0.004 * ns * nz["fin"]), 0),
        "flow_out_lpm": np.maximum(out["fout"] * (1 + flow_out_offset + 0.012 * ns * nz["fout"]), 0),
        "pit_volume_m3": out["pit"] + 0.03 * ns * nz["pit"],
        "mud_weight_sg": out["mw"] + 0.002 * ns * nz["mw"],
        "gas_units": np.maximum(gas, 0),
    }
    for ch, bias in drift_bias.items():
        meas[ch] = meas[ch] + bias

    # ---- labels ----
    types = list(PROFILE.anomaly_types)
    stack = np.vstack([a[t] for t in types])           # (types, n)
    active = stack > 0
    primary = np.where(active.any(axis=0), stack.argmax(axis=0), -1)
    names = np.array(types + ["Normal"], dtype=object)
    anomaly_type = names[primary]                      # -1 -> "Normal"
    active_str = np.array(["|".join(t for t, on in zip(types, col) if on) for col in active.T], dtype=object)
    event_scale = np.where(primary >= 0, stack.max(axis=0), 0.0)
    rules = PROFILE.severity_rules

    df = pd.DataFrame({
        "timestamp": pd.date_range(START_TIME, periods=n, freq="1min"),
        "well_id": well_id,
        "rig_state": rig_state,
        **{k: np.round(v, 4) for k, v in meas.items()},
        "anomaly_type": anomaly_type,
        "anomaly_flag": (primary >= 0).astype(int),
        "active_anomalies": active_str,
        "event_scale": np.round(event_scale, 3),
    })
    df["severity"] = df["anomaly_type"].map(lambda t: rules[t].severity if t in rules else "None")
    df["priority_score"] = df["anomaly_type"].map(lambda t: rules[t].priority if t in rules else 0)
    df["recommended_response"] = df["anomaly_type"].map(
        lambda t: rules[t].response if t in rules else "No Action Required")
    return df, events


def generate_dataset(n_wells=10, days=2.0, seed=42, events_per_day=0.8, noise_scale=1.0,
                     pit_transfers_per_day=2.0, pit_restore_tau_min=360.0):
    """Several wells; each well has its own SeedSequence child, so wells are independent."""
    children = np.random.SeedSequence(seed).spawn(n_wells)
    frames, rows = [], []
    for w, child in enumerate(children, start=1):
        wid = f"WELL-{w:03d}"
        df, events = generate_well(child.generate_state(1)[0], wid, days,
                                   events_per_day=events_per_day, noise_scale=noise_scale,
                                   pit_transfers_per_day=pit_transfers_per_day,
                                   pit_restore_tau_min=pit_restore_tau_min)
        frames.append(df)
        for ev in events:
            rows.append({"well_id": wid, **asdict(ev), "end": ev.end})
    events_df = pd.DataFrame(rows, columns=["well_id", "event_id", "type", "start", "ramp", "hold",
                                            "release", "scale", "channel", "sign", "end"])
    return pd.concat(frames, ignore_index=True), events_df

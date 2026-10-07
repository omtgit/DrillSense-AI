"""Drilling profile: channels, anomaly types, physics features, severity rules, report wording.

Physical meaning of every channel and anomaly signature is in docs/PHYSICS.md (items marked
NEEDS REVIEW have not been checked by a drilling engineer).
"""
import numpy as np

from .base import Channel, PhysicsFeature, Profile, SeverityRule

CHANNELS = {
    "depth_m": Channel("m", "bit depth (m)"),
    "wob_kn": Channel("kN", "weight on bit (kN)"),
    "rpm": Channel("rpm", "rotary speed (rpm)"),
    "torque_knm": Channel("kN·m", "torque (kN·m)"),
    "rop_mph": Channel("m/h", "rate of penetration (m/h)"),
    "spp_bar": Channel("bar", "standpipe pressure (bar)"),
    "flow_in_lpm": Channel("L/min", "flow in (L/min)"),
    "flow_out_lpm": Channel("L/min", "flow out (L/min)"),
    "pit_volume_m3": Channel("m3", "pit volume (m³)"),
    "mud_weight_sg": Channel("sg", "mud weight (sg)"),
    "gas_units": Channel("units", "total gas (units)"),
}

ANOMALY_TYPES = ("Kick", "Lost Circulation", "Stuck Pipe", "Washout", "Pack-off", "Sensor Drift")

# NEEDS REVIEW: severities, priorities and responses are a lookup invented for this project.
SEVERITY_RULES = {
    "Kick": SeverityRule("Critical", 100, "Flow check; if flow continues with pumps off, shut in and follow well-control procedure"),
    "Stuck Pipe": SeverityRule("High", 90, "Stop applying weight; work the string within torque and overpull limits"),
    "Lost Circulation": SeverityRule("High", 85, "Reduce pump rate, check returns and pit level, consider loss-control material"),
    "Pack-off": SeverityRule("High", 80, "Reduce flow rate, avoid pressure spikes, work the string and restore circulation"),
    "Washout": SeverityRule("Medium", 70, "Compare pump pressure with pump rate history; plan a string inspection"),
    "Sensor Drift": SeverityRule("Low", 40, "Cross-check the channel against a redundant sensor and recalibrate"),
}

# Hypotheses only: surface signals cannot confirm a downhole cause.
VOCAB = {
    "Kick": (
        "formation fluid entering the wellbore (flow-out above flow-in and pit volume rising)",
        "Loss of well control if the influx is not confirmed and managed.",
    ),
    "Lost Circulation": (
        "drilling fluid being lost to the formation (flow-out below flow-in and pit volume falling)",
        "Loss of hydrostatic head, which can allow an influx, plus lost fluid cost.",
    ),
    "Stuck Pipe": (
        "the string sticking in the hole (torque rising while rate of penetration collapses)",
        "Lost time, possible fishing operation and loss of the string or bottom-hole assembly.",
    ),
    "Washout": (
        "a leak in the drill string (standpipe pressure falling at constant pump rate)",
        "String failure (twist-off) if not found early, and poorer hole cleaning.",
    ),
    "Pack-off": (
        "cuttings or collapsed formation restricting the annulus (standpipe pressure rising, returns falling)",
        "Stuck pipe, lost circulation from pressure build-up and loss of hole cleaning.",
    ),
    "Sensor Drift": (
        "a sensor drifting or needing calibration rather than a real process change",
        "Decisions taken on unreliable readings; the real condition may be hidden.",
    ),
}

BIT_AREA_M2 = 0.0366  # 8.5 in bit, an assumption (NEEDS REVIEW)
NOMINAL_FLOW_LPM = 2000.0  # normalisation constant for spp_norm, not a measurement
STEPS_PER_HOUR = 60  # features assume 1-minute sampling


def _flow_delta(g):
    return (g["flow_out_lpm"] - g["flow_in_lpm"]).rolling(5, min_periods=1).mean()


def _pit_rate(g):
    return g["pit_volume_m3"].diff(10) * (STEPS_PER_HOUR / 10)


def _depth_progress(g):
    return g["depth_m"].diff(10) * (STEPS_PER_HOUR / 10)


def _spp_norm(g):
    f = g["flow_in_lpm"]
    norm = g["spp_bar"] / (f / NOMINAL_FLOW_LPM) ** 1.8
    return norm.where(f > 500)  # undefined with the pumps (nearly) off


def _mse(g):
    """Teale mechanical specific energy in MPa; only defined while drilling on bottom."""
    rop_mpm = g["rop_mph"] / 60.0
    e_kpa = g["wob_kn"] / BIT_AREA_M2 + 2 * np.pi * g["rpm"] * g["torque_knm"] / (BIT_AREA_M2 * rop_mpm)
    return (e_kpa / 1000.0).where((rop_mpm > 0.05) & (g["rpm"] > 5))


FEATURES = (
    PhysicsFeature("flow_delta_lpm", "flow_out - flow_in, 5-min trailing mean (kick / loss indicator)", _flow_delta),
    PhysicsFeature("pit_rate_m3ph", "pit volume change over the last 10 min, per hour", _pit_rate),
    PhysicsFeature("depth_progress_mph", "depth change over the last 10 min, per hour (stalls when stuck)", _depth_progress),
    PhysicsFeature("spp_norm", "standpipe pressure / (flow_in/2000)^1.8; falls with a washout, rises with a pack-off", _spp_norm),
    PhysicsFeature("mse_mpa", "Teale mechanical specific energy (MPa), on-bottom only", _mse),
)

PROFILE = Profile(
    name="drilling",
    description="Rotary drilling on a rig: depth, WOB, RPM, torque, ROP, standpipe pressure, flow in/out, pit volume, mud weight, gas.",
    channels=CHANNELS,
    anomaly_types=ANOMALY_TYPES,
    physics_features=FEATURES,
    severity_rules=SEVERITY_RULES,
    vocab=VOCAB,
)

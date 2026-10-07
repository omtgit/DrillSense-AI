"""Production-style profile: the original DrillSense variables, moved here unchanged
(channel labels and report wording are the ones report.py used before profiles existed;
the severity lookup is the one in notebooks/01_generate_synthetic_data.py;
the features are the ones in notebooks/02_feature_engineering.py).

The notebooks still generate this data. This profile only describes it.
"""
from .base import Channel, PhysicsFeature, Profile, SeverityRule

CHANNELS = {
    "pressure_psi": Channel("psi", "pressure (psi)"),
    "temperature_c": Channel("degC", "temperature (°C)"),
    "flow_rate_bpd": Channel("BPD", "flow rate (BPD)"),
    "vibration": Channel("", "vibration"),
    "gas_ratio": Channel("", "gas ratio"),
    "pump_rpm": Channel("rpm", "pump RPM"),
    "torque": Channel("", "torque"),
    "rop": Channel("", "rate of penetration"),
}

SEVERITY_RULES = {
    "Kick Risk": SeverityRule("Critical", 100, "Immediate inspection and pressure control check"),
    "Leak Risk": SeverityRule("High", 90, "Inspect within 2 hours"),
    "High Vibration": SeverityRule("High", 85, "Inspect rotating equipment within 4 hours"),
    "Pump Failure": SeverityRule("Medium", 70, "Maintenance within 6 hours"),
    "Sensor Drift": SeverityRule("Low", 40, "Calibrate sensor during next maintenance"),
}

# Hypotheses only: the data cannot confirm a root cause.
VOCAB = {
    "Pump Failure": (
        "pump degradation or loss of pump efficiency",
        "Loss of circulation or lift capacity and possible equipment damage.",
    ),
    "Leak Risk": (
        "a leak or loss of containment in the flow path",
        "Fluid loss, pressure loss and possible environmental release.",
    ),
    "Kick Risk": (
        "formation fluid or gas entering the wellbore",
        "Loss of well control if the influx is not confirmed and managed.",
    ),
    "Sensor Drift": (
        "sensor drift or calibration error rather than a real process change",
        "Decisions taken on unreliable readings; the real condition may be hidden.",
    ),
}


def _diff(col):
    return lambda g: g[col].diff()


def _roll(col):
    return lambda g: g[col].rolling(12, min_periods=1).mean()


FEATURES = (
    PhysicsFeature("pressure_change", "1-step change in pressure_psi", _diff("pressure_psi")),
    PhysicsFeature("temperature_change", "1-step change in temperature_c", _diff("temperature_c")),
    PhysicsFeature("flow_change", "1-step change in flow_rate_bpd", _diff("flow_rate_bpd")),
    PhysicsFeature("vibration_change", "1-step change in vibration", _diff("vibration")),
    PhysicsFeature("rolling_pressure_mean", "12-step trailing mean of pressure_psi", _roll("pressure_psi")),
    PhysicsFeature("rolling_temperature_mean", "12-step trailing mean of temperature_c", _roll("temperature_c")),
    PhysicsFeature("rolling_flow_mean", "12-step trailing mean of flow_rate_bpd", _roll("flow_rate_bpd")),
    PhysicsFeature("rolling_vibration_mean", "12-step trailing mean of vibration", _roll("vibration")),
)

PROFILE = Profile(
    name="production",
    description="Production-style variables of the original synthetic dataset (stub until real data is added).",
    channels=CHANNELS,
    anomaly_types=tuple(SEVERITY_RULES),
    physics_features=FEATURES,
    severity_rules=SEVERITY_RULES,
    vocab=VOCAB,
)

"""Human-readable names for model feature columns such as ``pit_volume_m3__dev180``."""
import re

SUFFIXES = {
    "mean15": "average over the last 15 min",
    "std15": "variability over the last 15 min",
    "dev180": "change vs its 3-hour average",
    "d5": "change over the last 5 min",
}

# Derived (physics) signals of the drilling profile; channel names come from the profile itself.
PHYSICS_NAMES = {
    "flow_delta_lpm": "Flow out minus flow in",
    "pit_rate_m3ph": "Pit volume rate of change",
    "depth_progress_mph": "Drilling progress",
    "spp_norm": "Standpipe pressure (corrected for flow)",
    "mse_mpa": "Mechanical specific energy",
}


def _short(label):
    """'pit volume (m³)' -> 'Pit volume'."""
    text = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    return text[:1].upper() + text[1:]


def feature_label(name, profile):
    """'pit_volume_m3__dev180' -> 'Pit volume: change vs its 3-hour average'."""
    base, _, suffix = name.partition("__")
    if base in profile.channels:
        title = _short(profile.channels[base].report_label)
    elif base in PHYSICS_NAMES:
        title = PHYSICS_NAMES[base]
    else:
        title = _short(base.replace("_", " "))
    return f"{title}: {SUFFIXES[suffix]}" if suffix in SUFFIXES else f"{title}: current value"

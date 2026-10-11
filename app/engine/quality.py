"""Data quality monitoring for real-time telemetry.

Identifies:
- Stale data / flatlines (unchanged readings over a rolling window)
- Gaps (timestamp intervals exceeding expected cadence)
- Spikes (unphysical rate of change between adjacent readings)
- Out-of-range sensor values (violating physical or operational limits)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

# Standard operational physical bounds for rotary drilling channels
CHANNEL_BOUNDS = {
    "wob_kn": (-10.0, 450.0),
    "rpm": (0.0, 300.0),
    "torque_knm": (0.0, 80.0),
    "rop_mph": (0.0, 200.0),
    "spp_bar": (0.0, 450.0),
    "flow_in_lpm": (0.0, 6000.0),
    "flow_out_lpm": (0.0, 6000.0),
    "pit_volume_m3": (0.0, 500.0),
    "mud_weight_sg": (0.8, 2.8),
    "gas_units": (0.0, 5000.0),
}

# Max allowed rate of change per minute
MAX_DIFF_PER_MIN = {
    "mud_weight_sg": 0.5,
    "pit_volume_m3": 50.0,
    "flow_in_lpm": 3000.0,
    "spp_bar": 250.0,
}


@dataclass
class ChannelQuality:
    channel: str
    is_valid: bool = True
    issues: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    well_id: str
    timestamp: str
    is_good: bool = True
    overall_score: float = 1.0  # 1.0 = perfect quality, 0.0 = severe issues
    channel_issues: Dict[str, List[str]] = field(default_factory=dict)
    summary_flags: List[str] = field(default_factory=list)


class DataQualityMonitor:
    """Evaluates telemetry data quality per row and rolling window per well."""

    def __init__(self, flatline_window: int = 15, expected_cadence_sec: float = 60.0):
        self.flatline_window = flatline_window
        self.expected_cadence_sec = expected_cadence_sec
        # Channel history per well for flatline and rate of change checks
        self._history: Dict[str, Dict[str, List[float]]] = {}
        self._last_time: Dict[str, pd.Timestamp] = {}

    def assess_row(self, well_id: str, timestamp: str, channels: Dict[str, float]) -> QualityReport:
        issues: Dict[str, List[str]] = {}
        flags: List[str] = []

        # 1. Gap detection
        curr_ts = pd.to_datetime(timestamp)
        if well_id in self._last_time:
            delta_sec = (curr_ts - self._last_time[well_id]).total_seconds()
            if delta_sec > self.expected_cadence_sec * 3.0:
                flags.append(f"Telemetry gap: {delta_sec:.0f}s elapsed (expected {self.expected_cadence_sec:.0f}s)")
        self._last_time[well_id] = curr_ts

        # 2. Channel-level validation
        if well_id not in self._history:
            self._history[well_id] = {ch: [] for ch in channels}

        hist = self._history[well_id]

        for ch, val in channels.items():
            ch_issues = []
            if ch not in hist:
                hist[ch] = []

            # Check NaN / Missing
            if np.isnan(val) or val is None:
                ch_issues.append("missing_value")
            else:
                # Check Out-of-Range
                if ch in CHANNEL_BOUNDS:
                    lo, hi = CHANNEL_BOUNDS[ch]
                    if val < lo or val > hi:
                        ch_issues.append(f"out_of_range ({val:.2f} not in [{lo}, {hi}])")

                # Check Spike / Excessive delta
                if hist[ch] and ch in MAX_DIFF_PER_MIN:
                    last_val = hist[ch][-1]
                    if abs(val - last_val) > MAX_DIFF_PER_MIN[ch]:
                        ch_issues.append(f"spike_detected (Δ {abs(val - last_val):.2f})")

                # Track rolling values for flatline detection
                hist[ch].append(val)
                if len(hist[ch]) > self.flatline_window:
                    hist[ch].pop(0)

                # Check Flatline (constant value when channel should vary, e.g. non-zero pumps)
                if len(hist[ch]) == self.flatline_window:
                    variance = float(np.var(hist[ch]))
                    # If pumps or pressure flatline exactly to 6 decimal places over 15 min
                    if variance == 0.0 and ch in ("spp_bar", "flow_out_lpm", "torque_knm") and hist[ch][0] > 10.0:
                        ch_issues.append("sensor_flatline")

            if ch_issues:
                issues[ch] = ch_issues

        is_good = len(issues) == 0 and len(flags) == 0
        score = max(0.0, 1.0 - (len(issues) * 0.15 + len(flags) * 0.2))

        return QualityReport(
            well_id=well_id,
            timestamp=str(timestamp),
            is_good=is_good,
            overall_score=round(score, 2),
            channel_issues=issues,
            summary_flags=flags,
        )

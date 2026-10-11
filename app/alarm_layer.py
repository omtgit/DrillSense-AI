"""Alarm management layer: persistence filter, hysteresis, cooldown, and multi-tier severity.

Prevents single-row spurious spikes, suppresses alarm ringing during pump connections,
and enforces realistic operational warning/alarm thresholds.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

PERSISTENCE_N = 3      # At least N out of M rows must exceed threshold
PERSISTENCE_M = 5
HYSTERESIS_RATIO = 0.8  # Drop alarm only when score falls below 80% of threshold
COOLDOWN_MINUTES = 10   # Suppress re-alarm chatter within cooldown window


@dataclass
class AlarmState:
    level: str  # "normal", "watch", "warning", "alarm"
    is_alarm: bool
    filtered_score: float


def apply_alarm_layer(
    scores: np.ndarray,
    threshold: float,
    states: Optional[np.ndarray] = None,
    n_persist: int = PERSISTENCE_N,
    m_window: int = PERSISTENCE_M,
    hysteresis: float = HYSTERESIS_RATIO,
    cooldown: int = COOLDOWN_MINUTES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply persistence filtering, state conditioning, and hysteresis to raw score array.

    Returns:
        (alarm_flags, alarm_levels)
        where alarm_flags is a boolean array and alarm_levels is an array of strings
        ("normal", "watch", "warning", "alarm").
    """
    total = len(scores)
    alarm_flags = np.zeros(total, bool)
    alarm_levels = np.full(total, "normal", dtype=object)

    if total == 0 or np.isnan(threshold) or threshold <= 0:
        return alarm_flags, alarm_levels

    thr_alarm = threshold
    thr_warning = threshold * 0.70
    thr_watch = threshold * 0.40
    thr_reset = threshold * hysteresis

    # State condition: during connections, require stronger evidence or specific flow check
    active_alarm = False
    cooldown_counter = 0

    # Moving window queue
    recent_high = []

    for i in range(total):
        s = scores[i]
        is_conn = (states[i] == "connection") if states is not None else False

        # In connection state, raise raw bar slightly against connection transients
        effective_score = s * 0.85 if is_conn else s

        recent_high.append(effective_score >= thr_alarm)
        if len(recent_high) > m_window:
            recent_high.pop(0)

        n_high = sum(recent_high)

        if active_alarm:
            if effective_score < thr_reset:
                active_alarm = False
                cooldown_counter = cooldown
            else:
                alarm_flags[i] = True
                alarm_levels[i] = "alarm"
        else:
            if cooldown_counter > 0:
                cooldown_counter -= 1
                alarm_flags[i] = False
                alarm_levels[i] = "watch" if effective_score >= thr_watch else "normal"
            elif n_high >= n_persist:
                active_alarm = True
                alarm_flags[i] = True
                alarm_levels[i] = "alarm"
            elif effective_score >= thr_warning:
                alarm_levels[i] = "warning"
            elif effective_score >= thr_watch:
                alarm_levels[i] = "watch"
            else:
                alarm_levels[i] = "normal"

    return alarm_flags, alarm_levels

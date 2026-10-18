"""Dataset quality audit utility for loaded telemetry data.

Analyzes:
- Missing value percentages per channel
- Flatlines / frozen sensor signals
- Out-of-range sensor readings
- Sampling interval cadence (median, min, max, irregular rate)
- Rig-state distribution mix (drilling, connection, circulating, off_bottom)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from app.engine.quality import CHANNEL_BOUNDS
from app.rig_state import (
    STATE_CIRCULATING,
    STATE_CONNECTION,
    STATE_DRILLING,
    STATE_OFF_BOTTOM,
    infer_rig_state,
)


@dataclass
class DatasetQualitySummary:
    total_rows: int
    well_count: int
    sampling_interval_sec: float
    sampling_interval_str: str
    channel_missing_pct: Dict[str, float]
    channel_flatlines_pct: Dict[str, float]
    channel_out_of_range_pct: Dict[str, float]
    rig_state_mix: Dict[str, float]  # state -> percentage
    summary_flags: List[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Tabulate channel audit into a clean DataFrame."""
        channels = sorted(list(self.channel_missing_pct.keys()))
        rows = []
        for ch in channels:
            rows.append({
                "Channel": ch,
                "Missing (%)": round(self.channel_missing_pct.get(ch, 0.0), 2),
                "Flatline (%)": round(self.channel_flatlines_pct.get(ch, 0.0), 2),
                "Out of Range (%)": round(self.channel_out_of_range_pct.get(ch, 0.0), 2),
            })
        return pd.DataFrame(rows)


def audit_dataset_quality(df: pd.DataFrame, channels: Optional[List[str]] = None) -> DatasetQualitySummary:
    """Run an audit of data quality on any loaded drilling DataFrame."""
    if df.empty:
        return DatasetQualitySummary(
            total_rows=0,
            well_count=0,
            sampling_interval_sec=0.0,
            sampling_interval_str="N/A",
            channel_missing_pct={},
            channel_flatlines_pct={},
            channel_out_of_range_pct={},
            rig_state_mix={},
            summary_flags=["Empty dataset"],
        )

    n_rows = len(df)
    n_wells = df["well_id"].nunique() if "well_id" in df.columns else 1

    # Audit channels
    if channels is None:
        exclude = {"well_id", "timestamp", "anomaly_flag", "anomaly_type", "rig_state", "date", "activity", "remarks"}
        channels = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # 1. Missing %
    missing_pct = {}
    for ch in channels:
        missing_count = df[ch].isna().sum()
        missing_pct[ch] = (missing_count / n_rows) * 100.0

    # 2. Flatline % (consecutive identical values > 10 rows)
    flatline_pct = {}
    for ch in channels:
        series = df[ch].dropna()
        if len(series) < 5:
            flatline_pct[ch] = 0.0
            continue
        # diff == 0 count
        zero_diffs = (series.diff() == 0).sum()
        flatline_pct[ch] = (zero_diffs / max(len(series), 1)) * 100.0

    # 3. Out of Range %
    out_of_range_pct = {}
    for ch in channels:
        if ch in CHANNEL_BOUNDS:
            lo, hi = CHANNEL_BOUNDS[ch]
            oor_count = ((df[ch] < lo) | (df[ch] > hi)).sum()
            out_of_range_pct[ch] = (oor_count / n_rows) * 100.0
        else:
            out_of_range_pct[ch] = 0.0

    # 4. Sampling interval
    median_interval = 60.0
    interval_str = "60s (estimated)"
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
        if len(ts) > 1:
            diffs = ts.diff().dt.total_seconds().dropna()
            diffs_pos = diffs[diffs > 0]
            if not diffs_pos.empty:
                median_interval = float(diffs_pos.median())
                if median_interval < 60:
                    interval_str = f"{median_interval:.1f} s"
                elif median_interval < 3600:
                    interval_str = f"{median_interval / 60.0:.1f} min"
                else:
                    interval_str = f"{median_interval / 3600.0:.1f} hours"

    # 5. Rig state mix
    try:
        states = infer_rig_state(df)
        state_counts = states.value_counts(normalize=True) * 100.0
        rig_state_mix = {
            STATE_DRILLING: round(float(state_counts.get(STATE_DRILLING, 0.0)), 1),
            STATE_CONNECTION: round(float(state_counts.get(STATE_CONNECTION, 0.0)), 1),
            STATE_CIRCULATING: round(float(state_counts.get(STATE_CIRCULATING, 0.0)), 1),
            STATE_OFF_BOTTOM: round(float(state_counts.get(STATE_OFF_BOTTOM, 0.0)), 1),
        }
    except Exception:
        rig_state_mix = {
            STATE_DRILLING: 0.0,
            STATE_CONNECTION: 0.0,
            STATE_CIRCULATING: 0.0,
            STATE_OFF_BOTTOM: 0.0,
        }

    flags = []
    if any(pct > 20.0 for pct in missing_pct.values()):
        flags.append("High missing rate (>20%) detected in one or more channels")
    if any(pct > 10.0 for pct in out_of_range_pct.values()):
        flags.append("Out-of-range sensor readings detected (>10%)")
    if any(pct > 50.0 for pct in flatline_pct.values()):
        flags.append("Sensors exhibit extended flatlines (>50% repeated readings)")

    return DatasetQualitySummary(
        total_rows=n_rows,
        well_count=n_wells,
        sampling_interval_sec=median_interval,
        sampling_interval_str=interval_str,
        channel_missing_pct=missing_pct,
        channel_flatlines_pct=flatline_pct,
        channel_out_of_range_pct=out_of_range_pct,
        rig_state_mix=rig_state_mix,
        summary_flags=flags,
    )

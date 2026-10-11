"""Real-time engine: Sliding-window online feature buffer, detector scoring,
alarm layer, and alert state machine.

Ensures that ONLINE features computed on streaming rows exactly match BATCH features.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple
import hashlib
import numpy as np
import pandas as pd

from app.alarm_layer import apply_alarm_layer
from app.engine.quality import DataQualityMonitor, QualityReport
from app.engine.storage import AlertRecord, RuntimeStorage
from app.features import add_context, add_profile_features
from app.ingest.sources import TelemetryRow
from app.rig_state import infer_rig_state
from profiles import get_profile


@dataclass
class EngineTickResult:
    well_id: str
    timestamp: str
    rig_state: str
    risk_score: float
    alarm_level: str  # "normal", "watch", "warning", "alarm"
    is_alarm: bool
    predicted_anomaly: str
    predicted_severity: str
    quality: QualityReport
    alert_record: Optional[AlertRecord] = None


class WellBuffer:
    """Maintains a rolling sliding-window buffer of raw telemetry per well."""

    def __init__(self, well_id: str, max_window_rows: int = 240):
        self.well_id = well_id
        self.max_window_rows = max_window_rows
        self._rows: List[Dict[str, Any]] = []

    def append(self, row: TelemetryRow) -> None:
        d = row.to_dict()
        self._rows.append(d)
        if len(self._rows) > self.max_window_rows:
            self._rows.pop(0)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._rows)
        return df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._rows)


class LiveEngine:
    """Stateful online processing engine per well."""

    def __init__(
        self,
        profile_name: str = "drilling",
        detector_instance: Optional[Any] = None,
        storage: Optional[RuntimeStorage] = None,
        quality_monitor: Optional[DataQualityMonitor] = None,
        window_size: int = 240,
    ):
        self.profile = get_profile(profile_name)
        self.detector = detector_instance
        self.storage = storage or RuntimeStorage()
        self.quality_monitor = quality_monitor or DataQualityMonitor()
        self.window_size = window_size
        self._buffers: Dict[str, WellBuffer] = {}
        # Track active alerts per well: well_id -> current alert_id
        self._active_alert_ids: Dict[str, str] = {}
        # Track last alert generation timestamp for cooldown
        self._last_alert_time: Dict[str, pd.Timestamp] = {}

    def _get_buffer(self, well_id: str) -> WellBuffer:
        if well_id not in self._buffers:
            self._buffers[well_id] = WellBuffer(well_id, max_window_rows=self.window_size)
        return self._buffers[well_id]

    def _generate_alert_id(self, well_id: str, timestamp: str, anomaly_type: str) -> str:
        raw_token = f"{well_id}:{timestamp}:{anomaly_type}"
        h = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()[:12]
        return f"ALT-{well_id}-{h}"

    def process_row(self, row: TelemetryRow) -> EngineTickResult:
        """Process one incoming telemetry row in online real-time mode."""
        buf = self._get_buffer(row.well_id)
        buf.append(row)

        # 1. Data Quality Check
        quality = self.quality_monitor.assess_row(row.well_id, str(row.timestamp), row.channels)

        # 2. Window-based Feature Computation (Online equals Batch)
        df_win = buf.to_dataframe()
        df_feat = add_profile_features(df_win, self.profile)
        raw_cols = [c for c in self.profile.channels if c not in self.profile.trend_channels]
        phys_cols = [f.name for f in self.profile.physics_features]
        df_prepared = add_context(df_feat, raw_cols + phys_cols)

        # Current row features (last in window)
        latest_row = df_prepared.iloc[-1:]

        # 3. Model Scoring
        if self.detector is not None:
            scored_win = self.detector.predict(df_prepared)
            cur_scored = scored_win.iloc[-1]
            p_anom = str(cur_scored["predicted_anomaly"])
            p_sev = str(cur_scored["predicted_severity"])
            risk_score = float(cur_scored["predicted_risk_score"])
            # All raw risk scores in window for alarm layer
            scores_arr = scored_win["predicted_risk_score"].values / 100.0
        else:
            # Fallback heuristic
            p_anom = "Normal"
            p_sev = "None"
            risk_score = 10.0
            scores_arr = np.array([0.1])

        # 4. Rig-State Inference
        state_series = infer_rig_state(df_win)
        curr_state = str(state_series.iloc[-1])

        # 5. Alarm Layer (Persistence, Hysteresis, Cooldown)
        flags, levels = apply_alarm_layer(scores_arr, threshold=0.5, states=state_series.values)
        is_alarm = bool(flags[-1])
        alarm_level = str(levels[-1])

        # 6. Alert Lifecycle & State Machine
        alert_record = None
        now_ts = pd.to_datetime(row.timestamp)

        if is_alarm and p_anom != "Normal":
            # Check if active alert already exists for this well
            existing_id = self._active_alert_ids.get(row.well_id)
            if existing_id:
                rec = self.storage.get_alert(existing_id)
                if rec and rec.status == "snoozed":
                    # Check if snooze duration expired
                    if rec.snooze_until and pd.to_datetime(rec.snooze_until) > now_ts:
                        # Still snoozed
                        alert_record = rec
                    else:
                        # Re-open from snooze
                        self.storage.update_alert_status(existing_id, "open")
                        rec.status = "open"
                        alert_record = rec
                elif rec and rec.status in ("open", "acknowledged"):
                    # Existing open/acknowledged alert continues
                    alert_record = rec
                else:
                    # Closed or missing -> create new alert
                    alert_record = self._create_alert(row, p_anom, p_sev, risk_score, quality)
            else:
                alert_record = self._create_alert(row, p_anom, p_sev, risk_score, quality)
        else:
            # Normal or sub-alarm -> close active alert if present and open
            existing_id = self._active_alert_ids.get(row.well_id)
            if existing_id:
                rec = self.storage.get_alert(existing_id)
                if rec and rec.status in ("open", "snoozed"):
                    self.storage.update_alert_status(existing_id, "closed")
                self._active_alert_ids.pop(row.well_id, None)

        return EngineTickResult(
            well_id=row.well_id,
            timestamp=str(row.timestamp),
            rig_state=curr_state,
            risk_score=risk_score,
            alarm_level=alarm_level,
            is_alarm=is_alarm,
            predicted_anomaly=p_anom,
            predicted_severity=p_sev,
            quality=quality,
            alert_record=alert_record,
        )

    def _create_alert(
        self,
        row: TelemetryRow,
        anomaly_type: str,
        severity: str,
        risk_score: float,
        quality: QualityReport
    ) -> AlertRecord:
        alt_id = self._generate_alert_id(row.well_id, str(row.timestamp), anomaly_type)
        quality_flag = "; ".join(quality.summary_flags) if not quality.is_good else None
        rec = AlertRecord(
            alert_id=alt_id,
            well_id=row.well_id,
            timestamp=str(row.timestamp),
            anomaly_type=anomaly_type,
            severity=severity,
            risk_score=risk_score,
            status="open",
            data_quality_flag=quality_flag,
        )
        self.storage.save_alert(rec)
        self._active_alert_ids[row.well_id] = alt_id
        return rec

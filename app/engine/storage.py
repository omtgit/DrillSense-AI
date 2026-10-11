"""SQLite persistence layer for alerts, events, operator acknowledgements, and feedback.

Stored strictly in data/runtime/drill_runtime.db (gitignored).
Contains no personal data.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import sqlite3

RUNTIME_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "runtime"
DB_PATH = RUNTIME_DIR / "drill_runtime.db"


@dataclass
class AlertRecord:
    alert_id: str
    well_id: str
    timestamp: str
    anomaly_type: str
    severity: str
    risk_score: float
    status: str  # "open", "acknowledged", "snoozed", "closed"
    snooze_until: Optional[str] = None
    data_quality_flag: Optional[str] = None
    operator_feedback: Optional[str] = None  # "true_event", "false_alarm", "unsure"
    feedback_notes: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class RuntimeStorage:
    """Read/write storage for operational alert lifecycle and operator interactions."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    well_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    snooze_until TEXT,
                    data_quality_flag TEXT,
                    operator_feedback TEXT,
                    feedback_notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_well_status 
                ON alerts (well_id, status)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_snapshots (
                    well_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (well_id, timestamp)
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_alert(self, alert: AlertRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        created = alert.created_at or now
        updated = now
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO alerts (
                    alert_id, well_id, timestamp, anomaly_type, severity, risk_score,
                    status, snooze_until, data_quality_flag, operator_feedback,
                    feedback_notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id, alert.well_id, alert.timestamp, alert.anomaly_type,
                alert.severity, alert.risk_score, alert.status, alert.snooze_until,
                alert.data_quality_flag, alert.operator_feedback, alert.feedback_notes,
                created, updated
            ))
            conn.commit()
        finally:
            conn.close()

    def get_alert(self, alert_id: str) -> Optional[AlertRecord]:
        conn = self._get_conn()
        try:
            cur = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            row = cur.fetchone()
            if row:
                return AlertRecord(**dict(row))
            return None
        finally:
            conn.close()

    def list_alerts(self, well_id: Optional[str] = None, status: Optional[str] = None, limit: int = 100) -> List[AlertRecord]:
        query = "SELECT * FROM alerts WHERE 1=1"
        params: List[Any] = []
        if well_id:
            query += " AND well_id = ?"
            params.append(well_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self._get_conn()
        try:
            cur = conn.execute(query, params)
            return [AlertRecord(**dict(row)) for row in cur.fetchall()]
        finally:
            conn.close()

    def update_alert_status(
        self,
        alert_id: str,
        status: str,
        snooze_until: Optional[str] = None,
        operator_feedback: Optional[str] = None,
        feedback_notes: Optional[str] = None
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            conn.execute("""
                UPDATE alerts
                SET status = ?,
                    snooze_until = COALESCE(?, snooze_until),
                    operator_feedback = COALESCE(?, operator_feedback),
                    feedback_notes = COALESCE(?, feedback_notes),
                    updated_at = ?
                WHERE alert_id = ?
            """, (status, snooze_until, operator_feedback, feedback_notes, now, alert_id))
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    def clear_runtime_data(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM alerts")
            conn.execute("DELETE FROM telemetry_snapshots")
            conn.commit()
        finally:
            conn.close()

"""Ingest layer: Read-only telemetry data source interfaces and implementations.

Strictly read-only: this module never writes back to a rig, database, or WITSML server.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union
import time
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class TelemetryRow:
    """A single timestamped telemetry record mapped to profile channel names."""
    timestamp: Union[str, datetime]
    well_id: str
    channels: Dict[str, float]
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"timestamp": str(self.timestamp), "well_id": self.well_id}
        d.update(self.channels)
        return d


class BaseTelemetrySource(ABC):
    """Abstract base class for all telemetry sources (strictly READ-ONLY)."""

    @abstractmethod
    def stream_rows(self) -> Iterator[TelemetryRow]:
        """Yield timestamped rows keyed by profile channel names."""
        pass

    @property
    @abstractmethod
    def well_ids(self) -> List[str]:
        """List of available well IDs in the source."""
        pass


class ReplaySource(BaseTelemetrySource):
    """Replays historical telemetry (from DataFrame, CSV, Parquet, or generated dataset).

    Supports:
    - Speed multiplier (1.0 = 1 sec per row, 10.0 = 10x speed, 0 = non-blocking batch stream)
    - Pause & resume
    - Seeking (by index or percentage)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        speed_multiplier: float = 0.0,
        loop: bool = False,
    ):
        self._df = df.copy().sort_values(["timestamp", "well_id"]).reset_index(drop=True)
        self.speed_multiplier = speed_multiplier
        self.loop = loop
        self._cursor = 0
        self._paused = False

    @property
    def well_ids(self) -> List[str]:
        return sorted(self._df["well_id"].unique().tolist())

    @property
    def total_rows(self) -> int:
        return len(self._df)

    @property
    def current_index(self) -> int:
        return self._cursor

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def is_paused(self) -> bool:
        return self._paused

    def seek(self, target_row: int) -> None:
        """Seek cursor to specified row index."""
        self._cursor = max(0, min(target_row, len(self._df) - 1))

    def seek_fraction(self, fraction: float) -> None:
        """Seek cursor to a fraction [0.0, 1.0] of total rows."""
        target = int(len(self._df) * max(0.0, min(1.0, fraction)))
        self.seek(target)

    def seek_next_event(self) -> Optional[int]:
        """Jump cursor to the next row where anomaly_flag == 1 (if available)."""
        if "anomaly_flag" not in self._df.columns:
            return None
        sub = self._df.iloc[self._cursor + 1:]
        events = sub[sub["anomaly_flag"] == 1]
        if not events.empty:
            next_idx = int(events.index[0])
            self.seek(next_idx)
            return next_idx
        return None

    def stream_rows(self) -> Iterator[TelemetryRow]:
        """Stream telemetry rows sequentially with pause and speed multiplier support."""
        while self._cursor < len(self._df):
            if self._paused:
                time.sleep(0.05)
                continue

            row = self._df.iloc[self._cursor]
            self._cursor += 1

            well = str(row["well_id"])
            ts = row["timestamp"]
            # Extract channel values
            exclude = {"well_id", "timestamp", "rig_state", "anomaly_flag", "anomaly_type", "event_scale",
                       "severity", "priority_score", "recommended_response", "active_anomalies"}
            ch_dict = {}
            for col in self._df.columns:
                if col not in exclude:
                    val = row[col]
                    try:
                        ch_dict[col] = float(val) if pd.notnull(val) else np.nan
                    except (ValueError, TypeError):
                        pass

            if self.speed_multiplier > 0:
                delay = 1.0 / self.speed_multiplier
                time.sleep(min(delay, 1.0))

            yield TelemetryRow(
                timestamp=ts,
                well_id=well,
                channels=ch_dict,
                raw_metadata={"anomaly_flag": int(row.get("anomaly_flag", 0)) if "anomaly_flag" in row else None,
                              "anomaly_type": str(row.get("anomaly_type", "Normal")) if "anomaly_type" in row else None}
            )

            if self.loop and self._cursor >= len(self._df):
                self._cursor = 0


class CsvSource(BaseTelemetrySource):
    """Ingests generic external CSV files using a YAML column mapping configuration.

    The mapping maps client/rig column names to standardized profile channels,
    applying optional scaling, offsets, and unit conversions.
    """

    def __init__(self, file_path: Union[str, Path], mapping_yaml_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.mapping_yaml_path = Path(mapping_yaml_path)
        with open(self.mapping_yaml_path, "r", encoding="utf-8") as f:
            self.mapping_config = yaml.safe_load(f)

        self._df = self._load_and_map()

    def _load_and_map(self) -> pd.DataFrame:
        raw_df = pd.read_csv(self.file_path)
        mapped_df = pd.DataFrame()

        well_col = self.mapping_config.get("well_id_column", "well_id")
        ts_col = self.mapping_config.get("timestamp_column", "timestamp")

        mapped_df["well_id"] = raw_df[well_col].astype(str) if well_col in raw_df.columns else "WELL-001"
        mapped_df["timestamp"] = raw_df[ts_col] if ts_col in raw_df.columns else pd.date_range("2026-01-01", periods=len(raw_df), freq="1min")

        columns_map = self.mapping_config.get("channels", {})
        for raw_col, spec in columns_map.items():
            if raw_col not in raw_df.columns:
                continue
            target_ch = spec.get("target_channel", raw_col)
            scale = float(spec.get("scale", 1.0))
            offset = float(spec.get("offset", 0.0))
            mapped_df[target_ch] = raw_df[raw_col].astype(float) * scale + offset

        return mapped_df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)

    @property
    def well_ids(self) -> List[str]:
        return sorted(self._df["well_id"].unique().tolist())

    def stream_rows(self) -> Iterator[TelemetryRow]:
        replay = ReplaySource(self._df, speed_multiplier=0.0)
        yield from replay.stream_rows()


class WitsmlEtpSourceInterface(BaseTelemetrySource):
    """Specification and documentation interface for real-time WITSML / ETP telemetry sources.

    Design Notes:
    - In WITSML (1.4.1.1 or 2.0) and ETP (Energistics Transfer Protocol via WebSockets):
      * Channel metadata (channel mnemonic, uom, log curve data) are subscribed to asynchronously.
      * The adapter buffers incoming protocol frames, maps XML/Avro records into TelemetryRow objects,
        and yields them into the engine pipeline.
    - Strictly READ-ONLY: DrillSense AI only subscribes (ETP Store / Discovery / ChannelStreaming)
      and never publishes control signals, channel updates, or write commands back to the rig server.
    """

    def __init__(self, endpoint_url: str = "", credentials_ref: str = ""):
        self.endpoint_url = endpoint_url
        self.credentials_ref = credentials_ref

    @property
    def well_ids(self) -> List[str]:
        return []

    def stream_rows(self) -> Iterator[TelemetryRow]:
        raise NotImplementedError(
            "WITSML/ETP network transport is an interface specification. "
            "Network access is disabled in this build per security policy."
        )

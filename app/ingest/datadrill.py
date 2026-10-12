"""DataDrill adapter: CSV loader mapping DataDRILL channels to the drilling profile.

Strictly READ-ONLY: never writes to external databases or servers.
Based on the DataDRILL research dataset structure (Arifeen et al., Zenodo).
"""
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np
import pandas as pd
import yaml

from app.ingest.sources import BaseTelemetrySource, ReplaySource, TelemetryRow


class DataDrillSource(BaseTelemetrySource):
    """Loads DataDrill CSV files and maps columns to DrillSense drilling channels.
    
    Supports:
    - Default or custom YAML mapping configuration
    - Missing value imputation and null handling
    - Telemetry row generation for live engine or offline analysis
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        mapping_yaml_path: Optional[Union[str, Path]] = None,
    ):
        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"DataDrill CSV file not found: {self.file_path}")

        default_yaml = Path(__file__).resolve().parent.parent.parent / "config" / "datadrill_mapping.yaml"
        self.mapping_yaml_path = Path(mapping_yaml_path) if mapping_yaml_path else default_yaml

        self.mapping_config = {}
        if self.mapping_yaml_path.is_file():
            with open(self.mapping_yaml_path, "r", encoding="utf-8") as f:
                self.mapping_config = yaml.safe_load(f)

        self._df = self._load_and_map()

    def _load_and_map(self) -> pd.DataFrame:
        raw = pd.read_csv(self.file_path)
        out = pd.DataFrame()

        well_col = self.mapping_config.get("well_id_column", "Well")
        ts_col = self.mapping_config.get("timestamp_column", "Timestamp")

        # Well ID
        if well_col in raw.columns:
            out["well_id"] = raw[well_col].astype(str)
        elif "well_id" in raw.columns:
            out["well_id"] = raw["well_id"].astype(str)
        else:
            out["well_id"] = "DATADRILL-001"

        # Timestamp
        if ts_col in raw.columns:
            out["timestamp"] = pd.to_datetime(raw[ts_col], errors="coerce")
        elif "timestamp" in raw.columns:
            out["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
        else:
            out["timestamp"] = pd.date_range("2026-01-01 00:00:00", periods=len(raw), freq="1min")

        # Fallback for invalid timestamps
        if out["timestamp"].isna().any():
            out["timestamp"] = out["timestamp"].ffill().bfill()

        # Map channels
        ch_configs = self.mapping_config.get("channels", {})
        # Normalize column mapping lookup
        col_lookup = {c.strip().lower(): c for c in raw.columns}

        for spec_col, spec in ch_configs.items():
            col_match = None
            if spec_col in raw.columns:
                col_match = spec_col
            elif spec_col.strip().lower() in col_lookup:
                col_match = col_lookup[spec_col.strip().lower()]

            if col_match:
                target = spec.get("target_channel", spec_col.lower())
                scale = float(spec.get("scale", 1.0))
                offset = float(spec.get("offset", 0.0))
                num_series = pd.to_numeric(raw[col_match], errors="coerce")
                out[target] = num_series * scale + offset

        # Transfer labels if present (e.g., kick label or anomaly flag)
        label_candidates = ["anomaly_flag", "anomaly_type", "kick_label", "Kick", "Label", "label"]
        for cand in label_candidates:
            if cand in raw.columns and cand not in out.columns:
                out[cand] = raw[cand]

        return out.sort_values(["well_id", "timestamp"]).reset_index(drop=True)

    @property
    def well_ids(self) -> List[str]:
        if self._df.empty or "well_id" not in self._df.columns:
            return []
        return sorted(self._df["well_id"].unique().tolist())

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    def stream_rows(self) -> Iterator[TelemetryRow]:
        if self._df.empty:
            return
        replay = ReplaySource(self._df, speed_multiplier=0.0)
        yield from replay.stream_rows()

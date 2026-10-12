"""Volve data adapter: WITSML real-time XML log parser and Daily Drilling Report (DDR) parser.

Strictly READ-ONLY: never writes to any rig, database, or WITSML server.

Index Support:
- TIME-INDEXED logs (indexType == 'date time'): Fully supported for real-time telemetry streaming,
  rolling feature computation, rig-state inference, and anomaly detection.
- DEPTH-INDEXED logs (indexType == 'measured depth'): Supported for structural inspection and depth-profile
  analysis. When mapped to time-series pipelines without explicit timestamps, pseudo-chronological progression
  is generated based on measured depth progression, or indexed directly by depth.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import yaml

from app.ingest.sources import BaseTelemetrySource, ReplaySource, TelemetryRow


def _strip_ns(tag: str) -> str:
    """Strip XML namespace prefix from element tags."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_witsml_log_xml(
    xml_path: Union[str, Path],
    mapping_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse a WITSML log XML file into a pandas DataFrame.
    
    Handles:
    - Namespace stripping
    - Log curve mnemonic extraction (logCurveInfo)
    - Missing value representations (null, -999.25, -9999, empty strings)
    - Unit conversion and channel mapping via mapping_config
    - Index type identification ('date time' vs 'measured depth')

    Returns:
        (df, metadata_dict)
    """
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"WITSML log XML file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    # Find log element
    log_elem = None
    for elem in root.iter():
        if _strip_ns(elem.tag).lower() == "log":
            log_elem = elem
            break

    if log_elem is None:
        raise ValueError(f"No <log> object found in WITSML file: {path}")

    # Metadata extraction
    well_name = "VOLVE-UNKNOWN"
    wellbore_name = ""
    index_type = "date time"
    null_value_str = "-999.25"

    for child in log_elem:
        tag_name = _strip_ns(child.tag).lower()
        if tag_name == "namewell" and child.text:
            well_name = child.text.strip()
        elif tag_name == "namewellbore" and child.text:
            wellbore_name = child.text.strip()
        elif tag_name == "indextype" and child.text:
            index_type = child.text.strip().lower()
        elif tag_name == "nullvalue" and child.text:
            null_value_str = child.text.strip()

    # Curve information
    mnemonics = []
    units = []
    for curve in log_elem.iter():
        if _strip_ns(curve.tag).lower() == "logcurveinfo":
            m_text = ""
            u_text = ""
            for c_child in curve:
                c_tag = _strip_ns(c_child.tag).lower()
                if c_tag == "mnemonic" and c_child.text:
                    m_text = c_child.text.strip().upper()
                elif c_tag == "unit" and c_child.text:
                    u_text = c_child.text.strip()
            if m_text:
                mnemonics.append(m_text)
                units.append(u_text)

    # Log data parsing
    data_rows = []
    for data_elem in log_elem.iter():
        if _strip_ns(data_elem.tag).lower() == "logdata":
            for row_elem in data_elem:
                if _strip_ns(row_elem.tag).lower() in ("data", "record"):
                    line = row_elem.text.strip() if row_elem.text else ""
                    if line:
                        tokens = [t.strip() for t in line.split(",")]
                        data_rows.append(tokens)

    if not data_rows:
        return pd.DataFrame(), {
            "well_name": well_name,
            "wellbore_name": wellbore_name,
            "index_type": index_type,
            "mnemonics": mnemonics,
            "total_rows": 0,
        }

    raw_df = pd.DataFrame(data_rows)
    if mnemonics and len(raw_df.columns) <= len(mnemonics):
        raw_df.columns = mnemonics[: len(raw_df.columns)]
    else:
        # Fallback names
        raw_df.columns = [f"COL_{i}" for i in range(len(raw_df.columns))]

    # Replace null representations with NaN
    null_tokens = {null_value_str, "-999.25", "-9999", "-999.0", "NULL", "null", "NaN", "nan", ""}
    for col in raw_df.columns:
        raw_df[col] = raw_df[col].replace(list(null_tokens), np.nan)

    # Identify index column
    index_col = raw_df.columns[0]
    out_df = pd.DataFrame()
    out_df["well_id"] = well_name

    is_time_indexed = "time" in index_type or "date" in index_type

    if is_time_indexed:
        out_df["timestamp"] = pd.to_datetime(raw_df[index_col], errors="coerce")
        # Forward fill or interpolate missing timestamps if any
        if out_df["timestamp"].isna().any():
            out_df["timestamp"] = out_df["timestamp"].ffill().bfill()
    else:
        # Depth indexed log
        depth_val = pd.to_numeric(raw_df[index_col], errors="coerce").fillna(0.0)
        out_df["depth_m"] = depth_val
        # Synthesize timestamps at 1-min increments for compatibility with time-series detector
        out_df["timestamp"] = pd.date_range("2026-01-01 00:00:00", periods=len(raw_df), freq="1min")

    # Channel mapping
    channel_maps = mapping_config.get("channels", {}) if mapping_config else {}

    for col in raw_df.columns:
        if col == index_col:
            continue
        col_clean = str(col).strip().upper()
        numeric_series = pd.to_numeric(raw_df[col], errors="coerce")

        if col_clean in channel_maps:
            spec = channel_maps[col_clean]
            target_ch = spec.get("target_channel", col_clean.lower())
            scale = float(spec.get("scale", 1.0))
            offset = float(spec.get("offset", 0.0))
            out_df[target_ch] = numeric_series * scale + offset
        else:
            # Keep unmapped column with lowercase name
            out_df[col_clean.lower()] = numeric_series

    metadata = {
        "well_name": well_name,
        "wellbore_name": wellbore_name,
        "index_type": index_type,
        "is_time_indexed": is_time_indexed,
        "mnemonics": mnemonics,
        "units": units,
        "total_rows": len(out_df),
    }

    return out_df, metadata


def parse_daily_drilling_report_xml(xml_path: Union[str, Path]) -> pd.DataFrame:
    """Parse a Daily Drilling Report (DDR) XML file into a structured table.
    
    Extracts:
    - well_id / well_name
    - report_date
    - activity_code / activity
    - remarks / description
    - depth (from/to) if recorded

    Returns:
        DataFrame with columns ['well_id', 'date', 'activity', 'remarks', 'depth_from', 'depth_to']
    """
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"DDR XML file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    well_name = "VOLVE-WELL"
    report_date = ""

    # Search for header info
    for elem in root.iter():
        tag = _strip_ns(elem.tag).lower()
        if tag in ("namewell", "wellname") and elem.text:
            well_name = elem.text.strip()
        elif tag in ("datereport", "reportdate", "dstart") and elem.text:
            report_date = elem.text.strip()

    records = []
    # Search for activity intervals (WITSML activity / opsReport structure)
    for elem in root.iter():
        tag = _strip_ns(elem.tag).lower()
        if tag in ("activity", "operation", "interval", "entry"):
            act_text = ""
            remarks_text = ""
            depth_from = np.nan
            depth_to = np.nan
            act_date = report_date

            for child in elem:
                c_tag = _strip_ns(child.tag).lower()
                c_val = child.text.strip() if child.text else ""
                if c_tag in ("activitycode", "type", "operation", "activity") and c_val:
                    act_text = c_val
                elif c_tag == "phase" and c_val and not act_text:
                    act_text = c_val
                elif c_tag in ("comments", "remark", "remarks", "description", "text") and c_val:
                    remarks_text = c_val
                elif c_tag in ("md", "depthfrom", "fromdepth", "mdstart") and c_val:
                    try:
                        depth_from = float(c_val)
                    except ValueError:
                        pass
                elif c_tag in ("mdstop", "depthto", "todepth", "mdend") and c_val:
                    try:
                        depth_to = float(c_val)
                    except ValueError:
                        pass
                elif c_tag in ("datetime", "datestart", "date") and c_val:
                    act_date = c_val

            if act_text or remarks_text:
                records.append({
                    "well_id": well_name,
                    "date": act_date,
                    "activity": act_text or "General Operations",
                    "remarks": remarks_text or "-",
                    "depth_from": depth_from,
                    "depth_to": depth_to,
                })

    if not records:
        # Fallback to general report remarks if no distinct activity rows exist
        remarks_all = []
        for elem in root.iter():
            tag = _strip_ns(elem.tag).lower()
            if tag in ("comments", "summary", "reportsummary") and elem.text:
                remarks_all.append(elem.text.strip())

        records.append({
            "well_id": well_name,
            "date": report_date or "2026-01-01",
            "activity": "Summary Report",
            "remarks": " ".join(remarks_all) if remarks_all else "No operations logged",
            "depth_from": np.nan,
            "depth_to": np.nan,
        })

    return pd.DataFrame(records)


class VolveFolderSource(BaseTelemetrySource):
    """Source adapter for a local folder containing Volve WITSML XML files.
    
    Strictly READ-ONLY.
    """

    def __init__(
        self,
        folder_path: Union[str, Path],
        mapping_yaml_path: Optional[Union[str, Path]] = None,
    ):
        self.folder_path = Path(folder_path)
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Volve directory not found: {self.folder_path}")

        self.mapping_config = {}
        if mapping_yaml_path:
            with open(mapping_yaml_path, "r", encoding="utf-8") as f:
                self.mapping_config = yaml.safe_load(f)

        self._df, self.metadata = self._load_folder()

    def _load_folder(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        xml_files = sorted(list(self.folder_path.glob("*.xml")) + list(self.folder_path.glob("*.XML")))
        frames = []
        metas = []

        for p in xml_files:
            try:
                df, meta = parse_witsml_log_xml(p, self.mapping_config)
                if not df.empty:
                    meta["file_name"] = p.name
                    frames.append(df)
                    metas.append(meta)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame(), metas

        combined = pd.concat(frames, ignore_index=True)
        if "timestamp" in combined.columns:
            combined = combined.sort_values(["well_id", "timestamp"]).reset_index(drop=True)
        return combined, metas

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

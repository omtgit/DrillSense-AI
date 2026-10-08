"""Deterministic engineering report built from a structured finding. No API, no network.

build_finding() turns the model detections for one well into a plain dict; render_report()
turns that dict into markdown. The optional Gemini step (gemini_utils.py) may only rephrase
the rendered text, and numbers_are_grounded() rejects output that introduces new numbers.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from profiles import get_profile  # noqa: E402

DEFAULT_PROFILE = "production"


def _mad_sigma(s):
    return 1.4826 * (s - s.median()).abs().median()


def top_contributors(well_df, event_mask, k=3, profile=None):
    """Channels whose mean during the event sits furthest from the well's own normal level.

    This is a robust z-score (median and MAD of the well's rows not flagged by the model).
    It describes which signals moved; it is not model attribution (SHAP comes later).
    """
    labels = (profile or get_profile(DEFAULT_PROFILE)).channel_labels
    base = well_df[well_df["predicted_anomaly"].eq("Normal")]
    event = well_df[event_mask]
    out = []
    if base.empty or event.empty:
        return out
    skip = set((profile or get_profile(DEFAULT_PROFILE)).trend_channels)
    for col in labels:
        if col not in well_df.columns or col in skip:
            continue
        sigma = _mad_sigma(base[col])
        if not sigma or pd.isna(sigma):
            continue
        baseline = float(base[col].median())
        observed = float(event[col].mean())
        z = (observed - baseline) / sigma
        out.append({"channel": col, "label": labels[col], "baseline": baseline,
                    "observed": observed, "z": float(z)})
    out.sort(key=lambda c: (-abs(c["z"]), c["channel"]))
    return out[:k]


GROUND_TRUTH_SOURCE = "ground-truth lookup in this dataset"
PROFILE_RULE_SOURCE = "profile rule for the predicted class"


def build_finding(df, row, k=3, profile=None, rules_from_profile=False):
    """Structured facts for the event containing `row` (a selected well's top-risk row).

    By default severity and response are read from the row (production data carries them as a
    ground-truth lookup). With `rules_from_profile` they come from the profile's severity rules
    for the PREDICTED class, so a report never leans on ground-truth labels.
    """
    prof = profile or get_profile(DEFAULT_PROFILE)
    severity, response, source = str(row["severity"]), str(row["recommended_response"]), GROUND_TRUTH_SOURCE
    if rules_from_profile:
        rule = prof.severity_rules.get(str(row["predicted_anomaly"]))
        severity = rule.severity if rule else "Unrated"
        response = rule.response if rule else "No rule defined for this class"
        source = PROFILE_RULE_SOURCE
    well = df[df["well_id"] == row["well_id"]].sort_values("timestamp").reset_index(drop=True)
    flagged = well["predicted_anomaly"].ne("Normal")
    run_id = (flagged != flagged.shift()).cumsum()
    at = well.index[well["timestamp"] == row["timestamp"]]
    finding = {
        "well_id": str(row["well_id"]),
        "predicted_anomaly": str(row["predicted_anomaly"]),
        "risk_score": float(row["predicted_risk_score"]),
        "severity": severity,
        "recommended_response": response,
        "rule_source": source,
        "detected": False,
        "contributors": [],
    }
    if len(at) and flagged.iloc[at[0]]:
        mask = run_id == run_id.iloc[at[0]]
        ev = well[mask]
        finding.update(
            detected=True,
            n_rows=int(mask.sum()),
            start=str(ev["timestamp"].iloc[0]),
            end=str(ev["timestamp"].iloc[-1]),
            contributors=top_contributors(well, mask, k, prof),
        )
    return finding


def _fmt(x):
    return f"{x:,.2f}"


def render_report(f, profile=None):
    p = profile or get_profile(DEFAULT_PROFILE)
    cause, risk = p.vocab.get(f["predicted_anomaly"], p.default_vocab)
    if not f["detected"]:
        return (
            "## Operational Summary\n"
            f"Well {f['well_id']}: the model flagged no anomaly at the selected time "
            f"(predicted class: {f['predicted_anomaly']}).\n\n"
            "## Immediate Actions\nNo action required from this tool's output."
        )
    src = f.get("rule_source", GROUND_TRUTH_SOURCE)
    lines = [
        "## Operational Summary",
        f"Well {f['well_id']} shows a model-detected **{f['predicted_anomaly']}** event from "
        f"{f['start']} to {f['end']} ({f['n_rows']} readings). Predicted risk score: "
        f"{f['risk_score']:g}. Severity ({src}): {f['severity']}.",
        "",
        "## Key Signals",
    ]
    for c in f["contributors"]:
        direction = "above" if c["z"] > 0 else "below"
        lines.append(
            f"- {c['label']}: mean {_fmt(c['observed'])} during the event versus a normal level "
            f"of {_fmt(c['baseline'])} for this well ({abs(c['z']):.1f} robust standard "
            f"deviations {direction} normal)."
        )
    if not f["contributors"]:
        lines.append("- No channel stood out against this well's normal level.")
    lines += [
        "",
        "## Possible Root Cause",
        f"The pattern is consistent with {cause}. This is a hypothesis from signal "
        "movement only and is not confirmed.",
        "",
        "## Operational Risk",
        risk,
        "",
        "## Immediate Actions",
        f"Recommended response ({src}): {f['recommended_response']}. "
        "Verify against field data before acting."
        + (f" {p.review_note}" if p.review_note else ""),
        "",
        "## Longer-Term Recommendation",
        "Review the sensor history for this well, confirm the cause on site, and record the "
        "outcome so thresholds can be tuned.",
    ]
    return "\n".join(lines)


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text):
    return {m.rstrip(",").replace(",", "") for m in _NUM.findall(text)}


def numbers_are_grounded(candidate, source):
    """True when every number in `candidate` already appears in `source`."""
    return _numbers(candidate) <= _numbers(source)

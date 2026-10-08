"""Profile = everything domain-specific. The engine (loader, features, detectors, report) only
reads these objects, so a new domain is a new profile module and nothing else.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class Channel:
    unit: str
    report_label: str  # wording used in generated reports


@dataclass(frozen=True)
class SeverityRule:
    severity: str
    priority: int
    response: str


@dataclass(frozen=True)
class PhysicsFeature:
    """A derived signal. `fn` gets ONE well's rows sorted by time and must be past-only
    (diff, trailing rolling window), so it never looks ahead."""
    name: str
    description: str
    fn: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    channels: Dict[str, Channel]
    anomaly_types: Tuple[str, ...]
    physics_features: Tuple[PhysicsFeature, ...]
    severity_rules: Dict[str, SeverityRule]
    # predicted class -> (possible cause wording, operational risk wording); hypotheses only
    vocab: Dict[str, Tuple[str, str]]
    default_vocab: Tuple[str, str] = (
        "an operational change that this tool cannot attribute to a specific cause",
        "Unclear; treat as unverified until checked by an engineer.",
    )

    # channels that move with normal progress (e.g. depth); left out of "which signals moved"
    trend_channels: Tuple[str, ...] = ()
    # caveat printed in reports; empty when nothing needs flagging
    review_note: str = ""

    @property
    def channel_labels(self):
        return {k: c.report_label for k, c in self.channels.items()}

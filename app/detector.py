"""Row-level anomaly detector for the drilling profile, trained at app start-up.

Why trained at start-up and not a committed model file: a pickled booster is a binary blob
that is tied to one xgboost version and cannot be reviewed in a diff, while this training run
takes a few seconds on seeded generated data and is deterministic. The app caches it
(st.cache_resource), so it runs once per server process.

Training and validation wells are GENERATED here with seeds different from the committed sample
(scripts/generate_drilling_data.py --sample uses seed 11), so the sample wells the dashboard
shows are never trained on and the validation wells are never trained on either.

Model: XGBoost, 7 classes (Normal + the profile's anomaly types) on raw channels + physics features
+ past-only rolling context, the "raw+physics+context" set of scripts/evaluate.py. A row is flagged
when P(Normal) < 0.5; its class is the most likely non-Normal class and its risk score is
100 * (1 - P(Normal)). This is a simple fixed rule: no threshold is tuned.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import add_context, add_profile_features  # noqa: E402
from generators.drilling import generate_dataset  # noqa: E402

TRAIN_SEED = 1001
VALID_SEED = 2002
TRAIN_WELLS = 8
VALID_WELLS = 4
DAYS = 3.0
EVENTS_PER_DAY = 1.5
XGB_ROUNDS = 150
XGB_PARAMS = dict(eta=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  tree_method="hist", nthread=4, verbosity=0, objective="multi:softprob")
FLAG_BELOW_P_NORMAL = 0.5
GRACE_MIN = 15  # an alarm up to this many minutes after an event ends still counts for it


def raw_channels(profile):
    return [c for c in profile.channels if c not in profile.trend_channels]


def prepare(df, profile):
    """Sort per well, add the profile's physics features and the rolling context."""
    df = df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)
    df = add_profile_features(df, profile)
    return add_context(df, raw_channels(profile) + [f.name for f in profile.physics_features])


@dataclass
class Detector:
    profile: object
    booster: xgb.Booster
    columns: list
    classes: list
    info: dict = field(default_factory=dict)

    def _proba(self, frame):
        X = frame[self.columns].values.astype(np.float32)
        return self.booster.predict(xgb.DMatrix(X))

    def predict(self, prepared):
        """Return a copy of `prepared` with predicted_anomaly, predicted_risk_score,
        predicted_severity and predicted_response (rules from the profile, by predicted class)."""
        proba = self._proba(prepared)
        p_normal = proba[:, 0]
        best = proba[:, 1:].argmax(axis=1) + 1
        flagged = p_normal < FLAG_BELOW_P_NORMAL
        names = np.array(self.classes, dtype=object)
        out = prepared.copy()
        out["predicted_anomaly"] = np.where(flagged, names[best], "Normal")
        out["predicted_risk_score"] = np.round(100.0 * (1.0 - p_normal), 1)
        rules = self.profile.severity_rules
        out["predicted_severity"] = out["predicted_anomaly"].map(lambda c: rules[c].severity if c in rules else "None")
        out["predicted_response"] = out["predicted_anomaly"].map(lambda c: rules[c].response if c in rules else "No action")
        return out


def _feature_columns(prepared, profile):
    return raw_channels(profile) + [f.name for f in profile.physics_features] + [c for c in prepared.columns if "__" in c]


def event_recall(scored, events):
    """Share of events with at least one flagged row inside [start, end + GRACE_MIN)."""
    hit = 0
    for ev in events.itertuples():
        g = scored[scored["well_id"] == ev.well_id].reset_index(drop=True)
        window = g.iloc[ev.start: ev.end + GRACE_MIN]
        hit += int((window["predicted_anomaly"] != "Normal").any())
    return hit / len(events) if len(events) else float("nan")


def score_rows(scored):
    """Row-level precision/recall/F1 of 'flagged' against the injected anomaly_flag."""
    pred = (scored["predicted_anomaly"] != "Normal").values
    true = scored["anomaly_flag"].astype(bool).values
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return precision, recall, f1


def train_detector(profile, train_wells=TRAIN_WELLS, valid_wells=VALID_WELLS, days=DAYS,
                   train_seed=TRAIN_SEED, valid_seed=VALID_SEED, rounds=XGB_ROUNDS):
    classes = ["Normal"] + list(profile.anomaly_types)
    tr_raw, _ = generate_dataset(train_wells, days, train_seed, EVENTS_PER_DAY)
    tr = prepare(tr_raw, profile)
    cols = _feature_columns(tr, profile)
    y = tr["anomaly_type"].map({c: i for i, c in enumerate(classes)}).values
    params = dict(XGB_PARAMS, num_class=len(classes), seed=0)
    booster = xgb.train(params, xgb.DMatrix(tr[cols].values.astype(np.float32), label=y), rounds)
    det = Detector(profile, booster, cols, classes)

    va_raw, va_events = generate_dataset(valid_wells, days, valid_seed, EVENTS_PER_DAY)
    va = det.predict(prepare(va_raw, profile))
    p, r, f1 = score_rows(va)
    det.info = {
        "train_wells": train_wells, "train_rows": int(len(tr)), "train_seed": train_seed,
        "valid_wells": valid_wells, "valid_rows": int(len(va)), "valid_seed": valid_seed,
        "valid_anomaly_rows": int(va["anomaly_flag"].sum()), "valid_events": int(len(va_events)),
        "days": days, "precision": p, "recall": r, "f1": f1,
        "event_recall": event_recall(va, va_events),
    }
    return det

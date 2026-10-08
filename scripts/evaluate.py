"""Evaluate anomaly detectors on the seeded synthetic drilling data.

    python scripts/evaluate.py                       # full run, writes docs/eval_results.json
    python scripts/evaluate.py --quick               # small smoke run (does not touch docs/)
    python scripts/evaluate.py --render              # rebuild the tables in docs/EVAL.md from the JSON

What is compared (all scored per row, then turned into alarms at a threshold):
  zscore   naive per-well robust z-score (max |z| over the raw channels)
  rule     physics rule: flow-out minus flow-in and pit-volume rate must agree in sign (kick/loss rule)
  iforest  Isolation Forest (unsupervised, trained on the training wells)
  xgb_bin  XGBoost, normal vs anomalous        xgb_multi  XGBoost, 7 classes (score = 1 - P(Normal))
Feature ablation for the learned models: raw channels / + physics features / + rolling context.

Protocol
  * "wells":  5-fold cross-validation grouped by WELL (no well is in both train and test).
  * "time":   train on the first 70% of every well, test on the last 30%.
  * Thresholds come from the TRAINING part only: the lowest threshold whose false alarms per well-day
    on training scores stay within a budget. For learned models the training scores are out-of-fold
    (inner cross-validation over the training wells), so they are not inflated by overfitting.
  * No hyper-parameters are tuned; XGBoost / Isolation Forest settings are fixed constants below.
  * Everything is repeated over several seeds (each seed = a new set of wells); mean and std reported.

Event-level metrics: an event is DETECTED if an alarm row falls inside [start, end + GRACE). Detection
delay = first such alarm row minus event start (minutes). An alarm episode (alarm rows merged when
<= MERGE_GAP minutes apart) that overlaps no event window is a FALSE ALARM.

These are results on synthetic data. See docs/EVAL.md for what they do and do not say.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.features import add_profile_features  # noqa: E402
from generators.drilling import generate_dataset  # noqa: E402
from profiles import get_profile  # noqa: E402

PROFILE = get_profile("drilling")
RESULTS_PATH = ROOT / "docs" / "eval_results.json"
EVAL_MD = ROOT / "docs" / "EVAL.md"

MERGE_GAP = 10          # minutes; alarm rows closer than this form one episode
GRACE = 15              # minutes after an event ends in which an alarm still counts as that event
BUDGETS = (1.0, 0.25)   # false alarms per well-day allowed on the training scores
PRIMARY_BUDGET = 1.0
TIME_CUT = 0.7
OUTER_FOLDS = 5
INNER_FOLDS = 3
STRONG = 0.1            # rows of an event with effective strength below this are "barely visible"
SCALE_BINS = ((0.0, 0.3, "small (<0.3)"), (0.3, 0.6, "medium (0.3-0.6)"), (0.6, 1.01, "large (>=0.6)"))
CLASSES = ["Normal"] + list(PROFILE.anomaly_types)
XGB_ROUNDS = 150
XGB_PARAMS = dict(eta=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  tree_method="hist", nthread=4, verbosity=0)
IF_PARAMS = dict(n_estimators=200, max_samples=1024, n_jobs=4)

RAW = [c for c in PROFILE.channels if c != "depth_m"]   # absolute depth only trends with time
PHYS = [f.name for f in PROFILE.physics_features]
CTX_BASE = RAW + PHYS

# (kind, feature set); the first five are the headline table, all learned ones form the ablation.
SPECS = (
    [("zscore", "raw"), ("rule", "physics"), ("iforest", "physics_only")]
    + [(k, fs) for k in ("iforest", "xgb_bin", "xgb_multi") for fs in ("raw", "raw+physics", "raw+physics+context")]
)
HEADLINE = ["zscore|raw", "rule|physics", "iforest|physics_only",
            "xgb_bin|raw+physics+context", "xgb_multi|raw+physics+context"]


def spec_name(spec):
    return f"{spec[0]}|{spec[1]}"


# ---------------------------------------------------------------- data and features

def add_context(df):
    """Past-only rolling context per well: 15-min mean/std, deviation from a 180-min trailing mean, 5-min diff."""
    parts = []
    for _, g in df.groupby("well_id", sort=False):
        cols = {}
        for c in CTX_BASE:
            s = g[c]
            cols[f"{c}__mean15"] = s.rolling(15, min_periods=5).mean()
            cols[f"{c}__std15"] = s.rolling(15, min_periods=5).std()
            cols[f"{c}__dev180"] = s - s.rolling(180, min_periods=30).mean()
            cols[f"{c}__d5"] = s.diff(5)
        parts.append(pd.DataFrame(cols, index=g.index))
    return pd.concat([df, pd.concat(parts)], axis=1)


def feature_sets(df):
    ctx = [c for c in df.columns if "__" in c]
    return {"raw": RAW, "raw+physics": RAW + PHYS, "raw+physics+context": RAW + PHYS + ctx, "physics_only": PHYS}


def build_dataset(seed, n_wells, days, events_per_day):
    df, ev = generate_dataset(n_wells=n_wells, days=days, seed=seed, events_per_day=events_per_day)
    df = add_profile_features(df, PROFILE)
    df["t"] = df.groupby("well_id").cumcount()
    df = add_context(df).reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    ev = ev[["well_id", "event_id", "type", "start", "end", "scale"]].reset_index(drop=True)
    return df, ev


# ---------------------------------------------------------------- scorers

def _robust_scale(x):
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    return med, max(mad, 1e-9)


def score_zscore(ap, ref):
    """max over raw channels of |x - median| / MAD, using each well's own reference rows (no labels)."""
    med = ref.groupby("well_id")[RAW].median()
    mad = (ref[RAW] - med.reindex(ref["well_id"]).values).abs().groupby(ref["well_id"].values).median() * 1.4826
    m = med.reindex(ap["well_id"]).values
    s = np.maximum(mad.reindex(ap["well_id"]).values, 1e-3 * np.abs(m) + 1e-6)
    return np.nanmax(np.abs((ap[RAW].values - m) / s), axis=1)


def score_rule(ap, tr):
    """Kick / loss rule: flow delta and pit rate (robust z from training rows) must have the same sign."""
    mf, sf = _robust_scale(tr["flow_delta_lpm"].values)
    mp, sp = _robust_scale(tr["pit_rate_m3ph"].values)
    zf = (ap["flow_delta_lpm"].values - mf) / sf
    zp = (ap["pit_rate_m3ph"].values - mp) / sp
    s = np.where(np.sign(zf) == np.sign(zp), np.minimum(np.abs(zf), np.abs(zp)), 0.0)
    return np.nan_to_num(s)


def _xgb_fit(X, y, num_class, seed):
    p = dict(XGB_PARAMS, seed=seed)
    if num_class:
        p.update(objective="multi:softprob", num_class=num_class)
    else:
        p.update(objective="binary:logistic")
    return xgb.train(p, xgb.DMatrix(X.astype(np.float32), label=y), XGB_ROUNDS)


def score_spec(spec, cols_by_set, tr, ap, ref, seed):
    """Return (score, proba or None, booster or None) for rows `ap`, fitted on rows `tr`."""
    kind, fs = spec
    if kind == "zscore":
        return score_zscore(ap, ref), None, None
    if kind == "rule":
        return score_rule(ap, tr), None, None
    cols = cols_by_set[fs]
    if kind == "iforest":
        med = tr[cols].median()
        model = IsolationForest(random_state=seed, **IF_PARAMS).fit(tr[cols].fillna(med).values)
        return -model.score_samples(ap[cols].fillna(med).values), None, None
    if kind == "xgb_bin":
        bst = _xgb_fit(tr[cols].values, tr["anomaly_flag"].values, 0, seed)
        return bst.predict(xgb.DMatrix(ap[cols].values.astype(np.float32))), None, bst
    y = tr["anomaly_type"].map({c: i for i, c in enumerate(CLASSES)}).values
    bst = _xgb_fit(tr[cols].values, y, len(CLASSES), seed)
    proba = bst.predict(xgb.DMatrix(ap[cols].values.astype(np.float32)))
    return 1.0 - proba[:, 0], proba, bst


def train_scores(spec, cols_by_set, tr, ref, seed):
    """Scores on the training rows used to set thresholds: out-of-fold for learned models."""
    if spec[0] in ("zscore", "rule"):
        return score_spec(spec, cols_by_set, tr, tr, ref, seed)[0]
    out = np.zeros(len(tr))
    for a, b in GroupKFold(min(INNER_FOLDS, tr["well_id"].nunique())).split(tr, groups=tr["well_id"]):
        out[b] = score_spec(spec, cols_by_set, tr.iloc[a], tr.iloc[b], None, seed)[0]
    return out


# ---------------------------------------------------------------- event-level machinery

def alarm_eval(wells, t, alarm, events):
    """Event detection and false alarms for one set of rows.

    wells/t/alarm: per-row arrays (rows of a well in increasing t). events: DataFrame with well_id,
    start, end. Returns dict(fa, detected (bool per event, order of `events`), delay (minutes or nan)).
    """
    ev_start = events["start"].values
    ev_end = events["end"].values + GRACE
    ev_pos = pd.Series(np.arange(len(events))).groupby(events["well_id"].values).indices
    detected = np.zeros(len(events), bool)
    delay = np.full(len(events), np.nan)
    fa = 0
    for w, pos in pd.Series(np.arange(len(wells))).groupby(wells).indices.items():
        a = t[pos][alarm[pos]]
        ev = ev_pos.get(w, np.array([], dtype=int))
        if len(a) == 0:
            continue
        brk = np.where(np.diff(a) > MERGE_GAP)[0]
        first, last = a[np.r_[0, brk + 1]], a[np.r_[brk, len(a) - 1]]
        if len(ev):
            hit = ((first[:, None] < ev_end[ev][None, :]) & (last[:, None] >= ev_start[ev][None, :])).any(axis=1)
            fa += int((~hit).sum())
            i = np.searchsorted(a, ev_start[ev])
            ok = i < len(a)
            ok[ok] = a[i[ok]] < ev_end[ev][ok]
            detected[ev] = ok
            delay[ev[ok]] = a[i[ok]] - ev_start[ev][ok]
        else:
            fa += len(first)
    return dict(fa=fa, detected=detected, delay=delay)


def fa_per_day(wells, t, alarm, events):
    return alarm_eval(wells, t, alarm, events)["fa"] / (len(wells) / 1440.0)


def pick_thresholds(scores, wells, t, events, budgets=BUDGETS):
    """Lowest threshold whose false alarms per well-day on the (training) scores fit each budget."""
    qs = 1 - np.logspace(-1, -4.3, 60)
    cands = np.unique(np.quantile(scores, qs))
    curve = [fa_per_day(wells, t, scores >= c, events) for c in cands]
    out = {}
    for b in budgets:
        ok = [c for c, f in zip(cands, curve) if f <= b]
        out[b] = float(ok[0]) if ok else float(np.nextafter(scores.max(), np.inf))
    return out


# ---------------------------------------------------------------- one protocol run

def run_protocol(df, events, mode, seed, specs, cols_by_set, shap_acc):
    """Out-of-sample scores and alarms for every spec. Returns dict spec_name -> arrays over df rows."""
    n = len(df)
    cut = int(df["t"].max() + 1) * TIME_CUT if mode == "time" else None
    if mode == "wells":
        splits = [(a, b) for a, b in GroupKFold(min(OUTER_FOLDS, df["well_id"].nunique())).split(df, groups=df["well_id"])]
    else:
        tm = df["t"].values
        splits = [(np.where(tm < cut)[0], np.where(tm >= cut)[0])]
    res = {spec_name(s): dict(score=np.full(n, np.nan), alarm={b: np.zeros(n, bool) for b in BUDGETS},
                              proba=np.full((n, len(CLASSES)), np.nan) if s[0] == "xgb_multi" else None)
           for s in specs}
    rng = np.random.default_rng(seed)
    for tr_i, te_i in splits:
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        tr_ev = events[events["well_id"].isin(tr["well_id"].unique())]
        if mode == "time":
            tr_ev = tr_ev[tr_ev["start"] < cut]
        ref_te = tr if mode == "time" else te
        for spec in specs:
            name = spec_name(spec)
            s_tr = train_scores(spec, cols_by_set, tr, tr, seed)
            thr = pick_thresholds(s_tr, tr["well_id"].values, tr["t"].values, tr_ev)
            s_te, proba, bst = score_spec(spec, cols_by_set, tr, te, ref_te, seed)
            r = res[name]
            r["score"][te_i] = s_te
            for b in BUDGETS:
                r["alarm"][b][te_i] = s_te >= thr[b]
            if proba is not None:
                r["proba"][te_i] = proba
            if mode == "wells" and bst is not None and spec[1] == "raw+physics+context":
                _collect_shap(shap_acc, spec[0], bst, cols_by_set[spec[1]], te, rng)
    return res


def _collect_shap(acc, kind, bst, cols, te, rng, cap=1500):
    """Mean |TreeSHAP| per feature for the rows of each true anomaly type (XGBoost pred_contribs)."""
    F = len(cols)
    if kind == "xgb_bin":
        pos = np.where(te["anomaly_flag"].values == 1)[0]
        if len(pos) < 30:
            return
        pos = rng.choice(pos, min(cap * 2, len(pos)), replace=False)
        c = bst.predict(xgb.DMatrix(te[cols].values[pos].astype(np.float32)), pred_contribs=True)
        acc.setdefault("binary", {}).setdefault("All anomalies", []).append(np.abs(c[:, :F]).mean(axis=0))
        return
    for k, typ in enumerate(CLASSES):
        if k == 0:
            continue
        rows = np.where(te["anomaly_type"].values == typ)[0]
        if len(rows) < 30:
            continue
        rows = rng.choice(rows, min(cap, len(rows)), replace=False)
        c = bst.predict(xgb.DMatrix(te[cols].values[rows].astype(np.float32)), pred_contribs=True)
        acc.setdefault("multiclass", {}).setdefault(typ, []).append(np.abs(c[:, k, :F]).mean(axis=0))


# ---------------------------------------------------------------- metrics

def _bin_label(x):
    for lo, hi, name in SCALE_BINS:
        if lo <= x < hi:
            return name
    return SCALE_BINS[-1][2]


def evaluate_pooled(df, events, mode, r, specname):
    """Row-level and event-level metrics from pooled out-of-sample scores/alarms."""
    if mode == "time":
        cut = int(df["t"].max() + 1) * TIME_CUT
        valid = (df["t"].values >= cut)
        straddle = events[(events["start"] < cut) & (events["end"] + GRACE > cut)]
        for _, e in straddle.iterrows():           # rows still inside an event that began in training
            valid &= ~((df["well_id"].values == e["well_id"]) & (df["t"].values < e["end"] + GRACE))
        ev = events[events["start"] >= cut]
    else:
        valid, ev = np.ones(len(df), bool), events
    ev = ev.reset_index(drop=True)
    score = np.where(np.isnan(r["score"]), np.nanmin(r["score"]), r["score"])
    y = df["anomaly_flag"].values
    sv, yv = score[valid], y[valid]
    strong = ~((y == 1) & (df["event_scale"].values < STRONG))
    sm = valid & strong
    out = dict(
        row=dict(prevalence=float(yv.mean()), pr_auc=float(average_precision_score(yv, sv)),
                 auroc=float(roc_auc_score(yv, sv)),
                 pr_auc_strong=float(average_precision_score(y[sm], score[sm]))),
        by_budget={}, by_type={}, by_scale={},
    )
    typ = df["anomaly_type"].values
    for t_name in PROFILE.anomaly_types:
        m = valid & ((typ == t_name) | (typ == "Normal"))
        a = float(roc_auc_score(typ[m] == t_name, score[m])) if (typ[m] == t_name).any() else None
        out["by_type"][t_name] = dict(auroc_vs_normal=a, budgets={})
    for _, _, name in SCALE_BINS:
        out["by_scale"][name] = dict(budgets={})
    ev_type = ev["type"].values
    ev_bin = np.array([_bin_label(x) for x in ev["scale"].values])
    days = valid.sum() / 1440.0
    for b in BUDGETS:
        alarm = r["alarm"][b] & valid
        res = alarm_eval(df["well_id"].values, df["t"].values, alarm, ev)
        det, dl = res["detected"], res["delay"]
        entry = dict(n_events=int(len(ev)), detected=int(det.sum()),
                     detect_share=float(det.mean()) if len(ev) else None,
                     fa=int(res["fa"]), days=float(days), fa_per_well_day=float(res["fa"] / days),
                     median_delay_min=float(np.nanmedian(dl)) if det.any() else None)
        if r["proba"] is not None and det.any():
            entry["type_match_share"] = _type_match(df, ev, det, alarm, r["proba"])
        out["by_budget"][str(b)] = entry
        for key, labels, names in (("by_type", ev_type, list(PROFILE.anomaly_types)),
                                   ("by_scale", ev_bin, [s[2] for s in SCALE_BINS])):
            for nm in names:
                m = labels == nm
                out[key][nm]["budgets"][str(b)] = dict(
                    n=int(m.sum()), detected=int(det[m].sum()),
                    delays=[float(x) for x in dl[m & det]])
    return out


def _type_match(df, ev, det, alarm, proba):
    """Among detected events: share where the class with the most probability mass on alarm rows is the event type."""
    wells, t = df["well_id"].values, df["t"].values
    pos_by_w = pd.Series(np.arange(len(df))).groupby(wells).indices
    hits = tot = 0
    for j in np.where(det)[0]:
        e = ev.iloc[j]
        p = pos_by_w[e["well_id"]]
        m = alarm[p] & (t[p] >= e["start"]) & (t[p] < e["end"] + GRACE)
        if not m.any():
            continue
        mass = np.nansum(proba[p][m][:, 1:], axis=0)
        tot += 1
        hits += int(PROFILE.anomaly_types[int(mass.argmax())] == e["type"])
    return hits / tot if tot else None


# ---------------------------------------------------------------- aggregation across seeds

def _ms(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return dict(mean=None, std=None, n=0)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)))


def summarise(per_seed):
    """per_seed: {seed: {mode: {spec: metrics}}} -> mean/std over seeds, breakdowns pooled over seeds."""
    seeds = list(per_seed)
    out = {}
    modes = per_seed[seeds[0]].keys()
    for mode in modes:
        out[mode] = {}
        for name in per_seed[seeds[0]][mode]:
            runs = [per_seed[s][mode][name] for s in seeds]
            e = dict(row={k: _ms([r["row"][k] for r in runs]) for k in runs[0]["row"]}, by_budget={},
                     by_type={}, by_scale={})
            for b in map(str, BUDGETS):
                e["by_budget"][b] = {k: _ms([r["by_budget"][b].get(k) for r in runs])
                                     for k in ("detect_share", "fa_per_well_day", "median_delay_min", "type_match_share")}
                e["by_budget"][b]["n_events"] = int(sum(r["by_budget"][b]["n_events"] for r in runs))
            for key in ("by_type", "by_scale"):
                for nm in runs[0][key]:
                    ent = dict(auroc_vs_normal=_ms([r[key][nm].get("auroc_vs_normal") for r in runs])["mean"]
                               if key == "by_type" else None, budgets={})
                    for b in map(str, BUDGETS):
                        n = sum(r[key][nm]["budgets"][b]["n"] for r in runs)
                        d = sum(r[key][nm]["budgets"][b]["detected"] for r in runs)
                        delays = [x for r in runs for x in r[key][nm]["budgets"][b]["delays"]]
                        ent["budgets"][b] = dict(n=int(n), detected=int(d),
                                                 detect_share=d / n if n else None,
                                                 median_delay_min=float(np.median(delays)) if delays else None)
                    e[key][nm] = ent
            out[mode][name] = e
    return out


def summarise_shap(acc, cols):
    fam = lambda c: "raw" if c in RAW else ("physics" if c in PHYS else "context")  # noqa: E731
    out = {}
    for kind, d in acc.items():
        out[kind] = {}
        for typ, vecs in d.items():
            v = np.mean(vecs, axis=0)
            tot = v.sum() or 1.0
            top = np.argsort(-v)[:10]
            out[kind][typ] = dict(
                top=[dict(feature=cols[i], mean_abs_shap=float(v[i]), share=float(v[i] / tot)) for i in top],
                family_share={f: float(sum(v[i] for i, c in enumerate(cols) if fam(c) == f) / tot)
                              for f in ("raw", "physics", "context")},
                n_folds=len(vecs))
    return out


# ---------------------------------------------------------------- docs tables

LABELS = {"zscore|raw": "Per-well robust z-score (raw channels)", "rule|physics": "Physics rule (flow delta + pit rate)",
          "iforest|physics_only": "Isolation Forest (physics features only)",
          "xgb_bin|raw+physics+context": "XGBoost binary", "xgb_multi|raw+physics+context": "XGBoost multiclass"}
MODES = {"wells": "Well-held-out (5-fold CV grouped by well)", "time": "Time-held-out (first 70% train, last 30% test)"}


def _f(m, d=3, pct=False):
    if not isinstance(m, dict) or m.get("mean") is None:
        return "n/a"
    k = 100 if pct else 1
    sfx = "%" if pct else ""
    return f"{m['mean'] * k:.{d}f}{sfx} ± {m['std'] * k:.{d}f}"


def _share(e):
    return "n/a" if e["detect_share"] is None else f"{e['detect_share']:.0%} ({e['detected']}/{e['n']})"


def _table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def tables_markdown(res):
    S, cfg, out = res["summary"], res["config"], []
    pb = str(cfg["primary_budget"])
    head = ["Method", "Row PR-AUC", "Row AUROC", "Events detected", "False alarms / well-day", "Median delay (min)"]
    for mode, title in MODES.items():
        prev = S[mode][HEADLINE[0]]["row"]["prevalence"]["mean"]
        out.append(f"#### {title}, threshold budget {pb} false alarm / well-day on training data\n")
        out.append(f"Share of rows inside an event: {prev:.1%} (this is the PR-AUC of a random scorer).\n")
        rows = []
        for m in HEADLINE:
            e = S[mode][m]
            bb = e["by_budget"][pb]
            rows.append([LABELS[m], _f(e["row"]["pr_auc"]), _f(e["row"]["auroc"]), _f(bb["detect_share"], 0, True),
                         _f(bb["fa_per_well_day"], 2), _f(bb["median_delay_min"], 0)])
        out.append(_table(head, rows) + "\n")
    out.append("#### Operating-point sensitivity (well-held-out): budget 0.25 false alarm / well-day\n")
    b2 = str(BUDGETS[1])
    rows = [[LABELS[m], _f(S["wells"][m]["by_budget"][b2]["detect_share"], 0, True),
             _f(S["wells"][m]["by_budget"][b2]["fa_per_well_day"], 2), _f(S["wells"][m]["by_budget"][b2]["median_delay_min"], 0)]
            for m in HEADLINE]
    out.append(_table(["Method", "Events detected", "False alarms / well-day", "Median delay (min)"], rows) + "\n")
    for key, title, names in (("by_type", "anomaly type", list(PROFILE.anomaly_types)),
                              ("by_scale", "event_scale bin", [x[2] for x in SCALE_BINS])):
        out.append(f"#### Events detected by {title} (well-held-out, budget {pb}; pooled over seeds, n = events)\n")
        rows = []
        for nm in names:
            rows.append([nm] + [_share(S["wells"][m][key][nm]["budgets"][pb]) for m in HEADLINE])
        out.append(_table(["" + title.capitalize()] + [LABELS[m].split(" (")[0] for m in HEADLINE], rows) + "\n")
        rows = []
        for nm in names:
            rows.append([nm] + ["n/a" if S["wells"][m][key][nm]["budgets"][pb]["median_delay_min"] is None
                                else f"{S['wells'][m][key][nm]['budgets'][pb]['median_delay_min']:.0f}" for m in HEADLINE])
        out.append(f"Median detection delay in minutes, same breakdown:\n")
        out.append(_table(["" + title.capitalize()] + [LABELS[m].split(" (")[0] for m in HEADLINE], rows) + "\n")
    out.append("#### Per-type row AUROC (type vs normal rows, well-held-out)\n")
    rows = [[t] + ["n/a" if S["wells"][m]["by_type"][t]["auroc_vs_normal"] is None
                   else f"{S['wells'][m]['by_type'][t]['auroc_vs_normal']:.3f}" for m in HEADLINE]
            for t in PROFILE.anomaly_types]
    out.append(_table(["Type"] + [LABELS[m].split(" (")[0] for m in HEADLINE], rows) + "\n")
    out.append("#### Feature ablation (well-held-out and time-held-out)\n")
    for mode, title in MODES.items():
        out.append(f"{title}, budget {pb}:\n")
        rows = []
        for kind, nm in (("iforest", "Isolation Forest"), ("xgb_bin", "XGBoost binary"), ("xgb_multi", "XGBoost multiclass")):
            for fs in ("raw", "raw+physics", "raw+physics+context"):
                e = S[mode][f"{kind}|{fs}"]
                bb = e["by_budget"][pb]
                rows.append([nm, fs, str(cfg["feature_counts"][fs]), _f(e["row"]["pr_auc"]), _f(e["row"]["auroc"]),
                             _f(bb["detect_share"], 0, True), _f(bb["fa_per_well_day"], 2), _f(bb["median_delay_min"], 0)])
        out.append(_table(["Model", "Features", "# feats", "Row PR-AUC", "Row AUROC", "Events detected",
                           "FA / well-day", "Median delay (min)"], rows) + "\n")
    mm = S["wells"]["xgb_multi|raw+physics+context"]["by_budget"][pb]["type_match_share"]
    out.append(f"XGBoost multiclass type attribution (detected events whose dominant predicted class is the event type, "
               f"well-held-out): {_f(mm, 0, True)}.\n")
    out.append("#### Top XGBoost features per anomaly type (TreeSHAP, mean |contribution|, well-held-out folds)\n")
    rows = []
    for typ, e in res["shap"].get("multiclass", {}).items():
        top = ", ".join(f"`{x['feature']}` ({x['share']:.0%})" for x in e["top"][:5])
        fam = e["family_share"]
        rows.append([typ, top, f"{fam['raw']:.0%} / {fam['physics']:.0%} / {fam['context']:.0%}"])
    out.append(_table(["Type", "Top 5 features (share of |SHAP|)", "raw / physics / context share"], rows) + "\n")
    return "\n".join(out)


def render_docs(res, path):
    start, end = "<!-- TABLES:START -->", "<!-- TABLES:END -->"
    text = Path(path).read_text()
    a, b = text.index(start) + len(start), text.index(end)
    Path(path).write_text(text[:a] + "\n\n" + tables_markdown(res) + "\n" + text[b:])
    print(f"updated tables in {path}")


# ---------------------------------------------------------------- driver

def run(args):
    t0 = time.time()
    per_seed, shap_acc, cols_ref = {}, {}, None
    n_rows = n_events = 0
    for seed in args.seeds:
        ts = time.time()
        df, events = build_dataset(seed, args.wells, args.days, args.events_per_day)
        cols_by_set = feature_sets(df)
        cols_ref = cols_by_set["raw+physics+context"]
        n_rows, n_events = n_rows + len(df), n_events + len(events)
        per_seed[seed] = {}
        for mode in ("wells", "time"):
            res = run_protocol(df, events, mode, seed, SPECS, cols_by_set, shap_acc if mode == "wells" else {})
            per_seed[seed][mode] = {nm: evaluate_pooled(df, events, mode, r, nm) for nm, r in res.items()}
        print(f"seed {seed}: {len(df):,} rows, {len(events)} events, {time.time() - ts:.0f}s", flush=True)
    results = dict(
        config=dict(wells=args.wells, days=args.days, events_per_day=args.events_per_day, seeds=list(args.seeds),
                    rows_total=n_rows, events_total=n_events, budgets=list(BUDGETS), primary_budget=PRIMARY_BUDGET,
                    time_cut=TIME_CUT, outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS, merge_gap_min=MERGE_GAP,
                    grace_min=GRACE, xgb_rounds=XGB_ROUNDS, xgb_params=XGB_PARAMS, iforest_params=IF_PARAMS,
                    feature_counts={k: len(v) for k, v in feature_sets(df).items()},
                    scale_bins=[s[2] for s in SCALE_BINS], headline_methods=HEADLINE,
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                  sklearn=sklearn.__version__, xgboost=xgb.__version__),
                    runtime_s=round(time.time() - t0)),
        summary=summarise(per_seed),
        shap=summarise_shap(shap_acc, cols_ref),
        per_seed={str(s): v for s, v in per_seed.items()},
    )
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wells", type=int, default=12)
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--events-per-day", type=float, default=0.8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    ap.add_argument("--quick", action="store_true", help="small smoke run, writes to the scratch path only if --out is given")
    ap.add_argument("--render", action="store_true", help="only rebuild the tables in docs/EVAL.md from --out")
    a = ap.parse_args()
    if a.render:
        render_docs(json.loads(a.out.read_text()), EVAL_MD)
        return
    if a.quick:
        a.wells, a.days, a.seeds = 6, 1.0, [0]
        a.events_per_day = max(a.events_per_day, 3.0)
    res = run(a)
    if a.quick and a.out == RESULTS_PATH:
        print("--quick: not overwriting docs/eval_results.json (pass --out to save)")
        return
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out} ({res['config']['runtime_s']}s)")


if __name__ == "__main__":
    main()

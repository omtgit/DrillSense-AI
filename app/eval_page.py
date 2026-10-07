"""Model Evaluation page: shows docs/eval_results.json. Nothing is trained or scored here."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px

RESULTS = Path(__file__).resolve().parent.parent / "docs" / "eval_results.json"

LABELS = {
    "zscore|raw": "Per-well robust z-score",
    "rule|physics": "Physics rule (flow delta + pit rate)",
    "iforest|physics_only": "Isolation Forest (physics features)",
    "xgb_bin|raw+physics+context": "XGBoost binary",
    "xgb_multi|raw+physics+context": "XGBoost multiclass",
}
MODES = {"wells": "Well-held-out (5-fold CV grouped by well)",
         "time": "Time-held-out (first 70% train, last 30% test)"}
KIND_NAMES = {"iforest": "Isolation Forest", "xgb_bin": "XGBoost binary", "xgb_multi": "XGBoost multiclass"}


def load_results(path=RESULTS):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _ms(m, digits=3, pct=False):
    if not m or m.get("mean") is None:
        return "n/a"
    k = 100 if pct else 1
    sfx = "%" if pct else ""
    return f"{m['mean'] * k:.{digits}f}{sfx} ± {m['std'] * k:.{digits}f}"


def headline_frame(res, mode, budget):
    rows = []
    for key, label in LABELS.items():
        e = res["summary"][mode][key]
        b = e["by_budget"][budget]
        rows.append({
            "Method": label,
            "Row PR-AUC": _ms(e["row"]["pr_auc"]),
            "Row AUROC": _ms(e["row"]["auroc"]),
            "Events detected": _ms(b["detect_share"], 0, True),
            "False alarms / well-day": _ms(b["fa_per_well_day"], 2),
            "Median delay (min)": _ms(b["median_delay_min"], 0),
        })
    return pd.DataFrame(rows)


def breakdown_frame(res, mode, budget, key):
    names = list(res["summary"][mode][next(iter(LABELS))][key])
    rows = []
    for nm in names:
        row = {"Group": nm}
        for k, label in LABELS.items():
            e = res["summary"][mode][k][key][nm]["budgets"][budget]
            row[label] = "n/a" if e["detect_share"] is None else f"{e['detect_share']:.0%} ({e['detected']}/{e['n']})"
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_frame(res, mode, budget):
    rows = []
    for kind, name in KIND_NAMES.items():
        for fs in ("raw", "raw+physics", "raw+physics+context"):
            e = res["summary"][mode][f"{kind}|{fs}"]
            b = e["by_budget"][budget]
            rows.append({"Model": name, "Features": fs, "# feats": res["config"]["feature_counts"][fs],
                         "Row PR-AUC": _ms(e["row"]["pr_auc"]), "Events detected": _ms(b["detect_share"], 0, True),
                         "False alarms / well-day": _ms(b["fa_per_well_day"], 2)})
    return pd.DataFrame(rows)


def render(st):
    st.header("Model Evaluation")
    res = load_results()
    if res is None:
        st.info("docs/eval_results.json not found. Generate it with `python scripts/evaluate.py` "
                "(see docs/EVAL.md). The app never trains models.")
        return
    cfg = res["config"]
    st.warning(
        "**Synthetic data.** These numbers come from the seeded simulator in `generators/drilling.py`, not from "
        "real rigs. They compare methods against each other on this simulator; they do not predict real-world "
        "performance. See docs/EVAL.md, Limitations."
    )
    st.caption(
        f"{len(cfg['seeds'])} seeds x {cfg['wells']} wells x {cfg['days']:g} days at 1-minute sampling "
        f"({cfg['rows_total']:,} rows, {cfg['events_total']} injected events in total). Mean ± std over seeds. "
        "Thresholds were chosen on training data only; nothing was tuned on test wells."
    )
    budgets = [str(b) for b in cfg["budgets"]]
    c1, c2 = st.columns(2)
    mode = c1.radio("Split", list(MODES), format_func=MODES.get, horizontal=False)
    budget = c2.radio("False-alarm budget used to set thresholds (per well-day, on training data)", budgets,
                      index=budgets.index(str(cfg["primary_budget"])))

    st.subheader("Headline comparison")
    st.dataframe(headline_frame(res, mode, budget), hide_index=True, use_container_width=True)
    st.caption("Row PR-AUC / AUROC are threshold-free. Events detected, false alarms and delay are event-level at the "
               "threshold chosen on training data; delay is the median over detected events only.")

    d = res["summary"][mode]
    chart = pd.DataFrame([{"Method": LABELS[k], "Events detected (%)": d[k]["by_budget"][budget]["detect_share"]["mean"] * 100,
                           "False alarms / well-day": d[k]["by_budget"][budget]["fa_per_well_day"]["mean"]} for k in LABELS])
    fig = px.scatter(chart, x="False alarms / well-day", y="Events detected (%)", color="Method", text="Method")
    fig.update_traces(marker_size=14, textposition="top center")
    fig.update_layout(showlegend=False, yaxis_range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Where does it fail? Detection by anomaly type and event size")
    st.caption("Pooled over seeds; (detected/events). Small events are the hard cases. event_scale is the event's "
               "peak strength, 1.0 = the full effect listed in docs/PHYSICS.md.")
    st.markdown("**By anomaly type**")
    st.dataframe(breakdown_frame(res, mode, budget, "by_type"), hide_index=True, use_container_width=True)
    st.markdown("**By event_scale bin**")
    st.dataframe(breakdown_frame(res, mode, budget, "by_scale"), hide_index=True, use_container_width=True)

    st.subheader("Feature ablation")
    st.dataframe(ablation_frame(res, mode, budget), hide_index=True, use_container_width=True)

    st.subheader("What XGBoost uses (TreeSHAP, well-held-out folds)")
    multi = res.get("shap", {}).get("multiclass", {})
    if multi:
        typ = st.selectbox("Anomaly type", list(multi))
        top = pd.DataFrame(multi[typ]["top"])
        fig = px.bar(top.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h",
                     labels={"mean_abs_shap": "mean |SHAP| (log-odds)", "feature": ""})
        st.plotly_chart(fig, use_container_width=True)
        fam = multi[typ]["family_share"]
        st.caption(f"Share of attribution by feature family: raw {fam['raw']:.0%}, physics {fam['physics']:.0%}, "
                   f"rolling context {fam['context']:.0%}. Names ending __mean15 / __std15 / __dev180 / __d5 are "
                   "past-only rolling features of the named channel.")
    with st.expander("Run configuration"):
        st.json({k: cfg[k] for k in ("wells", "days", "events_per_day", "seeds", "time_cut", "outer_folds",
                                       "inner_folds", "merge_gap_min", "grace_min", "xgb_rounds", "versions")})

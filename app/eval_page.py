"""Model Evaluation page: shows docs/eval_results.json. Nothing is trained or scored here."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px

from labels import feature_label
from profiles import get_profile
from ui import apply_plotly_theme

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
FEATURE_SETS = {"raw": "Raw sensors only", "raw+physics": "Raw + physics signals",
                "raw+physics+context": "Raw + physics + recent history"}
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
            rows.append({"Model": name, "Inputs": FEATURE_SETS[fs], "Number of inputs": res["config"]["feature_counts"][fs],
                         "Row PR-AUC": _ms(e["row"]["pr_auc"]), "Events detected": _ms(b["detect_share"], 0, True),
                         "False alarms / well-day": _ms(b["fa_per_well_day"], 2)})
    return pd.DataFrame(rows)


GUIDE = [
    ("Row PR-AUC", "How well a method puts the truly abnormal minutes above the normal ones when abnormal "
                   "minutes are rare (1.0 is perfect; a method with no skill scores about the share of abnormal minutes)."),
    ("Row AUROC", "The chance that a randomly chosen abnormal minute scores higher than a randomly chosen "
                  "normal one (0.5 is a coin flip, 1.0 is perfect)."),
    ("Events detected", "The share of simulated incidents where the method raised at least one alarm during the "
                        "incident or shortly after it."),
    ("False alarms / well-day", "How many alarms you would get on one well in one day when nothing is wrong; "
                                "lower means fewer needless call-outs."),
    ("Median delay (min)", "For the incidents that were caught, the typical number of minutes between the "
                           "incident starting and the first alarm (half were quicker, half slower)."),
]


def run_summary_frame(cfg, mode, budget):
    """Plain-language protocol table; every number is read from the results file."""
    split = {
        "wells": f"Each test well is held out: models are trained on other wells and scored on a well they have "
                 f"never seen ({cfg['outer_folds']}-fold, grouped by well).",
        "time": f"Models are trained on the first {cfg['time_cut']:.0%} of every well and scored on the last "
                f"{1 - cfg['time_cut']:.0%}.",
    }[mode]
    return pd.DataFrame([
        {"Topic": "How the data was split", "In plain words": split},
        {"Topic": "False-alarm budget", "In plain words":
            f"Each method's alarm threshold is set on training data only, so that it stays within about "
            f"{float(budget):g} false alarm per well per day. Test wells never influenced a threshold."},
        {"Topic": "When an incident counts as caught", "In plain words":
            f"At least one alarm between the start of the incident and {cfg['grace_min']} minutes after it ends. "
            f"Alarms less than {cfg['merge_gap_min']} minutes apart count as one."},
        {"Topic": "Where the data comes from", "In plain words":
            "A seeded simulator, not real rigs. The results compare methods with each other; they say little "
            "about field performance."},
    ])


def render(st):
    st.header("Model Evaluation")
    st.caption("Results come from simulated wells, so they compare methods with each other, not with real rigs.")
    res = load_results()
    if res is None:
        st.info("docs/eval_results.json not found. Generate it with `python scripts/evaluate.py` "
                "(see docs/EVAL.md). The app never trains models.")
        return
    cfg = res["config"]
    with st.expander("How to read this page: five numbers, one sentence each"):
        for name, sentence in GUIDE:
            st.markdown(f"**{name}.** {sentence}")

    budgets = [str(b) for b in cfg["budgets"]]
    c1, c2 = st.columns(2)
    mode = c1.radio("Split", list(MODES), format_func=MODES.get, horizontal=False)
    budget = c2.radio("False-alarm budget used to set thresholds (per well-day, on training data)", budgets,
                      index=budgets.index(str(cfg["primary_budget"])))

    st.subheader("How this evaluation was run")
    t1, t2, t3, t4 = st.columns(4)
    t1.container(border=True).metric("Repeats (random seeds)", len(cfg["seeds"]), help="Number of random seed repetitions to evaluate stability")
    t2.container(border=True).metric("Wells per repeat", cfg["wells"], help="Number of distinct simulated wells per evaluation run")
    t3.container(border=True).metric("Days per well", f"{cfg['days']:g}", help="Simulated duration in days for each well")
    t4.container(border=True).metric("Simulated incidents in total", cfg["events_total"], help="Total count of planted operational incidents")
    st.table(run_summary_frame(cfg, mode, budget).set_index("Topic"))
    st.caption("Tables show the mean ± the spread (standard deviation) over the repeats.")

    st.subheader("Headline comparison")
    st.dataframe(headline_frame(res, mode, budget), hide_index=True, width="stretch")
    st.caption("Row PR-AUC / AUROC are threshold-free. Events detected, false alarms and delay are event-level at the "
               "threshold chosen on training data; delay is the median over detected events only.")

    d = res["summary"][mode]
    chart = pd.DataFrame([{"Method": LABELS[k], "Events detected (%)": d[k]["by_budget"][budget]["detect_share"]["mean"] * 100,
                           "False alarms / well-day": d[k]["by_budget"][budget]["fa_per_well_day"]["mean"]} for k in LABELS])
    fig = px.scatter(chart, x="False alarms / well-day", y="Events detected (%)", color="Method", text="Method")
    fig.update_traces(marker_size=14, textposition="top center")
    fig.update_layout(showlegend=False, yaxis_range=[0, 105])
    apply_plotly_theme(fig)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Where does it fail? Detection by anomaly type and incident size")
    st.caption("Pooled over seeds; (detected/events). Small events are the hard cases. Event size is the incident's "
               "peak strength, 1.0 = the full effect listed in docs/PHYSICS.md.")
    st.markdown("**By anomaly type**")
    st.dataframe(breakdown_frame(res, mode, budget, "by_type"), hide_index=True, width="stretch")
    st.markdown("**By event size**")
    st.dataframe(breakdown_frame(res, mode, budget, "by_scale"), hide_index=True, width="stretch")

    st.subheader("Feature ablation")
    st.dataframe(ablation_frame(res, mode, budget), hide_index=True, width="stretch")

    st.subheader("What the XGBoost model pays attention to (SHAP values, on wells it had not seen)")
    multi = res.get("shap", {}).get("multiclass", {})
    if multi:
        typ = st.selectbox("Anomaly type", list(multi))
        profile = get_profile("drilling")
        top = pd.DataFrame(multi[typ]["top"])
        top["Feature"] = top["feature"].map(lambda f: feature_label(f, profile))
        fig = px.bar(top.iloc[::-1], x="mean_abs_shap", y="Feature", orientation="h",
                     custom_data=["feature"],
                     labels={"mean_abs_shap": "How much this feature moved the prediction", "Feature": ""})
        fig.update_traces(hovertemplate="%{y}<br>moved the prediction by %{x:.2f}<br>technical name: %{customdata[0]}"
                                        "<extra></extra>")
        apply_plotly_theme(fig)
        st.plotly_chart(fig, width="stretch")
        fam = multi[typ]["family_share"]
        st.caption(f"Share of the model's attention by kind of signal: raw sensor readings {fam['raw']:.0%}, "
                   f"derived physics signals {fam['physics']:.0%}, recent history of a signal {fam['context']:.0%}. "
                   "Longer bars mean the feature pushed the prediction harder (mean absolute SHAP value, in "
                   "log-odds, per feature). Only past readings are used, never future ones.")
    with st.expander("Technical details (for reproducibility)"):
        st.json({k: cfg[k] for k in ("wells", "days", "events_per_day", "seeds", "time_cut", "outer_folds",
                                       "inner_folds", "merge_gap_min", "grace_min", "xgb_rounds", "versions")},
                expanded=False)

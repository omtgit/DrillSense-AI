"""Instant answers to the suggested questions, worked out in code from the derived facts.

No API call and no key. Free-text questions go to Gemini (gemini_utils.answer_question).
"""
from report import _fmt, _minutes, derive_facts

SUGGESTED = {
    "movers": "Which signal moved the most?",
    "timing": "When did it start and how long did it last?",
    "severity": "How severe is it and what comes first?",
}
MEASURE = "distance from this well's normal in robust standard deviations"
INSTANT_LABEL = "Instant answer from the incident facts"


def gemini_label(model):
    return f"Gemini answer ({model}), checked against the incident facts"


def instant_answer(key, finding):
    d = derive_facts(finding)
    if d is None:
        return "The model flagged no event here, so there is no incident to describe."
    if key == "movers":
        if not d["ranked"]:
            return "No channel stood out against this well's normal level."
        top, rest = d["ranked"][0], d["ranked"][1:]
        text = (f"**{top['label']}** moved the most: {top['deviation']:.1f} robust standard deviations "
                f"{top['direction']} this well's normal (mean {_fmt(top['observed'])} during the event versus "
                f"{_fmt(top['baseline'])} normally).")
        if rest:
            text += " Next: " + "; ".join(
                f"{r['label']} ({r['deviation']:.1f}, {r['direction']} normal)" for r in rest) + "."
        return text + f" Measure used: {MEASURE}."
    if key == "timing":
        return (f"The {d['anomaly']} event started at {d['start']} and ended at {d['end']}: "
                f"{_minutes(d['duration_minutes'])} minutes from the first to the last flagged reading "
                f"({d['n_rows']} readings).")
    if key == "severity":
        return (f"Severity is **{d['severity']}** ({d['rule_source']}). The first response listed is: "
                f"{d['response']}.")
    raise KeyError(key)

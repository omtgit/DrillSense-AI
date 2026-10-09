"""Engineering-report block shared by the production and drilling Decision Center pages."""
import streamlit as st

from assistant import INSTANT_LABEL, SUGGESTED, gemini_label, instant_answer
from gemini_utils import AUDIENCES, LANGUAGES, answer_question, has_key, model_name, write_for
from waiting import run_with_status
from report import build_finding, render_derived_facts, render_report
from ui import severity_chip_html


def rank_by_risk(frame):
    """Highest predicted risk first; ties go to the most recent row, then the lowest well id.

    Many rows can share the maximum score, so sorting on the score alone gives an arbitrary pick.
    """
    return frame.sort_values(
        ["predicted_risk_score", "timestamp", "well_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )


def show_warning(warning):
    """Friendly message for visitors; the technical text, if any, stays in a collapsed expander."""
    st.warning(str(warning))
    detail = getattr(warning, "detail", "")
    if detail:
        with st.expander("Technical details"):
            st.code(detail)


def report_section(df, row, profile, rules_from_profile=False):
    finding = build_finding(df, row, profile=profile, rules_from_profile=rules_from_profile)
    template = render_report(finding, profile)

    severity = (
        finding.get("severity")
        or finding.get("predicted_severity")
        or (row.get("predicted_severity") if "predicted_severity" in row else row.get("severity", "Medium"))
    )

    with st.container(border=True):
        col_title, col_chip = st.columns([3, 1])
        with col_title:
            st.subheader("Engineering Report")
            st.caption("Built from the detected event by a fixed template; no API involved.")
        with col_chip:
            chip = severity_chip_html(str(severity))
            st.markdown(f'<div style="text-align:right; padding-top:12px;">{chip}</div>', unsafe_allow_html=True)

        st.markdown(template)

        st.download_button(
            label="Download report (.md)",
            data=template,
            file_name=f"drillsense_report_{finding['well_id']}_{finding.get('start', 'event')}.md",
            mime="text/markdown",
            help="Download this incident report as a Markdown document",
        )

    with st.container(border=True):
        assistant_block(template, finding)


def assistant_block(template, finding):
    """Suggested questions answered in code; optional Gemini step for other readers and free-text questions."""
    st.subheader("AI assistant (optional)")
    api_key = st.session_state.get("gemini_api_key") or None
    keyed = has_key(api_key)
    model = st.session_state.get("gemini_model") or None
    event_id = f"{finding['well_id']}|{finding.get('start')}|{finding['predicted_anomaly']}"
    language = "English"

    if keyed:
        st.caption("Same facts, different reader. Every number in the output is checked against the report "
                   "above; if anything does not match, you get the template report instead.")
        c1, c2 = st.columns(2)
        audience = c1.selectbox("Write it for", list(AUDIENCES))
        language = c2.selectbox("Language", LANGUAGES)
        if st.button("Generate"):
            text, warning = run_with_status(write_for, template, audience, language, api_key, model,
                                            fallback_text="Writing it up...")
            st.session_state["assistant_written"] = (event_id, audience, language, text, warning)
        written = st.session_state.get("assistant_written")
        if written and written[0] == event_id:
            _, aud, lang, text, warning = written
            if warning:
                show_warning(warning)
            else:
                st.markdown(f"**{aud} ({lang})**")
                st.markdown(text)
    else:
        st.info("The report above is complete without it. To have the same facts written for a specific "
                "reader, in English or Hindi, or to ask your own questions, add a free Google AI Studio "
                "key in the sidebar. The suggested questions below work without one.")

    st.markdown("**Ask about this incident**")
    cols = st.columns(len(SUGGESTED))
    for col, (key, label) in zip(cols, SUGGESTED.items()):
        if col.button(label, key=f"suggest_{key}"):
            st.session_state["assistant_answer"] = (event_id, label, instant_answer(key, finding), None,
                                                    INSTANT_LABEL)
    if keyed:
        facts = template + "\n\n" + render_derived_facts(finding)
        question = st.text_input("Or ask your own question", key="assistant_question",
                                 placeholder="For example: is the pump the likely cause?")
        if st.button("Ask") and question.strip():
            answer, warning = run_with_status(answer_question, facts, question, language, api_key, model,
                                              fallback_text="Looking it up in the report...")
            used = getattr(answer, "model", "") or (model or "").strip() or model_name()
            st.session_state["assistant_answer"] = (event_id, question, answer, warning, gemini_label(used))
    asked = st.session_state.get("assistant_answer")
    if asked and asked[0] == event_id:
        _, q, answer, warning, source = asked
        if warning:
            show_warning(warning)
        else:
            st.markdown(f"*{q}*")
            st.markdown(answer)
            st.caption(source)

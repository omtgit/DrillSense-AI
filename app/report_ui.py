"""Engineering-report block shared by the production and drilling Decision Center pages."""
import streamlit as st

from gemini_utils import AUDIENCES, LANGUAGES, answer_question, has_key, write_for
from report import build_finding, render_report


def rank_by_risk(frame):
    """Highest predicted risk first; ties go to the most recent row, then the lowest well id.

    Many rows can share the maximum score, so sorting on the score alone gives an arbitrary pick.
    """
    return frame.sort_values(
        ["predicted_risk_score", "timestamp", "well_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )


def report_section(df, row, profile, rules_from_profile=False):
    finding = build_finding(df, row, profile=profile, rules_from_profile=rules_from_profile)
    template = render_report(finding, profile)

    st.subheader("Engineering Report")
    st.caption("Built from the detected event by a fixed template; no API involved.")
    st.markdown(template)

    assistant_block(template, finding)


def assistant_block(template, finding):
    """Optional Gemini step: same facts for a chosen reader, and questions answered from the report only."""
    st.subheader("AI assistant (optional)")
    api_key = st.session_state.get("gemini_api_key") or None
    if not has_key(api_key):
        st.info("The report above is complete without it. To have the same facts written for a specific "
                "reader, in English or Hindi, or to ask questions about this event, add a free Google AI "
                "Studio key in the sidebar.")
        return
    model = st.session_state.get("gemini_model") or None
    event_id = f"{finding['well_id']}|{finding.get('start')}|{finding['predicted_anomaly']}"
    st.caption("Same facts, different reader. Every number in the output is checked against the report above; "
               "if anything does not match, you get the template report instead.")

    c1, c2 = st.columns(2)
    audience = c1.selectbox("Write it for", list(AUDIENCES))
    language = c2.selectbox("Language", LANGUAGES)
    if st.button("Generate"):
        with st.spinner("Writing it up..."):
            text, warning = write_for(template, audience, language, api_key, model)
        st.session_state["assistant_written"] = (event_id, audience, language, text, warning)
    written = st.session_state.get("assistant_written")
    if written and written[0] == event_id:
        _, aud, lang, text, warning = written
        if warning:
            st.warning(warning)
        else:
            st.markdown(f"**{aud} ({lang})**")
            st.markdown(text)

    st.markdown("**Ask about this incident**")
    question = st.text_input("Your question", key="assistant_question",
                             placeholder="For example: which signal moved the most?")
    if st.button("Ask") and question.strip():
        with st.spinner("Looking it up in the report..."):
            answer, warning = answer_question(template, question, language, api_key, model)
        st.session_state["assistant_answer"] = (event_id, question, answer, warning)
    asked = st.session_state.get("assistant_answer")
    if asked and asked[0] == event_id:
        _, q, answer, warning = asked
        if warning:
            st.warning(warning)
        else:
            st.markdown(f"*{q}*")
            st.markdown(answer)

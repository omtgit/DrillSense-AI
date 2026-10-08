"""Engineering-report block shared by the production and drilling Decision Center pages."""
import streamlit as st

from gemini_utils import rephrase_report
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

    if st.button("Rephrase with Gemini"):
        with st.spinner("Rephrasing..."):
            text, warning = rephrase_report(
                template,
                st.session_state.get("gemini_api_key") or None,
                st.session_state.get("gemini_model") or None,
            )
        if warning:
            st.warning(warning)
        else:
            st.subheader("Rephrased by Gemini")
            st.caption("Same facts as the template report; numbers were checked against it.")
            st.markdown(text)

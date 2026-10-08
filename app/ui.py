"""Small visitor-facing helpers shared by the pages."""
import streamlit as st

REPO_URL = "https://github.com/omtgit/DrillSense-AI"
DOCS_URL = f"{REPO_URL}/blob/main/docs"
SIM_NOTICE = "Simulated data. Educational demo, not for operational decisions."


def show_error(what, exc):
    """A friendly message first; the technical text stays available for whoever needs it."""
    st.error(f"Sorry, {what}. Please reload the page; if it keeps happening, open the details below "
             "and share them with the maintainer.")
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")

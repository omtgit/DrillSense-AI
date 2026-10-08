"""Runs app/main.py headlessly with Streamlit's AppTest: every page, both profiles, no exception."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

MAIN = str(Path(__file__).resolve().parent.parent / "app" / "main.py")
PAGES = ["Executive Dashboard", "Well Explorer", "AI Decision Center",
         "Model Evaluation", "CPU vs GPU Benchmark", "About"]


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(MAIN, default_timeout=180).run()


def test_default_profile_is_drilling_and_dashboard_separates_truth_from_model(app):
    assert not app.exception
    assert app.sidebar.selectbox[0].value == "Drilling"
    labels = [m.label for m in app.metric]
    assert any("ground truth" in l for l in labels) and any("Model detections" in l for l in labels)
    assert any("held-out" in s.value for s in app.subheader)


@pytest.mark.parametrize("profile", ["Drilling", "Production"])
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, profile, page):
    app.sidebar.selectbox[0].set_value(profile)
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception


def test_model_evaluation_page_is_readable(app):
    app.sidebar.selectbox[0].set_value("Drilling")
    app.sidebar.radio[0].set_value("Model Evaluation").run()
    assert not app.exception
    assert any("How this evaluation was run" in s.value for s in app.subheader)
    labels = [m.label for m in app.metric]
    for want in ("Repeats (random seeds)", "Wells per repeat", "Days per well", "Simulated incidents in total"):
        assert want in labels
    expanders = [e.label for e in app.expander]
    assert any(l.startswith("How to read this page") for l in expanders)
    assert "Technical details (for reproducibility)" in expanders
    assert "Run configuration" not in expanders
    # the raw JSON lives only inside the collapsed expander
    assert len(app.json) == 1


def _decision_center(monkeypatch=None, key=None):
    at = AppTest.from_file(MAIN, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("AI Decision Center")
    if key:
        at.session_state["gemini_api_key"] = key
    return at.run()


def test_assistant_without_key_shows_hint_not_controls(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import gemini_utils
    monkeypatch.setattr(gemini_utils, "load_dotenv", lambda: None)
    at = _decision_center()
    assert not at.exception
    assert "AI assistant (optional)" in [e.label for e in at.sidebar.expander]
    assert any("AI assistant (optional)" in s.value for s in at.subheader)
    assert any("complete without it" in i.value for i in at.info)
    assert not [b for b in at.button if b.label in ("Generate", "Ask")]


def test_assistant_with_key_shows_controls_and_falls_back_on_bad_output(monkeypatch):
    import gemini_utils
    monkeypatch.setattr(gemini_utils, "write_for", lambda t, a, l, k=None, m=None: (t, "Gemini call failed"))
    import report_ui
    monkeypatch.setattr(report_ui, "write_for", gemini_utils.write_for)
    at = _decision_center(key="test-key")
    assert not at.exception
    labels = [s.label for s in at.selectbox]
    assert "Write it for" in labels and "Language" in labels
    gen = next(b for b in at.button if b.label == "Generate")
    gen.click().run()
    assert not at.exception
    assert any("showing the template report" in w.value or "Gemini call failed" in w.value for w in at.warning)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_shows_the_simulated_data_notice(app, page):
    app.sidebar.selectbox[0].set_value("Drilling")
    app.sidebar.radio[0].set_value(page).run()
    assert any("Simulated data. Educational demo, not for operational decisions." in c.value for c in app.caption)


def test_sidebar_data_line_is_plain_and_about_links(app):
    app.sidebar.selectbox[0].set_value("Drilling")
    app.sidebar.radio[0].set_value("About").run()
    assert any(c.value == "Data: built-in simulated demo wells" for c in app.sidebar.caption)
    text = " ".join(m.value for m in app.markdown)
    for needle in ("docs/PHYSICS.md", "docs/EVAL.md", "github.com/omtgit/DrillSense-AI"):
        assert needle in text


def test_load_error_is_friendly_with_technical_text_hidden(monkeypatch):
    import data_source

    def boom(*a, **k):
        raise FileNotFoundError("Drilling data not found: /x/y.csv")

    import streamlit as st
    st.cache_data.clear()  # an earlier test may already have cached the scored wells
    monkeypatch.setattr(data_source, "load_drilling", boom)
    at = AppTest.from_file(MAIN, default_timeout=180).run()
    assert not at.exception
    assert any("could not be loaded" in e.value for e in at.error)
    assert all("FileNotFoundError" not in e.value for e in at.error)
    assert any("FileNotFoundError" in c.value for c in at.code)
    assert "Technical details" in [e.label for e in at.expander]


@pytest.mark.parametrize("profile", ["Drilling", "Production"])
def test_decision_center_table_has_plain_headers(app, profile):
    app.sidebar.selectbox[0].set_value(profile)
    app.sidebar.radio[0].set_value("AI Decision Center").run()
    cols = list(app.dataframe[0].value.columns)
    assert not [c for c in cols if "_" in c], cols
    assert "Well" in cols and "Predicted issue" in cols

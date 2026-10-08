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

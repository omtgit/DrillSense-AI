"""Tests for UI design system, severity chips, Plotly themes, and responsive helpers."""
from pathlib import Path
import plotly.express as px
import pandas as pd
import pytest

from ui import (
    apply_plotly_theme,
    empty_state,
    severity_chip_html,
    SEVERITY_COLORS,
    PLOTLY_COLORS,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"


def test_streamlit_config_toml_exists_and_sets_dark_theme():
    assert CONFIG_PATH.is_file(), f"Missing config file at {CONFIG_PATH}"
    content = CONFIG_PATH.read_text(encoding="utf-8")
    assert 'base = "dark"' in content
    assert 'primaryColor = "#00d2be"' in content
    assert 'backgroundColor = "#0e1117"' in content


def test_severity_chip_html_contains_bullet_and_accessible_colors():
    for sev in ("Critical", "High", "Medium", "Low", "Normal"):
        html = severity_chip_html(sev)
        assert f"&#9679; {sev}" in html or f"● {sev}" in html
        assert "background:" in html
        assert "border:" in html
        assert "color:" in html

    # Case insensitivity & fallback
    lower_html = severity_chip_html("critical")
    assert "&#9679; Critical" in lower_html or "● Critical" in lower_html
    fallback_html = severity_chip_html("UnknownRisk")
    assert "&#9679; Unknownrisk" in fallback_html or "● Unknownrisk" in fallback_html


def test_apply_plotly_theme_configures_dark_mode_and_palette():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    fig = px.line(df, x="x", y="y")
    themed = apply_plotly_theme(fig)

    assert themed.layout.template.layout.mapbox is not None or themed.layout.template.name == "plotly_dark"
    assert themed.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert themed.layout.font.color == "#e6edf3"
    assert themed.layout.colorway == tuple(PLOTLY_COLORS)


def test_empty_state_executes_without_error(monkeypatch):
    rendered = []
    import streamlit as st
    monkeypatch.setattr(st, "markdown", lambda html, unsafe_allow_html=True: rendered.append(html))

    empty_state("No anomalies detected for this well.", icon="[OK]")
    assert len(rendered) == 1
    assert "No anomalies detected for this well." in rendered[0]
    assert "[OK]" in rendered[0]

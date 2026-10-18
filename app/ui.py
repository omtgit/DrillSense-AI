"""Small visitor-facing helpers shared by the pages and design system."""
import streamlit as st

REPO_URL = "https://github.com/omtgit/DrillSense-AI"
DOCS_URL = f"{REPO_URL}/blob/main/docs"
FOOTER = "Demo with simulated data. Not for operational use."

# Accessible color palette for Plotly figures & UI chips
COLOR_ACCENT = "#00d2be"
COLOR_BG_CARD = "#161b22"
COLOR_BORDER = "#30363d"
COLOR_TEXT = "#e6edf3"
COLOR_MUTED = "#8b949e"

SEVERITY_COLORS = {
    "Critical": {"bg": "rgba(238, 82, 83, 0.2)", "border": "#ee5253", "text": "#ff6b6b"},
    "High": {"bg": "rgba(255, 159, 67, 0.2)", "border": "#ff9f43", "text": "#ffa502"},
    "Medium": {"bg": "rgba(254, 202, 87, 0.2)", "border": "#feca57", "text": "#ffd32a"},
    "Low": {"bg": "rgba(84, 160, 255, 0.2)", "border": "#54a0ff", "text": "#70a1ff"},
    "Normal": {"bg": "rgba(16, 172, 132, 0.2)", "border": "#10ac84", "text": "#2ed573"},
}

PLOTLY_COLORS = ["#00d2be", "#ff9f43", "#54a0ff", "#ee5253", "#10ac84", "#feca57", "#a55eea"]


def inject_design_system():
    """Injects global CSS variables, typography, card hover lift, shimmer, and accessibility styles."""
    css = """
    <style>
    /* DrillSense AI Design System Tokens */
    :root {
        --ds-bg-primary: #0e1117;
        --ds-bg-card: #161b22;
        --ds-bg-hover: #1f242c;
        --ds-border: #30363d;
        --ds-border-light: #484f58;
        --ds-accent: #00d2be;
        --ds-accent-hover: #33ddcb;
        --ds-text: #e6edf3;
        --ds-text-muted: #8b949e;
        --ds-success: #10ac84;
        --ds-warning: #ff9f43;
        --ds-danger: #ee5253;
        --ds-space-xs: 4px;
        --ds-space-sm: 8px;
        --ds-space-md: 16px;
        --ds-space-lg: 24px;
        --ds-space-xl: 32px;
        --ds-radius-sm: 6px;
        --ds-radius-md: 10px;
        --ds-radius-lg: 14px;
        --ds-font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    body {
        font-family: var(--ds-font-stack);
        color: var(--ds-text);
        letter-spacing: -0.01em;
    }

    /* Container constraints & padding */
    .main .block-container {
        max-width: 1220px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    /* Polish top Streamlit header bar */
    header[data-testid="stHeader"] {
        background: rgba(14, 17, 23, 0.85) !important;
        backdrop-filter: blur(8px) !important;
        border-bottom: 1px solid rgba(48, 54, 61, 0.4) !important;
    }

    /* Polish Streamlit Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid var(--ds-border) !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem !important;
    }

    /* Radio button navigation styling */
    div[data-testid="stRadio"] > div {
        gap: 4px !important;
    }

    div[data-testid="stRadio"] label {
        padding: 6px 12px !important;
        border-radius: var(--ds-radius-sm) !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
        margin: 0 !important;
    }

    div[data-testid="stRadio"] label:hover {
        background-color: var(--ds-bg-hover) !important;
    }

    /* Card container styling & gentle hover lift */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: var(--ds-radius-md) !important;
        border-color: var(--ds-border) !important;
        background-color: var(--ds-bg-card);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-1px);
        border-color: var(--ds-border-light) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.22);
    }

    /* Metric cards inside bordered containers */
    [data-testid="stMetric"] {
        background: transparent;
        padding: 0.25rem 0.5rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--ds-text-muted) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: var(--ds-text) !important;
        font-variant-numeric: tabular-nums;
    }

    /* Responsive touch targets for buttons */
    button[kind="primary"], button[kind="secondary"], .stButton > button {
        min-height: 40px !important;
        border-radius: var(--ds-radius-sm) !important;
        font-weight: 500 !important;
        transition: all 0.15s ease-in-out !important;
    }

    button:focus-visible {
        outline: 2px solid var(--ds-accent) !important;
        outline-offset: 2px !important;
    }

    /* Table horizontal scroll container */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        overflow-x: auto !important;
        border-radius: var(--ds-radius-sm) !important;
    }

    /* Severity Chip Badge */
    .ds-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Hero header */
    .ds-hero {
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--ds-border);
    }
    .ds-hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.01em;
    }
    .ds-hero-subtitle {
        color: var(--ds-text-muted);
        font-size: 0.92rem;
        margin-bottom: 0;
        line-height: 1.45;
    }

    /* Status Strip */
    .ds-status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 1rem 0;
    }
    .ds-status-item {
        flex: 1;
        min-width: 140px;
        background: var(--ds-bg-card);
        border: 1px solid var(--ds-border);
        border-radius: var(--ds-radius-sm);
        padding: 10px 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .ds-status-item-label {
        font-size: 0.8rem;
        color: var(--ds-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .ds-status-item-val {
        font-size: 1.25rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }

    /* Feature card container */
    .ds-feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 16px;
        margin: 1.5rem 0;
    }
    .ds-feature-card {
        background: var(--ds-bg-card);
        border: 1px solid var(--ds-border);
        border-radius: var(--ds-radius-md);
        padding: 18px 20px;
        transition: transform 0.18s ease, border-color 0.18s ease;
    }
    .ds-feature-card:hover {
        transform: translateY(-2px);
        border-color: var(--ds-accent);
    }
    .ds-feature-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 6px;
        color: var(--ds-text);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .ds-feature-desc {
        font-size: 0.88rem;
        color: var(--ds-text-muted);
        line-height: 1.45;
        margin: 0;
    }

    /* Responsive adjustments for mobile screens */
    @media (max-width: 768px) {
        .ds-status-strip {
            flex-direction: column !important;
        }
        .ds-hero-title {
            font-size: 1.45rem !important;
        }
        .ds-status-item {
            width: 100% !important;
        }
    }

    /* Respect accessibility motion preference */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def empty_state(message: str, icon: str = "&#128269;"):
    """Displays a friendly, accessible empty state card."""
    st.markdown(
        f"""
        <div style="text-align:center; padding: 2.5rem 1.5rem; border: 1px dashed var(--ds-border, #30363d); border-radius: 10px; background: rgba(22, 27, 34, 0.45); margin: 1.25rem 0;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
            <p style="color: var(--ds-text-muted, #8b949e); margin: 0; font-size: 0.95rem; line-height: 1.5;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_chip_html(severity: str) -> str:
    """Returns an accessible HTML chip badge for severity."""
    clean = str(severity).strip().capitalize()
    style = SEVERITY_COLORS.get(clean, SEVERITY_COLORS["Medium"])
    return (
        f'<span class="ds-chip" style="background:{style["bg"]}; border:1px solid {style["border"]}; color:{style["text"]};">'
        f'&#9679; {clean}</span>'
    )


def apply_plotly_theme(fig):
    """Applies a consistent accessible dark theme to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22, 27, 34, 0.4)",
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', color="#e6edf3"),
        colorway=PLOTLY_COLORS,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            bgcolor="rgba(22, 27, 34, 0.7)",
            bordercolor="#30363d",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    return fig


def show_error(what, exc):
    """A friendly message first; the technical text stays available for whoever needs it."""
    st.error(f"Sorry, {what}. Please reload the page; if it keeps happening, open the details below "
             "and share them with the maintainer.")
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")


def footer():
    """One short grey line at the very bottom of every page."""
    st.caption(FOOTER)

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
    }

    /* Page load subtle fade-in */
    @keyframes dsFadeIn {
        from {
            opacity: 0;
            transform: translateY(4px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .main .block-container {
        animation: dsFadeIn 0.25s ease-out;
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Loading shimmer animation */
    @keyframes dsShimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .ds-shimmer {
        background: linear-gradient(90deg, rgba(255,255,255,0.02) 25%, rgba(255,255,255,0.07) 50%, rgba(255,255,255,0.02) 75%);
        background-size: 200% 100%;
        animation: dsShimmer 2s infinite;
    }

    /* Card container styling & gentle hover lift */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: var(--ds-radius-md) !important;
        border-color: var(--ds-border) !important;
        background-color: var(--ds-bg-card);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    /* Enhanced Card hover lift and subtle glow */
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 210, 190, 0.45) !important;
        box-shadow: 0 8px 24px -4px rgba(0, 210, 190, 0.15);
    }

    /* Enhanced Tab Navigation */
    [data-baseweb="tab-list"] {
        gap: 8px !important;
        border-bottom: 1px solid var(--ds-border) !important;
        padding-bottom: 2px !important;
    }
    [data-baseweb="tab"] {
        border-radius: var(--ds-radius-sm) !important;
        padding: 8px 18px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        background: transparent !important;
    }
    [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.04) !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(0, 210, 190, 0.12) !important;
        color: var(--ds-accent) !important;
        border-bottom: 2px solid var(--ds-accent) !important;
    }

    /* Micro-Animations: Pulse, Radar, Mud Flow, Bubbles */
    @keyframes dsPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(1.2); }
    }
    @keyframes dsRadar {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(238, 82, 83, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(238, 82, 83, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(238, 82, 83, 0); }
    }
    .ds-pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
        animation: dsPulse 2s infinite ease-in-out;
    }
    .ds-radar-alarm {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #ee5253;
        margin-right: 6px;
        vertical-align: middle;
        animation: dsRadar 1.8s infinite ease-in-out;
    }
    @keyframes dsMudFlow {
        0% { stroke-dashoffset: 24; }
        100% { stroke-dashoffset: 0; }
    }
    .ds-mud-flow {
        stroke-dasharray: 6 6;
        animation: dsMudFlow 0.9s linear infinite;
    }
    @keyframes dsBubbleRise {
        0% { transform: translateY(70px) scale(0.6); opacity: 0; }
        20% { opacity: 0.9; }
        80% { opacity: 0.9; }
        100% { transform: translateY(0px) scale(1.1); opacity: 0; }
    }
    .ds-bubble-1 { animation: dsBubbleRise 3.2s infinite ease-in; }
    .ds-bubble-2 { animation: dsBubbleRise 3.8s 1.2s infinite ease-in; }
    .ds-bubble-3 { animation: dsBubbleRise 2.9s 2.1s infinite ease-in; }

    /* Metric cards inside bordered containers */
    [data-testid="stMetric"] {
        background: transparent;
        padding: 0.25rem 0.5rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--ds-text-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: var(--ds-text) !important;
    }

    /* Responsive touch targets for buttons */
    button[kind="primary"], button[kind="secondary"], .stButton > button {
        min-height: 42px !important;
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
        font-size: 0.82rem;
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
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .ds-hero-subtitle {
        color: var(--ds-text-muted);
        font-size: 0.95rem;
        margin-bottom: 0;
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
        padding: 8px 12px;
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
        font-size: 1.15rem;
        font-weight: 700;
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
        <div style="text-align:center; padding: 2rem 1rem; border: 1px dashed var(--ds-border, #30363d); border-radius: 8px; background: rgba(22, 27, 34, 0.4); margin: 1rem 0;">
            <div style="font-size: 1.8rem; margin-bottom: 0.35rem;">{icon}</div>
            <p style="color: var(--ds-text-muted, #8b949e); margin: 0; font-size: 0.95rem;">{message}</p>
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

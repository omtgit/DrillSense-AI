"""Premium design system for DrillSense AI.

Provides:
- CSS design tokens with light + dark theme variables
- Glass morphism cards, polished sidebar, backdrop blur header
- Smooth CSS transitions and micro-interactions
- Theme toggle stored in session state
- Severity chips, Plotly theme, empty states, error handling
"""
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
    "Critical": {"bg": "rgba(238, 82, 83, 0.18)", "border": "#ee5253", "text": "#ff6b6b"},
    "High":     {"bg": "rgba(255, 159, 67, 0.18)", "border": "#ff9f43", "text": "#ffa502"},
    "Medium":   {"bg": "rgba(254, 202, 87, 0.18)", "border": "#feca57", "text": "#ffd32a"},
    "Low":      {"bg": "rgba(84, 160, 255, 0.18)",  "border": "#54a0ff", "text": "#70a1ff"},
    "Normal":   {"bg": "rgba(16, 172, 132, 0.18)",  "border": "#10ac84", "text": "#2ed573"},
}

PLOTLY_COLORS = ["#00d2be", "#ff9f43", "#54a0ff", "#ee5253", "#10ac84", "#feca57", "#a55eea"]

# ──────────────────────────────────────────────────────────────────────────────
# Theme helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_light() -> bool:
    """Return True when user has toggled to light mode."""
    return bool(st.session_state.get("ds_light_mode", False))


def render_theme_toggle():
    """Render a compact theme-toggle button in the sidebar."""
    icon = "☀️" if _is_light() else "🌙"
    label = "Light mode" if _is_light() else "Dark mode"
    if st.sidebar.button(f"{icon} {label}", key="ds_theme_btn", use_container_width=True):
        st.session_state["ds_light_mode"] = not _is_light()
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Core design system injection
# ──────────────────────────────────────────────────────────────────────────────

def inject_design_system():
    """Inject global CSS: design tokens, typography, glass cards, animations, and a/light theme."""

    # ---- Dark palette (default) ----
    dark_vars = """
        --ds-bg-app:       #090c12;
        --ds-bg-primary:   #0e1117;
        --ds-bg-card:      #161b22;
        --ds-bg-card-2:    #1c2130;
        --ds-bg-glass:     rgba(22, 27, 34, 0.72);
        --ds-bg-hover:     #1f2937;
        --ds-border:       #30363d;
        --ds-border-light: #484f58;
        --ds-border-glass: rgba(255,255,255,0.06);
        --ds-accent:       #00d2be;
        --ds-accent-dim:   rgba(0, 210, 190, 0.15);
        --ds-accent-hover: #33ddcb;
        --ds-text:         #e6edf3;
        --ds-text-muted:   #8b949e;
        --ds-text-dim:     #555f6d;
        --ds-success:      #10ac84;
        --ds-warning:      #ff9f43;
        --ds-danger:       #ee5253;
        --ds-info:         #54a0ff;
        --ds-shadow-sm:    0 1px 4px rgba(0,0,0,0.35);
        --ds-shadow-md:    0 4px 20px rgba(0,0,0,0.45);
        --ds-shadow-lg:    0 12px 40px rgba(0,0,0,0.55);
        --ds-shadow-glow:  0 0 18px rgba(0, 210, 190, 0.12);
    """

    # ---- Light palette (toggled) ----
    light_vars = """
        --ds-bg-app:       #f0f4f8;
        --ds-bg-primary:   #f7f8fa;
        --ds-bg-card:      #ffffff;
        --ds-bg-card-2:    #f0f4f8;
        --ds-bg-glass:     rgba(255,255,255,0.80);
        --ds-bg-hover:     #eaf0f9;
        --ds-border:       #d0d7de;
        --ds-border-light: #b0bbc5;
        --ds-border-glass: rgba(0,0,0,0.08);
        --ds-accent:       #00a99d;
        --ds-accent-dim:   rgba(0,169,157,0.12);
        --ds-accent-hover: #00c4b7;
        --ds-text:         #1c2333;
        --ds-text-muted:   #4b5563;
        --ds-text-dim:     #9ca3af;
        --ds-success:      #0e9f6e;
        --ds-warning:      #e07c1f;
        --ds-danger:       #d63031;
        --ds-info:         #2563eb;
        --ds-shadow-sm:    0 1px 4px rgba(0,0,0,0.08);
        --ds-shadow-md:    0 4px 20px rgba(0,0,0,0.10);
        --ds-shadow-lg:    0 12px 40px rgba(0,0,0,0.12);
        --ds-shadow-glow:  0 0 18px rgba(0, 169, 157, 0.10);
    """

    active_theme_vars = light_vars if _is_light() else dark_vars

    # ---- Streamlit overrides for light mode ----
    light_overrides = """
    body, .stApp {
        background-color: #f7f8fa !important;
        color: #1c2333 !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2236 0%, #0f1623 100%) !important;
        border-right: 1px solid rgba(0,0,0,0.12) !important;
    }
    """ if _is_light() else ""

    css = f"""
    <style>
    /* ======================================================
       DrillSense AI — Premium Design System v2
       Dark + Light themes, glass morphism, smooth transitions
    ====================================================== */

    :root {{
        {active_theme_vars}

        /* Spacing scale */
        --ds-space-xs:  4px;
        --ds-space-sm:  8px;
        --ds-space-md:  16px;
        --ds-space-lg:  24px;
        --ds-space-xl:  40px;

        /* Radius scale */
        --ds-radius-sm: 6px;
        --ds-radius-md: 12px;
        --ds-radius-lg: 18px;
        --ds-radius-xl: 24px;

        /* Typography */
        --ds-font-stack: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                         "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        --ds-font-mono:  "SF Mono", "Cascadia Code", "Fira Code", monospace;

        /* Transition speeds */
        --ds-t-fast:   120ms;
        --ds-t-base:   220ms;
        --ds-t-slow:   380ms;
        --ds-ease:     cubic-bezier(0.4, 0, 0.2, 1);
    }}

    /* ─── Base ───────────────────────────────────────── */
    *, *::before, *::after {{
        box-sizing: border-box;
    }}

    body, .stApp {{
        font-family: var(--ds-font-stack) !important;
        color: var(--ds-text) !important;
        background-color: var(--ds-bg-primary) !important;
        letter-spacing: -0.01em;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    /* ─── Page enter animation ───────────────────────── */
    @keyframes ds-fade-up {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes ds-fade-in {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}
    @keyframes ds-slide-right {{
        from {{ opacity: 0; transform: translateX(-12px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes ds-pulse-ring {{
        0%  {{ box-shadow: 0 0 0 0 rgba(0,210,190,0.35); }}
        70% {{ box-shadow: 0 0 0 8px rgba(0,210,190,0); }}
        100%{{ box-shadow: 0 0 0 0 rgba(0,210,190,0); }}
    }}
    @keyframes ds-shimmer {{
        0%   {{ background-position: -800px 0; left: -100%; }}
        50%  {{ left: 150%; }}
        100% {{ background-position: 800px 0; left: 150%; }}
    }}

    .main .block-container {{
        max-width: 1280px;
        padding-top: 1rem;
        padding-bottom: 3.5rem;
        animation: ds-fade-up 0.3s var(--ds-ease) both;
    }}

    /* ─── Page background subtle gradient ────────────── */
    .stApp {{
        background:
            radial-gradient(ellipse at 20% 0%, rgba(0,210,190,0.035) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 100%, rgba(84,160,255,0.03) 0%, transparent 50%),
            var(--ds-bg-primary) !important;
    }}

    /* ─── Streamlit Header Bar (glass) ───────────────── */
    header[data-testid="stHeader"] {{
        background: rgba(9, 12, 18, 0.82) !important;
        backdrop-filter: saturate(180%) blur(16px) !important;
        -webkit-backdrop-filter: saturate(180%) blur(16px) !important;
        border-bottom: 1px solid rgba(255,255,255,0.06) !important;
        transition: background var(--ds-t-base) var(--ds-ease) !important;
    }}

    /* ─── Sidebar ─────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #111827 0%, #0b0f1a 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
        box-shadow: 2px 0 20px rgba(0,0,0,0.35) !important;
    }}
    {light_overrides}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
    }}

    /* Sidebar text overrides for dark sidebar in all modes */
    section[data-testid="stSidebar"] * {{
        color: #c9d1d9 !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #e6edf3 !important;
    }}

    /* ─── Navigation radio pills ──────────────────────── */
    div[data-testid="stRadio"] > div {{
        gap: 2px !important;
    }}

    div[data-testid="stRadio"] label {{
        display: flex !important;
        align-items: center !important;
        padding: 8px 14px !important;
        border-radius: var(--ds-radius-sm) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #8b949e !important;
        transition:
            background-color var(--ds-t-fast) var(--ds-ease),
            color var(--ds-t-fast) var(--ds-ease),
            transform var(--ds-t-fast) var(--ds-ease) !important;
        cursor: pointer !important;
        margin: 0 !important;
        letter-spacing: -0.005em;
    }}

    div[data-testid="stRadio"] label:hover {{
        background-color: rgba(255,255,255,0.07) !important;
        color: #e6edf3 !important;
        transform: translateX(2px) !important;
    }}

    /* Active navigation item */
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked),
    div[data-testid="stRadio"] label:has(input[value="true"]) {{
        background: linear-gradient(135deg, rgba(0,210,190,0.18), rgba(0,210,190,0.08)) !important;
        color: #00d2be !important;
        border-left: 3px solid #00d2be !important;
        padding-left: 11px !important;
        box-shadow: 0 0 12px rgba(0,210,190,0.08) !important;
    }}

    /* ─── Card borders / glass containers ────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: var(--ds-radius-md) !important;
        border: 1px solid var(--ds-border) !important;
        background: var(--ds-bg-glass) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        box-shadow: var(--ds-shadow-sm) !important;
        transition:
            transform var(--ds-t-base) var(--ds-ease),
            box-shadow var(--ds-t-base) var(--ds-ease),
            border-color var(--ds-t-base) var(--ds-ease) !important;
        animation: ds-fade-up 0.32s var(--ds-ease) both;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        transform: translateY(-2px) !important;
        border-color: var(--ds-border-light) !important;
        box-shadow: var(--ds-shadow-md), 0 0 0 1px var(--ds-accent-dim) !important;
    }}

    /* ─── Metric cards ────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: transparent !important;
        padding: 0.4rem 0.7rem !important;
        transition: transform var(--ds-t-fast) var(--ds-ease) !important;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-1px) !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: var(--ds-text-muted) !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 1.72rem !important;
        font-weight: 700 !important;
        color: var(--ds-text) !important;
        font-variant-numeric: tabular-nums !important;
        letter-spacing: -0.025em !important;
        line-height: 1.1 !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 0.8rem !important;
        font-variant-numeric: tabular-nums !important;
    }}

    /* ─── Buttons ─────────────────────────────────────── */
    .stButton > button {{
        min-height: 38px !important;
        border-radius: var(--ds-radius-sm) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        letter-spacing: -0.005em !important;
        transition:
            all var(--ds-t-fast) var(--ds-ease) !important;
        position: relative !important;
        overflow: hidden !important;
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #00d2be, #00bfad) !important;
        border: none !important;
        color: #0e1117 !important;
        box-shadow: 0 2px 10px rgba(0,210,190,0.28) !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #33ddcb, #00d2be) !important;
        box-shadow: 0 4px 16px rgba(0,210,190,0.40) !important;
        transform: translateY(-1px) !important;
    }}

    .stButton > button[kind="secondary"] {{
        border: 1px solid var(--ds-border) !important;
        background: transparent !important;
        color: var(--ds-text-muted) !important;
    }}

    .stButton > button[kind="secondary"]:hover {{
        border-color: var(--ds-border-light) !important;
        color: var(--ds-text) !important;
        background: rgba(255,255,255,0.04) !important;
        transform: translateY(-1px) !important;
    }}

    button:focus-visible {{
        outline: 2px solid var(--ds-accent) !important;
        outline-offset: 2px !important;
    }}

    /* ─── Tabs ────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px !important;
        background: transparent !important;
        border-bottom: 1px solid var(--ds-border) !important;
        padding-bottom: 0 !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: var(--ds-radius-sm) var(--ds-radius-sm) 0 0 !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--ds-text-muted) !important;
        padding: 8px 16px !important;
        background: transparent !important;
        border: none !important;
        transition: color var(--ds-t-fast) var(--ds-ease) !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: var(--ds-text) !important;
        background: rgba(255,255,255,0.04) !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: var(--ds-accent) !important;
        border-bottom: 2px solid var(--ds-accent) !important;
        background: transparent !important;
        font-weight: 600 !important;
    }}

    /* ─── Selectboxes & Inputs ───────────────────────── */
    .stSelectbox > div > div {{
        border-radius: var(--ds-radius-sm) !important;
        border-color: var(--ds-border) !important;
        background: var(--ds-bg-card) !important;
        transition: border-color var(--ds-t-fast) var(--ds-ease) !important;
    }}

    .stSelectbox > div > div:hover,
    .stSelectbox > div > div:focus-within {{
        border-color: var(--ds-accent) !important;
        box-shadow: 0 0 0 3px var(--ds-accent-dim) !important;
    }}

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        border-radius: var(--ds-radius-sm) !important;
        border-color: var(--ds-border) !important;
        background: var(--ds-bg-card) !important;
        transition: border-color var(--ds-t-fast) var(--ds-ease) !important;
    }}

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: var(--ds-accent) !important;
        box-shadow: 0 0 0 3px var(--ds-accent-dim) !important;
    }}

    /* ─── Multiselect ────────────────────────────────── */
    .stMultiSelect > div {{
        border-radius: var(--ds-radius-sm) !important;
    }}

    /* ─── Expanders ──────────────────────────────────── */
    .streamlit-expanderHeader {{
        border-radius: var(--ds-radius-sm) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: background var(--ds-t-fast) var(--ds-ease) !important;
    }}

    .streamlit-expanderHeader:hover {{
        background: rgba(255,255,255,0.04) !important;
    }}

    /* ─── DataFrames ─────────────────────────────────── */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        border-radius: var(--ds-radius-md) !important;
        overflow: hidden !important;
        border: 1px solid var(--ds-border) !important;
        box-shadow: var(--ds-shadow-sm) !important;
    }}

    /* ─── Alerts & status boxes ──────────────────────── */
    .stAlert {{
        border-radius: var(--ds-radius-md) !important;
        border-left-width: 4px !important;
        font-size: 0.9rem !important;
    }}

    /* ─── Progress / spinner ─────────────────────────── */
    .stProgress > div > div {{
        background: linear-gradient(90deg, var(--ds-accent), var(--ds-accent-hover)) !important;
        border-radius: 999px !important;
    }}

    /* ─── Divider ────────────────────────────────────── */
    hr {{
        border: none !important;
        border-top: 1px solid var(--ds-border) !important;
        margin: 1.5rem 0 !important;
    }}

    /* ─── Code blocks ────────────────────────────────── */
    code, pre {{
        font-family: var(--ds-font-mono) !important;
        border-radius: var(--ds-radius-sm) !important;
        background: rgba(0,0,0,0.25) !important;
    }}

    /* ─── Scrollbar ──────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: var(--ds-border);
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--ds-border-light);
    }}

    /* ─── DS Component Classes ───────────────────────── */

    /* Severity Chip */
    .ds-chip {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        white-space: nowrap;
        transition: transform var(--ds-t-fast) var(--ds-ease);
    }}
    .ds-chip:hover {{ transform: scale(1.04); }}

    /* Page hero header */
    .ds-page-header {{
        padding: 0 0 1.5rem 0;
        margin-bottom: 1.75rem;
        border-bottom: 1px solid var(--ds-border);
        animation: ds-fade-up 0.28s var(--ds-ease) both;
        position: relative;
    }}
    .ds-page-header::after {{
        content: "";
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 48px;
        height: 2px;
        background: linear-gradient(90deg, var(--ds-accent), transparent);
        border-radius: 1px;
    }}
    .ds-page-title {{
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.028em;
        color: var(--ds-text);
        margin: 0 0 0.25rem 0;
        line-height: 1.15;
    }}
    .ds-page-subtitle {{
        font-size: 0.88rem;
        color: var(--ds-text-muted);
        margin: 0;
        line-height: 1.55;
    }}

    /* Hero banner (dashboard) */
    .ds-hero {{
        position: relative;
        border-radius: var(--ds-radius-lg);
        border: 1px solid var(--ds-border-glass);
        background: linear-gradient(135deg,
            rgba(22, 27, 34, 0.96) 0%,
            rgba(14, 17, 23, 0.98) 55%,
            rgba(0, 210, 190, 0.05) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 32px 36px;
        margin-bottom: 28px;
        overflow: hidden;
        animation: ds-fade-up 0.4s var(--ds-ease) both;
        box-shadow: var(--ds-shadow-lg), 0 0 0 1px rgba(0,210,190,0.06) inset;
    }}
    .ds-hero::before {{
        content: "";
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        background:
            radial-gradient(ellipse at 85% 30%, rgba(0,210,190,0.10) 0%, transparent 55%),
            radial-gradient(ellipse at 15% 80%, rgba(84,160,255,0.05) 0%, transparent 50%);
        pointer-events: none;
    }}
    .ds-hero::after {{
        content: "";
        position: absolute;
        top: 0; left: -100%;
        width: 60%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.022), transparent);
        animation: ds-shimmer 6s ease-in-out infinite;
        pointer-events: none;
    }}
    .ds-hero-wordmark {{
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 10px;
    }}
    .ds-hero-icon {{
        width: 44px; height: 44px;
        background: linear-gradient(135deg, #00d2be22, #00d2be44);
        border: 1px solid rgba(0,210,190,0.4);
        border-radius: var(--ds-radius-md);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem;
        flex-shrink: 0;
        box-shadow: 0 0 12px rgba(0,210,190,0.15);
    }}
    .ds-hero-name {{
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--ds-text);
        line-height: 1;
    }}
    .ds-hero-tagline {{
        font-size: 0.85rem;
        color: var(--ds-accent);
        font-weight: 500;
        letter-spacing: 0.01em;
    }}
    .ds-hero-desc {{
        color: var(--ds-text-muted);
        font-size: 0.92rem;
        max-width: 820px;
        line-height: 1.6;
        margin: 12px 0 18px 0;
    }}
    .ds-hero-badges {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }}
    .ds-badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid;
        transition: all var(--ds-t-fast) var(--ds-ease);
    }}
    .ds-badge:hover {{
        transform: translateY(-1px);
        box-shadow: var(--ds-shadow-sm);
    }}

    /* Status strip (fleet overview) */
    .ds-status-strip {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin: 20px 0;
    }}
    .ds-status-item {{
        background: var(--ds-bg-glass);
        backdrop-filter: blur(8px);
        border: 1px solid var(--ds-border);
        border-radius: var(--ds-radius-md);
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        gap: 6px;
        transition: all var(--ds-t-base) var(--ds-ease);
        animation: ds-fade-up 0.35s var(--ds-ease) both;
    }}
    .ds-status-item:hover {{
        transform: translateY(-2px);
        box-shadow: var(--ds-shadow-md);
        border-color: var(--ds-border-light);
    }}
    .ds-status-item-label {{
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--ds-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }}
    .ds-status-item-val {{
        font-size: 2rem;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.03em;
        line-height: 1;
    }}

    /* Feature grid */
    .ds-feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 14px;
        margin: 16px 0 24px;
    }}
    .ds-feature-card {{
        background: var(--ds-bg-glass);
        backdrop-filter: blur(8px);
        border: 1px solid var(--ds-border);
        border-radius: var(--ds-radius-md);
        padding: 18px 20px;
        transition: all var(--ds-t-base) var(--ds-ease);
        animation: ds-fade-up 0.4s var(--ds-ease) both;
    }}
    .ds-feature-card:hover {{
        transform: translateY(-3px);
        border-color: var(--ds-accent);
        box-shadow: var(--ds-shadow-md), 0 0 0 1px var(--ds-accent-dim);
    }}
    .ds-feature-icon {{
        font-size: 1.4rem;
        margin-bottom: 10px;
        display: block;
    }}
    .ds-feature-title {{
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 6px;
        color: var(--ds-text);
        letter-spacing: -0.01em;
    }}
    .ds-feature-desc {{
        font-size: 0.84rem;
        color: var(--ds-text-muted);
        line-height: 1.5;
        margin: 0;
    }}

    /* Inline KV property list */
    .ds-kv-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 8px 24px;
        font-size: 0.9rem;
        padding: 4px 0;
    }}
    .ds-kv-item {{ display: flex; flex-direction: column; gap: 2px; }}
    .ds-kv-label {{
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--ds-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }}
    .ds-kv-value {{
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--ds-text);
    }}
    .ds-kv-value.accent {{ color: var(--ds-accent) !important; }}
    .ds-kv-value.danger {{ color: var(--ds-danger) !important; }}

    /* Sidebar branding block */
    .ds-sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 16px 0;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 12px;
    }}
    .ds-sidebar-logo {{
        width: 34px; height: 34px;
        background: linear-gradient(135deg, rgba(0,210,190,0.25), rgba(0,210,190,0.45));
        border: 1px solid rgba(0,210,190,0.5);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }}
    .ds-sidebar-title {{
        font-size: 1rem;
        font-weight: 700;
        color: #e6edf3 !important;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }}
    .ds-sidebar-version {{
        font-size: 0.7rem;
        color: #484f58 !important;
        letter-spacing: 0.02em;
    }}

    /* Section label dividers */
    .ds-section-label {{
        font-size: 0.67rem;
        font-weight: 800;
        color: #484f58 !important;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        padding: 10px 0 5px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .ds-section-label::after {{
        content: "";
        flex: 1;
        height: 1px;
        background: rgba(255,255,255,0.06);
    }}

    /* Live indicator dot */
    .ds-live-dot {{
        display: inline-block;
        width: 7px;
        height: 7px;
        background: var(--ds-success);
        border-radius: 50%;
        animation: ds-pulse-ring 2s ease infinite;
        vertical-align: middle;
        margin-right: 5px;
    }}

    /* Empty state card */
    .ds-empty-state {{
        text-align: center;
        padding: 3rem 2rem;
        border: 1px dashed var(--ds-border);
        border-radius: var(--ds-radius-lg);
        background: var(--ds-bg-glass);
        backdrop-filter: blur(6px);
        margin: 1.5rem 0;
        animation: ds-fade-in 0.4s var(--ds-ease) both;
    }}
    .ds-empty-icon {{ font-size: 2.2rem; margin-bottom: 12px; display: block; }}
    .ds-empty-msg {{
        color: var(--ds-text-muted);
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.55;
    }}

    /* Benchmark metric cards */
    .ds-bench-card {{
        background: var(--ds-bg-glass);
        border: 1px solid var(--ds-border);
        border-radius: var(--ds-radius-md);
        padding: 20px 24px;
        text-align: center;
        transition: all var(--ds-t-base) var(--ds-ease);
    }}
    .ds-bench-card:hover {{
        transform: translateY(-2px);
        box-shadow: var(--ds-shadow-md);
    }}
    .ds-bench-val {{
        font-size: 2.2rem;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.03em;
        line-height: 1;
        margin-bottom: 4px;
    }}
    .ds-bench-label {{
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--ds-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }}

    /* Responsive */
    @media (max-width: 900px) {{
        .ds-status-strip {{ grid-template-columns: repeat(3, 1fr); }}
        .ds-hero-name {{ font-size: 1.5rem; }}
        .ds-status-item-val {{ font-size: 1.6rem; }}
    }}
    @media (max-width: 600px) {{
        .ds-status-strip {{ grid-template-columns: 1fr 1fr; }}
        .ds-hero {{ padding: 20px; }}
        .ds-hero-badges {{ flex-direction: column; }}
    }}

    /* Respect reduced motion */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Public component helpers
# ──────────────────────────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Renders a compact, animated page header used by every screen."""
    icon_html = f'<span style="margin-right:8px;">{icon}</span>' if icon else ""
    sub_html = f'<p class="ds-page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="ds-page-header">
            <h1 class="ds-page-title">{icon_html}{title}</h1>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message: str, icon: str = "&#128269;"):
    """Displays a polished, accessible empty-state card."""
    st.markdown(
        f"""
        <div class="ds-empty-state">
            <span class="ds-empty-icon">{icon}</span>
            <p class="ds-empty-msg">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_chip_html(severity: str) -> str:
    """Returns an accessible HTML chip badge for severity."""
    clean = str(severity).strip().capitalize()
    style = SEVERITY_COLORS.get(clean, SEVERITY_COLORS["Medium"])
    return (
        f'<span class="ds-chip" style="background:{style["bg"]}; '
        f'border:1px solid {style["border"]}; color:{style["text"]};">'
        f'&#9679; {clean}</span>'
    )


def kv_grid(items: dict, accent_keys: list = None, danger_keys: list = None):
    """Render a responsive key-value property grid.

    Args:
        items: Dict of label -> value pairs.
        accent_keys: Labels to render in accent color.
        danger_keys: Labels to render in danger color.
    """
    accent_keys = accent_keys or []
    danger_keys = danger_keys or []
    cells = ""
    for k, v in items.items():
        if k in accent_keys:
            cls = "ds-kv-value accent"
        elif k in danger_keys:
            cls = "ds-kv-value danger"
        else:
            cls = "ds-kv-value"
        cells += (
            f'<div class="ds-kv-item">'
            f'<span class="ds-kv-label">{k}</span>'
            f'<span class="{cls}">{v}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="ds-kv-grid">{cells}</div>',
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig, height: int = None):
    """Applies a consistent, accessible dark theme to Plotly figures."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.4)",
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            color="#e6edf3",
            size=12,
        ),
        colorway=PLOTLY_COLORS,
        margin=dict(l=40, r=20, t=44, b=40),
        legend=dict(
            bgcolor="rgba(22,27,34,0.75)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#1c2130",
            bordercolor="#30363d",
            font_size=12,
        ),
    )
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor="#1c2130",
        zerolinecolor="#30363d",
        linecolor="#30363d",
        tickfont=dict(size=11),
    )
    fig.update_yaxes(
        gridcolor="#1c2130",
        zerolinecolor="#30363d",
        linecolor="#30363d",
        tickfont=dict(size=11),
    )
    return fig


def show_error(what, exc):
    """A friendly message first; technical text in a collapsible expander."""
    st.error(
        f"Sorry, {what}. Please reload the page; if the problem continues, "
        "open the details below and share them with the maintainer."
    )
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")


def footer():
    """One short grey disclaimer at the very bottom of every page."""
    st.caption(FOOTER)

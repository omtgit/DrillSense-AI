"""Premium landing hero for DrillSense AI Executive Dashboard.

Renders an executive-grade hero banner with:
- Branded product identity (icon, name, tagline)
- Core value proposition description
- Capability badges (Physics-Conditioned, Rig-State Aware, etc.)
- Smooth glass-morphism card styling
"""
import streamlit as st


def render_home_hero(active_profile: str = "drilling", on_navigate=None):
    """Render the premium product hero banner."""

    profile_badge = (
        '<span class="ds-badge" style="background:rgba(0,210,190,0.12); '
        'border-color:#00d2be; color:#00d2be;">⛽ Drilling Profile Active</span>'
        if active_profile == "drilling"
        else
        '<span class="ds-badge" style="background:rgba(84,160,255,0.12); '
        'border-color:#54a0ff; color:#70a1ff;">🏭 Production Profile Active</span>'
    )

    st.markdown(
        f"""
        <div class="ds-hero">
            <div class="ds-hero-wordmark">
                <div class="ds-hero-icon">⛽</div>
                <div>
                    <div class="ds-hero-name">DrillSense AI</div>
                    <div class="ds-hero-tagline">Energy-Intelligence &amp; Physics-Conditioned Anomaly Detection</div>
                </div>
            </div>
            <p class="ds-hero-desc">
                Real-time surface drilling telemetry surveillance with physical mass-balance flow checks
                and automated engineering incident reporting. Designed for petroleum engineers to detect
                formation influxes, circulation losses, and sensor drift — offline, without cloud dependencies.
            </p>
            <div class="ds-hero-badges">
                {profile_badge}
                <span class="ds-badge" style="background:rgba(0,210,190,0.10); border-color:rgba(0,210,190,0.4); color:#00d2be;">
                    ⚛ Physics Conditioned
                </span>
                <span class="ds-badge" style="background:rgba(84,160,255,0.10); border-color:rgba(84,160,255,0.4); color:#70a1ff;">
                    🎛 Rig-State Aware
                </span>
                <span class="ds-badge" style="background:rgba(16,172,132,0.10); border-color:rgba(16,172,132,0.4); color:#2ed573;">
                    ✈ Offline Capable
                </span>
                <span class="ds-badge" style="background:rgba(254,202,87,0.10); border-color:rgba(254,202,87,0.4); color:#ffd32a;">
                    📄 WITSML &amp; CSV Ready
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

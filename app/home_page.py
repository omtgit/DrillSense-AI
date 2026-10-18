"""Landing / Overview Welcome Section for DrillSense AI.

Provides a clear, executive-grade introduction to DrillSense AI:
- Product identity and core value proposition
- Quick-action shortcuts to jump to Live Monitor, BYOD, or Well Explorer
- System status overview (profile, detector status, active models)
- Real workflow overview (Detection -> Rig-State Conditioning -> Engineering Reporting)
"""
import streamlit as st
from app.ui import severity_chip_html


def render_home_hero(active_profile="drilling", on_navigate=None):
    """Render the executive product hero banner with quick start actions."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(14, 17, 23, 0.98) 100%);
                    border: 1px solid var(--ds-border, #30363d);
                    border-radius: var(--ds-radius-md, 10px);
                    padding: 24px 28px;
                    margin-bottom: 24px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <span style="font-size: 2rem;">⛽</span>
                <div>
                    <h1 style="font-size: 2rem; font-weight: 700; margin: 0; color: #e6edf3; letter-spacing: -0.02em;">
                        DrillSense AI
                    </h1>
                    <div style="font-size: 0.95rem; color: #00d2be; font-weight: 500; margin-top: 2px;">
                        Energy-Intelligence & Physics-Conditioned Anomaly Detection for Oilfield Telemetry
                    </div>
                </div>
            </div>
            <p style="color: #8b949e; font-size: 0.95rem; max-width: 820px; margin: 12px 0 18px 0; line-height: 1.55;">
                Real-time surface drilling telemetry surveillance, physical mass-balance flow checks, 
                and automated engineering incident reports. Designed for petroleum engineers and operations centers 
                to mitigate formation influxes (kicks), circulation losses, and sensor drift.
            </p>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">
                <span class="ds-chip" style="background: rgba(0, 210, 190, 0.15); border: 1px solid #00d2be; color: #00d2be;">
                    &#9679; Physics Conditioned
                </span>
                <span class="ds-chip" style="background: rgba(84, 160, 255, 0.15); border: 1px solid #54a0ff; color: #70a1ff;">
                    &#9679; Rig-State Aware
                </span>
                <span class="ds-chip" style="background: rgba(16, 172, 132, 0.15); border: 1px solid #10ac84; color: #2ed573;">
                    &#9679; Offline Capable
                </span>
                <span class="ds-chip" style="background: rgba(254, 202, 87, 0.15); border: 1px solid #feca57; color: #ffd32a;">
                    &#9679; WITSML & CSV Ready
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

import os
from dotenv import load_dotenv
from google import genai



def get_client():
    """Build the Gemini client on first use so the app starts without a key."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set; add it to your environment or .env to generate reports.")
    return genai.Client(api_key=api_key)


def generate_report(row):

    prompt = f"""
You are an experienced petroleum production engineer.

Analyze the following well telemetry summary.

Well ID: {row['well_id']}

Predicted Issue:
{row['predicted_anomaly']}

Severity:
{row['severity']}

Risk Score:
{row['predicted_risk_score']}

Pressure:
{row['pressure_psi']} psi

Temperature:
{row['temperature_c']} °C

Flow Rate:
{row['flow_rate_bpd']} BPD

Gas Ratio:
{row['gas_ratio']}

Pump RPM:
{row['pump_rpm']}

Torque:
{row['torque']}

Rate of Penetration:
{row['rop']}

Recommended Response:
{row['recommended_response']}

Write a professional engineering report with the following sections:

1. Operational Summary
2. Possible Root Cause
3. Operational Risk
4. Immediate Actions
5. Longer-Term Recommendation

Keep the response under 250 words.
"""

    response = get_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
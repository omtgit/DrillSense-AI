import os
from dotenv import load_dotenv
from google import genai



# Verify this model name is still served before relying on it; override with GEMINI_MODEL.
DEFAULT_MODEL = "gemini-2.5-flash"


def model_name():
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def get_client(api_key=None):
    """Build the Gemini client on first use so the app starts without a key.

    A key typed into the sidebar wins over GEMINI_API_KEY from the environment.
    """
    load_dotenv()
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No Gemini API key: paste one in the sidebar or set GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


PROMPT = """Rewrite the engineering report below so it reads more smoothly.
Rules:
- Use only the facts in the report. Do not add numbers, causes, wells, dates or recommendations.
- Copy every number exactly as written. Do not round or convert units.
- Keep the same section headings, in the same order, as markdown "##" headings.
- Keep it under 250 words.

REPORT:
{report}
"""


def rephrase_report(template_report, api_key=None):
    """Ask Gemini to rephrase the template report. Returns (text, error_or_None).

    Falls back to the template text if the call fails or the reply contains a number
    that is not in the template. The key is never logged or included in error text.
    """
    from report import numbers_are_grounded

    try:
        response = get_client(api_key).models.generate_content(
            model=model_name(), contents=PROMPT.format(report=template_report)
        )
        text = response.text or ""
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if api_key:
            msg = msg.replace(api_key, "***")
        return template_report, f"Gemini call failed ({msg}); showing the template report."
    if not text.strip():
        return template_report, "Gemini returned no text; showing the template report."
    if not numbers_are_grounded(text, template_report):
        return template_report, "Gemini introduced numbers not in the facts; showing the template report."
    return text, None

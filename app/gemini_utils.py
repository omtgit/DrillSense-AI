import os
from dotenv import load_dotenv
from google import genai



# Not verified against a live key; override with GEMINI_MODEL or the sidebar field.
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def model_name():
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def is_model_not_found(exc):
    """True if the exception looks like an API 404 / model-not-found error."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = f"{type(exc).__name__} {exc}".lower()
    return code == 404 or "not_found" in text or "not found" in text or "404" in text


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


def rephrase_report(template_report, api_key=None, model=None):
    """Ask Gemini to rephrase the template report. Returns (text, error_or_None).

    Falls back to the template text if the call fails or the reply contains a number
    that is not in the template. The key is never logged or included in error text.
    """
    from report import numbers_are_grounded

    try:
        response = get_client(api_key).models.generate_content(
            model=(model or "").strip() or model_name(), contents=PROMPT.format(report=template_report)
        )
        text = response.text or ""
    except Exception as exc:
        if is_model_not_found(exc):
            used = (model or "").strip() or model_name()
            return template_report, (
                f"Model '{used}' was not found by the Gemini API. Try another model name in the "
                "sidebar; showing the template report."
            )
        msg = f"{type(exc).__name__}: {exc}"
        if api_key:
            msg = msg.replace(api_key, "***")
        return template_report, f"Gemini call failed ({msg}); showing the template report."
    if not text.strip():
        return template_report, "Gemini returned no text; showing the template report."
    if not numbers_are_grounded(text, template_report):
        return template_report, "Gemini introduced numbers not in the facts; showing the template report."
    return text, None

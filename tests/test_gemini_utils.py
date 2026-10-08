import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import gemini_utils  # noqa: E402


class NotFound(Exception):
    code = 404


def fake_client(exc=None, text="ok"):
    seen = {}

    def generate_content(model, contents):
        seen["model"] = model
        if exc:
            raise exc
        return SimpleNamespace(text=text)

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client, seen


def test_default_model_and_env_override(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert gemini_utils.model_name() == "gemini-3.5-flash-lite"
    monkeypatch.setenv("GEMINI_MODEL", "my-model")
    assert gemini_utils.model_name() == "my-model"


def test_explicit_model_wins(monkeypatch):
    client, seen = fake_client(text="same")
    monkeypatch.setattr(gemini_utils, "get_client", lambda key=None: client)
    gemini_utils.rephrase_report("report", "k", model="custom-model")
    assert seen["model"] == "custom-model"


def test_not_found_gives_clear_message(monkeypatch):
    client, _ = fake_client(exc=NotFound("404 NOT_FOUND: models/x is not found"))
    monkeypatch.setattr(gemini_utils, "get_client", lambda key=None: client)
    text, err = gemini_utils.rephrase_report("report", "secret-key", model="x")
    assert text == "report"
    assert "try another model name" in err.lower()
    assert "Traceback" not in err and "secret-key" not in err


def test_other_errors_still_fall_back(monkeypatch):
    client, _ = fake_client(exc=RuntimeError("boom"))
    monkeypatch.setattr(gemini_utils, "get_client", lambda key=None: client)
    text, err = gemini_utils.rephrase_report("report", "k")
    assert text == "report" and "failed" in err


REPORT = "## Operational Summary\nWell W1 risk 87.5 from 10:00 to 10:20 (21 readings)."


def _patch(monkeypatch, text, seen=None):
    client, s = fake_client(text=text)

    def gen(model, contents):
        if seen is not None:
            seen.append(contents)
        return SimpleNamespace(text=text)

    client.models.generate_content = gen
    monkeypatch.setattr(gemini_utils, "get_client", lambda key=None: client)


def test_write_for_includes_audience_and_language(monkeypatch):
    seen = []
    _patch(monkeypatch, "Handover: W1 risk 87.5, 21 readings.", seen)
    text, err = gemini_utils.write_for(REPORT, "Shift handover note", "Hindi", "k")
    assert err is None and "87.5" in text
    assert "Hindi" in seen[0] and "handover note" in seen[0] and REPORT in seen[0]
    assert "Do not add numbers, causes" in seen[0]


def test_write_for_rejects_invented_numbers(monkeypatch):
    _patch(monkeypatch, "W1 risk 87.5 and 99 readings.")
    text, err = gemini_utils.write_for(REPORT, "Management summary", "English", "k")
    assert text == REPORT and "numbers" in err


def test_answer_question_grounded_and_refusal_in_prompt(monkeypatch):
    seen = []
    _patch(monkeypatch, "The risk score is 87.5.", seen)
    ans, err = gemini_utils.answer_question(REPORT, "What is the risk?", "English", "k")
    assert err is None and "87.5" in ans
    assert gemini_utils.NOT_IN_DATA in seen[0] and "untrusted" in seen[0]


def test_answer_question_rejects_ungrounded_and_hides_key(monkeypatch):
    _patch(monkeypatch, "It is 4242.")
    ans, err = gemini_utils.answer_question(REPORT, "q", "English", "k")
    assert ans is None and "numbers" in err
    client, _ = fake_client(exc=RuntimeError("bad key sk-secret"))
    monkeypatch.setattr(gemini_utils, "get_client", lambda key=None: client)
    ans, err = gemini_utils.answer_question(REPORT, "q", "English", "sk-secret")
    assert ans is None and "sk-secret" not in err


def test_has_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_utils, "load_dotenv", lambda: None)
    assert not gemini_utils.has_key(None)
    assert gemini_utils.has_key("k")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert gemini_utils.has_key(None)


def _closed_client_guard(monkeypatch):
    """Make the real SDK refuse requests once its HTTP client is closed; no network is used."""
    import json
    from google.genai import _api_client

    def request(self, http_method, path, request_dict, http_options=None):
        if self._httpx_client.is_closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        body = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok 87.5"}]}}]}
        return _api_client.SdkHttpResponse(headers={}, body=json.dumps(body))

    monkeypatch.setattr(_api_client.BaseApiClient, "request", request)


def test_real_client_is_not_closed_before_the_request(monkeypatch):
    """Regression: get_client(key).models.generate_content(...) let the Client be garbage-collected
    (closing its HTTP client) before the request went out."""
    _closed_client_guard(monkeypatch)
    text, err = gemini_utils.rephrase_report("report 87.5", "dummy-key")
    assert err is None and text == "ok 87.5"
    ans, err = gemini_utils.answer_question("report 87.5", "what?", "English", "dummy-key")
    assert err is None and ans == "ok 87.5"


def test_old_call_pattern_fails_with_the_real_client(monkeypatch):
    """Documents the bug: the temporary-Client pattern hits the closed-client error."""
    _closed_client_guard(monkeypatch)
    try:
        gemini_utils.get_client("dummy-key").models.generate_content(model="m", contents="x")
    except RuntimeError as exc:
        assert "closed" in str(exc)
    else:
        raise AssertionError("expected the temporary-Client pattern to fail")


def test_answer_accepts_numbers_from_the_question(monkeypatch):
    _patch(monkeypatch, "No, 500 is not the risk score; it is 87.5.")
    ans, err = gemini_utils.answer_question(REPORT, "Is the risk 500?", "English", "k")
    assert err is None and "500" in ans
    # ...but a number in neither the report nor the question is still rejected.
    _patch(monkeypatch, "It is 501.")
    ans, err = gemini_utils.answer_question(REPORT, "Is the risk 500?", "English", "k")
    assert ans is None and "numbers" in err


def test_hindi_refusal_is_shown_in_hindi(monkeypatch):
    seen = []
    _patch(monkeypatch, gemini_utils.NOT_IN_DATA, seen)
    ans, err = gemini_utils.answer_question(REPORT, "Who is the CEO?", "Hindi", "k")
    assert err is None and ans == gemini_utils.NOT_IN_DATA_HI
    assert gemini_utils.NOT_IN_DATA_HI in seen[0]
    ans, _ = gemini_utils.answer_question(REPORT, "Who is the CEO?", "English", "k")
    assert ans == gemini_utils.NOT_IN_DATA


def test_devanagari_digits_are_normalised(monkeypatch):
    _patch(monkeypatch, "जोखिम स्कोर ८७.५ है और २१ रीडिंग हैं।")
    ans, err = gemini_utils.answer_question(REPORT, "जोखिम क्या है?", "Hindi", "k")
    assert err is None and "८७.५" in ans
    _patch(monkeypatch, "जोखिम स्कोर ९९ है।")
    ans, err = gemini_utils.answer_question(REPORT, "जोखिम क्या है?", "Hindi", "k")
    assert ans is None and "numbers" in err

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

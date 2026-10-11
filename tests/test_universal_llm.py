"""Unit tests for universal LLM provider layer (Phase 4f).

Tests:
- Dynamic model discovery & filtering
- Automatic model ranking / recommendation across various model lists
- Fallback chain on errors
- Localhost-only HTTP check for OpenAI-compatible provider
- Zero API key leaks in error messages or technical details
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from llm.base import BaseProvider, GenerationResult, ModelInfo
from llm.gemini_provider import (
    GeminiProvider,
    parse_model_score,
    rank_models,
    sanitize_message,
)
from llm.openai_provider import OpenAIProvider, validate_base_url
import gemini_utils


def test_model_ranking_heuristics():
    # 1. Flash-lite preferred over flash and pro in same generation
    candidates = ["gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
    assert rank_models(candidates)[0] == "gemini-2.5-flash-lite"
    assert rank_models(candidates)[1] == "gemini-2.5-flash"
    assert rank_models(candidates)[2] == "gemini-2.5-pro"

    # 2. Newer generation preferred over older generation
    candidates_gen = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    assert rank_models(candidates_gen)[0] == "gemini-2.5-flash"

    # 3. Stable general-release preferred over preview/experimental
    candidates_preview = ["gemini-2.5-flash-preview", "gemini-2.0-flash", "gemini-2.5-flash-exp"]
    # Even if 2.5 is higher gen, non-preview 2.0 is preferred over 2.5 preview
    assert rank_models(candidates_preview)[0] == "gemini-2.0-flash"

    # 4. Unknown future versions (e.g. gemini-4.0-flash) cleanly win
    future_candidates = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-4.0-flash"]
    assert rank_models(future_candidates)[0] == "gemini-4.0-flash"


def test_gemini_model_discovery_filtering():
    raw_models = [
        SimpleNamespace(name="models/gemini-2.5-flash-lite", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-2.5-pro", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/text-embedding-004", supported_actions=["embedContent"]),
        SimpleNamespace(name="models/imagen-3.0-generate", supported_actions=["generateImages"]),
        SimpleNamespace(name="models/gemini-live-audio", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-2.0-flash", supported_actions=["generateContent"]),
    ]

    class FakeModelsApi:
        def list(self):
            return raw_models

    fake_client = SimpleNamespace(models=FakeModelsApi())
    provider = GeminiProvider(api_key="fake-test-key")
    provider._get_client = lambda: fake_client

    discovered = provider.list_models()
    ids = [m.id for m in discovered]

    # Non-text / embedding / imagen / audio must be excluded
    assert "text-embedding-004" not in ids
    assert "imagen-3.0-generate" not in ids
    assert "gemini-live-audio" not in ids

    # Text models present and sorted (2.5-flash-lite, 2.5-pro, 2.0-flash)
    assert ids == ["gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.0-flash"]
    assert discovered[0].is_recommended is True


def test_openai_validate_base_url():
    # Localhost allowed with http
    assert validate_base_url("http://localhost:11434/v1")[0] is True
    assert validate_base_url("http://127.0.0.1:8000/v1")[0] is True
    assert validate_base_url("http://0.0.0.0:8000/v1")[0] is True

    # Remote plain http rejected
    ok, reason = validate_base_url("http://api.together.xyz/v1")
    assert ok is False
    assert "Plain http://" in reason

    # Remote https allowed
    assert validate_base_url("https://api.openai.com/v1")[0] is True
    assert validate_base_url("https://openrouter.ai/api/v1")[0] is True

    # Bad scheme rejected
    assert validate_base_url("ftp://localhost:8000")[0] is False


def test_openai_provider_generate_with_mock_client(monkeypatch):
    import httpx

    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "OpenAI answer 42."}}
                ]
            }

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers=None, json=None):
            return MockResponse()

    monkeypatch.setattr(httpx, "Client", MockClient)

    p = OpenAIProvider(base_url="http://localhost:11434/v1", api_key="test-key", model="llama3")
    res = p.generate("Hello world")
    assert res.error is None
    assert res.text == "OpenAI answer 42."
    assert res.model == "llama3"
    assert res.provider == "OpenAI-compatible"


def test_key_is_never_logged_or_in_error_messages():
    secret_key = "AIzaSySecretApiKey123456789"

    # Test sanitizer
    msg = f"Error communicating with server using key {secret_key}."
    clean = sanitize_message(msg, secret_key)
    assert secret_key not in clean
    assert "***" in clean

    # Test OpenAI provider error
    p = OpenAIProvider(base_url="http://bad.remote.host/v1", api_key=secret_key)
    res = p.generate("test")
    assert secret_key not in (res.error or "")
    assert secret_key not in (res.detail or "")


def test_fallback_candidate_used_in_guarded_call(monkeypatch):
    class ApiError(Exception):
        code = 404

    attempts = []

    def mock_gen(model, contents):
        attempts.append(model)
        if model == "primary-model":
            raise ApiError("Model primary-model not found")
        return SimpleNamespace(text="Output from fallback 87.5.")

    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=mock_gen))
    monkeypatch.setattr(gemini_utils, "get_client", lambda key=None: fake_client)

    text, err = gemini_utils._guarded_call(
        prompt="Explain well 87.5",
        source="Well 87.5",
        fallback="Fallback template",
        api_key="k",
        model="primary-model",
        fallback_candidate="secondary-model",
    )

    assert err is None
    assert text == "Output from fallback 87.5."
    assert text.model == "secondary-model"
    assert attempts == ["primary-model", "secondary-model"]

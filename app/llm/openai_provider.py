"""OpenAI-compatible provider using httpx directly (no additional SDK dependencies).

Allows connecting to local servers (e.g. Ollama, vLLM, LM Studio) or third-party OpenAI-compatible endpoints.
Enforces security check: plain HTTP allowed ONLY for localhost / 127.0.0.1.
Never logs or reveals the API key in error messages.
"""
import os
import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from llm.base import BaseProvider, GenerationResult, ModelInfo


def sanitize_message(text: str, api_key: Optional[str]) -> str:
    """Ensure the API key never appears in any string, replacing it with ***."""
    if not text or not api_key:
        return text or ""
    return text.replace(api_key, "***")


def validate_base_url(url: str) -> Tuple[bool, str]:
    """Validate that plain HTTP is only used for localhost/127.0.0.1.

    Returns (is_valid, error_reason).
    """
    if not url:
        return False, "Base URL cannot be empty."
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()

    if scheme not in ("http", "https"):
        return False, f"Invalid scheme '{scheme}'. Must be http:// or https://."

    if scheme == "http":
        is_local = hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
        if not is_local:
            return False, "Plain http:// is only permitted for localhost (e.g. http://localhost:11434/v1). Remote endpoints must use https://."

    return True, ""


class OpenAIProvider(BaseProvider):
    """OpenAI-compatible HTTP provider using httpx."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or "").strip() or os.getenv("OPENAI_BASE_URL", "").strip() or "http://localhost:11434/v1"
        self.api_key = (api_key or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        self.default_model = (model or "").strip() or os.getenv("OPENAI_MODEL", "").strip() or "llama3"

    @property
    def name(self) -> str:
        return "OpenAI-compatible"

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> List[ModelInfo]:
        """Fetch model list from /models endpoint."""
        valid, reason = validate_base_url(self.base_url)
        if not valid:
            raise ValueError(reason)

        url = self.base_url.rstrip("/") + "/models"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, headers=self._get_headers())
                if resp.status_code != 200:
                    return [ModelInfo(id=self.default_model, display_name=self.default_model, is_recommended=True)]
                data = resp.json()
                items = data.get("data", [])
                models = []
                for item in items:
                    mid = item.get("id")
                    if mid:
                        models.append(ModelInfo(id=mid, display_name=mid, is_recommended=(mid == self.default_model)))
                if not models:
                    models = [ModelInfo(id=self.default_model, display_name=self.default_model, is_recommended=True)]
                return models
        except Exception:
            return [ModelInfo(id=self.default_model, display_name=self.default_model, is_recommended=True)]

    def generate(self, prompt: str, model: Optional[str] = None) -> GenerationResult:
        """Call /chat/completions endpoint."""
        valid, reason = validate_base_url(self.base_url)
        if not valid:
            return GenerationResult(
                text="",
                model=model or self.default_model,
                provider=self.name,
                error=reason,
                detail=reason,
            )

        used_model = (model or "").strip() or self.default_model
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": used_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=self._get_headers(), json=payload)
                if resp.status_code == 404:
                    msg = f"Model '{used_model}' or endpoint not found on {self.base_url}. Showing template report."
                    return GenerationResult(text="", model=used_model, provider=self.name, error=msg, detail=sanitize_message(resp.text, self.api_key))
                if resp.status_code == 429:
                    msg = f"Rate limit reached on {self.base_url}. Try again shortly."
                    return GenerationResult(text="", model=used_model, provider=self.name, error=msg, detail=sanitize_message(resp.text, self.api_key))
                if resp.status_code >= 400:
                    msg = f"Endpoint returned HTTP {resp.status_code}. Check the URL and model name."
                    return GenerationResult(text="", model=used_model, provider=self.name, error=msg, detail=sanitize_message(resp.text, self.api_key))

                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return GenerationResult(text="", model=used_model, provider=self.name, error="Endpoint returned no completion choices.", detail="Empty choices list.")
                text = choices[0].get("message", {}).get("content", "")
                return GenerationResult(text=text, model=used_model, provider=self.name)
        except Exception as exc:
            err_str = sanitize_message(f"{type(exc).__name__}: {exc}", self.api_key)
            if "refused" in err_str.lower() or "connecterror" in err_str.lower():
                msg = f"Failed to connect to {self.base_url}. If using Ollama, make sure Ollama is running ('ollama run {used_model}' in terminal). Showing template report."
            else:
                msg = f"Failed to connect to {self.base_url}. Showing template report."
            return GenerationResult(text="", model=used_model, provider=self.name, error=msg, detail=err_str)

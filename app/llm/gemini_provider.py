"""Gemini provider using google-genai SDK.

Discovers models dynamically, filters text models, scores and selects recommended default,
and executes generate requests with retries and fallback.
"""
import os
import re
import time
from typing import List, Optional, Tuple

from google import genai
from llm.base import BaseProvider, GenerationResult, ModelInfo

RETRY_CODES = {429, 500, 503}
BACKOFF_SECONDS = (1, 2, 4)
_sleep = time.sleep


def error_code(exc: Exception) -> Optional[int]:
    """HTTP status of an API error: from exception attributes or string match."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    text = f"{type(exc).__name__} {exc}"
    for word, number in (("RESOURCE_EXHAUSTED", 429), ("UNAVAILABLE", 503), ("INTERNAL", 500)):
        if word in text:
            return number
    m = re.search(r"\b(4\d\d|5\d\d)\b", text)
    return int(m.group(1)) if m else None


def is_not_found(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = f"{type(exc).__name__} {exc}".lower()
    return code == 404 or "not_found" in text or "not found" in text or "404" in text


def sanitize_message(text: str, api_key: Optional[str]) -> str:
    """Ensure the API key never appears in any string, replacing it with ***."""
    if not text or not api_key:
        return text or ""
    return text.replace(api_key, "***")


def parse_model_score(name: str) -> Tuple[int, int, int]:
    """Score a Gemini model name to rank best defaults without hard-coding specific models.

    Returns:
        (preview_score, generation_score, tier_score)
        where higher values are preferred.
    """
    clean = name.lower()
    if clean.startswith("models/"):
        clean = clean[len("models/"):]

    # 1. Preview/Experimental penalty: prefer non-preview/experimental
    # 1 = stable/general release, 0 = preview/experimental
    is_preview = bool(re.search(r"(preview|exp\b|experimental)", clean))
    preview_score = 0 if is_preview else 1

    # 2. Generation / Version number: extract version like 2.5, 3.0, 1.5, 2
    # Convert major.minor into integer score (e.g. 2.5 -> 250, 1.5 -> 150)
    gen_match = re.search(r"gemini[^\d]*(\d+)(?:\.(\d+))?", clean)
    if gen_match:
        major = int(gen_match.group(1))
        minor = int(gen_match.group(2)) if gen_match.group(2) else 0
        gen_score = major * 100 + minor
    else:
        gen_score = 0

    # 3. Tier preference: Flash-Lite > Flash > Pro > Other
    if "flash-lite" in clean or "flash_lite" in clean:
        tier_score = 3
    elif "flash" in clean:
        tier_score = 2
    elif "pro" in clean:
        tier_score = 1
    else:
        tier_score = 0

    return (preview_score, gen_score, tier_score)


def rank_models(model_names: List[str]) -> List[str]:
    """Sort model names from highest preference to lowest."""
    return sorted(model_names, key=parse_model_score, reverse=True)


class GeminiProvider(BaseProvider):
    """Google Gemini provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()

    @property
    def name(self) -> str:
        return "Gemini"

    def _get_client(self) -> genai.Client:
        if not self.api_key:
            raise RuntimeError("No Gemini API key: paste one in the sidebar or set GEMINI_API_KEY.")
        return genai.Client(api_key=self.api_key)

    def list_models(self) -> List[ModelInfo]:
        """Discover models via client.models.list(), keep text models, choose auto default."""
        client = self._get_client()
        raw_list = client.models.list()

        discovered = []
        for m in raw_list:
            m_id = getattr(m, "name", "") or getattr(m, "id", "")
            if not m_id:
                continue
            # Strip models/ prefix if present for uniform identifiers
            clean_id = m_id[len("models/"):] if m_id.startswith("models/") else m_id
            lower_id = clean_id.lower()

            # Filter supported_actions
            actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
            if actions:
                actions_str = [str(a).lower() for a in actions]
                if not any("generatecontent" in a for a in actions_str):
                    continue

            # Drop embedding, image, audio, speech, tts, live, etc.
            if any(term in lower_id for term in ("embedding", "embed", "imagen", "image", "audio", "speech", "tts", "whisper", "live")):
                continue

            # Keep only text gemini models
            if "gemini" not in lower_id:
                continue

            disp_name = getattr(m, "display_name", "") or clean_id
            desc = getattr(m, "description", "") or ""
            discovered.append(clean_id)

        if not discovered:
            # Fallback if list returned nothing matching
            discovered = ["gemini-2.5-flash"]

        ranked = rank_models(discovered)
        recommended_id = ranked[0]

        models_info = []
        for mid in ranked:
            models_info.append(
                ModelInfo(
                    id=mid,
                    display_name=mid,
                    description="",
                    is_recommended=(mid == recommended_id)
                )
            )
        return models_info

    def pick_default_model(self, available_models: Optional[List[str]] = None) -> str:
        """Choose the best default model from the available list."""
        if not available_models:
            models = self.list_models()
            available_models = [m.id for m in models]
        ranked = rank_models(available_models)
        return ranked[0] if ranked else "gemini-2.5-flash"

    def generate(self, prompt: str, model: Optional[str] = None, fallback_candidate: Optional[str] = None) -> GenerationResult:
        """Generate text using Gemini SDK, retrying on 429/500/503 and falling back if needed."""
        client = self._get_client()
        used = (model or "").strip()
        if not used or used.lower() == "auto":
            try:
                models = self.list_models()
                used = self.pick_default_model([m.id for m in models])
            except Exception:
                used = "gemini-2.5-flash"

        def _call_single(target_model: str):
            for attempt in range(len(BACKOFF_SECONDS) + 1):
                try:
                    return client.models.generate_content(model=target_model, contents=prompt)
                except Exception as exc:
                    if error_code(exc) not in RETRY_CODES or attempt == len(BACKOFF_SECONDS):
                        raise
                    _sleep(BACKOFF_SECONDS[attempt])

        try:
            resp = _call_single(used)
            text = resp.text or ""
            return GenerationResult(text=text, model=used, provider=self.name)
        except Exception as exc:
            # Check if fallback candidate is available
            can_fallback = (
                (error_code(exc) in RETRY_CODES or is_not_found(exc))
                and fallback_candidate
                and fallback_candidate != used
            )
            if can_fallback:
                try:
                    resp = client.models.generate_content(model=fallback_candidate, contents=prompt)
                    text = resp.text or ""
                    return GenerationResult(text=text, model=fallback_candidate, provider=self.name)
                except Exception as fallback_exc:
                    err_msg, detail = self._friendly_error(fallback_exc, fallback_candidate)
                    return GenerationResult(text="", model=fallback_candidate, provider=self.name, error=err_msg, detail=detail)

            err_msg, detail = self._friendly_error(exc, used)
            return GenerationResult(text="", model=used, provider=self.name, error=err_msg, detail=detail)

    def _friendly_error(self, exc: Exception, model: str) -> Tuple[str, str]:
        code = error_code(exc)
        detail = sanitize_message(f"{type(exc).__name__}: {exc}", self.api_key)
        if code == 404 or is_not_found(exc):
            msg = f"Model '{model}' was not found by the Gemini API. Try another model name in the sidebar; showing the template report."
        elif code == 429:
            msg = "This key has hit Google's free-tier limit for now. Wait a minute and try again, or use a different key."
        elif code in (500, 503):
            msg = "Google's model is busy right now. Try again in a minute, or change the model name in the sidebar."
        elif code in (400, 401, 403):
            msg = "Google did not accept the request. Check the API key and the model name in the sidebar."
        else:
            msg = "Gemini call failed; showing the template report."
        return msg, detail

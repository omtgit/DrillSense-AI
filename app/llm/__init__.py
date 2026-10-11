"""Universal LLM provider package for DrillSense AI.

Supports Gemini (google-genai) and OpenAI-compatible endpoints via httpx directly.
"""
from llm.base import BaseProvider, GenerationResult, ModelInfo
from llm.gemini_provider import GeminiProvider
from llm.openai_provider import OpenAIProvider

__all__ = ["BaseProvider", "GenerationResult", "ModelInfo", "GeminiProvider", "OpenAIProvider"]

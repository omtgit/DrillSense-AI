"""Base classes and data models for LLM providers."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelInfo:
    """Discovered or configured model metadata."""
    id: str
    display_name: str
    description: str = ""
    is_recommended: bool = False


@dataclass
class GenerationResult:
    """Result of an LLM generation call."""
    text: str
    model: str
    provider: str
    error: Optional[str] = None
    detail: Optional[str] = None


class BaseProvider:
    """Abstract base class for LLM providers."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    def list_models(self) -> List[ModelInfo]:
        """Discover or return available models."""
        raise NotImplementedError

    def generate(self, prompt: str, model: Optional[str] = None) -> GenerationResult:
        """Generate text completion for prompt."""
        raise NotImplementedError

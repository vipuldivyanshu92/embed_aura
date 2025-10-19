"""Base interface for Small Language Model clients."""

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from app.models import Hypothesis, MediaType


class GeneratedAnswer(TypedDict):
    """Structured answer payload returned by SLM clients."""

    answer: str
    supporting_points: list[str]
    confidence: float


class SLMClient(ABC):
    """Abstract base class for SLM client implementations."""

    @abstractmethod
    async def generate_hypotheses(
        self,
        user_input: str | None,
        context: dict[str, Any],
        count: int = 3,
        media_type: MediaType = MediaType.TEXT,
        media_url: str | None = None,
        media_base64: str | None = None,
    ) -> list[Hypothesis]:
        """
        Generate hypotheses about user intent (multi-modal support).

        Args:
            user_input: The user's text input
            context: Context including memories, persona facets, etc.
            count: Number of hypotheses to generate (2-3)
            media_type: Type of media (text, image, audio, video)
            media_url: URL to media file
            media_base64: Base64 encoded media

        Returns:
            List of Hypothesis objects sorted by confidence
        """
        pass

    @abstractmethod
    async def summarize(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """
        Summarize text to fit within token budget.

        Args:
            text: Text to summarize
            max_tokens: Maximum tokens for summary

        Returns:
            Summarized text
        """
        pass

    @abstractmethod
    async def generate_answer(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> GeneratedAnswer:
        """Generate a structured answer from an enriched prompt."""
        pass

    @abstractmethod
    async def describe_media(
        self,
        media_type: MediaType,
        media_url: str | None = None,
        media_base64: str | None = None,
    ) -> dict[str, Any]:
        """Generate a structured description and classification tags for media inputs.

        Args:
            media_type: Type of media (image, audio, video, etc.)
            media_url: Optional URL to the media resource
            media_base64: Optional base64 encoded media payload

        Returns:
            Dictionary containing keys like ``caption`` and ``tags`` that describe
            the media content. Implementations should return best-effort results
            and may fall back to generic placeholders if detailed analysis is
            unavailable.
        """
        pass

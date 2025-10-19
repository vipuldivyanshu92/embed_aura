"""Base interface for Small Language Model clients."""

from abc import ABC, abstractmethod
from typing import Any

from app.models import Hypothesis, MediaType


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

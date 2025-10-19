"""Base interface for Small Language Model clients."""

from abc import ABC, abstractmethod
from typing import Any

from app.models import Hypothesis


class SLMClient(ABC):
    """Abstract base class for SLM client implementations."""

    @abstractmethod
    async def generate_hypotheses(
        self,
        user_input: str,
        context: dict[str, Any],
        count: int = 3,
    ) -> list[Hypothesis]:
        """
        Generate hypotheses about user intent.

        Args:
            user_input: The user's short input text
            context: Context including memories, persona facets, etc.
            count: Number of hypotheses to generate (2-3)

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

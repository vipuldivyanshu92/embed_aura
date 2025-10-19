"""HTTP-based SLM client for remote small language models."""

from typing import Any

import httpx
import structlog

from app.clients.slm.base import SLMClient
from app.clients.slm.local import LocalSLM
from app.models import Hypothesis

logger = structlog.get_logger()


class HttpSLM(SLMClient):
    """
    HTTP client for remote Small Language Model.

    Falls back to LocalSLM if the remote service is unreachable.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0) -> None:
        """
        Initialize HTTP SLM client.

        Args:
            base_url: Base URL of the SLM service
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.fallback = LocalSLM()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def generate_hypotheses(
        self,
        user_input: str,
        context: dict[str, Any],
        count: int = 3,
    ) -> list[Hypothesis]:
        """
        Generate hypotheses via HTTP API.

        Args:
            user_input: User's input text
            context: Context dictionary
            count: Number of hypotheses to generate

        Returns:
            List of hypotheses
        """
        try:
            client = await self._get_client()

            payload = {
                "input": user_input,
                "context": context,
                "count": count,
                "task": "hypothesize",
            }

            response = await client.post(
                f"{self.base_url}/v1/generate",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            hypotheses = [
                Hypothesis(
                    id=h.get("id", f"h{i+1}"),
                    question=h["question"],
                    rationale=h.get("rationale", ""),
                    confidence=h.get("confidence", 0.5),
                )
                for i, h in enumerate(data.get("hypotheses", []))
            ]

            return hypotheses[:count]

        except (httpx.HTTPError, httpx.TimeoutException, KeyError) as e:
            logger.warning(
                "slm_http_failed_fallback_to_local",
                error=str(e),
                endpoint="generate_hypotheses",
            )
            # Fallback to local
            return await self.fallback.generate_hypotheses(user_input, context, count)

    async def summarize(self, text: str, max_tokens: int) -> str:
        """
        Summarize text via HTTP API.

        Args:
            text: Text to summarize
            max_tokens: Maximum token budget

        Returns:
            Summarized text
        """
        try:
            client = await self._get_client()

            payload = {
                "text": text,
                "max_tokens": max_tokens,
                "task": "summarize",
            }

            response = await client.post(
                f"{self.base_url}/v1/generate",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            return data.get("summary", text)

        except (httpx.HTTPError, httpx.TimeoutException, KeyError) as e:
            logger.warning(
                "slm_http_failed_fallback_to_local",
                error=str(e),
                endpoint="summarize",
            )
            # Fallback to local
            return await self.fallback.summarize(text, max_tokens)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

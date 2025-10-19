"""Telemetry and observability service."""

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog

logger = structlog.get_logger()


class TelemetryService:
    """
    Handles structured logging, request tracking, and metrics.
    """

    def __init__(self) -> None:
        """Initialize telemetry service."""
        self.hypothesis_ctr: dict[str, dict[str, Any]] = {}
        # Stores: hypothesis_id -> {shown_at, selected_at, user_id}

    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())

    @asynccontextmanager
    async def track_request(
        self, request_id: str, endpoint: str, **kwargs: Any
    ) -> AsyncIterator[None]:
        """
        Context manager to track request timing and outcome.

        Args:
            request_id: Unique request identifier
            endpoint: Endpoint name
            **kwargs: Additional context fields

        Yields:
            None
        """
        start_time = time.time()

        logger.info(
            "request_started",
            request_id=request_id,
            endpoint=endpoint,
            **kwargs,
        )

        try:
            yield
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "request_failed",
                request_id=request_id,
                endpoint=endpoint,
                duration_ms=int(duration * 1000),
                error=str(e),
                **kwargs,
            )
            raise
        else:
            duration = time.time() - start_time
            logger.info(
                "request_completed",
                request_id=request_id,
                endpoint=endpoint,
                duration_ms=int(duration * 1000),
                **kwargs,
            )

    def track_hypothesis_shown(self, user_id: str, hypothesis_ids: list[str]) -> None:
        """
        Track when hypotheses are shown to a user.

        Args:
            user_id: User identifier
            hypothesis_ids: List of hypothesis IDs shown
        """
        timestamp = time.time()

        for hyp_id in hypothesis_ids:
            self.hypothesis_ctr[hyp_id] = {
                "user_id": user_id,
                "shown_at": timestamp,
                "selected_at": None,
            }

        logger.debug(
            "hypotheses_tracked",
            user_id=user_id,
            count=len(hypothesis_ids),
        )

    def track_hypothesis_selected(self, user_id: str, hypothesis_id: str) -> None:
        """
        Track when a hypothesis is selected.

        Args:
            user_id: User identifier
            hypothesis_id: Selected hypothesis ID
        """
        timestamp = time.time()

        if hypothesis_id in self.hypothesis_ctr:
            shown_at = self.hypothesis_ctr[hypothesis_id].get("shown_at")
            self.hypothesis_ctr[hypothesis_id]["selected_at"] = timestamp

            # Calculate CTR metrics
            if shown_at:
                time_to_select = timestamp - shown_at
                logger.info(
                    "hypothesis_selected",
                    user_id=user_id,
                    hypothesis_id=hypothesis_id,
                    time_to_select_ms=int(time_to_select * 1000),
                )
        else:
            logger.warning(
                "hypothesis_selected_without_show",
                user_id=user_id,
                hypothesis_id=hypothesis_id,
            )

    def get_hypothesis_ctr(self) -> dict[str, float]:
        """
        Calculate click-through rate for hypotheses.

        Returns:
            Dictionary of hypothesis position -> CTR
        """
        position_stats: dict[str, dict[str, int]] = {
            "h1": {"shown": 0, "selected": 0},
            "h2": {"shown": 0, "selected": 0},
            "h3": {"shown": 0, "selected": 0},
        }

        for hyp_id, data in self.hypothesis_ctr.items():
            if hyp_id in position_stats:
                position_stats[hyp_id]["shown"] += 1
                if data.get("selected_at") is not None:
                    position_stats[hyp_id]["selected"] += 1

        # Calculate CTR
        ctr = {}
        for position, stats in position_stats.items():
            if stats["shown"] > 0:
                ctr[position] = stats["selected"] / stats["shown"]
            else:
                ctr[position] = 0.0

        return ctr

    def log_token_estimate(self, context: str, tokens: int, **kwargs: Any) -> None:
        """
        Log token estimation for monitoring.

        Args:
            context: Context identifier (e.g., "enriched_prompt")
            tokens: Estimated token count
            **kwargs: Additional context
        """
        logger.debug(
            "token_estimate",
            context=context,
            tokens=tokens,
            **kwargs,
        )

"""Hypothesis generation service."""

from typing import Any

import structlog

from app.clients.slm.base import SLMClient
from app.config import get_settings
from app.memory.base import MemoryProvider
from app.models import Hypothesis, MediaType, MemoryType
from app.services.persona import PersonaService

logger = structlog.get_logger()


class HypothesizerService:
    """
    Generates hypotheses about user intent from short inputs.
    """

    def __init__(
        self,
        slm_client: SLMClient,
        memory_provider: MemoryProvider,
        persona_service: PersonaService,
    ) -> None:
        """
        Initialize hypothesizer service.

        Args:
            slm_client: SLM client for hypothesis generation
            memory_provider: Memory provider for context retrieval
            persona_service: Persona service for user preferences
        """
        self.slm_client = slm_client
        self.memory_provider = memory_provider
        self.persona_service = persona_service
        self.settings = get_settings()

    async def generate_hypotheses(
        self,
        user_id: str,
        input_text: str | None = None,
        media_type: MediaType = MediaType.TEXT,
        media_url: str | None = None,
        media_base64: str | None = None,
        count: int = 3,
    ) -> tuple[list[Hypothesis], bool]:
        """
        Generate hypotheses about user intent (multi-modal support).

        Args:
            user_id: User identifier
            input_text: User's text input
            media_type: Type of media (text, image, audio, video)
            media_url: URL to media file
            media_base64: Base64 encoded media
            count: Number of hypotheses to generate (max 3)

        Returns:
            Tuple of (hypotheses list, auto_advance flag)
        """
        # Limit to max 3
        count = min(count, 3)

        # Get context
        context = await self._build_context(
            user_id, input_text, media_type, media_url, media_base64
        )

        # Generate hypotheses via SLM
        hypotheses = await self.slm_client.generate_hypotheses(
            input_text, context, count, media_type, media_url, media_base64
        )

        # Check auto-advance threshold
        auto_advance = False
        if hypotheses and hypotheses[0].confidence >= self.settings.auto_advance_confidence:
            auto_advance = True
            logger.info(
                "auto_advance_suggested",
                user_id=user_id,
                hypothesis_id=hypotheses[0].id,
                confidence=hypotheses[0].confidence,
            )

        # Log for learning loop
        await self._log_hypotheses_shown(user_id, hypotheses)

        return hypotheses, auto_advance

    async def _build_context(
        self,
        user_id: str,
        input_text: str | None,
        media_type: MediaType,
        media_url: str | None,
        media_base64: str | None,
    ) -> dict[str, Any]:
        """
        Build context for hypothesis generation (multi-modal support).

        Retrieves top-K memories of types: GOAL, PREFERENCE, STYLE
        and user persona facets.

        Args:
            user_id: User identifier
            input_text: User's text input
            media_type: Type of media
            media_url: URL to media
            media_base64: Base64 encoded media

        Returns:
            Context dictionary
        """
        # Get persona
        persona = await self.persona_service.get_or_create_persona(user_id)

        # Get relevant memories
        goals = await self.memory_provider.get_memories(user_id, MemoryType.GOAL, limit=3)
        preferences = await self.memory_provider.get_memories(
            user_id, MemoryType.PREFERENCE, limit=5
        )
        styles = await self.memory_provider.get_memories(user_id, MemoryType.STYLE, limit=3)

        context = {
            "persona_facets": persona.facets,
            "interaction_count": persona.interaction_count,
            "recent_goals": [m.content or m.media_description for m in goals if m.content or m.media_description],
            "preferences": [m.content or m.media_description for m in preferences if m.content or m.media_description],
            "styles": [m.content or m.media_description for m in styles if m.content or m.media_description],
            "media_type": media_type.value,
            "has_media": media_url is not None or media_base64 is not None,
        }

        return context

    async def _log_hypotheses_shown(self, user_id: str, hypotheses: list[Hypothesis]) -> None:
        """
        Log that hypotheses were shown to user.

        This creates a learning signal for future improvements.

        Args:
            user_id: User identifier
            hypotheses: List of hypotheses shown
        """
        logger.info(
            "hypotheses_generated",
            user_id=user_id,
            count=len(hypotheses),
            top_confidence=hypotheses[0].confidence if hypotheses else 0.0,
            hypothesis_ids=[h.id for h in hypotheses],
        )

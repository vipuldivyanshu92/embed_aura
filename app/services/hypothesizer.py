"""Hypothesis generation service."""

from typing import Any

import structlog

from app.clients.slm.base import SLMClient
from app.config import get_settings
from app.memory.base import MemoryProvider
from app.models import Hypothesis, MediaType, MemoryType
from app.services.persona import PersonaService
from app.utils.embeddings import generate_embedding

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
        count = min(count, 4)

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

        OPTIMIZED: Uses semantic search to find the most RELEVANT memories
        based on the current input, not just the most recent ones.

        Args:
            user_id: User identifier
            input_text: User's text input
            media_type: Type of media
            media_url: URL to media
            media_base64: Base64 encoded media

        Returns:
            Context dictionary with relevant memories
        """
        # Get persona
        persona = await self.persona_service.get_or_create_persona(user_id)

        # Generate embedding for semantic search
        query_embedding, query_description = generate_embedding(
            text=input_text,
            media_type=media_type,
            media_url=media_url,
            media_base64=media_base64,
        )

        # Use semantic search to find RELEVANT memories (not just recent)
        relevant_memories = await self.memory_provider.search_memories(
            user_id,
            query_embedding,
            limit=20,  # Get top 20 most relevant memories
        )

        # Categorize memories by type for better context building
        goals = []
        preferences = []
        styles = []
        history = []
        artifacts = []

        for memory in relevant_memories:
            if memory.mtype == MemoryType.GOAL:
                goals.append(memory)
            elif memory.mtype == MemoryType.PREFERENCE:
                preferences.append(memory)
            elif memory.mtype == MemoryType.STYLE:
                styles.append(memory)
            elif memory.mtype == MemoryType.HISTORY:
                history.append(memory)
            elif memory.mtype == MemoryType.ARTIFACT:
                artifacts.append(memory)

        # Build enriched context with relevant memories
        context = {
            "persona_facets": persona.facets,
            "interaction_count": persona.interaction_count,
            # Most relevant goals (semantically similar to current input)
            "relevant_goals": [
                m.content or m.media_description 
                for m in goals[:3] 
                if m.content or m.media_description
            ],
            # Most relevant preferences
            "relevant_preferences": [
                m.content or m.media_description 
                for m in preferences[:5] 
                if m.content or m.media_description
            ],
            # Style preferences
            "styles": [
                m.content or m.media_description 
                for m in styles[:3] 
                if m.content or m.media_description
            ],
            # Similar past interactions (helps understand patterns)
            "similar_history": [
                m.content or m.media_description 
                for m in history[:3] 
                if m.content or m.media_description
            ],
            # Relevant artifacts (code snippets, documents, etc.)
            "relevant_artifacts": [
                m.content or m.media_description 
                for m in artifacts[:2] 
                if m.content or m.media_description
            ],
            "media_type": media_type.value,
            "has_media": media_url is not None or media_base64 is not None,
            "total_relevant_memories": len(relevant_memories),
            # Content description (for images, this is the vision model's description)
            "content_description": query_description,
        }

        logger.debug(
            "hypothesis_context_built",
            user_id=user_id,
            goals=len(goals),
            preferences=len(preferences),
            history=len(history),
            artifacts=len(artifacts),
            total_memories=len(relevant_memories),
        )

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

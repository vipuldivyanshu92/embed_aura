"""Persona management and learning."""

from datetime import datetime
from typing import Any

import numpy as np
import structlog

from app.config import get_settings
from app.memory.base import MemoryProvider
from app.models import Persona
from app.clients.slm.base import SLMClient

logger = structlog.get_logger()


class PersonaService:
    """
    Manages user persona and preference learning.

    Maintains a persona vector (aggregated representation of user preferences)
    and facet sliders (concise, formal, code_first, etc.)
    """

    # Default facet values for new users
    DEFAULT_FACETS = {
        "concise": 0.5,
        "formal": 0.5,
        "code_first": 0.5,
        "step_by_step": 0.5,
    }

    def __init__(
        self, 
        memory_provider: MemoryProvider,
        slm_client: SLMClient | None = None,
    ) -> None:
        """
        Initialize persona service.

        Args:
            memory_provider: Memory provider for persona storage
            slm_client: Optional SLM client for enhanced learning (vLLM)
        """
        self.memory_provider = memory_provider
        self.slm_client = slm_client
        self.settings = get_settings()

    async def get_or_create_persona(self, user_id: str) -> Persona:
        """
        Get existing persona or create a new one.

        Args:
            user_id: User identifier

        Returns:
            Persona object
        """
        persona = await self.memory_provider.get_persona(user_id)

        if persona is None:
            # Create new persona with default values
            persona = Persona(
                user_id=user_id,
                vector=self._create_default_vector(),
                facets=self.DEFAULT_FACETS.copy(),
                last_updated=datetime.utcnow(),
                interaction_count=0,
            )
            await self.memory_provider.store_persona(persona)
            logger.info("created_new_persona", user_id=user_id)

        return persona

    def _create_default_vector(self) -> list[float]:
        """Create a default persona vector (small random initialization)."""
        embed_dims = self.settings.embed_dims
        # Small random vector near zero
        vec = np.random.normal(0, 0.01, embed_dims)
        return vec.tolist()

    async def update_persona(
        self,
        user_id: str,
        signals: dict[str, Any],
    ) -> Persona:
        """
        Update persona based on interaction signals.

        Signals can include:
        - selected_hypothesis_id: Which hypothesis was chosen
        - success: Boolean indicating task success
        - feedback: Explicit feedback
        - embedding: New embedding to incorporate

        Args:
            user_id: User identifier
            signals: Dictionary of learning signals

        Returns:
            Updated persona
        """
        persona = await self.get_or_create_persona(user_id)

        # Update interaction count
        persona.interaction_count += 1

        # Update vector if embedding provided
        if "embedding" in signals:
            new_embedding = signals["embedding"]
            persona.vector = self._update_vector(
                persona.vector, new_embedding, persona.interaction_count
            )

        # Update facets based on signals
        facet_updates = await self._compute_facet_updates(signals)
        for facet, delta in facet_updates.items():
            if facet in persona.facets:
                new_value = persona.facets[facet] + delta
                # Clamp to [0, 1]
                persona.facets[facet] = max(
                    self.settings.persona_facet_clamp_min,
                    min(self.settings.persona_facet_clamp_max, new_value),
                )

        persona.last_updated = datetime.utcnow()

        # Persist
        await self.memory_provider.store_persona(persona)

        logger.debug(
            "updated_persona",
            user_id=user_id,
            facets=persona.facets,
            interaction_count=persona.interaction_count,
        )

        return persona

    def _update_vector(
        self,
        current_vector: list[float],
        new_embedding: list[float],
        interaction_count: int,
    ) -> list[float]:
        """
        Update persona vector using recency-weighted moving average.

        Args:
            current_vector: Current persona vector
            new_embedding: New embedding to incorporate
            interaction_count: Total interaction count

        Returns:
            Updated vector
        """
        current = np.array(current_vector)
        new = np.array(new_embedding)

        # Recency weight: newer interactions have more weight
        alpha = self.settings.persona_update_rate

        # Weighted average
        updated = (1 - alpha) * current + alpha * new

        return updated.tolist()

    async def _compute_facet_updates(self, signals: dict[str, Any]) -> dict[str, float]:
        """
        Compute facet updates based on signals.

        Uses vLLM for intelligent facet learning if available,
        otherwise falls back to heuristic model.

        Args:
            signals: Interaction signals

        Returns:
            Dictionary of facet deltas
        """
        # Try vLLM-based learning if available
        if self.slm_client and hasattr(self.slm_client, 'generate_followup_questions'):
            try:
                updates = await self._vllm_compute_facet_updates(signals)
                if updates:
                    return updates
            except Exception as e:
                logger.warning("vllm_facet_learning_failed", error=str(e))
        
        # Fallback to heuristic model
        return self._heuristic_compute_facet_updates(signals)
    
    def _heuristic_compute_facet_updates(self, signals: dict[str, Any]) -> dict[str, float]:
        """Heuristic-based facet update computation (original logic)."""
        updates: dict[str, float] = {}
        learning_rate = self.settings.persona_update_rate

        # If user selected hypothesis #1 (typically most direct), increase concise
        if signals.get("selected_hypothesis_id") == "h1":
            updates["concise"] = learning_rate * 0.05

        # If user selected hypothesis #2 or #3, decrease concise slightly
        elif signals.get("selected_hypothesis_id") in ["h2", "h3"]:
            updates["concise"] = -learning_rate * 0.02

        # If success signal is high, reinforce current preferences
        if signals.get("success", False):
            # Minimal update - just a signal we're on track
            for facet in self.DEFAULT_FACETS:
                updates[facet] = updates.get(facet, 0.0) + learning_rate * 0.01

        # Explicit facet updates from signals
        explicit_facets = signals.get("facet_updates", {})
        for facet, delta in explicit_facets.items():
            updates[facet] = updates.get(facet, 0.0) + delta * learning_rate

        return updates
    
    async def _vllm_compute_facet_updates(self, signals: dict[str, Any]) -> dict[str, float] | None:
        """
        Use vLLM to intelligently compute facet updates based on signals.
        
        Args:
            signals: Interaction signals
            
        Returns:
            Dictionary of facet deltas or None if failed
        """
        import json
        
        prompt = f"""Analyze the following user interaction signals and determine how to update their preference facets.

Signals: {json.dumps(signals, indent=2)}

Available facets (range 0.0-1.0):
- concise: preference for brief vs detailed responses
- formal: preference for formal vs casual language
- code_first: preference for code examples vs explanations
- step_by_step: preference for step-by-step vs high-level guidance

Provide facet adjustments as small deltas (-0.1 to +0.1) in JSON format:
{{
  "facet_name": delta_value,
  ...
}}

Consider:
- If they selected hypothesis 1, they may prefer concise
- If they selected hypothesis 2-3, they may prefer detailed
- Success signals reinforce current preferences
- Explicit feedback should have strong impact

JSON Response:"""

        try:
            # Use the client's chat completion method
            response = await self.slm_client._chat_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200,
            )
            
            # Parse JSON response
            updates_raw = json.loads(response.strip())
            
            # Apply learning rate and validate
            learning_rate = self.settings.persona_update_rate
            updates = {}
            for facet, delta in updates_raw.items():
                if facet in self.DEFAULT_FACETS and isinstance(delta, (int, float)):
                    # Clamp delta to reasonable range
                    clamped_delta = max(-0.1, min(0.1, float(delta)))
                    updates[facet] = clamped_delta * learning_rate
            
            logger.debug("vllm_facet_updates_computed", updates=updates)
            return updates
            
        except Exception as e:
            logger.error("vllm_facet_update_error", error=str(e))
            return None

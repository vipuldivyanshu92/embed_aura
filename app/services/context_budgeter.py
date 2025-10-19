"""Context budgeting and token allocation."""

import structlog

from app.clients.slm.base import SLMClient
from app.config import get_settings
from app.models import ContextSection
from app.utils.tokens import estimate_tokens

logger = structlog.get_logger()


class ContextBudgeter:
    """
    Manages token budget allocation across context sections.

    Allocates tokens according to configured weights and summarizes
    sections that exceed their budget.
    """

    def __init__(self, slm_client: SLMClient) -> None:
        """
        Initialize context budgeter.

        Args:
            slm_client: SLM client for summarization
        """
        self.slm_client = slm_client
        self.settings = get_settings()
        self.weights = self.settings.get_budget_weights_dict()

    async def allocate_budget(
        self,
        sections: dict[str, str],
    ) -> tuple[list[ContextSection], dict[str, int]]:
        """
        Allocate token budget across sections.

        Args:
            sections: Dictionary mapping section name to content

        Returns:
            Tuple of (list of ContextSection objects, token breakdown dict)
        """
        total_budget = self.settings.token_budget

        # Map section names to weight keys
        section_weight_map = {
            "goal_summary": "goal",
            "preferences_style": "pref",
            "critical_artifacts": "artifacts",
            "recent_history": "history",
            "constraints": "constraints",
            "task_specific_retrieval": "task",
            "safety_system": "safety",
        }

        allocated_sections: list[ContextSection] = []
        breakdown: dict[str, int] = {}

        for section_name, content in sections.items():
            # Get weight for this section
            weight_key = section_weight_map.get(section_name, section_name)
            weight = self.weights.get(weight_key, 0.0)

            # Calculate allocation
            allocated_tokens = int(total_budget * weight)

            # Estimate current token count
            current_tokens = estimate_tokens(content)

            # Summarize if over budget
            summarized = False
            final_content = content

            if current_tokens > allocated_tokens and allocated_tokens > 0:
                logger.debug(
                    "section_over_budget",
                    section=section_name,
                    current=current_tokens,
                    allocated=allocated_tokens,
                )

                # Summarize to fit budget
                final_content = await self.slm_client.summarize(content, allocated_tokens)
                summarized = True

                logger.debug(
                    "section_summarized",
                    section=section_name,
                    original_tokens=current_tokens,
                    new_tokens=estimate_tokens(final_content),
                )

            final_tokens = estimate_tokens(final_content)

            section = ContextSection(
                name=section_name,
                content=final_content,
                tokens=final_tokens,
                allocated=allocated_tokens,
                summarized=summarized,
            )

            allocated_sections.append(section)
            breakdown[section_name] = final_tokens

        logger.info(
            "budget_allocated",
            total_budget=total_budget,
            total_used=sum(breakdown.values()),
            breakdown=breakdown,
        )

        return allocated_sections, breakdown

    def format_sections(self, sections: list[ContextSection]) -> str:
        """
        Format context sections into a single string.

        Args:
            sections: List of context sections

        Returns:
            Formatted context string
        """
        parts = []

        for section in sections:
            if section.content:
                # Add section header
                header = f"\n{'='*60}\n{section.name.upper()}\n{'='*60}\n"
                parts.append(header)
                parts.append(section.content)

        return "\n".join(parts)

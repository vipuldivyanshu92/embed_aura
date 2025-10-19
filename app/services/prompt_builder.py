"""Prompt construction from context sections."""

import structlog

from app.models import Hypothesis, MemoryType, Persona, RankedMemory
from app.services.context_budgeter import ContextBudgeter
from app.services.safety import SafetyService

logger = structlog.get_logger()


class PromptBuilder:
    """
    Builds enriched prompts from memories, persona, and user input.
    """

    PROMPT_TEMPLATE = """You are an expert teacher crafting a guided answer for {user_name}.

Primary goal:
{goal_summary}

User input:
{user_input}

Selected question (intent):
{selected_hypothesis}

Preferences & style:
{preferences_style}

Constraints:
{constraints}

Key context & artifacts:
{ranked_artifacts}

Recent history:
{recent_history}

Output contract:
{output_contract}

Instructions:
- Provide a clear, teacher-quality answer in markdown.
- Include 2-4 supporting bullet points referencing relevant memories when available.
- Return a JSON object with fields: answer (string), supporting_points (array of strings), confidence (0-1).
- Confidence should reflect how well the context supports your guidance.
"""

    def __init__(
        self,
        context_budgeter: ContextBudgeter,
        safety_service: SafetyService,
    ) -> None:
        """
        Initialize prompt builder.

        Args:
            context_budgeter: Context budgeting service
            safety_service: Safety/PII redaction service
        """
        self.context_budgeter = context_budgeter
        self.safety_service = safety_service

    async def build_prompt(
        self,
        user_id: str,
        user_input: str,
        selected_hypothesis: Hypothesis,
        ranked_memories: list[RankedMemory],
        persona: Persona,
    ) -> tuple[str, dict[str, int]]:
        """
        Build enriched prompt from inputs.

        Args:
            user_id: User identifier
            user_input: User's original input
            selected_hypothesis: Selected hypothesis
            ranked_memories: Ranked and filtered memories
            persona: User persona

        Returns:
            Tuple of (enriched_prompt, context_breakdown)
        """
        # Group memories by type
        memory_groups = self._group_memories_by_type(ranked_memories)

        # Build sections
        sections = {
            "goal_summary": self._build_goal_summary(memory_groups, persona),
            "preferences_style": self._build_preferences_style(memory_groups, persona),
            "critical_artifacts": self._build_artifacts(memory_groups),
            "recent_history": self._build_history(memory_groups),
            "constraints": self._build_constraints(persona),
            "task_specific_retrieval": self._build_task_specific(user_input, memory_groups),
            "safety_system": self._build_safety_system(),
        }

        # Apply safety redaction to all sections
        for key in sections:
            sections[key] = self.safety_service.redact_pii(sections[key])

        # Allocate budget
        allocated_sections, breakdown = await self.context_budgeter.allocate_budget(sections)

        # Extract section contents
        section_content = {s.name: s.content for s in allocated_sections}

        # Build final prompt
        enriched_prompt = self.PROMPT_TEMPLATE.format(
            user_name=user_id,
            goal_summary=section_content.get("goal_summary", "No specific goals defined."),
            user_input=user_input,
            selected_hypothesis=selected_hypothesis.question,
            preferences_style=section_content.get("preferences_style", "No specific preferences."),
            constraints=section_content.get("constraints", "Follow best practices."),
            ranked_artifacts=section_content.get("critical_artifacts", "No artifacts available."),
            recent_history=section_content.get("recent_history", "No recent history available."),
            output_contract=self._build_output_contract(persona),
        )

        return enriched_prompt, breakdown

    def _group_memories_by_type(
        self, ranked_memories: list[RankedMemory]
    ) -> dict[MemoryType, list[RankedMemory]]:
        """Group memories by type."""
        groups: dict[MemoryType, list[RankedMemory]] = {}

        for ranked in ranked_memories:
            mtype = ranked.memory.mtype
            if mtype not in groups:
                groups[mtype] = []
            groups[mtype].append(ranked)

        return groups

    def _build_goal_summary(
        self, memory_groups: dict[MemoryType, list[RankedMemory]], persona: Persona
    ) -> str:
        """Build goal summary section."""
        goals = memory_groups.get(MemoryType.GOAL, [])

        if not goals:
            return "No specific long-term goals defined. Focus on immediate task completion."

        parts = []
        for i, ranked in enumerate(goals[:3], 1):  # Top 3 goals
            parts.append(f"{i}. {ranked.memory.content}")

        return "\n".join(parts)

    def _build_preferences_style(
        self, memory_groups: dict[MemoryType, list[RankedMemory]], persona: Persona
    ) -> str:
        """Build preferences and style section."""
        prefs = memory_groups.get(MemoryType.PREFERENCE, [])
        styles = memory_groups.get(MemoryType.STYLE, [])

        parts = []

        # Add persona facets
        parts.append("User preferences (learned):")
        for facet, value in persona.facets.items():
            if value > 0.6:
                parts.append(f"  - Prefers {facet.replace('_', ' ')}")
            elif value < 0.4:
                parts.append(f"  - Less emphasis on {facet.replace('_', ' ')}")

        # Add explicit preferences
        if prefs:
            parts.append("\nExplicit preferences:")
            for ranked in prefs[:5]:
                parts.append(f"  - {ranked.memory.content}")

        # Add style preferences
        if styles:
            parts.append("\nStyle preferences:")
            for ranked in styles[:3]:
                parts.append(f"  - {ranked.memory.content}")

        return "\n".join(parts) if parts else "No specific preferences defined."

    def _build_artifacts(self, memory_groups: dict[MemoryType, list[RankedMemory]]) -> str:
        """Build artifacts section."""
        artifacts = memory_groups.get(MemoryType.ARTIFACT, [])

        if not artifacts:
            return "No artifacts available."

        parts = []
        for i, ranked in enumerate(artifacts[:5], 1):  # Top 5 artifacts
            parts.append(f"\nArtifact {i}:")
            parts.append(f"{ranked.memory.content}")

        return "\n".join(parts)

    def _build_history(self, memory_groups: dict[MemoryType, list[RankedMemory]]) -> str:
        """Build recent history section."""
        history = memory_groups.get(MemoryType.HISTORY, [])

        if not history:
            return "No recent interaction history."

        parts = []
        for ranked in history[:5]:  # Top 5 history items
            parts.append(f"- {ranked.memory.content}")

        return "\n".join(parts)

    def _build_constraints(self, persona: Persona) -> str:
        """Build constraints section."""
        constraints = [
            "Follow security best practices",
            "Ensure code is well-documented",
            "Provide error handling",
        ]

        # Add persona-specific constraints
        if persona.facets.get("formal", 0.5) > 0.7:
            constraints.append("Maintain professional tone and formatting")

        if persona.facets.get("code_first", 0.5) > 0.7:
            constraints.append("Prioritize code examples over explanations")

        return "\n".join(f"- {c}" for c in constraints)

    def _build_task_specific(
        self, user_input: str, memory_groups: dict[MemoryType, list[RankedMemory]]
    ) -> str:
        """Build task-specific retrieval section."""
        # This is a placeholder for task-specific context
        # Could be enhanced with more sophisticated retrieval
        profile = memory_groups.get(MemoryType.PROFILE, [])

        if profile:
            parts = ["Relevant context:"]
            for ranked in profile[:3]:
                parts.append(f"- {ranked.memory.content}")
            return "\n".join(parts)

        return f"Task: {user_input}"

    def _build_safety_system(self) -> str:
        """Build safety and system section."""
        return """Safety reminders:
- Do not generate harmful, unethical, or dangerous content
- Respect user privacy and data protection
- Follow applicable laws and regulations
- Clarify if uncertain about requirements"""

    def _build_output_contract(self, persona: Persona) -> str:
        """Build output contract based on persona."""
        contract = ["Provide clear, actionable output"]

        if persona.facets.get("code_first", 0.5) > 0.6:
            contract.append("Include working code examples")

        if persona.facets.get("step_by_step", 0.5) > 0.6:
            contract.append("Break down into clear steps")

        if persona.facets.get("concise", 0.5) > 0.6:
            contract.append("Be concise and to the point")

        return "\n".join(f"- {c}" for c in contract)

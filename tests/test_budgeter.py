"""Tests for context budgeter."""

import pytest

from app.clients.slm.local import LocalSLM
from app.services.context_budgeter import ContextBudgeter


@pytest.fixture
def context_budgeter() -> ContextBudgeter:
    """Create a context budgeter instance."""
    slm_client = LocalSLM()
    return ContextBudgeter(slm_client)


@pytest.mark.asyncio
async def test_allocate_budget_within_limits(
    context_budgeter: ContextBudgeter,
) -> None:
    """Test that budget allocation stays within configured limits."""
    sections = {
        "goal_summary": "This is a short goal.",
        "preferences_style": "User prefers concise responses.",
        "critical_artifacts": "Some artifact content here.",
        "recent_history": "Recent interaction history.",
        "constraints": "Follow best practices.",
        "task_specific_retrieval": "Task-specific context.",
        "safety_system": "Safety guidelines.",
    }

    allocated_sections, breakdown = await context_budgeter.allocate_budget(sections)

    # Check that all sections are present
    assert len(allocated_sections) == len(sections)

    # Check that breakdown sums to approximately token budget
    total_tokens = sum(breakdown.values())
    # Allow some variance
    assert total_tokens <= context_budgeter.settings.token_budget * 1.2


@pytest.mark.asyncio
async def test_summarize_oversized_sections(
    context_budgeter: ContextBudgeter,
) -> None:
    """Test that oversized sections are summarized."""
    # Create a very long section
    long_content = " ".join(["This is a long sentence."] * 1000)

    sections = {
        "goal_summary": long_content,
        "preferences_style": "Short",
        "critical_artifacts": "Short",
        "recent_history": "Short",
        "constraints": "Short",
        "task_specific_retrieval": "Short",
        "safety_system": "Short",
    }

    allocated_sections, breakdown = await context_budgeter.allocate_budget(sections)

    # Find goal_summary section
    goal_section = next(s for s in allocated_sections if s.name == "goal_summary")

    # Should be marked as summarized
    assert goal_section.summarized is True

    # Should be within allocated budget
    assert goal_section.tokens <= goal_section.allocated * 1.1  # Allow small margin


@pytest.mark.asyncio
async def test_budget_weights_applied(
    context_budgeter: ContextBudgeter,
) -> None:
    """Test that budget weights are correctly applied."""
    sections = {
        "goal_summary": "A",
        "preferences_style": "B",
        "critical_artifacts": "C",
        "recent_history": "D",
        "constraints": "E",
        "task_specific_retrieval": "F",
        "safety_system": "G",
    }

    allocated_sections, breakdown = await context_budgeter.allocate_budget(sections)

    # Artifacts should get most tokens (25% = 0.25)
    artifacts_section = next(s for s in allocated_sections if s.name == "critical_artifacts")

    # Safety should get least (5% = 0.05)
    safety_section = next(s for s in allocated_sections if s.name == "safety_system")

    # Check allocation proportions
    total_budget = context_budgeter.settings.token_budget
    assert artifacts_section.allocated == int(total_budget * 0.25)
    assert safety_section.allocated == int(total_budget * 0.05)


@pytest.mark.asyncio
async def test_format_sections(context_budgeter: ContextBudgeter) -> None:
    """Test section formatting."""
    sections = {
        "goal_summary": "Goal content",
        "preferences_style": "Preference content",
        "critical_artifacts": "Artifact content",
        "recent_history": "History content",
        "constraints": "Constraint content",
        "task_specific_retrieval": "Task content",
        "safety_system": "Safety content",
    }

    allocated_sections, _ = await context_budgeter.allocate_budget(sections)

    formatted = context_budgeter.format_sections(allocated_sections)

    # Should contain all section content
    assert "Goal content" in formatted
    assert "Preference content" in formatted
    assert "Artifact content" in formatted

    # Should have section headers
    assert "GOAL_SUMMARY" in formatted or "goal_summary" in formatted.lower()

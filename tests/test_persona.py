"""Tests for persona service."""

import pytest

from app.memory.local import LocalMemoryProvider
from app.services.persona import PersonaService


@pytest.fixture
def persona_service() -> PersonaService:
    """Create a persona service instance."""
    memory_provider = LocalMemoryProvider()
    return PersonaService(memory_provider)


@pytest.mark.asyncio
async def test_create_new_persona(persona_service: PersonaService) -> None:
    """Test creating a new persona."""
    persona = await persona_service.get_or_create_persona("new_user")

    assert persona.user_id == "new_user"
    assert len(persona.vector) == 384  # Default embed dims
    assert persona.facets == {
        "concise": 0.5,
        "formal": 0.5,
        "code_first": 0.5,
        "step_by_step": 0.5,
    }
    assert persona.interaction_count == 0


@pytest.mark.asyncio
async def test_update_persona_increments_interaction(
    persona_service: PersonaService,
) -> None:
    """Test that persona update increments interaction count."""
    persona = await persona_service.get_or_create_persona("user1")
    initial_count = persona.interaction_count

    updated = await persona_service.update_persona("user1", {"success": True})

    assert updated.interaction_count == initial_count + 1


@pytest.mark.asyncio
async def test_update_persona_with_embedding(
    persona_service: PersonaService,
) -> None:
    """Test persona vector update with new embedding."""
    persona = await persona_service.get_or_create_persona("user2")
    initial_vector = persona.vector.copy()

    new_embedding = [0.1] * 384
    updated = await persona_service.update_persona("user2", {"embedding": new_embedding})

    # Vector should have changed
    assert updated.vector != initial_vector


@pytest.mark.asyncio
async def test_update_facets_based_on_signals(
    persona_service: PersonaService,
) -> None:
    """Test facet updates based on interaction signals."""
    persona = await persona_service.get_or_create_persona("user3")
    initial_concise = persona.facets["concise"]

    # Select first hypothesis (should increase concise)
    await persona_service.update_persona("user3", {"selected_hypothesis_id": "h1"})

    updated = await persona_service.get_or_create_persona("user3")

    # Concise facet should have increased
    assert updated.facets["concise"] >= initial_concise


@pytest.mark.asyncio
async def test_facets_clamped_to_range(persona_service: PersonaService) -> None:
    """Test that facets are clamped to [0, 1]."""
    await persona_service.get_or_create_persona("user4")

    # Try to update with extreme values
    for _ in range(50):  # Many updates
        await persona_service.update_persona(
            "user4",
            {
                "facet_updates": {
                    "concise": 0.5,  # Push towards 1
                }
            },
        )

    updated = await persona_service.get_or_create_persona("user4")

    # Should be clamped to 1.0
    assert 0.0 <= updated.facets["concise"] <= 1.0


@pytest.mark.asyncio
async def test_persona_persistence(persona_service: PersonaService) -> None:
    """Test that persona changes persist."""
    # Create and update persona
    await persona_service.update_persona(
        "user5",
        {
            "selected_hypothesis_id": "h1",
            "embedding": [0.5] * 384,
        },
    )

    # Retrieve persona again
    persona = await persona_service.get_or_create_persona("user5")

    assert persona.interaction_count > 0

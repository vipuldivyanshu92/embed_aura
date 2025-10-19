"""Tests for ranking service."""

from datetime import datetime, timedelta

import pytest

from app.models import MemoryItem, MemoryType
from app.services.ranking import RankingService


@pytest.fixture
def ranking_service() -> RankingService:
    """Create a ranking service instance."""
    return RankingService()


@pytest.fixture
def sample_memories() -> list[MemoryItem]:
    """Create sample memories for testing."""
    now = datetime.utcnow()

    return [
        MemoryItem(
            id="mem1",
            user_id="user1",
            mtype=MemoryType.GOAL,
            content="Build a REST API",
            embedding=[1.0, 0.0, 0.0],
            confidence=0.9,
            created_at=now - timedelta(days=1),
            updated_at=now - timedelta(days=1),
        ),
        MemoryItem(
            id="mem2",
            user_id="user1",
            mtype=MemoryType.PREFERENCE,
            content="Prefer Python for backend",
            embedding=[0.8, 0.6, 0.0],
            confidence=0.8,
            created_at=now - timedelta(days=5),
            updated_at=now - timedelta(days=5),
        ),
        MemoryItem(
            id="mem3",
            user_id="user1",
            mtype=MemoryType.ARTIFACT,
            content="Sample code snippet",
            embedding=[0.5, 0.5, 0.7],
            confidence=0.7,
            created_at=now - timedelta(days=10),
            updated_at=now - timedelta(days=10),
        ),
    ]


async def test_rank_memories_by_cosine(
    ranking_service: RankingService, sample_memories: list[MemoryItem]
) -> None:
    """Test that memories are ranked by cosine similarity."""
    # Query embedding close to mem1
    query = [1.0, 0.0, 0.0]

    ranked = await ranking_service.rank_memories(sample_memories, query)

    # mem1 should be first (perfect match)
    assert ranked[0].memory.id == "mem1"
    assert ranked[0].cosine_score > 0.9


async def test_rank_memories_empty_list(ranking_service: RankingService) -> None:
    """Test ranking with empty memory list."""
    ranked = await ranking_service.rank_memories([], [1.0, 0.0, 0.0])
    assert ranked == []


async def test_recency_score_decreases_with_age(
    ranking_service: RankingService, sample_memories: list[MemoryItem]
) -> None:
    """Test that recency score decreases with age."""
    query = [0.0, 0.0, 1.0]  # Neutral query

    ranked = await ranking_service.rank_memories(sample_memories, query)

    # Find the most recent memory (mem1, 1 day old)
    mem1_ranked = next(r for r in ranked if r.memory.id == "mem1")
    mem3_ranked = next(r for r in ranked if r.memory.id == "mem3")

    assert mem1_ranked.recency_score > mem3_ranked.recency_score


async def test_combined_score(ranking_service: RankingService, sample_memories: list[MemoryItem]) -> None:
    """Test that combined score is calculated correctly."""
    query = [1.0, 0.0, 0.0]

    ranked = await ranking_service.rank_memories(sample_memories, query)

    for r in ranked:
        # Score should be weighted combination
        expected = 0.6 * r.cosine_score + 0.25 * r.recency_score + 0.15 * r.confidence_score
        assert abs(r.score - expected) < 0.001


async def test_deduplicate_similar_memories(ranking_service: RankingService) -> None:
    """Test deduplication of similar memories."""
    now = datetime.utcnow()

    memories = [
        MemoryItem(
            id="mem1",
            user_id="user1",
            mtype=MemoryType.GOAL,
            content="Build API",
            embedding=[1.0, 0.0, 0.0],
            confidence=0.9,
            created_at=now,
            updated_at=now,
        ),
        MemoryItem(
            id="mem2",
            user_id="user1",
            mtype=MemoryType.GOAL,
            content="Build API duplicate",
            embedding=[0.99, 0.01, 0.0],  # Very similar
            confidence=0.8,
            created_at=now,
            updated_at=now,
        ),
    ]

    ranked = await ranking_service.rank_memories(memories, [1.0, 0.0, 0.0])
    deduped = ranking_service.deduplicate(ranked, threshold=0.95)

    # Should keep only one (the higher scored)
    assert len(deduped) == 1
    assert deduped[0].memory.id == "mem1"


async def test_resolve_conflicts_by_confidence(ranking_service: RankingService) -> None:
    """Test conflict resolution keeps higher confidence memory."""
    now = datetime.utcnow()

    memories = [
        MemoryItem(
            id="mem1",
            user_id="user1",
            mtype=MemoryType.PREFERENCE,
            content="Prefer Python",
            embedding=[1.0, 0.0, 0.0],
            confidence=0.9,
            tags=["language"],
            created_at=now,
            updated_at=now,
        ),
        MemoryItem(
            id="mem2",
            user_id="user1",
            mtype=MemoryType.PREFERENCE,
            content="Prefer JavaScript",
            embedding=[0.8, 0.2, 0.0],
            confidence=0.6,
            tags=["language"],
            created_at=now,
            updated_at=now,
        ),
    ]

    ranked = await ranking_service.rank_memories(memories, [1.0, 0.0, 0.0])
    resolved = ranking_service.resolve_conflicts(ranked)

    # Should keep the higher confidence one
    assert any(r.memory.id == "mem1" for r in resolved)
    # mem2 might be filtered out due to conflict resolution

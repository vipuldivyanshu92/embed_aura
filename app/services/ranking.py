"""Memory ranking and relevance scoring."""

from datetime import datetime

import numpy as np
import structlog

from app.config import get_settings
from app.models import MemoryItem, RankedMemory

logger = structlog.get_logger()


class RankingService:
    """
    Ranks memories by combined score:
    score = 0.6 * cosine + 0.25 * recency + 0.15 * confidence

    Also handles deduplication and conflict resolution.
    """

    def __init__(self) -> None:
        """Initialize ranking service."""
        self.settings = get_settings()

    def rank_memories(
        self,
        memories: list[MemoryItem],
        query_embedding: list[float],
    ) -> list[RankedMemory]:
        """
        Rank memories by relevance score.

        Args:
            memories: List of memory items
            query_embedding: Query embedding vector

        Returns:
            List of ranked memories sorted by score (descending)
        """
        if not memories:
            return []

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)

        ranked: list[RankedMemory] = []
        now = datetime.utcnow()

        for memory in memories:
            # Cosine similarity
            cosine_score = self._compute_cosine(query_vec, query_norm, memory.embedding)

            # Recency score
            recency_score = self._compute_recency(memory.updated_at, now)

            # Confidence score (already in [0, 1])
            confidence_score = memory.confidence

            # Combined score
            total_score = (
                self.settings.ranking_cosine_weight * cosine_score
                + self.settings.ranking_recency_weight * recency_score
                + self.settings.ranking_confidence_weight * confidence_score
            )

            ranked.append(
                RankedMemory(
                    memory=memory,
                    score=total_score,
                    cosine_score=cosine_score,
                    recency_score=recency_score,
                    confidence_score=confidence_score,
                )
            )

        # Sort by total score
        ranked.sort(key=lambda r: r.score, reverse=True)

        return ranked

    def _compute_cosine(
        self,
        query_vec: np.ndarray,
        query_norm: float,
        embedding: list[float],
    ) -> float:
        """Compute cosine similarity."""
        mem_vec = np.array(embedding)
        mem_norm = np.linalg.norm(mem_vec)

        if query_norm > 0 and mem_norm > 0:
            return float(np.dot(query_vec, mem_vec) / (query_norm * mem_norm))
        return 0.0

    def _compute_recency(
        self,
        updated_at: datetime,
        now: datetime,
    ) -> float:
        """
        Compute recency score using exponential decay.

        Score = exp(-days_old / decay_period)

        Args:
            updated_at: When memory was last updated
            now: Current time

        Returns:
            Recency score in [0, 1]
        """
        age = now - updated_at
        days_old = age.total_seconds() / 86400.0  # Convert to days

        decay_period = self.settings.recency_decay_days
        score = np.exp(-days_old / decay_period)

        return float(score)

    def deduplicate(
        self,
        ranked_memories: list[RankedMemory],
        threshold: float | None = None,
    ) -> list[RankedMemory]:
        """
        Remove near-duplicate memories.

        Keeps the higher-scored item when duplicates are found.

        Args:
            ranked_memories: List of ranked memories
            threshold: Cosine similarity threshold for duplication
                      (default from settings)

        Returns:
            Deduplicated list
        """
        if threshold is None:
            threshold = self.settings.memory_dedup_threshold

        if not ranked_memories:
            return []

        # Already sorted by score (highest first)
        # Keep items that are not duplicates of any previously kept item
        kept: list[RankedMemory] = []

        for candidate in ranked_memories:
            is_duplicate = False

            for existing in kept:
                similarity = self._compute_embedding_similarity(
                    candidate.memory.embedding,
                    existing.memory.embedding,
                )

                if similarity >= threshold:
                    is_duplicate = True
                    logger.debug(
                        "dedup_removed",
                        removed_id=candidate.memory.id,
                        kept_id=existing.memory.id,
                        similarity=similarity,
                    )
                    break

            if not is_duplicate:
                kept.append(candidate)

        return kept

    def _compute_embedding_similarity(
        self,
        emb1: list[float],
        emb2: list[float],
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0 and norm2 > 0:
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        return 0.0

    def resolve_conflicts(
        self,
        ranked_memories: list[RankedMemory],
    ) -> list[RankedMemory]:
        """
        Resolve conflicting memories.

        If two memories have the same tag/topic but different content,
        keep the one with higher confidence or more recent.

        Args:
            ranked_memories: List of ranked memories

        Returns:
            Conflict-resolved list
        """
        # Group by tags
        tag_groups: dict[str, list[RankedMemory]] = {}

        for ranked in ranked_memories:
            for tag in ranked.memory.tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(ranked)

        # For each tag, if multiple memories exist, keep the best one
        resolved: list[RankedMemory] = []
        processed_ids: set[str] = set()

        for tag, group in tag_groups.items():
            if len(group) > 1:
                # Sort by confidence, then recency
                group.sort(
                    key=lambda r: (r.memory.confidence, r.memory.updated_at),
                    reverse=True,
                )

                # Keep the best one
                best = group[0]
                if best.memory.id not in processed_ids:
                    resolved.append(best)
                    processed_ids.add(best.memory.id)

                    logger.debug(
                        "conflict_resolved",
                        tag=tag,
                        kept_id=best.memory.id,
                        discarded=[r.memory.id for r in group[1:]],
                    )
            else:
                # No conflict, keep it
                if group[0].memory.id not in processed_ids:
                    resolved.append(group[0])
                    processed_ids.add(group[0].memory.id)

        # Add memories with no tags
        for ranked in ranked_memories:
            if not ranked.memory.tags and ranked.memory.id not in processed_ids:
                resolved.append(ranked)
                processed_ids.add(ranked.memory.id)

        # Re-sort by score
        resolved.sort(key=lambda r: r.score, reverse=True)

        return resolved

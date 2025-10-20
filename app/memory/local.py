"""Local JSON-based memory provider for development and testing."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import structlog

from app.config import get_settings
from app.memory.base import MemoryProvider
from app.models import MemoryItem, MemoryType, Persona

logger = structlog.get_logger()


class LocalMemoryProvider(MemoryProvider):
    """
    In-memory storage with JSON persistence.

    Suitable for development and single-instance deployments.
    Data is persisted to JSON files on shutdown or periodically.
    """

    def __init__(self) -> None:
        """Initialize local memory provider."""
        self.settings = get_settings()
        self.data_dir = Path(self.settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.memories_file = self.data_dir / "memories.json"
        self.personas_file = self.data_dir / "personas.json"

        # In-memory storage
        self.memories: dict[str, list[MemoryItem]] = {}
        self.personas: dict[str, Persona] = {}

        # Load existing data
        self._load()

    def _load(self) -> None:
        """Load data from JSON files."""
        # Load memories
        if self.memories_file.exists():
            try:
                with open(self.memories_file) as f:
                    data = json.load(f)
                    for user_id, items in data.items():
                        self.memories[user_id] = [MemoryItem(**item) for item in items]
                logger.info(
                    "loaded_memories",
                    user_count=len(self.memories),
                    file=str(self.memories_file),
                )
            except Exception as e:
                logger.error("failed_to_load_memories", error=str(e))

        # Load personas
        if self.personas_file.exists():
            try:
                with open(self.personas_file) as f:
                    data = json.load(f)
                    for user_id, persona_data in data.items():
                        self.personas[user_id] = Persona(**persona_data)
                logger.info(
                    "loaded_personas",
                    user_count=len(self.personas),
                    file=str(self.personas_file),
                )
            except Exception as e:
                logger.error("failed_to_load_personas", error=str(e))

    def _save(self) -> None:
        """Save data to JSON files."""
        # Save memories
        try:
            data = {
                user_id: [mem.model_dump(mode="json") for mem in mems]
                for user_id, mems in self.memories.items()
            }
            with open(self.memories_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug("saved_memories", file=str(self.memories_file))
        except Exception as e:
            logger.error("failed_to_save_memories", error=str(e))

        # Save personas
        try:
            data = {
                user_id: persona.model_dump(mode="json")
                for user_id, persona in self.personas.items()
            }
            with open(self.personas_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug("saved_personas", file=str(self.personas_file))
        except Exception as e:
            logger.error("failed_to_save_personas", error=str(e))

    async def store_memory(self, memory: MemoryItem) -> None:
        """Store a memory item."""
        if memory.user_id not in self.memories:
            self.memories[memory.user_id] = []

        # Update if exists (same ID), otherwise append
        existing_idx = next(
            (i for i, m in enumerate(self.memories[memory.user_id]) if m.id == memory.id),
            None,
        )

        if existing_idx is not None:
            self.memories[memory.user_id][existing_idx] = memory
        else:
            self.memories[memory.user_id].append(memory)

        self._save()
        logger.debug(
            "stored_memory",
            user_id=memory.user_id,
            memory_id=memory.id,
            mtype=memory.mtype,
        )

    async def get_memories(
        self,
        user_id: str,
        mtype: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Retrieve memories for a user."""
        user_memories = self.memories.get(user_id, [])

        # Filter by type if specified
        if mtype is not None:
            user_memories = [m for m in user_memories if m.mtype == mtype]

        # Filter out expired memories
        now = datetime.utcnow()
        user_memories = [m for m in user_memories if m.expires_at is None or m.expires_at > now]

        # Sort by updated_at (most recent first)
        user_memories.sort(key=lambda m: m.updated_at, reverse=True)

        return user_memories[:limit]

    async def delete_memories(
        self,
        user_id: str,
        mtype: MemoryType | None = None,
    ) -> int:
        """Delete memories for a user."""
        if user_id not in self.memories:
            return 0

        original_count = len(self.memories[user_id])

        if mtype is None:
            # Delete all memories for user
            del self.memories[user_id]
            deleted = original_count
        else:
            # Delete only specific type
            self.memories[user_id] = [m for m in self.memories[user_id] if m.mtype != mtype]
            deleted = original_count - len(self.memories[user_id])

        if deleted > 0:
            self._save()
            logger.info(
                "deleted_memories",
                user_id=user_id,
                mtype=mtype,
                count=deleted,
            )

        return deleted

    async def get_persona(self, user_id: str) -> Persona | None:
        """Retrieve user persona."""
        return self.personas.get(user_id)

    async def store_persona(self, persona: Persona) -> None:
        """Store user persona."""
        self.personas[persona.user_id] = persona
        self._save()
        logger.debug("stored_persona", user_id=persona.user_id)

    async def search_memories(
        self,
        user_id: str,
        query_embedding: list[float],
        mtype: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search memories by embedding similarity."""
        user_memories = await self.get_memories(user_id, mtype, limit=1000)

        if not user_memories:
            return []

        # Compute cosine similarity
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        query_dims = len(query_embedding)

        scored_memories = []
        skipped_count = 0
        
        for memory in user_memories:
            mem_vec = np.array(memory.embedding)
            mem_dims = len(memory.embedding)
            
            # Skip memories with mismatched dimensions
            if mem_dims != query_dims:
                logger.warning(
                    "embedding_dimension_mismatch",
                    memory_id=memory.id,
                    memory_dims=mem_dims,
                    query_dims=query_dims,
                    user_id=user_id,
                )
                skipped_count += 1
                continue
            
            mem_norm = np.linalg.norm(mem_vec)

            if query_norm > 0 and mem_norm > 0:
                similarity = np.dot(query_vec, mem_vec) / (query_norm * mem_norm)
            else:
                similarity = 0.0

            scored_memories.append((similarity, memory))

        if skipped_count > 0:
            logger.info(
                "skipped_memories_dimension_mismatch",
                user_id=user_id,
                skipped_count=skipped_count,
                total_memories=len(user_memories),
            )

        # Sort by similarity
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        return [mem for _, mem in scored_memories[:limit]]

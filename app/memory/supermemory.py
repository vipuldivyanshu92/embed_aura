"""Supermemory provider adapter with local fallback."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import structlog

from app.memory.base import MemoryProvider
from app.models import MemoryItem, MemoryType, Persona

logger = structlog.get_logger()


class SupermemoryProvider(MemoryProvider):
    """
    Supermemory provider adapter.

    Currently uses local JSON storage as a fallback until Supermemory SDK is integrated.

    TODO: Integrate with Supermemory SDK or API when available.

    Installation:
        # Install Supermemory client library when available
        # pip install supermemory

    Future Usage:
        from supermemory import Client

        self.client = Client(
            api_key=settings.supermemory_api_key,
            base_url=settings.supermemory_base_url
        )

    Reference:
        https://supermemory.ai/docs
    """

    def __init__(self, api_key: str, base_url: str) -> None:
        """
        Initialize Supermemory provider.

        Args:
            api_key: Supermemory API key (currently unused)
            base_url: Supermemory API base URL (currently unused)
        """
        self.api_key = api_key
        self.base_url = base_url

        # TODO: Initialize Supermemory client when SDK is available
        # self.client = Client(api_key=api_key, base_url=base_url)

        # Local fallback storage
        self.data_dir = Path("./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.memories_file = self.data_dir / "memories.json"
        self.personas_file = self.data_dir / "personas.json"

        # In-memory storage
        self.memories: dict[str, list[MemoryItem]] = {}
        self.personas: dict[str, Persona] = {}

        # Load existing data
        self._load()

        logger.info(
            "supermemory_provider_initialized",
            message="Using local JSON fallback until Supermemory SDK is integrated",
            api_key_set=bool(api_key),
            base_url=base_url,
        )

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
        """
        Store a memory in Supermemory.

        Currently uses local storage. TODO: Implement using Supermemory SDK:
            await self.client.memories.create(
                user_id=memory.user_id,
                content=memory.content,
                embedding=memory.embedding,
                metadata={
                    "type": memory.mtype.value,
                    "confidence": memory.confidence,
                    "tags": memory.tags,
                    ...
                }
            )
        """
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
        """
        Retrieve memories from Supermemory.

        Currently uses local storage. TODO: Implement using Supermemory SDK:
            filter_params = {"user_id": user_id}
            if mtype:
                filter_params["type"] = mtype.value

            memories = await self.client.memories.list(
                filters=filter_params,
                limit=limit
            )
            # Convert to MemoryItem objects
        """
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
        """
        Delete memories from Supermemory.

        Currently uses local storage. TODO: Implement using Supermemory SDK:
            filter_params = {"user_id": user_id}
            if mtype:
                filter_params["type"] = mtype.value

            deleted = await self.client.memories.delete(filters=filter_params)
            return deleted.count
        """
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
        """
        Retrieve persona from Supermemory.

        Currently uses local storage. TODO: Implement using Supermemory SDK.
        Consider using Supermemory's user profile/metadata features.
        """
        return self.personas.get(user_id)

    async def store_persona(self, persona: Persona) -> None:
        """
        Store persona in Supermemory.

        Currently uses local storage. TODO: Implement using Supermemory SDK:
            await self.client.users.update_profile(
                user_id=persona.user_id,
                profile={
                    "vector": persona.vector,
                    "facets": persona.facets,
                    "last_updated": persona.last_updated,
                    ...
                }
            )
        """
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
        """
        Search memories by embedding.

        Currently uses local storage. TODO: Implement using Supermemory SDK:
            results = await self.client.memories.search(
                user_id=user_id,
                embedding=query_embedding,
                filters={"type": mtype.value} if mtype else {},
                limit=limit
            )
            # Convert to MemoryItem objects
        """
        user_memories = await self.get_memories(user_id, mtype, limit=1000)

        if not user_memories:
            return []

        # Compute cosine similarity
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)

        scored_memories = []
        for memory in user_memories:
            mem_vec = np.array(memory.embedding)
            mem_norm = np.linalg.norm(mem_vec)

            if query_norm > 0 and mem_norm > 0:
                similarity = np.dot(query_vec, mem_vec) / (query_norm * mem_norm)
            else:
                similarity = 0.0

            scored_memories.append((similarity, memory))

        # Sort by similarity
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        return [mem for _, mem in scored_memories[:limit]]

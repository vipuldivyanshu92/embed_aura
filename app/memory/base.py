"""Base interface for memory providers."""

from abc import ABC, abstractmethod

from app.models import MemoryItem, MemoryType, Persona


class MemoryProvider(ABC):
    """Abstract base class for memory storage implementations."""

    @abstractmethod
    async def store_memory(self, memory: MemoryItem) -> None:
        """
        Store a memory item.

        Args:
            memory: Memory item to store
        """
        pass

    @abstractmethod
    async def get_memories(
        self,
        user_id: str,
        mtype: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """
        Retrieve memories for a user.

        Args:
            user_id: User identifier
            mtype: Optional memory type filter
            limit: Maximum number of memories to retrieve

        Returns:
            List of memory items
        """
        pass

    @abstractmethod
    async def delete_memories(
        self,
        user_id: str,
        mtype: MemoryType | None = None,
    ) -> int:
        """
        Delete memories for a user.

        Args:
            user_id: User identifier
            mtype: Optional memory type filter (if None, deletes all)

        Returns:
            Number of memories deleted
        """
        pass

    @abstractmethod
    async def get_persona(self, user_id: str) -> Persona | None:
        """
        Retrieve user persona.

        Args:
            user_id: User identifier

        Returns:
            Persona object or None if not found
        """
        pass

    @abstractmethod
    async def store_persona(self, persona: Persona) -> None:
        """
        Store user persona.

        Args:
            persona: Persona object to store
        """
        pass

    @abstractmethod
    async def search_memories(
        self,
        user_id: str,
        query_embedding: list[float],
        mtype: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """
        Search memories by embedding similarity.

        Args:
            user_id: User identifier
            query_embedding: Query embedding vector
            mtype: Optional memory type filter
            limit: Maximum results

        Returns:
            List of memory items ranked by similarity
        """
        pass

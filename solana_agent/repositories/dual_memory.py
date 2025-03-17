"""
Dual memory repository implementation.

This repository combines multiple memory repositories for different storage needs.
"""
from typing import Dict, List, Any

from solana_agent.interfaces import MemoryRepository
from solana_agent.repositories import ZepMemoryRepository
from solana_agent.repositories import MongoMemoryRepository
from solana_agent.domains import MemoryInsight


class DualMemoryRepository(MemoryRepository):
    """Repository that combines Zep for semantic memory and MongoDB for persistence."""

    def __init__(
        self,
        zep_repository: ZepMemoryRepository,
        mongo_repository: MongoMemoryRepository
    ):
        """Initialize the dual memory repository.

        Args:
            zep_repository: Zep memory repository for semantic memory
            mongo_repository: MongoDB repository for persistent storage
        """
        self.zep = zep_repository
        self.mongo = mongo_repository

    async def store_insight(self, user_id: str, insight: MemoryInsight) -> None:
        """Store a memory insight in both repositories.

        Args:
            user_id: ID of the user the insight relates to
            insight: Memory insight to store
        """
        # Store in Zep for semantic search and retrieval
        await self.zep.store_insight(user_id, insight)

        # Store in MongoDB for persistence and structured queries
        self.mongo.store_insight(user_id, insight)

    async def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory using Zep's semantic search capabilities.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of matching memory items
        """
        # Prefer Zep for semantic search
        results = await self.zep.search(query, limit)

        # Fall back to MongoDB if Zep search fails or returns no results
        if not results:
            results = self.mongo.search(query, limit)

        return results

    async def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history using MongoDB's structured storage.

        Args:
            user_id: User ID
            limit: Maximum number of items to return

        Returns:
            List of conversation history items
        """
        # Get from MongoDB for consistent history
        history = self.mongo.get_user_history(user_id, limit)

        # If MongoDB has no history, try Zep
        if not history:
            history = await self.zep.get_user_history(user_id, limit)

        return history

    async def delete_user_memory(self, user_id: str) -> bool:
        """Delete all memory from both repositories.

        Args:
            user_id: User ID

        Returns:
            True if successful
        """
        # Delete from both sources
        zep_success = await self.zep.delete_user_memory(user_id)
        mongo_success = self.mongo.delete_user_memory(user_id)

        return zep_success and mongo_success

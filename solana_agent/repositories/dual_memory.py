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
        try:
            # Prefer Zep for semantic search
            results = await self.zep.search(query, limit)
            if results:
                return results
        except Exception:
            # Fall back to MongoDB if Zep fails
            pass

        # Fall back to MongoDB if Zep returns no results or fails
        return self.mongo.search(query, limit)

    async def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get user interaction history.

        Args:
            user_id: User identifier
            limit: Maximum number of history entries to return

        Returns:
            List of user interaction records
        """
        try:
            # Prefer MongoDB for structured history retrieval
            history = self.mongo.get_user_history(user_id, limit)
            if history:
                return history
        except Exception:
            # Fall back to Zep if MongoDB fails
            pass

        # Fall back to Zep if MongoDB returns no results or fails
        return await self.zep.get_user_history(user_id, limit)

    async def delete_user_memory(self, user_id: str) -> bool:
        """Delete all memory for a user.

        Args:
            user_id: User identifier

        Returns:
            True if successful, False otherwise
        """
        # Delete from both repositories
        zep_success = await self.zep.delete_user_memory(user_id)
        mongo_success = self.mongo.delete_user_memory(user_id)

        # Only return True if both operations succeeded
        return zep_success and mongo_success

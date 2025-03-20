from abc import ABC, abstractmethod
from typing import Dict, List

from solana_agent.domains.memory import MemoryInsight


class MemoryRepository(ABC):
    """Interface for memory storage and retrieval."""

    @abstractmethod
    def store_insight(self, user_id: str, insight: MemoryInsight) -> None:
        """Store a memory insight.

        Args:
            user_id: ID of the user the insight relates to
            insight: Memory insight to store
        """
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory for insights matching a query.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of matching memory items
        """
        pass

    @abstractmethod
    def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history for a user.

        Args:
            user_id: User ID
            limit: Maximum number of items to return

        Returns:
            List of conversation history items
        """
        pass

    @abstractmethod
    def delete_user_memory(self, user_id: str) -> bool:
        """Delete all memory for a user.

        Args:
            user_id: User ID

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def get_user_history_paginated(self, user_id: str, page_num: int, page_size: int) -> List[Dict]:
        """Get paginated conversation history for a user.

        Args:
            user_id: User ID
            page_num: Page number (starting from 1)
            page_size: Number of items per page

        Returns:
            List of conversation history items
        """
        pass

    @abstractmethod
    def count_user_history(self, user_id: str) -> int:
        """Count the number of items in a user's conversation history.

        Args:
            user_id: User ID

        Returns:
            Number of items
        """
        pass

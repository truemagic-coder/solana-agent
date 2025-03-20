

from abc import ABC, abstractmethod
from typing import Dict, List

from solana_agent.domains.memory import MemoryInsight


class MemoryService(ABC):
    """Interface for memory management services."""

    @abstractmethod
    async def extract_insights(self, conversation: Dict[str, str]) -> List[MemoryInsight]:
        """Extract insights from a conversation.

        Args:
            conversation: Dictionary with 'message' and 'response' keys

        Returns:
            List of extracted memory insights
        """
        pass

    @abstractmethod
    async def store_insights(self, user_id: str, insights: List[MemoryInsight]) -> None:
        """Store multiple insights in memory.

        Args:
            user_id: ID of the user these insights relate to
            insights: List of insights to store
        """
        pass

    @abstractmethod
    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """Search collective memory for relevant insights.

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

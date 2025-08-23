from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class MemoryProvider(ABC):
    """Interface for conversation memory providers."""

    @abstractmethod
    async def store(self, user_id: str, messages: List[Dict[str, Any]]) -> None:
        """Store messages in memory."""
        pass

    @abstractmethod
    async def retrieve(self, user_id: str) -> str:
        """Retrieve memory for a user as formatted string."""
        pass

    @abstractmethod
    async def delete(self, user_id: str) -> None:
        """Delete memory for a user."""
        pass

    @abstractmethod
    def find(
        self,
        collection: str,
        query: Dict,
        sort: Optional[List[Tuple]] = None,
        limit: int = 0,
        skip: int = 0,
    ) -> List[Dict]:
        """Find documents matching query."""
        pass

    @abstractmethod
    def count_documents(self, collection: str, query: Dict) -> int:
        """Count documents matching query."""
        pass

    @abstractmethod
    async def save_capture(
        self,
        user_id: str,
        capture_name: str,
        agent_name: Optional[str],
        data: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Persist a structured capture for a user and return its ID if available."""
        pass

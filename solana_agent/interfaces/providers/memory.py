from abc import ABC, abstractmethod
from typing import Any, Dict, List


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

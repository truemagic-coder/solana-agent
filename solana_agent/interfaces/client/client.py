from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict


class SolanaAgent(ABC):
    """Interface for the Solana agent system."""

    @abstractmethod
    async def process(self, user_id: str, message: str) -> AsyncGenerator[str, None]:
        """Process a user message and return the response stream."""
        pass

    @abstractmethod
    async def get_user_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "desc"  # "asc" for oldest-first, "desc" for newest-first
    ) -> Dict[str, Any]:
        """Get paginated message history for a user."""
        pass

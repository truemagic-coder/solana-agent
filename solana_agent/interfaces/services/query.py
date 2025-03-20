from abc import ABC, abstractmethod
from typing import AsyncGenerator


class QueryService(ABC):
    """Interface for processing user queries."""

    @abstractmethod
    async def process(self, user_id: str, query: str) -> AsyncGenerator[str, None]:
        """Process a query from a user."""
        pass

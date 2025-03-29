from abc import ABC, abstractmethod
from typing import Any, Tuple


class RoutingService(ABC):
    """Interface for query routing services."""

    @abstractmethod
    async def route_query(self, query: str) -> str:
        """Route a query to the appropriate agent.

        Args:
            query: User query

        Returns:
            Tuple of (agent_name, ticket)
        """
        pass

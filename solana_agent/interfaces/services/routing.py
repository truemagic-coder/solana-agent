from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class RoutingService(ABC):
    """Interface for query routing services."""

    @abstractmethod
    async def route_query(
        self, query: str, runtime_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Route a query to the appropriate agent.

        Args:
            query: User query

        Returns:
            Tuple of (agent_name, ticket)
        """
        pass

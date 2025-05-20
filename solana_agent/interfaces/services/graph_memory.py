from abc import ABC, abstractmethod
from typing import Dict, List, Any


class GraphMemoryService(ABC):
    """
    Interface for a graph memory service.
    """

    @abstractmethod
    async def add_episode(
        self,
        user_message: str,
        assistant_message: str,
        user_id: str,
    ):
        """
        Add an episode to the graph memory.
        """
        pass

    @abstractmethod
    async def search(
        self, query: str, user_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the graph memory for relevant episodes.
        """
        pass

    @abstractmethod
    async def traverse(self, node_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Traverse the graph memory from a given node ID.
        """
        pass

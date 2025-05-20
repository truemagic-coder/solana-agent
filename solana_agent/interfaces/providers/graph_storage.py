from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class GraphStorageProvider(ABC):
    @abstractmethod
    async def add_node(self, node: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    async def add_edge(self, edge: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_edges(
        self, node_id: str, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def find_neighbors(
        self, node_id: str, depth: int = 1
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def temporal_query(
        self, node_id: str, start_time: Optional[str], end_time: Optional[str]
    ) -> List[Dict[str, Any]]:
        pass

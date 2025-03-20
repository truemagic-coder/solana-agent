from abc import ABC, abstractmethod
from typing import Dict, List


class VectorStorageProvider(ABC):
    """Interface for vector database providers."""

    @abstractmethod
    def store_vectors(self, vectors: List[Dict], namespace: str) -> None:
        """Store vectors in the database."""
        pass

    @abstractmethod
    def search_vectors(
        self, query_vector: List[float], namespace: str, limit: int = 5
    ) -> List[Dict]:
        """Search vectors by similarity."""
        pass

    @abstractmethod
    def delete_vector(self, id: str, namespace: str) -> None:
        """Delete a vector."""
        pass

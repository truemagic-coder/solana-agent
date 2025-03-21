from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class DataStorageProvider(ABC):
    """Interface for data storage providers."""

    @abstractmethod
    def create_collection(self, name: str) -> None:
        """Create a new collection."""
        pass

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        pass

    @abstractmethod
    def insert_one(self, collection: str, document: Dict) -> str:
        """Insert a document into a collection."""
        pass

    @abstractmethod
    def find_one(self, collection: str, query: Dict) -> Optional[Dict]:
        """Find a single document."""
        pass

    @abstractmethod
    def find(
        self, collection: str, query: Dict, sort: Optional[List] = None, limit: int = 0, skip: int = 0
    ) -> List[Dict]:
        """Find documents matching query."""
        pass

    @abstractmethod
    def update_one(self, collection: str, query: Dict, update: Dict, upsert: bool = False) -> bool:
        """Update a document."""
        pass

    @abstractmethod
    def delete_one(self, collection: str, query: Dict) -> bool:
        """Delete a document."""
        pass

    @abstractmethod
    def delete_all(self, collection: str, query: Dict) -> bool:
        """Delete all documents matching query."""
        pass

    @abstractmethod
    def create_index(self, collection: str, keys: List, **kwargs) -> None:
        """Create an index."""
        pass

    @abstractmethod
    def count_documents(self, collection: str, query: Dict) -> int:
        """Count documents matching query."""
        pass

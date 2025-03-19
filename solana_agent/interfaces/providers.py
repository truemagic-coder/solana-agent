"""
Provider interfaces for external service adapters.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional, Any, TypeVar, Type
from pydantic import BaseModel
from datetime import datetime

T = TypeVar('T', bound=BaseModel)


class LLMProvider(ABC):
    """Interface for language model providers."""

    @abstractmethod
    async def generate_text(
        self, user_id: str, prompt: str, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text from the language model."""
        pass

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        pass

    @abstractmethod
    async def parse_structured_output(
        self, prompt: str, system_prompt: str, model_class: Type[T], **kwargs
    ) -> T:
        """Generate structured output using a specific model class."""
        pass


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
    def create_index(self, collection: str, keys: List, **kwargs) -> None:
        """Create an index."""
        pass

    @abstractmethod
    def count_documents(self, collection: str, query: Dict) -> int:
        """Count documents matching query."""
        pass


class VectorStoreProvider(ABC):
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

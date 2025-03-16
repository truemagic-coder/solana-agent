"""
Provider interfaces for external service adapters.

These interfaces define the contracts for external service adapters,
such as LLM providers, memory systems, and embedding services.
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
        self, user_id: str, prompt: str, stream: bool = True, **kwargs
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

    @abstractmethod
    async def assess_content_safety(self, text: str) -> Dict[str, Any]:
        """Assess the safety of content."""
        pass


class MemoryProvider(ABC):
    """Interface for conversation memory providers."""

    @abstractmethod
    async def store(self, user_id: str, messages: List[Dict[str, Any]]) -> None:
        """Store messages in memory."""
        pass

    @abstractmethod
    async def retrieve(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve memory for a user."""
        pass

    @abstractmethod
    async def retrieve_as_string(self, user_id: str, limit: int = 10) -> str:
        """Retrieve memory for a user as formatted string."""
        pass

    @abstractmethod
    async def delete(self, user_id: str) -> bool:
        """Delete memory for a user."""
        pass

    @abstractmethod
    async def search(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memory for relevant messages."""
        pass

    @abstractmethod
    async def get_summary(self, user_id: str) -> str:
        """Get a summary of the conversation history."""
        pass


class VectorDBProvider(ABC):
    """Interface for vector database providers."""

    @abstractmethod
    def upsert(self, namespace: str, id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """Insert or update a vector in the database."""
        pass

    @abstractmethod
    def delete(self, namespace: str, id: str) -> bool:
        """Delete a vector from the database."""
        pass

    @abstractmethod
    def query(self, namespace: str, vector: List[float], top_k: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Query vectors by similarity."""
        pass

    @abstractmethod
    def create_namespace(self, namespace: str, dimension: int) -> bool:
        """Create a new namespace."""
        pass

    @abstractmethod
    def delete_namespace(self, namespace: str) -> bool:
        """Delete a namespace."""
        pass


class NotificationProvider(ABC):
    """Interface for sending notifications."""

    @abstractmethod
    async def send_notification(self, user_id: str, message: str, channel: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send a notification to a user."""
        pass

    @abstractmethod
    async def send_scheduled_notification(self, user_id: str, message: str, channel: str, schedule_time: datetime, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Schedule a notification to be sent later and return the schedule ID."""
        pass

    @abstractmethod
    async def cancel_scheduled_notification(self, schedule_id: str) -> bool:
        """Cancel a scheduled notification."""
        pass

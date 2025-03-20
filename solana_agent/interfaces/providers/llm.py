from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Type, TypeVar

from pydantic import BaseModel


T = TypeVar('T', bound=BaseModel)


class LLMProvider(ABC):
    """Interface for language model providers."""

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        needs_search: bool = False,
        **kwargs,
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

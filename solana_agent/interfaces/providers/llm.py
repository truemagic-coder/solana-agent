from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, BinaryIO, List, Literal, Type, TypeVar, Union

from pydantic import BaseModel


T = TypeVar('T', bound=BaseModel)


class LLMProvider(ABC):
    """Interface for language model providers."""

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> AsyncGenerator[str, None]:
        """Generate text from the language model."""
        pass

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        pass

    @abstractmethod
    async def parse_structured_output(
        self, prompt: str, system_prompt: str, model_class: Type[T],
    ) -> T:
        """Generate structured output using a specific model class."""
        pass

    @abstractmethod
    async def tts(
        self,
        text: str,
        instructions: str = "",
        voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                       "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
    ) -> AsyncGenerator[bytes, None]:
        """Stream text-to-speech audio from the language model."""
        pass

    @abstractmethod
    async def transcribe_audio(
        self,
        audio_file: Union[str, Path, BinaryIO],
    ) -> AsyncGenerator[str, None]:
        """Transcribe audio from the language model."""
        pass

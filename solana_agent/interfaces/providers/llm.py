from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Literal, Type, TypeVar, Union

from pydantic import BaseModel


T = TypeVar('T', bound=BaseModel)


class LLMProvider(ABC):
    """Interface for language model providers."""

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        internet_search: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Generate text from the language model."""
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
        instructions: str = "You speak in a friendly and helpful manner.",
        voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                       "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
        response_format: Literal['mp3', 'opus',
                                 'aac', 'flac', 'wav', 'pcm'] = "aac",
    ) -> AsyncGenerator[bytes, None]:
        """Stream text-to-speech audio from the language model."""
        pass

    @abstractmethod
    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
    ) -> AsyncGenerator[str, None]:
        """Transcribe audio from the language model."""
        pass

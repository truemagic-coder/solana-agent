from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Interface for language model providers."""

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Generate text from the language model."""
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response deltas and tool call deltas.

            Yields normalized events:
            - {"type": "content", "delta": str}
        - {"type": "tool_call_delta", "id": Optional[str], "index": Optional[int], "name": Optional[str], "arguments_delta": str}
            - {"type": "message_end", "finish_reason": str}
            - {"type": "error", "error": str}
        """
        pass

    @abstractmethod
    async def parse_structured_output(
        self,
        prompt: str,
        system_prompt: str,
        model_class: Type[T],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> T:
        """Generate structured output using a specific model class."""
        pass

    @abstractmethod
    async def tts(
        self,
        text: str,
        voice: Literal[
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "fable",
            "onyx",
            "nova",
            "sage",
            "shimmer",
        ] = "nova",
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "aac",
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

    @abstractmethod
    async def embed_text(
        self, text: str, model: Optional[str] = None, dimensions: Optional[int] = None
    ) -> List[float]:
        """
        Generate an embedding for the given text.

        Args:
            text: The text to embed.
            model: The embedding model to use.
            dimensions: Optional desired output dimensions for the embedding.

        Returns:
            A list of floats representing the embedding vector.
        """
        pass

    @abstractmethod
    async def generate_text_with_images(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        system_prompt: str = "",
        detail: Literal["low", "high", "auto"] = "auto",
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate text from the language model using images."""
        pass

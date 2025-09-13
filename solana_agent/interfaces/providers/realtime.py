from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Literal,
    Optional,
    Awaitable,
    Callable,
    List,
    Union,
)


@dataclass
class RealtimeSessionOptions:
    model: Optional[str] = None
    voice: Literal[
        "alloy",
        "ash",
        "ballad",
        "cedar",
        "coral",
        "echo",
        "marin",
        "sage",
        "shimmer",
        "verse",
    ] = "marin"
    vad_enabled: bool = True
    input_rate_hz: int = 24000
    output_rate_hz: int = 24000
    input_mime: str = "audio/pcm"  # 16-bit PCM
    output_mime: str = "audio/pcm"  # 16-bit PCM
    output_modalities: List[Literal["audio", "text"]] = None  # None means auto-detect
    instructions: Optional[str] = None
    # Optional: tools payload compatible with OpenAI Realtime session.update
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: str = "auto"
    # Tool execution behavior
    # Max time to allow a tool to run before timing out (seconds)
    tool_timeout_s: float = 300.0
    # Optional guard: if a tool takes longer than this to complete, skip sending
    # function_call_output to avoid stale/expired call_id issues. Set to None to always send.
    tool_result_max_age_s: Optional[float] = None
    # --- Realtime transcription configuration (optional) ---
    # When transcription_model is set, QueryService should skip the HTTP STT path and rely on
    # realtime websocket transcription events. Other fields customize that behavior.
    transcription_model: Optional[str] = None
    transcription_language: Optional[str] = None  # e.g. 'en'
    transcription_prompt: Optional[str] = None
    transcription_noise_reduction: Optional[bool] = None
    transcription_include_logprobs: bool = False


@dataclass
class RealtimeChunk:
    """Represents a chunk of data from a realtime session with its modality type."""

    modality: Literal["audio", "text"]
    data: Union[str, bytes]
    timestamp: Optional[float] = None  # Optional timestamp for ordering
    metadata: Optional[Dict[str, Any]] = None  # Optional additional metadata

    @property
    def is_audio(self) -> bool:
        """Check if this is an audio chunk."""
        return self.modality == "audio"

    @property
    def is_text(self) -> bool:
        """Check if this is a text chunk."""
        return self.modality == "text"

    @property
    def text_data(self) -> Optional[str]:
        """Get text data if this is a text chunk."""
        return self.data if isinstance(self.data, str) else None

    @property
    def audio_data(self) -> Optional[bytes]:
        """Get audio data if this is an audio chunk."""
        return self.data if isinstance(self.data, bytes) else None


async def separate_audio_chunks(
    chunks: AsyncGenerator[RealtimeChunk, None],
) -> AsyncGenerator[bytes, None]:
    """Extract only audio chunks from a stream of RealtimeChunk objects.

    Args:
        chunks: Stream of RealtimeChunk objects

    Yields:
        Audio data bytes from audio chunks only
    """
    async for chunk in chunks:
        if chunk.is_audio and chunk.audio_data:
            yield chunk.audio_data


async def separate_text_chunks(
    chunks: AsyncGenerator[RealtimeChunk, None],
) -> AsyncGenerator[str, None]:
    """Extract only text chunks from a stream of RealtimeChunk objects.

    Args:
        chunks: Stream of RealtimeChunk objects

    Yields:
        Text data from text chunks only
    """
    async for chunk in chunks:
        if chunk.is_text and chunk.text_data:
            yield chunk.text_data


async def demux_realtime_chunks(
    chunks: AsyncGenerator[RealtimeChunk, None],
) -> tuple[AsyncGenerator[bytes, None], AsyncGenerator[str, None]]:
    """Demux a stream of RealtimeChunk objects into separate audio and text streams.

    Note: This function consumes the input generator, so each output stream can only be consumed once.

    Args:
        chunks: Stream of RealtimeChunk objects

    Returns:
        Tuple of (audio_stream, text_stream) async generators
    """
    # Collect all chunks first since we can't consume the generator twice
    collected_chunks = []
    async for chunk in chunks:
        collected_chunks.append(chunk)

    async def audio_stream():
        for chunk in collected_chunks:
            if chunk.is_audio and chunk.audio_data:
                yield chunk.audio_data

    async def text_stream():
        for chunk in collected_chunks:
            if chunk.is_text and chunk.text_data:
                yield chunk.text_data

    return audio_stream(), text_stream()


class BaseRealtimeSession(ABC):
    """Abstract realtime session supporting bidirectional audio/text over WebSocket."""

    @abstractmethod
    async def connect(self) -> None:  # pragma: no cover
        pass

    @abstractmethod
    async def close(self) -> None:  # pragma: no cover
        pass

    # --- Client events ---
    @abstractmethod
    async def update_session(
        self, session_patch: Dict[str, Any]
    ) -> None:  # pragma: no cover
        pass

    @abstractmethod
    async def append_audio(self, pcm16_bytes: bytes) -> None:  # pragma: no cover
        """Append 16-bit PCM audio bytes (matching configured input rate/mime)."""
        pass

    @abstractmethod
    async def commit_input(self) -> None:  # pragma: no cover
        pass

    @abstractmethod
    async def clear_input(self) -> None:  # pragma: no cover
        pass

    @abstractmethod
    async def create_response(
        self, response_patch: Optional[Dict[str, Any]] = None
    ) -> None:  # pragma: no cover
        pass

    # --- Server events (demuxed) ---
    @abstractmethod
    def iter_events(self) -> AsyncGenerator[Dict[str, Any], None]:  # pragma: no cover
        pass

    @abstractmethod
    def iter_output_audio(self) -> AsyncGenerator[bytes, None]:  # pragma: no cover
        pass

    @abstractmethod
    def iter_input_transcript(self) -> AsyncGenerator[str, None]:  # pragma: no cover
        pass

    @abstractmethod
    def iter_output_transcript(self) -> AsyncGenerator[str, None]:  # pragma: no cover
        pass

    # --- Optional tool execution hook ---
    @abstractmethod
    def set_tool_executor(
        self,
        executor: Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]],
    ) -> None:  # pragma: no cover
        """Register a coroutine that executes a tool by name with arguments and returns a result dict."""
        pass

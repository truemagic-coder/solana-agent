from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Literal, Optional, Awaitable, Callable


@dataclass
class RealtimeSessionOptions:
    model: Optional[str] = None
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
    ] = "nova"
    vad_enabled: bool = True
    input_rate_hz: int = 24000
    output_rate_hz: int = 24000
    input_mime: str = "audio/pcm"  # 16-bit PCM
    output_mime: str = "audio/pcm"  # 16-bit PCM
    instructions: Optional[str] = None
    # Optional: tools payload compatible with OpenAI Realtime session.update
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: str = "auto"


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

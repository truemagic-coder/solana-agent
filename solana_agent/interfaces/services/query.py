from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel

from solana_agent.interfaces.services.routing import RoutingService as RoutingInterface
from solana_agent.interfaces.providers.realtime import RealtimeChunk


class QueryService(ABC):
    """Interface for processing user queries."""

    @abstractmethod
    async def process(
        self,
        user_id: str,
        query: Union[str, bytes],
        output_format: Literal["text", "audio"] = "text",
        realtime: bool = False,
        # Realtime minimal controls (voice/format come from audio_* args)
        vad: Optional[bool] = None,
        rt_encode_input: bool = False,
        rt_encode_output: bool = False,
        rt_output_modalities: Optional[List[Literal["audio", "text"]]] = None,
        rt_voice: Literal[
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
        ] = "marin",
        # Realtime transcription configuration (new)
        rt_transcription_model: Optional[str] = None,
        rt_transcription_language: Optional[str] = None,
        rt_transcription_prompt: Optional[str] = None,
        rt_transcription_noise_reduction: Optional[bool] = None,
        rt_transcription_include_logprobs: bool = False,
        # Prefer raw PCM passthrough for realtime output (overrides default aac when True and caller didn't request another format)
        rt_prefer_pcm: bool = False,
        # Optional override for realtime output sample rate (PCM). Defaults to provider/session default if None.
        rt_output_rate_hz: Optional[int] = None,
    # When True and output is audio, suppress emitting assistant text to caller while still persisting it to memory.
    rt_suppress_text_output: bool = False,
        audio_voice: Literal[
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
        audio_output_format: Literal[
            "mp3", "opus", "aac", "flac", "wav", "pcm"
        ] = "aac",
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
        prompt: Optional[str] = None,
        router: Optional[RoutingInterface] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        capture_schema: Optional[Dict[str, Any]] = None,
        capture_name: Optional[str] = None,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel, RealtimeChunk], None]:
        """Process the user request and generate a response."""
        pass

    @abstractmethod
    async def get_user_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "desc",  # "asc" for oldest-first, "desc" for newest-first
    ) -> Dict[str, Any]:
        """Get paginated message history for a user."""
        pass

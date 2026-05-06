"""Query orchestration for the thin single-agent public SDK."""

from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel

from solana_agent.interfaces.services.query import QueryService as QueryServiceInterface
from solana_agent.services.agent import AgentService


class QueryService(QueryServiceInterface):
    """Route every request to the configured single agent."""

    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service

    def _default_agent_name(self) -> str:
        agents = self.agent_service.get_all_ai_agents()
        if not agents:
            raise ValueError("No AI agent is configured.")
        return next(iter(agents))

    async def _normalize_query(
        self,
        query: Union[str, bytes],
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ],
    ) -> str:
        if isinstance(query, str):
            return query

        parts: list[str] = []
        async for chunk in self.agent_service.llm_provider.transcribe_audio(
            query,
            audio_input_format,
        ):
            parts.append(chunk)

        text = "".join(parts).strip()
        if not text:
            raise ValueError("Audio transcription returned no text")
        return text

    async def process(
        self,
        privy_user_id: str,
        query: Union[str, bytes],
        runtime_context: Optional[Dict[str, Any]] = None,
        output_format: Literal["text", "audio"] = "text",
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
        images: Optional[List[Union[str, bytes]]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        capture_schema: Optional[Dict[str, Any]] = None,
        capture_name: Optional[str] = None,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:  # pragma: no cover
        del capture_schema
        del capture_name

        user_text = await self._normalize_query(query, audio_input_format)
        async for chunk in self.agent_service.generate_response(
            agent_name=self._default_agent_name(),
            privy_user_id=privy_user_id,
            query=user_text,
            runtime_context=runtime_context,
            images=images,
            output_format=output_format,
            audio_voice=audio_voice,
            audio_output_format=audio_output_format,
            prompt=prompt,
            output_model=output_model,
        ):
            yield chunk

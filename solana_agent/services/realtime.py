from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, Optional

from solana_agent.interfaces.providers.realtime import (
    BaseRealtimeSession,
    RealtimeSessionOptions,
)
from solana_agent.interfaces.providers.audio import AudioTranscoder


class RealtimeService:
    """High-level service to manage a realtime audio session.

    Responsibilities:
    - Connect/close a realtime session (WebSocket-based)
    - Update voice and VAD at runtime via session.update
    - Append/commit/clear input audio buffers
    - Expose separate async generators for audio and input/output transcripts
    - Allow out-of-band response.create (e.g., text-to-speech without new audio)
    """

    def __init__(
        self,
        session: BaseRealtimeSession,
        options: Optional[RealtimeSessionOptions] = None,
        transcoder: Optional[AudioTranscoder] = None,
        accept_compressed_input: bool = False,
        client_input_mime: str = "audio/mp4",
        encode_output: bool = False,
        client_output_mime: str = "audio/aac",
    ) -> None:
        self._session = session
        self._options = options or RealtimeSessionOptions()
        self._connected = False
        self._lock = asyncio.Lock()
        self._transcoder = transcoder
        # Client-side transport controls (do not affect OpenAI session formats)
        self._accept_compressed_input = accept_compressed_input
        self._client_input_mime = client_input_mime
        self._encode_output = encode_output
        self._client_output_mime = client_output_mime

    async def start(self) -> None:
        async with self._lock:
            if self._connected:
                return
            await self._session.connect()
            self._connected = True

    async def stop(self) -> None:
        async with self._lock:
            if not self._connected:
                return
            await self._session.close()
            self._connected = False

    # --- Configuration ---
    async def configure(
        self,
        *,
        voice: Optional[str] = None,
        vad_enabled: Optional[bool] = None,
        instructions: Optional[str] = None,
        input_rate_hz: Optional[int] = None,
        output_rate_hz: Optional[int] = None,
        input_mime: Optional[str] = None,
        output_mime: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> None:
        """Update session settings (voice, VAD, formats, tools)."""
        patch: Dict[str, Any] = {}

        audio_patch: Dict[str, Any] = {}
        if input_mime or input_rate_hz is not None or vad_enabled is not None:
            turn_detection = None
            if vad_enabled is not None:
                if vad_enabled:
                    turn_detection = {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 200,
                        "create_response": True,
                        "interrupt_response": True,
                    }
                else:
                    turn_detection = None
            audio_patch["input"] = {
                "format": {
                    "type": input_mime or self._options.input_mime,
                    "rate": input_rate_hz or self._options.input_rate_hz,
                },
                "turn_detection": turn_detection,
            }

        if output_mime or output_rate_hz is not None or voice is not None:
            audio_patch["output"] = {
                "format": {
                    "type": output_mime or self._options.output_mime,
                    "rate": output_rate_hz or self._options.output_rate_hz,
                },
                "voice": voice or self._options.voice,
                "speed": 1.0,
            }

        if audio_patch:
            patch["audio"] = audio_patch

        if instructions is not None:
            patch["instructions"] = instructions
        if tools is not None:
            patch["tools"] = tools
        if tool_choice is not None:
            patch["tool_choice"] = tool_choice

        if patch:
            await self._session.update_session(patch)

        # Update local options snapshot
        if voice is not None:
            self._options.voice = voice
        if vad_enabled is not None:
            self._options.vad_enabled = vad_enabled
        if instructions is not None:
            self._options.instructions = instructions
        if input_rate_hz is not None:
            self._options.input_rate_hz = input_rate_hz
        if output_rate_hz is not None:
            self._options.output_rate_hz = output_rate_hz
        if input_mime is not None:
            self._options.input_mime = input_mime
        if output_mime is not None:
            self._options.output_mime = output_mime
        if tools is not None:
            self._options.tools = tools
        if tool_choice is not None:
            self._options.tool_choice = tool_choice

    # --- Audio input ---
    async def append_audio(self, chunk_bytes: bytes) -> None:
        """Accepts PCM16 by default; if accept_compressed_input is True, transcodes client audio to PCM16.

        This keeps the server session configured for PCM while allowing mobile clients to send MP4/AAC.
        """
        if self._accept_compressed_input:
            if not self._transcoder:
                raise ValueError(
                    "Compressed input enabled but no transcoder configured"
                )
            pcm16 = await self._transcoder.to_pcm16(
                chunk_bytes, self._client_input_mime, self._options.input_rate_hz
            )
            await self._session.append_audio(pcm16)
            return
        # Default: pass-through PCM16
        await self._session.append_audio(chunk_bytes)

    async def commit_input(self) -> None:
        await self._session.commit_input()

    async def clear_input(self) -> None:
        await self._session.clear_input()

    # --- Out-of-band response (e.g., TTS without new audio) ---
    async def create_response(
        self, response_patch: Optional[Dict[str, Any]] = None
    ) -> None:
        await self._session.create_response(response_patch)

    # --- Streams ---
    def iter_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        return self._session.iter_events()

    def iter_output_audio(self) -> AsyncGenerator[bytes, None]:
        return self._session.iter_output_audio()

    async def iter_output_audio_encoded(self) -> AsyncGenerator[bytes, None]:
        """If encode_output is True and a transcoder exists, encode PCM16 to client_output_mime (e.g., AAC)."""
        async for chunk in self._session.iter_output_audio():
            if self._encode_output and self._transcoder:
                yield await self._transcoder.from_pcm16(
                    chunk, self._client_output_mime, self._options.output_rate_hz
                )
            else:
                yield chunk

    def iter_input_transcript(self) -> AsyncGenerator[str, None]:
        return self._session.iter_input_transcript()

    def iter_output_transcript(self) -> AsyncGenerator[str, None]:
        return self._session.iter_output_transcript()

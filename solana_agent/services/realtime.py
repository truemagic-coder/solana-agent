from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from solana_agent.interfaces.providers.realtime import (
    BaseRealtimeSession,
    RealtimeSessionOptions,
    RealtimeChunk,
)
from solana_agent.interfaces.providers.audio import AudioTranscoder

logger = logging.getLogger(__name__)


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

    async def start(self) -> None:  # pragma: no cover
        async with self._lock:
            if self._connected:
                return
            logger.info("RealtimeService: starting session")
            await self._session.connect()
            self._connected = True

    async def stop(self) -> None:  # pragma: no cover
        async with self._lock:
            if not self._connected:
                return
            logger.info("RealtimeService: stopping session")
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
    ) -> None:  # pragma: no cover
        """Update session settings (voice, VAD, formats, tools)."""
        patch: Dict[str, Any] = {}

        audio_patch: Dict[str, Any] = {}
        if input_mime or input_rate_hz is not None or vad_enabled is not None:
            turn_detection = None
            if vad_enabled is not None:
                if vad_enabled:
                    turn_detection = {
                        "type": "semantic_vad",
                        "create_response": True,
                    }
                else:
                    turn_detection = None
            audio_patch["input"] = {
                "format": "pcm16",  # session is fixed to PCM16 server-side
                "turn_detection": turn_detection,
            }

        if output_mime or output_rate_hz is not None or voice is not None:
            # Only configure audio output if audio is in the output modalities
            modalities = (
                self._options.output_modalities
                if self._options.output_modalities is not None
                else ["audio"]
            )
            if "audio" in modalities:
                audio_patch["output"] = {
                    "format": "pcm16",  # session is fixed to PCM16 server-side
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
            logger.debug("RealtimeService.configure patch: %s", patch)
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
    async def append_audio(self, chunk_bytes: bytes) -> None:  # pragma: no cover
        """Accepts PCM16 by default; if accept_compressed_input is True, transcodes client audio to PCM16.

        This keeps the server session configured for PCM while allowing mobile clients to send MP4/AAC.
        """
        logger.debug(
            "RealtimeService.append_audio: len=%d, accept_compressed_input=%s, client_input_mime=%s",
            len(chunk_bytes),
            self._accept_compressed_input,
            self._client_input_mime,
        )
        if self._accept_compressed_input:
            if not self._transcoder:
                raise ValueError(
                    "Compressed input enabled but no transcoder configured"
                )
            pcm16 = await self._transcoder.to_pcm16(
                chunk_bytes, self._client_input_mime, self._options.input_rate_hz
            )
            await self._session.append_audio(pcm16)
            logger.debug("RealtimeService.append_audio: sent PCM16 len=%d", len(pcm16))
            return
        # Default: pass-through PCM16
        await self._session.append_audio(chunk_bytes)
        logger.debug(
            "RealtimeService.append_audio: sent passthrough len=%d", len(chunk_bytes)
        )

    async def commit_input(self) -> None:  # pragma: no cover
        logger.debug("RealtimeService.commit_input")
        await self._session.commit_input()

    async def clear_input(self) -> None:  # pragma: no cover
        logger.debug("RealtimeService.clear_input")
        await self._session.clear_input()

    # --- Out-of-band response (e.g., TTS without new audio) ---
    async def create_conversation_item(
        self, item: Dict[str, Any]
    ) -> None:  # pragma: no cover
        """Create a conversation item (e.g., for text input)."""
        await self._session.create_conversation_item(item)

    async def create_response(  # pragma: no cover
        self, response_patch: Optional[Dict[str, Any]] = None
    ) -> None:
        await self._session.create_response(response_patch)

    # --- Streams ---
    def iter_events(self) -> AsyncGenerator[Dict[str, Any], None]:  # pragma: no cover
        return self._session.iter_events()

    def iter_output_audio(self) -> AsyncGenerator[bytes, None]:  # pragma: no cover
        return self._session.iter_output_audio()

    def reset_output_stream(self) -> None:  # pragma: no cover
        try:
            if hasattr(self._session, "reset_output_stream"):
                self._session.reset_output_stream()
        except Exception:
            pass

    async def iter_output_audio_encoded(
        self,
    ) -> AsyncGenerator[RealtimeChunk, None]:  # pragma: no cover
        """Stream PCM16 audio as RealtimeChunk objects, tolerating long tool executions by waiting while calls are pending.

        - If no audio arrives immediately, we keep waiting as long as a function/tool call is pending.
        - Bridge across multiple audio segments (e.g., pre-call and post-call responses).
        - Only end the stream when no audio is available and no pending tool call remains.
        """

        def _has_pending_tool() -> bool:
            try:
                return bool(
                    getattr(self._session, "has_pending_tool_call", lambda: False)()
                )
            except Exception:
                return False

        async def _produce_pcm():
            max_wait_pending_sec = 600.0  # allow up to 10 minutes while tools run
            waited_while_pending = 0.0
            base_idle_timeout = 12.0
            idle_slice = 1.0

            while True:
                gen = self._session.iter_output_audio()
                try:
                    # Inner loop for one segment until generator ends
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                gen.__anext__(), timeout=idle_slice
                            )
                        except asyncio.TimeoutError:
                            if _has_pending_tool():
                                waited_while_pending += idle_slice
                                if waited_while_pending <= max_wait_pending_sec:
                                    continue
                                else:
                                    logger.warning(
                                        "RealtimeService: exceeded max pending-tool wait; ending stream"
                                    )
                                    return
                            else:
                                # No pending tool: accumulate idle time; stop after base timeout
                                waited_while_pending += idle_slice
                                if waited_while_pending >= base_idle_timeout:
                                    logger.warning(
                                        "RealtimeService: idle with no pending tool; ending stream"
                                    )
                                    return
                                continue
                        # Got a chunk; reset idle counter and yield
                        waited_while_pending = 0.0
                        if not chunk:
                            continue
                        yield chunk
                except StopAsyncIteration:
                    # Segment ended; if a tool is pending, continue to next segment
                    if _has_pending_tool():
                        await asyncio.sleep(0.25)
                        continue
                    # Otherwise, no more audio segments expected
                    return

        if self._encode_output and self._transcoder:
            async for out in self._transcoder.stream_from_pcm16(
                _produce_pcm(), self._client_output_mime, self._options.output_rate_hz
            ):
                yield RealtimeChunk(modality="audio", data=out)
        else:
            async for chunk in _produce_pcm():
                yield RealtimeChunk(modality="audio", data=chunk)

    async def iter_output_combined(
        self,
    ) -> AsyncGenerator[RealtimeChunk, None]:  # pragma: no cover
        """Stream both audio and text chunks as RealtimeChunk objects.

        This method combines audio and text streams when both modalities are enabled.
        Audio chunks are yielded as they arrive, and text chunks are yielded as transcript deltas arrive.
        """

        # Determine which modalities to stream based on session options
        modalities = (
            self._options.output_modalities
            if self._options.output_modalities is not None
            else ["audio"]
        )
        should_stream_audio = "audio" in modalities
        should_stream_text = "text" in modalities

        if not should_stream_audio and not should_stream_text:
            return  # No modalities requested

        # Create tasks for both streams if needed
        tasks = []
        queues = []

        if should_stream_audio:
            audio_queue = asyncio.Queue()
            queues.append(audio_queue)

            async def _collect_audio():
                try:
                    async for chunk in self.iter_output_audio_encoded():
                        await audio_queue.put(chunk)
                finally:
                    await audio_queue.put(None)  # Sentinel

            tasks.append(asyncio.create_task(_collect_audio()))

        if should_stream_text:
            text_queue = asyncio.Queue()
            queues.append(text_queue)

            async def _collect_text():
                try:
                    async for text_chunk in self.iter_output_transcript():
                        if text_chunk:  # Only yield non-empty text chunks
                            await text_queue.put(
                                RealtimeChunk(modality="text", data=text_chunk)
                            )
                finally:
                    await text_queue.put(None)  # Sentinel

            tasks.append(asyncio.create_task(_collect_text()))

        try:
            # Collect chunks from all queues
            active_queues = len(queues)

            while active_queues > 0:
                for queue in queues:
                    try:
                        chunk = queue.get_nowait()
                        if chunk is None:
                            active_queues -= 1
                        else:
                            yield chunk
                    except asyncio.QueueEmpty:
                        continue

                # Small delay to prevent busy waiting
                if active_queues > 0:
                    await asyncio.sleep(0.01)

        finally:
            # Cancel all tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

    def iter_input_transcript(self) -> AsyncGenerator[str, None]:  # pragma: no cover
        return self._session.iter_input_transcript()

    def iter_output_transcript(self) -> AsyncGenerator[str, None]:  # pragma: no cover
        return self._session.iter_output_transcript()


class TwinRealtimeService:
    """Orchestrates two realtime sessions in parallel:

    - conversation: full duplex (audio out + assistant transcript, tools, etc.)
    - transcription: transcription-only session per GA (input transcript deltas)

    Audio input is fanned out to both sessions. Output audio is sourced from the
    conversation session only. Input transcript is sourced from the transcription
    session only. This aligns with the GA guidance to use a dedicated
    transcription session for reliable realtime STT, while the conversation
    session handles assistant speech.
    """

    def __init__(
        self,
        conversation: BaseRealtimeSession,
        transcription: BaseRealtimeSession,
        *,
        conv_options: Optional[RealtimeSessionOptions] = None,
        trans_options: Optional[RealtimeSessionOptions] = None,
        transcoder: Optional[AudioTranscoder] = None,
        accept_compressed_input: bool = False,
        client_input_mime: str = "audio/mp4",
        encode_output: bool = False,
        client_output_mime: str = "audio/aac",
    ) -> None:
        self._conv = conversation
        self._trans = transcription
        self._conv_opts = conv_options or RealtimeSessionOptions()
        self._trans_opts = trans_options or RealtimeSessionOptions()
        self._transcoder = transcoder
        self._accept_compressed_input = accept_compressed_input
        self._client_input_mime = client_input_mime
        self._encode_output = encode_output
        self._client_output_mime = client_output_mime
        self._connected = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:  # pragma: no cover
        async with self._lock:
            if self._connected:
                return
            logger.info("TwinRealtimeService: starting conversation + transcription")
            await asyncio.gather(self._conv.connect(), self._trans.connect())
            self._connected = True

    async def stop(self) -> None:  # pragma: no cover
        async with self._lock:
            if not self._connected:
                return
            logger.info("TwinRealtimeService: stopping both sessions")
            try:
                await asyncio.gather(self._conv.close(), self._trans.close())
            finally:
                self._connected = False

    async def reconnect(self) -> None:  # pragma: no cover
        async with self._lock:
            try:
                await asyncio.gather(self._conv.close(), self._trans.close())
            except Exception:
                pass
            self._connected = False
            await self.start()

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
    ) -> None:  # pragma: no cover
        # Only the conversation session needs voice/tools; transcription session
        # already has its own VAD model configured at connect-time.
        patch: Dict[str, Any] = {}
        audio_patch: Dict[str, Any] = {}
        if (
            vad_enabled is not None
            or input_rate_hz is not None
            or input_mime is not None
        ):
            turn_detection = None
            if vad_enabled is not None:
                if vad_enabled:
                    turn_detection = {"type": "semantic_vad", "create_response": True}
                else:
                    turn_detection = None
            audio_patch["input"] = {"format": "pcm16", "turn_detection": turn_detection}
        if output_rate_hz is not None or output_mime is not None or voice is not None:
            # Only configure audio output if audio is in the output modalities
            modalities = (
                self._conv_opts.output_modalities
                if self._conv_opts.output_modalities is not None
                else ["audio"]
            )
            if "audio" in modalities:
                audio_patch["output"] = {
                    "format": "pcm16",
                    "voice": voice or self._conv_opts.voice,
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
            logger.debug("TwinRealtimeService.configure patch (conv): %s", patch)
            await self._conv.update_session(patch)

        # Update local snapshots
        if voice is not None:
            self._conv_opts.voice = voice
        if vad_enabled is not None:
            self._conv_opts.vad_enabled = vad_enabled
            self._trans_opts.vad_enabled = vad_enabled
        if instructions is not None:
            self._conv_opts.instructions = instructions
        if input_rate_hz is not None:
            self._conv_opts.input_rate_hz = input_rate_hz
            self._trans_opts.input_rate_hz = input_rate_hz
        if output_rate_hz is not None:
            self._conv_opts.output_rate_hz = output_rate_hz
        if input_mime is not None:
            self._conv_opts.input_mime = input_mime
            self._trans_opts.input_mime = input_mime
        if output_mime is not None:
            self._conv_opts.output_mime = output_mime
        if tools is not None:
            self._conv_opts.tools = tools
        if tool_choice is not None:
            self._conv_opts.tool_choice = tool_choice

    async def append_audio(self, chunk_bytes: bytes) -> None:  # pragma: no cover
        # Transcode once if needed, then fan out to both
        if self._accept_compressed_input:
            if not self._transcoder:
                raise ValueError(
                    "Compressed input enabled but no transcoder configured"
                )
            pcm16 = await self._transcoder.to_pcm16(
                chunk_bytes, self._client_input_mime, self._conv_opts.input_rate_hz
            )
            await asyncio.gather(
                self._conv.append_audio(pcm16), self._trans.append_audio(pcm16)
            )
            return
        await asyncio.gather(
            self._conv.append_audio(chunk_bytes),
            self._trans.append_audio(chunk_bytes),
        )

    async def commit_input(self) -> None:  # pragma: no cover
        await asyncio.gather(self._conv.commit_input(), self._trans.commit_input())

    async def commit_conversation(self) -> None:  # pragma: no cover
        await self._conv.commit_input()

    async def commit_transcription(self) -> None:  # pragma: no cover
        await self._trans.commit_input()

    async def clear_input(self) -> None:  # pragma: no cover
        await asyncio.gather(self._conv.clear_input(), self._trans.clear_input())

    async def create_conversation_item(
        self, item: Dict[str, Any]
    ) -> None:  # pragma: no cover
        """Create a conversation item (e.g., for text input)."""
        await self._conv.create_conversation_item(item)

    async def create_response(
        self, response_patch: Optional[Dict[str, Any]] = None
    ) -> None:  # pragma: no cover
        # Only conversation session creates assistant responses
        await self._conv.create_response(response_patch)

    # --- Streams ---
    def iter_events(self) -> AsyncGenerator[Dict[str, Any], None]:  # pragma: no cover
        # Prefer conversation events; caller can listen to transcription via iter_input_transcript
        return self._conv.iter_events()

    def iter_output_audio(self) -> AsyncGenerator[bytes, None]:  # pragma: no cover
        return self._conv.iter_output_audio()

    def reset_output_stream(self) -> None:  # pragma: no cover
        try:
            if hasattr(self._conv, "reset_output_stream"):
                self._conv.reset_output_stream()
        except Exception:
            pass

    async def iter_output_audio_encoded(
        self,
    ) -> AsyncGenerator[RealtimeChunk, None]:  # pragma: no cover
        # Reuse the same encoding pipeline as RealtimeService but source from conversation
        pcm_gen = self._conv.iter_output_audio()

        try:
            first_chunk = await asyncio.wait_for(pcm_gen.__anext__(), timeout=12.0)
        except StopAsyncIteration:
            logger.warning("TwinRealtimeService: no PCM produced (ended immediately)")
            return
        except asyncio.TimeoutError:
            logger.warning("TwinRealtimeService: no PCM within timeout; closing conv")
            try:
                # Close both sessions to ensure clean restart on next turn
                await asyncio.gather(self._conv.close(), self._trans.close())
                self._connected = False
            except Exception:
                pass
            return

        async def _pcm_iter():
            if first_chunk:
                yield first_chunk
            async for c in pcm_gen:
                if not c:
                    continue
                yield c

        if self._encode_output and self._transcoder:
            async for out in self._transcoder.stream_from_pcm16(
                _pcm_iter(), self._client_output_mime, self._conv_opts.output_rate_hz
            ):
                yield RealtimeChunk(modality="audio", data=out)
        else:
            async for chunk in _pcm_iter():
                yield RealtimeChunk(modality="audio", data=chunk)

    def iter_input_transcript(self) -> AsyncGenerator[str, None]:  # pragma: no cover
        return self._trans.iter_input_transcript()

    def iter_output_transcript(self) -> AsyncGenerator[str, None]:  # pragma: no cover
        return self._conv.iter_output_transcript()

    def iter_transcription_events(
        self,
    ) -> AsyncGenerator[Dict[str, Any], None]:  # pragma: no cover
        # Expose transcription session events for completion detection
        return self._trans.iter_events()

    def is_connected(self) -> bool:  # pragma: no cover
        return self._connected

    def set_tool_executor(self, executor) -> None:  # pragma: no cover
        # Forward to conversation session (tools only apply there)
        if hasattr(self._conv, "set_tool_executor"):
            self._conv.set_tool_executor(executor)

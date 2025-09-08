from __future__ import annotations

import base64
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional
import asyncio

import websockets

from solana_agent.interfaces.providers.realtime import (
    BaseRealtimeSession,
    RealtimeSessionOptions,
)

logger = logging.getLogger(__name__)


class OpenAIRealtimeWebSocketSession(BaseRealtimeSession):
    """OpenAI Realtime WebSocket session (server-to-server) with audio in/out.

    Notes:
    - Expects PCM16 audio at configured sample rate (default 24kHz).
    - Emits raw PCM16 bytes for output audio deltas.
    - Provides separate async generators for input/output transcripts.
    - You can toggle VAD via session.update and manually commit when disabled.
    """

    def __init__(
        self,
        api_key: str,
        url: str = "wss://api.openai.com/v1/realtime",
        options: Optional[RealtimeSessionOptions] = None,
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.options = options or RealtimeSessionOptions()

        self._ws = None
        self._event_queue = asyncio.Queue()
        self._audio_queue = asyncio.Queue()
        self._in_tr_queue = asyncio.Queue()
        self._out_tr_queue = asyncio.Queue()
        self._recv_task = None
        self._tool_executor = None
        self._pending_calls = {}
        self._last_input_item_id: Optional[str] = None
        self._commit_evt: asyncio.Event = asyncio.Event()

    async def connect(self) -> None:  # pragma: no cover
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
        ]
        model = self.options.model or "gpt-realtime"
        uri = f"{self.url}?model={model}"
        logger.info(
            "Realtime WS connecting: uri=%s, input=%s@%sHz, output=%s@%sHz, voice=%s, vad=%s",
            uri,
            self.options.input_mime,
            self.options.input_rate_hz,
            self.options.output_mime,
            self.options.output_rate_hz,
            self.options.voice,
            self.options.vad_enabled,
        )
        self._ws = await websockets.connect(
            uri, additional_headers=headers, max_size=None
        )
        logger.info("Connected to OpenAI Realtime WS: %s", uri)
        self._recv_task = asyncio.create_task(self._recv_loop())

        # Configure session (instructions, tools). VAD handled per-request or defaults.

        vad_cfg = (
            {"type": "semantic_vad", "create_response": True}
            if self.options.vad_enabled
            else None
        )

        # Build optional prompt block
        prompt_block = None
        if getattr(self.options, "prompt_id", None):
            prompt_block = {
                "id": self.options.prompt_id,
            }
            if getattr(self.options, "prompt_version", None):
                prompt_block["version"] = self.options.prompt_version
            if getattr(self.options, "prompt_variables", None):
                prompt_block["variables"] = self.options.prompt_variables

        # Note: keep output format as "pcm16" so downstream ffmpeg expects s16le input
        session_payload: Dict[str, Any] = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.options.model or "gpt-realtime",
                "output_modalities": ["audio", "text"],
                "audio": {
                    "input": {
                        "format": "pcm16",
                        "turn_detection": vad_cfg,  # null when VAD disabled
                    },
                    "output": {
                        "format": "pcm16",
                        "voice": self.options.voice or "nova",
                        "speed": float(
                            getattr(self.options, "voice_speed", 1.0) or 1.0
                        ),
                    },
                },
                # Optional server-stored prompt
                **({"prompt": prompt_block} if prompt_block else {}),
                # Direct overrides
                "instructions": self.options.instructions or "",
                "tools": self.options.tools or [],
                "tool_choice": self.options.tool_choice,
            },
        }
        logger.info(
            "Realtime WS: sending session.update (voice=%s, vad=%s, output=%s)",
            self.options.voice,
            self.options.vad_enabled,
            session_payload["session"]["audio"]["output"]["format"],
        )
        await self._send(session_payload)

    async def close(self) -> None:  # pragma: no cover
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None

    async def _send(self, payload: Dict[str, Any]) -> None:  # pragma: no cover
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        try:
            await self._ws.send(json.dumps(payload))
        finally:
            try:
                ptype = payload.get("type")
            except Exception:
                ptype = str(type(payload))
            logger.debug("WS send: %s", ptype)

    async def _recv_loop(self) -> None:  # pragma: no cover
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                    etype = data.get("type")
                    logger.debug("WS recv: %s", etype)
                    # Demux streams
                    if etype in ("response.output_audio.delta", "response.audio.delta"):
                        b64 = data.get("delta") or ""
                        if b64:
                            try:
                                chunk = base64.b64decode(b64)
                                self._audio_queue.put_nowait(chunk)
                                logger.debug("Audio delta bytes=%d", len(chunk))
                            except Exception:
                                pass
                    elif etype in ("response.text.delta", "response.output_text.delta"):
                        metadata = data.get("response", {}).get("metadata", {})
                        if metadata.get("type") == "transcription":
                            delta = data.get("delta") or ""
                            if delta:
                                self._in_tr_queue.put_nowait(delta)
                                logger.info("Input transcript delta: %r", delta[:120])
                            else:
                                logger.debug("Input transcript delta: empty")
                    elif etype == "conversation.item.input_audio_transcription.delta":
                        delta = data.get("delta") or ""
                        if delta:
                            self._in_tr_queue.put_nowait(delta)
                            logger.info("Input transcript delta (GA): %r", delta[:120])
                        else:
                            logger.debug("Input transcript delta (GA): empty")
                    elif etype in (
                        "response.output_audio_transcript.delta",
                        "response.audio_transcript.delta",
                    ):
                        delta = data.get("delta") or ""
                        if delta:
                            self._out_tr_queue.put_nowait(delta)
                            logger.debug("Output transcript delta: %r", delta[:120])
                    elif etype == "input_audio_buffer.committed":
                        # Track the committed audio item id for OOB transcription referencing
                        item_id = data.get("item_id") or data.get("id")
                        if item_id:
                            self._last_input_item_id = item_id
                            logger.info(
                                "Realtime WS: input_audio_buffer committed: item_id=%s",
                                item_id,
                            )
                            try:
                                self._commit_evt.set()
                            except Exception:
                                pass
                    elif etype in (
                        "response.output_audio.done",
                        "response.audio.done",
                    ):
                        # End of audio stream for the response; stop audio iterator but keep WS open for transcripts
                        logger.info(
                            "Realtime WS: output audio done; ending audio stream"
                        )
                        try:
                            self._audio_queue.put_nowait(None)
                        except Exception:
                            pass
                        # Don't break; we still want to receive input transcription events
                    elif etype in (
                        "response.completed",
                        "response.complete",
                        "response.done",
                    ):
                        metadata = data.get("response", {}).get("metadata", {})
                        if metadata.get("type") == "response":
                            logger.info(
                                "Realtime WS: main response completed; ending audio stream"
                            )
                            try:
                                self._audio_queue.put_nowait(None)
                            except Exception:
                                pass
                        elif metadata.get("type") == "transcription":
                            # Transcription completed
                            logger.info("Realtime WS: transcription response completed")
                    elif etype in ("response.text.done", "response.output_text.done"):
                        metadata = data.get("response", {}).get("metadata", {})
                        if metadata.get("type") == "transcription":
                            tr = data.get("text") or ""
                            if tr:
                                try:
                                    self._in_tr_queue.put_nowait(tr)
                                    logger.debug(
                                        "Input transcript completed: %r", tr[:120]
                                    )
                                except Exception:
                                    pass
                    elif (
                        etype == "conversation.item.input_audio_transcription.completed"
                    ):
                        tr = data.get("transcript") or ""
                        if tr:
                            try:
                                self._in_tr_queue.put_nowait(tr)
                                logger.debug(
                                    "Input transcript completed (GA): %r", tr[:120]
                                )
                            except Exception:
                                pass
                    elif etype == "session.updated":
                        sess = data.get("session", {})
                        voice = sess.get("voice")
                        instr = sess.get("instructions")
                        logger.info(
                            "Realtime WS: session updated: voice=%s, instructions=%s",
                            voice,
                            instr[:100] if instr else None,
                        )
                    elif etype == "session.created":
                        logger.info("Realtime WS: session created")
                    # Always also publish raw events
                    try:
                        self._event_queue.put_nowait(data)
                    except Exception:
                        pass
                    # Handle tool/function calls automatically if provided by server events
                    if etype == "response.function_call.delta":
                        # Accumulate arguments by call id
                        # The OpenAI Realtime server emits function call deltas; we buffer by call_id
                        call = data.get("function_call", {})
                        call_id = call.get("id") or data.get("id")
                        name = call.get("name")
                        arguments_delta = call.get("arguments_delta", "")
                        if not hasattr(self, "_pending_calls"):
                            self._pending_calls = {}
                        pc = self._pending_calls.setdefault(
                            call_id, {"name": name, "args": ""}
                        )
                        if name:
                            pc["name"] = name
                        if arguments_delta:
                            pc["args"] += arguments_delta
                        logger.debug(
                            "Function call delta: id=%s name=%s args_len=%d",
                            call_id,
                            name,
                            len(pc.get("args", "")),
                        )
                    elif etype == "response.function_call":
                        # Finalize call and execute
                        call = data.get("function_call", {})
                        call_id = call.get("id") or data.get("id")
                        pc = getattr(self, "_pending_calls", {}).pop(call_id, None)
                        if pc and self._tool_executor and pc.get("name"):
                            try:
                                logger.info(
                                    "Executing tool: id=%s name=%s",
                                    call_id,
                                    pc.get("name"),
                                )
                                args = pc.get("args") or "{}"
                                try:
                                    parsed = json.loads(args)
                                except Exception:
                                    parsed = {}
                                result = await self._tool_executor(pc["name"], parsed)
                                await self._send(
                                    {
                                        "type": "response.function_call.output",
                                        "function_call": {
                                            "id": call_id,
                                            "output": json.dumps(result),
                                        },
                                    }
                                )
                                logger.info("Tool result sent: id=%s", call_id)
                            except Exception:
                                # Send minimal failure output to avoid stalling the stream
                                try:
                                    await self._send(
                                        {
                                            "type": "response.function_call.output",
                                            "function_call": {
                                                "id": call_id,
                                                "output": json.dumps(
                                                    {"error": "tool_execution_failed"}
                                                ),
                                            },
                                        }
                                    )
                                    logger.warning(
                                        "Tool execution failed; failure output sent: id=%s",
                                        call_id,
                                    )
                                except Exception:
                                    pass
                except Exception:
                    continue
        except Exception:
            logger.exception("Realtime WS receive loop error")
        finally:
            # Close queues gracefully
            for q in (
                self._audio_queue,
                self._in_tr_queue,
                self._out_tr_queue,
                self._event_queue,
            ):
                try:
                    q.put_nowait(None)  # type: ignore
                except Exception:
                    pass

    # --- Client event helpers ---
    async def update_session(
        self, session_patch: Dict[str, Any]
    ) -> None:  # pragma: no cover
        await self._send({"type": "session.update", "session": session_patch})

    async def append_audio(self, pcm16_bytes: bytes) -> None:  # pragma: no cover
        b64 = base64.b64encode(pcm16_bytes).decode("ascii")
        await self._send({"type": "input_audio_buffer.append", "audio": b64})

    async def commit_input(self) -> None:  # pragma: no cover
        await self._send({"type": "input_audio_buffer.commit"})

    async def clear_input(self) -> None:  # pragma: no cover
        await self._send({"type": "input_audio_buffer.clear"})

    async def create_response(
        self, response_patch: Optional[Dict[str, Any]] = None
    ) -> None:  # pragma: no cover
        # Wait briefly for commit event so we can reference the audio item
        if not self._last_input_item_id:
            try:
                await asyncio.wait_for(self._commit_evt.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
        input_arr = []
        if self._last_input_item_id:
            input_arr.append({"type": "item_reference", "id": self._last_input_item_id})
        transcription_payload = {
            "type": "response.create",
            "response": {
                # Prefer out-of-band, but if no item id yet, allow default conversation
                **({"conversation": "none"} if input_arr else {}),
                "metadata": {"type": "transcription"},
                "modalities": ["text"],
                "instructions": "Transcribe the user's input audio accurately.",
                **({"input": input_arr} if input_arr else {}),
            },
        }
        await self._send(transcription_payload)

        # Then, create main response
        payload: Dict[str, Any] = {"type": "response.create"}
        if response_patch:
            payload["response"] = response_patch
        # Ensure default audio modality and voice are present if not explicitly set
        if "response" not in payload:
            payload["response"] = {}
        rp = payload["response"]
        rp.setdefault("conversation", "none")
        rp.setdefault(
            "modalities", ["audio"]
        )  # lock to audio-only unless client requests text
        rp.setdefault("audio", {"voice": self.options.voice, "format": "pcm16"})
        rp.setdefault("metadata", {"type": "response"})
        # Attach input reference so the model links this response to last audio
        if self._last_input_item_id and "input" not in rp:
            rp["input"] = [{"type": "item_reference", "id": self._last_input_item_id}]
        await self._send(payload)

    # --- Streams ---
    async def _iter_queue(self, q) -> AsyncGenerator[Any, None]:
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    def iter_events(self) -> AsyncGenerator[Dict[str, Any], None]:  # pragma: no cover
        return self._iter_queue(self._event_queue)

    def iter_output_audio(self) -> AsyncGenerator[bytes, None]:  # pragma: no cover
        return self._iter_queue(self._audio_queue)

    def iter_input_transcript(self) -> AsyncGenerator[str, None]:  # pragma: no cover
        return self._iter_queue(self._in_tr_queue)

    def iter_output_transcript(self) -> AsyncGenerator[str, None]:  # pragma: no cover
        return self._iter_queue(self._out_tr_queue)

    def set_tool_executor(self, executor):  # pragma: no cover
        self._tool_executor = executor


class OpenAITranscriptionWebSocketSession(BaseRealtimeSession):
    """OpenAI Realtime Transcription WebSocket session.

    This session is transcription-only per GA docs. It accepts PCM16 input and emits
    conversation.item.input_audio_transcription.* events.
    """

    def __init__(
        self,
        api_key: str,
        url: str = "wss://api.openai.com/v1/realtime",
        options: Optional[RealtimeSessionOptions] = None,
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.options = options or RealtimeSessionOptions()
        self._ws = None
        self._event_queue = asyncio.Queue()
        self._in_tr_queue = asyncio.Queue()
        self._recv_task = None
        self._last_input_item_id: Optional[str] = None

    async def connect(self) -> None:  # pragma: no cover
        headers = [("Authorization", f"Bearer {self.api_key}")]
        # Model is for TTS session; transcription model is set in session update
        model = self.options.model or "gpt-realtime"
        uri = f"{self.url}?model={model}"
        logger.info("Transcription WS connecting: uri=%s", uri)
        self._ws = await websockets.connect(
            uri, additional_headers=headers, max_size=None
        )
        self._recv_task = asyncio.create_task(self._recv_loop())

        # Transcription session config per GA
        ts_payload: Dict[str, Any] = {
            "type": "transcription_session.update",
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": getattr(self.options, "transcribe_model", None)
                or "gpt-4o-mini-transcribe",
                **(
                    {"prompt": getattr(self.options, "transcribe_prompt", "")}
                    if getattr(self.options, "transcribe_prompt", None) is not None
                    else {}
                ),
                **(
                    {"language": getattr(self.options, "transcribe_language", "en")}
                    if getattr(self.options, "transcribe_language", None) is not None
                    else {}
                ),
            },
            "turn_detection": (
                {"type": "server_vad"} if self.options.vad_enabled else None
            ),
            # Optionally include extra properties (e.g., logprobs)
            # "include": ["item.input_audio_transcription.logprobs"],
        }
        logger.info("Transcription WS: sending transcription_session.update")
        await self._send(ts_payload)

    async def close(self) -> None:  # pragma: no cover
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None

    async def _send(self, payload: Dict[str, Any]) -> None:  # pragma: no cover
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(json.dumps(payload))

    async def _recv_loop(self) -> None:  # pragma: no cover
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                    etype = data.get("type")
                    # Temporarily log at INFO to diagnose missing events
                    logger.info("Transcription WS recv: %s", etype)
                    if etype == "input_audio_buffer.committed":
                        self._last_input_item_id = data.get("item_id") or data.get("id")
                        if self._last_input_item_id:
                            logger.info(
                                "Transcription WS: input_audio_buffer committed: item_id=%s",
                                self._last_input_item_id,
                            )
                    elif etype in (
                        "conversation.item.input_audio_transcription.delta",
                        "input_audio_transcription.delta",
                        "response.input_audio_transcription.delta",
                    ) or (
                        isinstance(etype, str)
                        and etype.endswith("input_audio_transcription.delta")
                    ):
                        delta = data.get("delta") or ""
                        if delta:
                            self._in_tr_queue.put_nowait(delta)
                            logger.info("Transcription delta: %r", delta[:120])
                    elif etype in (
                        "conversation.item.input_audio_transcription.completed",
                        "input_audio_transcription.completed",
                        "response.input_audio_transcription.completed",
                    ) or (
                        isinstance(etype, str)
                        and etype.endswith("input_audio_transcription.completed")
                    ):
                        tr = data.get("transcript") or ""
                        if tr:
                            self._in_tr_queue.put_nowait(tr)
                            logger.debug("Transcription completed: %r", tr[:120])
                    # Always publish raw events
                    try:
                        self._event_queue.put_nowait(data)
                    except Exception:
                        pass
                except Exception:
                    continue
        except Exception:
            logger.exception("Transcription WS receive loop error")
        finally:
            for q in (self._in_tr_queue, self._event_queue):
                try:
                    q.put_nowait(None)  # type: ignore
                except Exception:
                    pass

    # --- Client events ---
    async def update_session(
        self, session_patch: Dict[str, Any]
    ) -> None:  # pragma: no cover
        # Allow updating transcription session fields
        patch = {"type": "transcription_session.update", **session_patch}
        await self._send(patch)

    async def append_audio(self, pcm16_bytes: bytes) -> None:  # pragma: no cover
        b64 = base64.b64encode(pcm16_bytes).decode("ascii")
        await self._send({"type": "input_audio_buffer.append", "audio": b64})
        logger.info("Transcription WS: appended bytes=%d", len(pcm16_bytes))

    async def commit_input(self) -> None:  # pragma: no cover
        await self._send({"type": "input_audio_buffer.commit"})
        logger.info("Transcription WS: input_audio_buffer.commit sent")

    async def clear_input(self) -> None:  # pragma: no cover
        await self._send({"type": "input_audio_buffer.clear"})

    async def create_response(
        self, response_patch: Optional[Dict[str, Any]] = None
    ) -> None:  # pragma: no cover
        # No responses in transcription session
        return

    # --- Streams ---
    async def _iter_queue(self, q) -> AsyncGenerator[Any, None]:
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    def iter_events(self) -> AsyncGenerator[Dict[str, Any], None]:  # pragma: no cover
        return self._iter_queue(self._event_queue)

    def iter_output_audio(self) -> AsyncGenerator[bytes, None]:  # pragma: no cover
        # No audio in transcription session
        async def _empty():
            if False:
                yield b""

        return _empty()

    def iter_input_transcript(self) -> AsyncGenerator[str, None]:  # pragma: no cover
        return self._iter_queue(self._in_tr_queue)

    def iter_output_transcript(self) -> AsyncGenerator[str, None]:  # pragma: no cover
        # No assistant transcript in transcription-only mode
        async def _empty():
            if False:
                yield ""

        return _empty()

    def set_tool_executor(self, executor):  # pragma: no cover
        # Not applicable for transcription-only
        return

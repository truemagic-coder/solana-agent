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

    async def connect(self) -> None:  # pragma: no cover
        headers = [("Authorization", f"Bearer {self.api_key}")]
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

        # Configure session (voice, VAD, formats)
        turn_detection = (
            {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 200,
                "create_response": True,
                "interrupt_response": True,
            }
            if self.options.vad_enabled
            else None
        )

        session_patch = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": self.options.instructions or "",
                "tools": self.options.tools or [],
                "tool_choice": self.options.tool_choice,
                "audio": {
                    "input": {
                        "format": {
                            "type": self.options.input_mime,
                            "rate": self.options.input_rate_hz,
                        },
                        "turn_detection": turn_detection,
                    },
                    "output": {
                        "format": {
                            "type": self.options.output_mime,
                            "rate": self.options.output_rate_hz,
                        },
                        "voice": self.options.voice,
                        "speed": 1.0,
                    },
                },
            },
        }
        logger.debug("Sending session.update: %s", session_patch)
        await self._send(session_patch)

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
                    elif etype in (
                        "conversation.item.input_audio_transcription.delta",
                        "response.input_audio_transcription.delta",
                    ):
                        delta = data.get("delta") or ""
                        if delta:
                            self._in_tr_queue.put_nowait(delta)
                            logger.debug("Input transcript delta: %r", delta[:120])
                    elif etype in (
                        "response.output_audio_transcript.delta",
                        "response.audio_transcript.delta",
                    ):
                        delta = data.get("delta") or ""
                        if delta:
                            self._out_tr_queue.put_nowait(delta)
                            logger.debug("Output transcript delta: %r", delta[:120])
                    elif etype in ("response.completed", "response.complete"):
                        # End of current response; close the session to finish the turn
                        logger.info("Realtime WS: response completed; closing session")
                        try:
                            await self.close()
                        except Exception:
                            pass
                        break
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
        payload: Dict[str, Any] = {"type": "response.create"}
        if response_patch:
            payload["response"] = response_patch
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

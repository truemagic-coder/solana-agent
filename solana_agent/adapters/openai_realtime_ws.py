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

    async def connect(self) -> None:  # pragma: no cover
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]
        model = self.options.model or "gpt-realtime"
        uri = f"{self.url}?model={model}"
        self._ws = await websockets.connect(uri, extra_headers=headers, max_size=None)
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
        await self._send(session_patch)

    async def close(self) -> None:  # pragma: no cover
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None

    async def _send(self, payload: Dict[str, Any]) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(json.dumps(payload))

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                    etype = data.get("type")
                    # Demux streams
                    if etype == "response.output_audio.delta":
                        b64 = data.get("delta") or ""
                        if b64:
                            try:
                                self._audio_queue.put_nowait(base64.b64decode(b64))
                            except Exception:
                                pass
                    elif etype == "conversation.item.input_audio_transcription.delta":
                        delta = data.get("delta") or ""
                        if delta:
                            self._in_tr_queue.put_nowait(delta)
                    elif etype == "response.output_audio_transcript.delta":
                        delta = data.get("delta") or ""
                        if delta:
                            self._out_tr_queue.put_nowait(delta)
                    # Always also publish raw events
                    try:
                        self._event_queue.put_nowait(data)
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
    async def update_session(self, session_patch: Dict[str, Any]) -> None:
        await self._send({"type": "session.update", "session": session_patch})

    async def append_audio(self, pcm16_bytes: bytes) -> None:
        b64 = base64.b64encode(pcm16_bytes).decode("ascii")
        await self._send({"type": "input_audio_buffer.append", "audio": b64})

    async def commit_input(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})

    async def clear_input(self) -> None:
        await self._send({"type": "input_audio_buffer.clear"})

    async def create_response(
        self, response_patch: Optional[Dict[str, Any]] = None
    ) -> None:
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

    def iter_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        return self._iter_queue(self._event_queue)

    def iter_output_audio(self) -> AsyncGenerator[bytes, None]:
        return self._iter_queue(self._audio_queue)

    def iter_input_transcript(self) -> AsyncGenerator[str, None]:
        return self._iter_queue(self._in_tr_queue)

    def iter_output_transcript(self) -> AsyncGenerator[str, None]:
        return self._iter_queue(self._out_tr_queue)

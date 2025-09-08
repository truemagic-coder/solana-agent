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

        # Queues and state
        self._ws = None
        self._event_queue = asyncio.Queue()
        self._audio_queue = asyncio.Queue()
        self._in_tr_queue = asyncio.Queue()
        self._out_tr_queue = asyncio.Queue()

        self._recv_task = None
        self._tool_executor = None
        self._pending_calls = {}
        self._active_tool_calls = 0

        self._last_input_item_id = None
        self._commit_evt = asyncio.Event()
        # Session state tracking for ordering/logging
        self._last_session_patch = {}
        self._last_session_updated_payload = {}
        self._session_updated_evt = asyncio.Event()
        self._awaiting_session_updated = False
        self._session_created_evt = asyncio.Event()
        # Outgoing event tracking for error correlation
        self._event_seq = 0
        self._sent_events = {}

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

        # Optionally wait briefly for session.created; some servers ignore pre-created updates
        try:
            await asyncio.wait_for(self._session_created_evt.wait(), timeout=1.0)
            logger.info(
                "Realtime WS: session.created observed before first session.update"
            )
        except asyncio.TimeoutError:
            logger.info(
                "Realtime WS: no session.created within 1.0s; sending session.update anyway"
            )

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

        # Note: use GA top-level schema so downstream ffmpeg expects s16le input
        session_payload: Dict[str, Any] = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": self.options.voice,
                "speed": float(getattr(self.options, "voice_speed", 1.0) or 1.0),
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                **({"turn_detection": vad_cfg} if vad_cfg is not None else {}),
                # Optional server-stored prompt
                **({"prompt": prompt_block} if prompt_block else {}),
                # Direct overrides
                "instructions": self.options.instructions or "",
                **({"tools": self.options.tools} if self.options.tools else {}),
                **(
                    {"tool_choice": self.options.tool_choice}
                    if getattr(self.options, "tool_choice", None)
                    else {}
                ),
            },
        }
        logger.info(
            "Realtime WS: sending session.update (voice=%s, vad=%s, output=%s)",
            self.options.voice,
            self.options.vad_enabled,
            session_payload["session"]["output_audio_format"],
        )
        # Log exact session.update payload and mark awaiting session.updated
        try:
            logger.info(
                "Realtime WS: sending session.update payload=%s",
                json.dumps(session_payload.get("session", {}), sort_keys=True),
            )
        except Exception:
            pass
        self._last_session_patch = session_payload.get("session", {})
        self._session_updated_evt = asyncio.Event()
        self._awaiting_session_updated = True
        # Quick sanity warnings
        try:
            sess = self._last_session_patch
            instr = sess.get("instructions")
            voice = sess.get("voice")
            if instr is None or (isinstance(instr, str) and instr.strip() == ""):
                logger.warning(
                    "Realtime WS: instructions missing/empty in session.update"
                )
            if not voice:
                logger.warning("Realtime WS: voice missing in session.update")
        except Exception:
            pass
        await self._send_tracked(session_payload, label="session.update:init")

    async def close(self) -> None:  # pragma: no cover
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None
        # Unblock any pending waiters to avoid dangling tasks
        try:
            self._commit_evt.set()
        except Exception:
            pass
        try:
            self._session_created_evt.set()
        except Exception:
            pass
        try:
            self._session_updated_evt.set()
        except Exception:
            pass

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

    def _next_event_id(self) -> str:
        try:
            self._event_seq += 1
            return f"evt-{self._event_seq}"
        except Exception:
            # Fallback to random when counters fail
            import uuid as _uuid

            return str(_uuid.uuid4())

    async def _send_tracked(
        self, payload: Dict[str, Any], label: Optional[str] = None
    ) -> str:
        """Attach an event_id and retain a snapshot for correlating error events."""
        try:
            eid = payload.get("event_id") or self._next_event_id()
            payload["event_id"] = eid
            self._sent_events[eid] = {
                "label": label or (payload.get("type") or "client.event"),
                "type": payload.get("type"),
                "payload": payload.copy(),
                "ts": asyncio.get_event_loop().time(),
            }
        except Exception:
            eid = payload.get("event_id") or ""
        await self._send(payload)
        return eid

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
                                logger.info("Audio delta bytes=%d", len(chunk))
                            except Exception:
                                pass
                    elif etype == "response.text.delta":
                        # Some servers emit generic text deltas with metadata marking transcription
                        metadata = data.get("response", {}).get("metadata", {})
                        if metadata.get("type") == "transcription":
                            delta = data.get("delta") or ""
                            if delta:
                                self._in_tr_queue.put_nowait(delta)
                                logger.info("Input transcript delta: %r", delta[:120])
                            else:
                                logger.debug("Input transcript delta: empty")
                    elif etype == "response.output_text.delta":
                        # Assistant text stream (not used for audio, but useful as transcript)
                        delta = data.get("delta") or ""
                        if delta:
                            self._out_tr_queue.put_nowait(delta)
                            logger.debug("Assistant text delta: %r", delta[:120])
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
                        # Do not terminate audio stream here; completion may be for a function_call
                        metadata = data.get("response", {}).get("metadata", {})
                        if metadata.get("type") == "transcription":
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
                        self._last_session_updated_payload = sess
                        # Mark that the latest update has been applied
                        try:
                            self._awaiting_session_updated = False
                            self._session_updated_evt.set()
                        except Exception:
                            pass
                        try:
                            logger.info(
                                "Realtime WS: session.updated payload=%s",
                                json.dumps(sess, sort_keys=True),
                            )
                        except Exception:
                            logger.info(
                                "Realtime WS: session updated (payload dump failed)"
                            )
                        # Extra guardrails: warn if key fields absent/empty
                        try:
                            instr = sess.get("instructions")
                            voice = sess.get("voice")
                            if instr is None or (
                                isinstance(instr, str) and instr.strip() == ""
                            ):
                                logger.warning(
                                    "Realtime WS: session.updated has empty/missing instructions"
                                )
                            if not voice:
                                logger.warning(
                                    "Realtime WS: session.updated missing voice"
                                )
                        except Exception:
                            pass
                    elif etype == "session.created":
                        logger.info("Realtime WS: session created")
                        try:
                            self._session_created_evt.set()
                        except Exception:
                            pass
                    elif etype == "error" or (
                        isinstance(etype, str)
                        and (
                            etype.endswith(".error")
                            or ("error" in etype)
                            or ("failed" in etype)
                        )
                    ):
                        # Surface server errors explicitly
                        try:
                            logger.error(
                                "Realtime WS error event: %s",
                                json.dumps(data, sort_keys=True),
                            )
                        except Exception:
                            logger.error(
                                "Realtime WS error event (payload dump failed)"
                            )
                        # If we're awaiting a session.updated, try a fallback legacy schema once
                        try:
                            if self._awaiting_session_updated:
                                logger.warning(
                                    "Realtime WS: session.update may have failed; attempting legacy schema fallback"
                                )
                                sess = self._last_session_patch or {}
                                # Build legacy nested payload
                                legacy_payload = {
                                    "type": "session.update",
                                    "session": {
                                        "type": "realtime",
                                        "model": self.options.model or "gpt-realtime",
                                        "output_modalities": ["audio", "text"],
                                        "audio": {
                                            "input": {
                                                "format": sess.get(
                                                    "input_audio_format", "pcm16"
                                                ),
                                                "turn_detection": sess.get(
                                                    "turn_detection"
                                                ),
                                            },
                                            "output": {
                                                "format": sess.get(
                                                    "output_audio_format", "pcm16"
                                                ),
                                                **(
                                                    {"voice": sess.get("voice")}
                                                    if sess.get("voice")
                                                    else {}
                                                ),
                                                **(
                                                    {"speed": sess.get("speed")}
                                                    if sess.get("speed") is not None
                                                    else {}
                                                ),
                                            },
                                        },
                                        **(
                                            {"prompt": sess.get("prompt")}
                                            if sess.get("prompt")
                                            else {}
                                        ),
                                        **(
                                            {"instructions": sess.get("instructions")}
                                            if sess.get("instructions") is not None
                                            else {}
                                        ),
                                        **(
                                            {"tools": sess.get("tools")}
                                            if sess.get("tools")
                                            else {}
                                        ),
                                        **(
                                            {"tool_choice": sess.get("tool_choice")}
                                            if sess.get("tool_choice") is not None
                                            else {}
                                        ),
                                    },
                                }
                                # Reset awaiting flag and attempt fallback
                                self._session_updated_evt = asyncio.Event()
                                self._awaiting_session_updated = True
                                await self._ws.send(json.dumps(legacy_payload))
                                logger.info(
                                    "Realtime WS: legacy session.update sent as fallback"
                                )
                        except Exception:
                            pass
                    # Always also publish raw events
                    try:
                        self._event_queue.put_nowait(data)
                    except Exception:
                        pass
                    # Handle tool/function calls (support GA and legacy shapes)
                    if etype in (
                        "response.function_call.delta",
                        "response.function_call_arguments.delta",
                    ):
                        # Accumulate arguments by call id
                        call = data.get("function_call", {})
                        call_id = call.get("id") or data.get("id")
                        name = call.get("name")
                        arguments_delta = call.get("arguments_delta", "") or call.get(
                            "arguments", ""
                        )
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
                            "Function call args delta: id=%s name=%s args_len=%d",
                            call_id,
                            name,
                            len(pc.get("args", "")),
                        )
                    elif etype == "response.function_call":
                        # Finalize and execute (legacy summary event)
                        call = data.get("function_call", {})
                        call_id = call.get("id") or data.get("id")
                        if not hasattr(self, "_pending_calls"):
                            self._pending_calls = {}
                        self._pending_calls.setdefault(
                            call_id,
                            {
                                "name": call.get("name"),
                                "args": call.get("arguments", ""),
                            },
                        )
                        await self._execute_pending_call(call_id)
                    elif etype in (
                        "response.done",
                        "response.completed",
                        "response.complete",
                    ):
                        # Also detect GA function_call items in response.output
                        try:
                            out_items = data.get("response", {}).get("output", [])
                            for item in out_items:
                                if item.get("type") == "function_call":
                                    call_id = item.get("call_id") or item.get("id")
                                    name = item.get("name")
                                    args = item.get("arguments") or "{}"
                                    if not hasattr(self, "_pending_calls"):
                                        self._pending_calls = {}
                                    self._pending_calls[call_id] = {
                                        "name": name,
                                        "args": args,
                                    }
                                    await self._execute_pending_call(call_id)
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
        # Translate nested audio.* to GA top-level fields
        raw = dict(session_patch or {})
        patch: Dict[str, Any] = {}
        try:
            audio = dict((raw.get("audio") or {}))
            inp = dict((audio.get("input") or {}))
            out = dict((audio.get("output") or {}))
            if inp:
                if inp.get("format"):
                    patch["input_audio_format"] = inp.get("format")
                if "turn_detection" in inp:
                    patch["turn_detection"] = inp.get("turn_detection")
            if out:
                if out.get("format"):
                    patch["output_audio_format"] = out.get("format")
                if out.get("voice"):
                    patch["voice"] = out.get("voice")
                if out.get("speed") is not None:
                    patch["speed"] = out.get("speed")
        except Exception:
            pass
        # Pass through known top-level fields directly
        for k in (
            "instructions",
            "tools",
            "tool_choice",
            "modalities",
            "voice",
            "speed",
            "input_audio_format",
            "output_audio_format",
            "turn_detection",
        ):
            if k in raw and raw.get(k) is not None:
                patch[k] = raw[k]
        payload = {"type": "session.update", "session": patch}
        # Mark awaiting updated and store last patch
        self._last_session_patch = patch or {}
        self._session_updated_evt = asyncio.Event()
        self._awaiting_session_updated = True
        # Log payload and warn if potentially clearing/omitting critical fields
        try:
            logger.info(
                "Realtime WS: sending session.update payload=%s",
                json.dumps(self._last_session_patch, sort_keys=True),
            )
            if "instructions" in self._last_session_patch and (
                (self._last_session_patch.get("instructions") or "").strip() == ""
            ):
                logger.warning(
                    "Realtime WS: session.update sets empty instructions; this clears them"
                )
            if "voice" in self._last_session_patch and not self._last_session_patch.get(
                "voice"
            ):
                logger.warning("Realtime WS: session.update provides empty voice")
            if "instructions" not in self._last_session_patch:
                logger.warning(
                    "Realtime WS: session.update omits instructions; relying on previous instructions"
                )
        except Exception:
            pass
        await self._send(payload)

    async def append_audio(self, pcm16_bytes: bytes) -> None:  # pragma: no cover
        b64 = base64.b64encode(pcm16_bytes).decode("ascii")
        await self._send_tracked(
            {"type": "input_audio_buffer.append", "audio": b64},
            label="input_audio_buffer.append",
        )

    async def commit_input(self) -> None:  # pragma: no cover
        await self._send_tracked(
            {"type": "input_audio_buffer.commit"}, label="input_audio_buffer.commit"
        )
        try:
            logger.info("Realtime WS: input_audio_buffer.commit sent")
        except Exception:
            pass

    async def clear_input(self) -> None:  # pragma: no cover
        await self._send_tracked(
            {"type": "input_audio_buffer.clear"}, label="input_audio_buffer.clear"
        )
        # Reset last input reference and commit event to avoid stale references
        try:
            self._last_input_item_id = None
            self._commit_evt = asyncio.Event()
        except Exception:
            pass

    async def create_response(
        self, response_patch: Optional[Dict[str, Any]] = None
    ) -> None:  # pragma: no cover
        # Wait briefly for commit event so we can reference the latest audio item when applicable
        if not self._last_input_item_id:
            try:
                await asyncio.wait_for(self._commit_evt.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        # Ensure the latest session.update (if any) has been applied before responding
        if self._awaiting_session_updated:
            # Prefer an explicit session.updated; if absent, accept session.created
            try:
                if self._session_updated_evt.is_set():
                    logger.info(
                        "Realtime WS: response.create proceeding after session.updated (pre-set)"
                    )
                else:
                    await asyncio.wait_for(
                        self._session_updated_evt.wait(), timeout=2.5
                    )
                    logger.info(
                        "Realtime WS: response.create proceeding after session.updated"
                    )
            except asyncio.TimeoutError:
                if self._session_created_evt.is_set():
                    logger.info(
                        "Realtime WS: response.create proceeding after session.created (no session.updated observed)"
                    )
                    # Best-effort: resend last session.update once more to apply voice/instructions
                    try:
                        if self._last_session_patch:
                            logger.info(
                                "Realtime WS: resending session.update to apply config before response"
                            )
                            # Reset awaiting flag and wait briefly again
                            self._session_updated_evt = asyncio.Event()
                            self._awaiting_session_updated = True
                            await self._send(
                                {
                                    "type": "session.update",
                                    "session": self._last_session_patch,
                                }
                            )
                            try:
                                await asyncio.wait_for(
                                    self._session_updated_evt.wait(), timeout=1.0
                                )
                                logger.info(
                                    "Realtime WS: proceeding after retry session.updated"
                                )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    "Realtime WS: retry session.update did not yield session.updated in time"
                                )
                    except Exception:
                        pass
                else:
                    try:
                        await asyncio.wait_for(
                            self._session_created_evt.wait(), timeout=2.5
                        )
                        logger.info(
                            "Realtime WS: response.create proceeding after session.created"
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Realtime WS: neither session.updated nor session.created received in time; proceeding"
                        )

        # Then, create main response
        payload: Dict[str, Any] = {"type": "response.create"}
        if response_patch:
            payload["response"] = response_patch
        # Ensure default audio modality and voice are present if not explicitly set
        if "response" not in payload:
            payload["response"] = {}
        rp = payload["response"]
        rp.setdefault(
            "modalities", ["audio"]
        )  # lock to audio-only unless client requests text
        # Do not set per-response voice; rely on session voice
        rp.setdefault("audio", {"format": "pcm16"})
        rp.setdefault("metadata", {"type": "response"})
        # Attach input reference so the model links this response to last audio
        if self._last_input_item_id and "input" not in rp:
            rp["input"] = [{"type": "item_reference", "id": self._last_input_item_id}]
        try:
            has_ref = bool(self._last_input_item_id)
            logger.info(
                "Realtime WS: sending response.create (input_ref=%s) modalities=%s audio=%s",
                has_ref,
                rp.get("modalities"),
                rp.get("audio"),
            )
        except Exception:
            pass
        await self._send_tracked(payload, label="response.create")

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

    # Expose whether a function/tool call is currently pending
    def has_pending_tool_call(self) -> bool:  # pragma: no cover
        try:
            return (
                bool(getattr(self, "_pending_calls", {}))
                or int(getattr(self, "_active_tool_calls", 0)) > 0
            )
        except Exception:
            return False

    # --- Internal helpers for GA tool execution ---
    async def _execute_pending_call(self, call_id: Optional[str]) -> None:
        if not call_id:
            return
        # Peek without popping so we remain in a "pending/active" state
        pc = getattr(self, "_pending_calls", {}).get(call_id)
        if not pc or not self._tool_executor or not pc.get("name"):
            return
        try:
            # Mark as active to keep timeouts from firing while tool runs
            try:
                self._active_tool_calls += 1
            except Exception:
                pass
            args_preview_len = len((pc.get("args") or ""))
            logger.info(
                "Executing tool: id=%s name=%s args_len=%d",
                call_id,
                pc.get("name"),
                args_preview_len,
            )
            args = pc.get("args") or "{}"
            try:
                parsed = json.loads(args)
            except Exception:
                parsed = {}
            start_ts = asyncio.get_event_loop().time()
            result = await self._tool_executor(pc["name"], parsed)
            dur = asyncio.get_event_loop().time() - start_ts
            try:
                result_summary = (
                    f"keys={list(result.keys())[:5]}"
                    if isinstance(result, dict)
                    else type(result).__name__
                )
            except Exception:
                result_summary = "<unavailable>"
            logger.info(
                "Tool done: id=%s name=%s dur=%.2fs result=%s",
                call_id,
                pc.get("name"),
                dur,
                result_summary,
            )
            # Provide tool result back as conversation.item.create per GA, then resume with audio response
            await self._send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    },
                }
            )
            await self._send(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio"],
                        "audio": {"format": "pcm16"},
                        "metadata": {"type": "response"},
                    },
                }
            )
            logger.info("Tool result delivered; continuing response (audio)")
        except Exception:
            try:
                await self._send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps({"error": "tool_execution_failed"}),
                        },
                    }
                )
                await self._send(
                    {
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio"],
                            "audio": {"format": "pcm16"},
                            "metadata": {"type": "response"},
                        },
                    }
                )
                logger.warning(
                    "Tool execution failed; sent error output and resumed response"
                )
            except Exception:
                pass
        finally:
            # Clear pending state and decrement active count
            try:
                getattr(self, "_pending_calls", {}).pop(call_id, None)
            except Exception:
                pass
            try:
                self._active_tool_calls = max(0, int(self._active_tool_calls) - 1)
                logger.debug(
                    "Pending tool calls decremented; active=%d", self._active_tool_calls
                )
            except Exception:
                pass


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

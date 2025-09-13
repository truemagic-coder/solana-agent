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
    - Expects 24kHz audio.
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

        # Tool/function state
        self._recv_task = None
        self._tool_executor = None
        self._pending_calls = {}
        self._active_tool_calls = 0
        # Map call_id or item_id -> asyncio.Event when server has created the function_call item
        self._call_ready_events = {}
        # Track identifiers already executed to avoid duplicates
        self._executed_call_ids = set()
        # Track mapping from tool call identifiers to the originating response.id
        # Prefer addressing function_call_output to a response to avoid conversation lookup races
        self._call_response_ids = {}
        # Accumulate function call arguments when streamed
        self._call_args_accum = {}
        self._call_names = {}
        # Map from function_call item_id -> call_id for reliable output addressing
        self._item_call_ids = {}

        # Input/response state
        self._pending_input_bytes = 0
        self._commit_evt = asyncio.Event()
        self._commit_inflight = False
        self._response_active = False
        self._server_auto_create_enabled = False
        self._last_commit_ts = 0.0

        # Session/event tracking
        self._session_created_evt = asyncio.Event()
        self._session_updated_evt = asyncio.Event()
        self._awaiting_session_updated = False
        self._last_session_patch = {}
        self._last_session_updated_payload = {}
        self._last_input_item_id = None
        # Response generation tracking to bind tool outputs to the active response
        self._response_generation = 0
        # Track the currently active response.id (fallback when some events omit it)
        self._active_response_id = None
        # Accumulate assistant output text by response.id and flush on response completion
        self._out_text_buffers = {}

        # Outbound event correlation
        self._event_seq = 0
        self._sent_events = {}

    async def connect(self) -> None:  # pragma: no cover
        # Defensive: ensure session events exist even if __init__ didn’t set them (older builds)
        if not hasattr(self, "_session_created_evt") or not isinstance(
            getattr(self, "_session_created_evt", None), asyncio.Event
        ):
            self._session_created_evt = asyncio.Event()
        if not hasattr(self, "_session_updated_evt") or not isinstance(
            getattr(self, "_session_updated_evt", None), asyncio.Event
        ):
            self._session_updated_evt = asyncio.Event()
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
        ]
        model = self.options.model or "gpt-realtime"
        uri = f"{self.url}?model={model}"

        # Determine if audio output should be configured for logging
        modalities = self.options.output_modalities or ["audio", "text"]
        should_configure_audio_output = "audio" in modalities

        if should_configure_audio_output:
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
        else:
            logger.info(
                "Realtime WS connecting: uri=%s, input=%s@%sHz, text-only output, vad=%s",
                uri,
                self.options.input_mime,
                self.options.input_rate_hz,
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

        # Configure session (instructions, tools). VAD handled per-request.
        # Per server schema, turn_detection belongs under audio.input; set to None to disable.
        # When VAD is disabled, explicitly set create_response to False if the server honors it
        td_input = (
            {"type": "server_vad", "create_response": True}
            if self.options.vad_enabled
            else {"type": "server_vad", "create_response": False}
        )

        # Build session.update per docs (nested audio object)
        def _strip_tool_strict(tools_val):
            try:
                tools_list = list(tools_val or [])
            except Exception:
                return tools_val
            cleaned = []
            for t in tools_list:
                try:
                    t2 = dict(t)
                    t2.pop("strict", None)
                    cleaned.append(t2)
                except Exception:
                    cleaned.append(t)
            return cleaned

        # Determine if audio output should be configured
        modalities = self.options.output_modalities or ["audio", "text"]
        should_configure_audio_output = "audio" in modalities

        # Build session.update per docs (nested audio object)
        session_payload: Dict[str, Any] = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": modalities,
                "audio": {
                    "input": {
                        "format": {
                            "type": self.options.input_mime or "audio/pcm",
                            "rate": int(self.options.input_rate_hz or 24000),
                        },
                        "turn_detection": td_input,
                    },
                    **(
                        {
                            "output": {
                                "format": {
                                    "type": self.options.output_mime or "audio/pcm",
                                    "rate": int(self.options.output_rate_hz or 24000),
                                },
                                "voice": self.options.voice,
                                "speed": float(
                                    getattr(self.options, "voice_speed", 1.0) or 1.0
                                ),
                            }
                        }
                        if should_configure_audio_output
                        else {}
                    ),
                },
                # Note: no top-level turn_detection; nested under audio.input
                **({"prompt": prompt_block} if prompt_block else {}),
                "instructions": self.options.instructions or "",
                **(
                    {"tools": _strip_tool_strict(self.options.tools)}
                    if self.options.tools
                    else {}
                ),
                **(
                    {"tool_choice": self.options.tool_choice}
                    if getattr(self.options, "tool_choice", None)
                    else {}
                ),
            },
        }
        # Optional realtime transcription configuration
        try:
            tr_model = getattr(self.options, "transcription_model", None)
            if tr_model:
                audio_obj = session_payload["session"].setdefault("audio", {})
                # Attach input transcription config per GA schema
                transcription_cfg: Dict[str, Any] = {"model": tr_model}
                lang = getattr(self.options, "transcription_language", None)
                if lang:
                    transcription_cfg["language"] = lang
                prompt_txt = getattr(self.options, "transcription_prompt", None)
                if prompt_txt is not None:
                    transcription_cfg["prompt"] = prompt_txt
                if getattr(self.options, "transcription_include_logprobs", False):
                    session_payload["session"].setdefault("include", []).append(
                        "item.input_audio_transcription.logprobs"
                    )
                nr = getattr(self.options, "transcription_noise_reduction", None)
                if nr is not None:
                    audio_obj["noise_reduction"] = bool(nr)
                # Place under audio.input.transcription per current server conventions
                audio_obj.setdefault("input", {}).setdefault(
                    "transcription", transcription_cfg
                )
        except Exception:
            logger.exception("Failed to attach transcription config to session.update")
        if should_configure_audio_output:
            logger.info(
                "Realtime WS: sending session.update (voice=%s, vad=%s, output=%s@%s)",
                self.options.voice,
                self.options.vad_enabled,
                (self.options.output_mime or "audio/pcm"),
                int(self.options.output_rate_hz or 24000),
            )
        else:
            logger.info(
                "Realtime WS: sending session.update (text-only, vad=%s)",
                self.options.vad_enabled,
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
            voice = ((sess.get("audio") or {}).get("output") or {}).get("voice")
            if instr is None or (isinstance(instr, str) and instr.strip() == ""):
                logger.warning(
                    "Realtime WS: instructions missing/empty in session.update"
                )
            if not voice and should_configure_audio_output:
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
                peid = payload.get("event_id")
            except Exception:
                ptype = str(type(payload))
                peid = None
            logger.debug("WS send: %s (event_id=%s)", ptype, peid)

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
                    eid = data.get("event_id")
                    logger.debug("WS recv: %s (event_id=%s)", etype, eid)
                    # Track active response.id from any response.* event that carries it
                    try:
                        if isinstance(etype, str) and etype.startswith("response."):
                            _resp = data.get("response") or {}
                            _rid = _resp.get("id")
                            if _rid:
                                self._active_response_id = _rid
                    except Exception:
                        pass
                    # Demux streams
                    if etype in ("response.output_audio.delta", "response.audio.delta"):
                        b64 = data.get("delta") or ""
                        if b64:
                            try:
                                chunk = base64.b64decode(b64)
                                self._audio_queue.put_nowait(chunk)
                                # Ownership/response tagging for diagnostics
                                try:
                                    owner = getattr(self, "_owner_user_id", None)
                                except Exception:
                                    owner = None
                                try:
                                    rid = getattr(self, "_active_response_id", None)
                                except Exception:
                                    rid = None
                                try:
                                    gen = int(getattr(self, "_response_generation", 0))
                                except Exception:
                                    gen = None
                                logger.info(
                                    "Audio delta bytes=%d owner=%s rid=%s gen=%s",
                                    len(chunk),
                                    owner,
                                    rid,
                                    gen,
                                )
                                try:
                                    # New response detected if we were previously inactive
                                    if not getattr(self, "_response_active", False):
                                        self._response_generation = (
                                            int(
                                                getattr(self, "_response_generation", 0)
                                            )
                                            + 1
                                        )
                                    self._response_active = True
                                except Exception:
                                    pass
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
                    elif etype in (
                        "response.function_call_arguments.delta",
                        "response.function_call_arguments.done",
                    ):
                        # Capture streamed function-call arguments early and mark readiness on .done
                        try:
                            resp = data.get("response") or {}
                            rid = resp.get("id") or getattr(
                                self, "_active_response_id", None
                            )
                            # Many servers include call_id at top-level for these events
                            call_id = data.get("call_id") or data.get("id")
                            # Some servers include the function call item under 'item'
                            if not call_id:
                                call_id = (data.get("item") or {}).get("call_id") or (
                                    data.get("item") or {}
                                ).get("id")
                            # Name can appear directly or under item
                            name = data.get("name") or (data.get("item") or {}).get(
                                "name"
                            )
                            if name:
                                self._call_names[call_id] = name
                            if rid and call_id:
                                self._call_response_ids[call_id] = rid
                            if etype.endswith("delta"):
                                delta = data.get("delta") or ""
                                if call_id and delta:
                                    self._call_args_accum[call_id] = (
                                        self._call_args_accum.get(call_id, "") + delta
                                    )
                            else:  # .done
                                if call_id:
                                    # Mark call ready and enqueue pending execution if not already
                                    ev = self._call_ready_events.get(call_id)
                                    if not ev:
                                        ev = asyncio.Event()
                                        self._call_ready_events[call_id] = ev
                                    ev.set()
                                    # Register pending call using accumulated args
                                    if call_id not in getattr(
                                        self, "_pending_calls", {}
                                    ):
                                        args_text = (
                                            self._call_args_accum.get(call_id, "{}")
                                            or "{}"
                                        )
                                        if not hasattr(self, "_pending_calls"):
                                            self._pending_calls = {}
                                        self._pending_calls[call_id] = {
                                            "name": self._call_names.get(call_id),
                                            "args": args_text,
                                            "gen": int(
                                                getattr(self, "_response_generation", 0)
                                            ),
                                            "call_id": call_id,
                                            "item_id": None,
                                        }
                                        self._executed_call_ids.add(call_id)
                                        await self._execute_pending_call(call_id)
                        except Exception:
                            pass
                    elif etype == "response.output_text.delta":
                        # Assistant textual output delta. Buffer per response.id.
                        # Prefer the audio transcript stream for final transcript; only use text
                        # deltas if no audio transcript arrives.
                        try:
                            rid = (data.get("response") or {}).get("id") or getattr(
                                self, "_active_response_id", None
                            )
                            delta = data.get("delta") or ""
                            if rid and delta:
                                buf = self._out_text_buffers.setdefault(
                                    rid, {"text": "", "has_audio": False}
                                )
                                # Only accumulate text when we don't yet have audio transcript
                                if not bool(buf.get("has_audio")):
                                    buf["text"] = str(buf.get("text", "")) + delta
                                    logger.debug(
                                        "Buffered assistant text delta (rid=%s, len=%d)",
                                        rid,
                                        len(delta),
                                    )
                        except Exception:
                            pass
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
                        # Assistant audio transcript delta (authoritative for spoken output)
                        try:
                            rid = (data.get("response") or {}).get("id") or getattr(
                                self, "_active_response_id", None
                            )
                            delta = data.get("delta") or ""
                            if rid and delta:
                                buf = self._out_text_buffers.setdefault(
                                    rid, {"text": "", "has_audio": False}
                                )
                                buf["has_audio"] = True
                                buf["text"] = str(buf.get("text", "")) + delta
                                logger.debug(
                                    "Buffered audio transcript delta (rid=%s, len=%d)",
                                    rid,
                                    len(delta),
                                )
                        except Exception:
                            pass
                    elif etype == "input_audio_buffer.committed":
                        # Track the committed audio item id for OOB transcription referencing
                        item_id = data.get("item_id") or data.get("id")
                        if item_id:
                            self._last_input_item_id = item_id
                            try:
                                self._last_commit_ts = asyncio.get_event_loop().time()
                            except Exception:
                                pass
                            logger.info(
                                "Realtime WS: input_audio_buffer committed: item_id=%s",
                                item_id,
                            )
                            # Clear commit in-flight flag on ack and signal waiters
                            try:
                                self._commit_inflight = False
                            except Exception:
                                pass
                            try:
                                self._commit_evt.set()
                            except Exception:
                                pass
                    elif etype in (
                        "response.output_audio.done",
                        "response.audio.done",
                    ):
                        # End of audio stream for the response; stop audio iterator but keep WS open for transcripts
                        try:
                            owner = getattr(self, "_owner_user_id", None)
                        except Exception:
                            owner = None
                        try:
                            rid = (data.get("response") or {}).get("id") or getattr(
                                self, "_active_response_id", None
                            )
                        except Exception:
                            rid = None
                        try:
                            gen = int(getattr(self, "_response_generation", 0))
                        except Exception:
                            gen = None
                        logger.info(
                            "Realtime WS: output audio done; owner=%s rid=%s gen=%s",
                            owner,
                            rid,
                            gen,
                        )
                        # If we have a buffered transcript for this response, flush it now
                        try:
                            rid = (data.get("response") or {}).get("id") or getattr(
                                self, "_active_response_id", None
                            )
                            if rid and rid in self._out_text_buffers:
                                final = str(
                                    self._out_text_buffers.get(rid, {}).get("text")
                                    or ""
                                )
                                if final:
                                    self._out_tr_queue.put_nowait(final)
                                    logger.debug(
                                        "Flushed assistant transcript on audio.done (rid=%s, len=%d)",
                                        rid,
                                        len(final),
                                    )
                                self._out_text_buffers.pop(rid, None)
                        except Exception:
                            pass
                        try:
                            self._audio_queue.put_nowait(None)
                        except Exception:
                            pass
                        try:
                            self._response_active = False
                        except Exception:
                            pass
                        try:
                            # Clear active response id when audio for that response is done
                            self._active_response_id = None
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
                        # Flush buffered assistant transcript, if any
                        try:
                            rid = (data.get("response") or {}).get("id")
                            if rid and rid in self._out_text_buffers:
                                final = str(
                                    self._out_text_buffers.get(rid, {}).get("text")
                                    or ""
                                )
                                if final:
                                    self._out_tr_queue.put_nowait(final)
                                    logger.debug(
                                        "Flushed assistant transcript on response.done (rid=%s, len=%d)",
                                        rid,
                                        len(final),
                                    )
                                self._out_text_buffers.pop(rid, None)
                        except Exception:
                            pass
                        try:
                            self._response_active = False
                        except Exception:
                            pass
                        try:
                            # Response lifecycle ended; clear active id
                            self._active_response_id = None
                        except Exception:
                            pass
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
                        else:
                            # For assistant text-only completions without audio, flush buffered text
                            try:
                                rid = (data.get("response") or {}).get("id") or getattr(
                                    self, "_active_response_id", None
                                )
                                if rid and rid in self._out_text_buffers:
                                    final = str(
                                        self._out_text_buffers.get(rid, {}).get("text")
                                        or ""
                                    )
                                    if final:
                                        self._out_tr_queue.put_nowait(final)
                                        logger.debug(
                                            "Flushed assistant transcript on text.done (rid=%s, len=%d)",
                                            rid,
                                            len(final),
                                        )
                                    self._out_text_buffers.pop(rid, None)
                                # Always terminate the output transcript stream for this response when text-only.
                                try:
                                    # Only enqueue sentinel when no audio modality is configured
                                    modalities = (
                                        getattr(self.options, "output_modalities", None)
                                        or []
                                    )
                                    if "audio" not in modalities:
                                        self._out_tr_queue.put_nowait(None)
                                        logger.debug(
                                            "Enqueued transcript termination sentinel (text-only response)"
                                        )
                                except Exception:
                                    pass
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
                        # Track server auto-create enablement (turn_detection.create_response)
                        try:
                            td_root = sess.get("turn_detection")
                            td_input = (
                                (sess.get("audio") or {})
                                .get("input", {})
                                .get("turn_detection")
                            )
                            td = td_root if td_root is not None else td_input
                            # If local VAD is disabled, force auto-create disabled
                            if not bool(getattr(self.options, "vad_enabled", False)):
                                self._server_auto_create_enabled = False
                            else:
                                self._server_auto_create_enabled = bool(
                                    isinstance(td, dict)
                                    and bool(td.get("create_response"))
                                )
                            logger.debug(
                                "Realtime WS: server_auto_create_enabled=%s",
                                self._server_auto_create_enabled,
                            )
                        except Exception:
                            pass
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
                            voice = sess.get("voice") or (
                                (sess.get("audio") or {}).get("output", {}).get("voice")
                            )
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
                        # Correlate to the originating client event by event_id
                        try:
                            eid = data.get("event_id") or data.get("error", {}).get(
                                "event_id"
                            )
                            if eid and eid in self._sent_events:
                                sent = self._sent_events.get(eid) or {}
                                logger.error(
                                    "Realtime WS error correlated: event_id=%s sent_label=%s sent_type=%s",
                                    eid,
                                    sent.get("label"),
                                    sent.get("type"),
                                )
                        except Exception:
                            pass
                        # No legacy fallback; rely on current config/state.
                    # Always also publish raw events
                    try:
                        self._event_queue.put_nowait(data)
                    except Exception:
                        pass
                    # Handle tool/function calls (GA-only triggers)
                    if etype in (
                        "conversation.item.created",
                        "conversation.item.added",
                    ):
                        try:
                            item = data.get("item") or {}
                            if item.get("type") == "function_call":
                                call_id = item.get("call_id")
                                item_id = item.get("id")
                                # Mark readiness for both identifiers if available
                                for cid in filter(None, [call_id, item_id]):
                                    ev = self._call_ready_events.get(cid)
                                    if not ev:
                                        ev = asyncio.Event()
                                        self._call_ready_events[cid] = ev
                                    ev.set()
                                logger.debug(
                                    "Call ready via item.%s: call_id=%s item_id=%s",
                                    "created" if etype.endswith("created") else "added",
                                    call_id,
                                    item_id,
                                )
                        except Exception:
                            pass
                    elif etype in (
                        "response.output_item.added",
                        "response.output_item.created",
                    ):
                        # Map response-scoped function_call items to response_id and ack function_call_output
                        try:
                            resp = data.get("response") or {}
                            rid = resp.get("id") or getattr(
                                self, "_active_response_id", None
                            )
                            item = data.get("item") or {}
                            itype = item.get("type")
                            if itype == "function_call":
                                call_id = item.get("call_id")
                                item_id = item.get("id")
                                name = item.get("name")
                                if name and call_id:
                                    self._call_names[call_id] = name
                                if call_id and item_id:
                                    self._item_call_ids[item_id] = call_id
                                # Bind call identifiers to the response id for response-scoped outputs
                                for cid in filter(None, [call_id, item_id]):
                                    if rid:
                                        self._call_response_ids[cid] = rid
                                # Signal readiness for these identifiers
                                for cid in filter(None, [call_id, item_id]):
                                    ev = self._call_ready_events.get(cid)
                                    if not ev:
                                        ev = asyncio.Event()
                                        self._call_ready_events[cid] = ev
                                    ev.set()
                                logger.debug(
                                    "Mapped function_call via response.%s: call_id=%s item_id=%s response_id=%s",
                                    "created" if etype.endswith("created") else "added",
                                    call_id,
                                    item_id,
                                    rid,
                                )
                        except Exception:
                            pass
                    elif etype in (
                        "response.done",
                        "response.completed",
                        "response.complete",
                    ):
                        # Also detect GA function_call items in response.output
                        try:
                            resp = data.get("response", {})
                            rid = resp.get("id")
                            out_items = resp.get("output", [])
                            for item in out_items:
                                if item.get("type") == "function_call":
                                    call_id = item.get("call_id")
                                    item_id = item.get("id")
                                    # Prefer a canonical id to reference later; try item_id first, then call_id
                                    canonical_id = item_id or call_id
                                    name = item.get("name")
                                    args = item.get("arguments") or "{}"
                                    status = item.get("status")
                                    if call_id and item_id:
                                        self._item_call_ids[item_id] = call_id
                                    # Map this call to the response id for response-scoped outputs
                                    for cid in filter(
                                        None, [call_id, item_id, canonical_id]
                                    ):
                                        if rid:
                                            self._call_response_ids[cid] = rid
                                    if (
                                        canonical_id in self._executed_call_ids
                                        or (
                                            call_id
                                            and call_id in self._executed_call_ids
                                        )
                                        or (
                                            item_id
                                            and item_id in self._executed_call_ids
                                        )
                                    ):
                                        continue
                                    if status and str(status).lower() not in (
                                        "completed",
                                        "complete",
                                        "done",
                                    ):
                                        # Wait for completion in a later event
                                        continue
                                    if not hasattr(self, "_pending_calls"):
                                        self._pending_calls = {}
                                    self._pending_calls[canonical_id] = {
                                        "name": name,
                                        "args": args,
                                        # Bind this call to the current response generation
                                        "gen": int(
                                            getattr(self, "_response_generation", 0)
                                        ),
                                        # Keep both identifiers for fallback when posting outputs
                                        "call_id": call_id,
                                        "item_id": item_id,
                                    }
                                    # Mark this call as ready (server has produced the item)
                                    for cid in filter(
                                        None, [canonical_id, call_id, item_id]
                                    ):
                                        ev = self._call_ready_events.get(cid)
                                        if not ev:
                                            ev = asyncio.Event()
                                            self._call_ready_events[cid] = ev
                                        ev.set()
                                    logger.debug(
                                        "Call ready via response.done: call_id=%s item_id=%s canonical_id=%s",
                                        call_id,
                                        item_id,
                                        canonical_id,
                                    )
                                    for cid in filter(
                                        None, [canonical_id, call_id, item_id]
                                    ):
                                        self._executed_call_ids.add(cid)
                                    await self._execute_pending_call(canonical_id)
                        except Exception:
                            pass

                    # No ack/error event mapping; server emits concrete lifecycle events
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
        # Build nested session.update per docs. Only include provided fields.
        raw = dict(session_patch or {})
        patch: Dict[str, Any] = {}
        audio_patch: Dict[str, Any] = {}

        try:
            audio = dict(raw.get("audio") or {})
            # Normalize turn_detection to audio.input per server schema
            include_td = False
            turn_det = None
            inp = dict(audio.get("input") or {})
            if "turn_detection" in inp:
                include_td = True
                turn_det = inp.get("turn_detection")
            if "turn_detection" in audio:
                include_td = True
                turn_det = audio.get("turn_detection")
            if "turn_detection" in raw:
                include_td = True
                turn_det = raw.get("turn_detection")

            inp = dict(audio.get("input") or {})
            out = dict(audio.get("output") or {})

            # Input format
            fmt_in = inp.get("format", raw.get("input_audio_format"))
            if fmt_in is not None:
                if isinstance(fmt_in, dict):
                    audio_patch.setdefault("input", {})["format"] = fmt_in
                else:
                    ftype = str(fmt_in)
                    if ftype == "pcm16":
                        ftype = "audio/pcm"
                    audio_patch.setdefault("input", {})["format"] = {
                        "type": ftype,
                        "rate": int(self.options.input_rate_hz or 24000),
                    }

            # Optional input extras
            for key in ("noise_reduction", "transcription"):
                if key in audio:
                    audio_patch[key] = audio.get(key)

            # Apply turn_detection under audio.input if provided (allow None to disable)
            if include_td:
                audio_patch.setdefault("input", {})["turn_detection"] = turn_det

            # Output format/voice/speed
            op: Dict[str, Any] = {}
            fmt_out = out.get("format", raw.get("output_audio_format"))
            if fmt_out is not None:
                if isinstance(fmt_out, dict):
                    op["format"] = fmt_out
                else:
                    ftype = str(fmt_out)
                    if ftype == "pcm16":
                        ftype = "audio/pcm"
                    op["format"] = {
                        "type": ftype,
                        "rate": int(self.options.output_rate_hz or 24000),
                    }
            if "voice" in out:
                op["voice"] = out.get("voice")
            if "speed" in out:
                op["speed"] = out.get("speed")
            # Convenience: allow top-level overrides
            if "voice" in raw and "voice" not in op:
                op["voice"] = raw.get("voice")
            if "speed" in raw and "speed" not in op:
                op["speed"] = raw.get("speed")
            if op:
                audio_patch.setdefault("output", {}).update(op)
        except Exception:
            pass

        if audio_patch:
            patch["audio"] = audio_patch

        # No top-level turn_detection

        def _strip_tool_strict(tools_val):
            try:
                tools_list = list(tools_val or [])
            except Exception:
                return tools_val
            cleaned = []
            for t in tools_list:
                try:
                    t2 = dict(t)
                    t2.pop("strict", None)
                    cleaned.append(t2)
                except Exception:
                    cleaned.append(t)
            return cleaned

        # Pass through other documented fields if present
        for k in (
            "model",
            "output_modalities",
            "prompt",
            "instructions",
            "tools",
            "tool_choice",
            "include",
            "max_output_tokens",
            "tracing",
            "truncation",
        ):
            if k in raw:
                if k == "tools":
                    patch[k] = _strip_tool_strict(raw[k])
                else:
                    patch[k] = raw[k]

        # --- Inject realtime transcription config if options were updated after initial connect ---
        try:
            tr_model = getattr(self.options, "transcription_model", None)
            if tr_model and isinstance(patch, dict):
                # Ensure audio/input containers exist without overwriting caller provided fields
                aud = patch.setdefault("audio", {})
                inp = aud.setdefault("input", {})
                # Only add if not explicitly provided in this patch
                if "transcription" not in inp:
                    transcription_cfg: Dict[str, Any] = {"model": tr_model}
                    lang = getattr(self.options, "transcription_language", None)
                    if lang:
                        transcription_cfg["language"] = lang
                    prompt_txt = getattr(self.options, "transcription_prompt", None)
                    if prompt_txt is not None:
                        transcription_cfg["prompt"] = prompt_txt
                    nr = getattr(self.options, "transcription_noise_reduction", None)
                    if nr is not None:
                        aud["noise_reduction"] = bool(nr)
                    if getattr(self.options, "transcription_include_logprobs", False):
                        patch.setdefault("include", [])
                        if (
                            "item.input_audio_transcription.logprobs"
                            not in patch["include"]
                        ):
                            patch["include"].append(
                                "item.input_audio_transcription.logprobs"
                            )
                    inp["transcription"] = transcription_cfg
                    try:
                        logger.debug(
                            "Realtime WS: update_session injected transcription config model=%s",
                            tr_model,
                        )
                    except Exception:
                        pass
        except Exception:
            logger.exception(
                "Realtime WS: failed injecting transcription config in update_session"
            )

        # Ensure tools are cleaned even if provided only under audio or elsewhere
        if "tools" in patch:
            patch["tools"] = _strip_tool_strict(patch["tools"])  # idempotent

        # Per server requirements, always include session.type and output_modalities
        try:
            patch["type"] = "realtime"
            # Preserve caller-provided output_modalities if present, otherwise default to configured modalities
            if "output_modalities" not in patch:
                patch["output_modalities"] = self.options.output_modalities or [
                    "audio",
                    "text",
                ]
        except Exception:
            pass

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
            out_cfg = (self._last_session_patch.get("audio") or {}).get("output") or {}
            if "voice" in out_cfg and not out_cfg.get("voice"):
                logger.warning("Realtime WS: session.update provides empty voice")
            if "instructions" not in self._last_session_patch:
                logger.warning(
                    "Realtime WS: session.update omits instructions; relying on previous instructions"
                )
        except Exception:
            pass
        # Use tracked send to attach an event_id and improve diagnostics
        await self._send_tracked(payload, label="session.update:patch")

    async def append_audio(self, pcm16_bytes: bytes) -> None:  # pragma: no cover
        b64 = base64.b64encode(pcm16_bytes).decode("ascii")
        await self._send_tracked(
            {"type": "input_audio_buffer.append", "audio": b64},
            label="input_audio_buffer.append",
        )
        try:
            self._pending_input_bytes += len(pcm16_bytes)
        except Exception:
            pass

    async def commit_input(self) -> None:  # pragma: no cover
        try:
            # If a previous response is still marked active, wait briefly, then proceed.
            # Skipping commits here can cause new turns to reference old audio and repeat answers.
            if bool(getattr(self, "_response_active", False)):
                logger.warning(
                    "Realtime WS: response active at commit; waiting briefly before proceeding"
                )
                for _ in range(5):  # up to ~0.5s
                    await asyncio.sleep(0.1)
                    if not bool(getattr(self, "_response_active", False)):
                        break
            # Avoid overlapping commits while awaiting server ack
            if bool(getattr(self, "_commit_inflight", False)):
                logger.warning("Realtime WS: skipping commit; commit in-flight")
                return
            # Avoid rapid duplicate commits
            last_commit = float(getattr(self, "_last_commit_ts", 0.0))
            if last_commit and (asyncio.get_event_loop().time() - last_commit) < 1.0:
                logger.warning("Realtime WS: skipping commit; committed recently")
                return
            # Require at least 100ms of audio (~4800 bytes at 24kHz mono 16-bit)
            min_bytes = int(0.1 * int(self.options.input_rate_hz or 24000) * 2)
        except Exception:
            min_bytes = 4800
        if int(getattr(self, "_pending_input_bytes", 0)) < min_bytes:
            try:
                logger.warning(
                    "Realtime WS: skipping commit; buffer too small bytes=%d < %d",
                    int(getattr(self, "_pending_input_bytes", 0)),
                    min_bytes,
                )
            except Exception:
                pass
            return
        # Reset commit event before sending a new commit and mark as in-flight
        try:
            self._commit_evt = asyncio.Event()
            self._commit_inflight = True
        except Exception:
            pass
        await self._send_tracked(
            {"type": "input_audio_buffer.commit"}, label="input_audio_buffer.commit"
        )
        try:
            logger.info("Realtime WS: input_audio_buffer.commit sent")
            self._pending_input_bytes = 0
            self._last_commit_ts = asyncio.get_event_loop().time()
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

    async def create_conversation_item(
        self, item: Dict[str, Any]
    ) -> None:  # pragma: no cover
        """Create a conversation item (e.g., for text input)."""
        payload = {"type": "conversation.item.create", "item": item}
        await self._send_tracked(payload, label="conversation.item.create")

    async def create_response(
        self, response_patch: Optional[Dict[str, Any]] = None
    ) -> None:  # pragma: no cover
        # Avoid duplicate responses: if server auto-creates after commit or one is already active, don't send.
        try:
            if getattr(self, "_response_active", False):
                logger.warning(
                    "Realtime WS: response.create suppressed — response already active"
                )
                return
            auto = bool(getattr(self, "_server_auto_create_enabled", False))
            last_commit = float(getattr(self, "_last_commit_ts", 0.0))
            if auto and last_commit:
                # If we committed very recently (<1.0s), assume server will auto-create
                if (asyncio.get_event_loop().time() - last_commit) < 1.0:
                    logger.info(
                        "Realtime WS: response.create skipped — server auto-create expected"
                    )
                    return
        except Exception:
            pass
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
                            # Ensure required session.type on retry
                            _sess = dict(self._last_session_patch or {})
                            _sess["type"] = "realtime"
                            await self._send(
                                {"type": "session.update", "session": _sess}
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
        # Ensure response object exists; rely on session defaults for modalities/audio
        if "response" not in payload:
            payload["response"] = {}
        rp = payload["response"]
        # Sanitize unsupported fields that servers may reject
        try:
            rp.pop("modalities", None)
            rp.pop("audio", None)
        except Exception:
            pass
        rp.setdefault("metadata", {"type": "response"})
        # Attach input reference so the model links this response to last audio
        if self._last_input_item_id and "input" not in rp:
            rp["input"] = [{"type": "item_reference", "id": self._last_input_item_id}]
        try:
            has_ref = bool(self._last_input_item_id)
            logger.info(
                "Realtime WS: sending response.create (input_ref=%s)",
                has_ref,
            )
        except Exception:
            pass
        # Increment response generation when we intentionally start a new response
        try:
            if not getattr(self, "_response_active", False):
                self._response_generation = (
                    int(getattr(self, "_response_generation", 0)) + 1
                )
        except Exception:
            pass
        await self._send_tracked(payload, label="response.create")
        try:
            self._response_active = True
        except Exception:
            pass

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

    def reset_output_stream(self) -> None:  # pragma: no cover
        """Drain any queued output audio and clear per-response text buffers.
        This avoids replaying stale audio if the client failed to consume previous chunks."""
        try:
            while True:
                try:
                    _ = self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                except Exception:
                    break
            try:
                self._out_text_buffers.clear()
            except Exception:
                pass
        except Exception:
            pass

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
            # Drop if this call was bound to a previous response generation
            try:
                call_gen = int(pc.get("gen", 0))
                cur_gen = int(getattr(self, "_response_generation", 0))
                if call_gen and call_gen != cur_gen:
                    logger.warning(
                        "Skipping stale tool call: id=%s name=%s call_gen=%d cur_gen=%d",
                        call_id,
                        pc.get("name"),
                        call_gen,
                        cur_gen,
                    )
                    return
            except Exception:
                pass
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
            timeout_s = float(getattr(self.options, "tool_timeout_s", 300.0) or 300.0)
            try:
                result = await asyncio.wait_for(
                    self._tool_executor(pc["name"], parsed), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Tool timeout: id=%s name=%s exceeded %.1fs",
                    call_id,
                    pc.get("name"),
                    timeout_s,
                )
                result = {"error": "tool_timeout"}
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
            # Ensure the server has created the function_call item before we post output
            try:
                ev = self._call_ready_events.get(call_id)
                if ev:
                    try:
                        await asyncio.wait_for(ev.wait(), timeout=1.5)
                    except asyncio.TimeoutError:
                        logger.debug(
                            "Call ready wait timed out; proceeding anyway: call_id=%s",
                            call_id,
                        )
                # Tiny jitter to help ordering on the server
                await asyncio.sleep(0.03)
            except Exception:
                pass

            # Send tool result via conversation.item.create, then trigger response.create (per docs)
            try:
                # Derive a valid call_id and avoid sending item_id as call_id
                derived_call_id = (
                    pc.get("call_id")
                    or self._item_call_ids.get(pc.get("item_id"))
                    or (
                        call_id
                        if isinstance(call_id, str) and call_id.startswith("call_")
                        else None
                    )
                )
                if not derived_call_id:
                    logger.error(
                        "Cannot send function_call_output: missing call_id (id=%s name=%s)",
                        call_id,
                        pc.get("name"),
                    )
                else:
                    await self._send_tracked(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": derived_call_id,
                                "output": json.dumps(result),
                            },
                        },
                        label="conversation.item.create:function_call_output",
                    )
                    logger.info(
                        "conversation.item.create(function_call_output) sent call_id=%s",
                        derived_call_id,
                    )
            except Exception:
                logger.exception(
                    "Failed to send function_call_output for call_id=%s", call_id
                )
            try:
                await asyncio.sleep(0.02)
                await self._send_tracked(
                    {
                        "type": "response.create",
                        "response": {"metadata": {"type": "response"}},
                    },
                    label="response.create:after_tool",
                )
                logger.info("response.create sent after tool output")
            except Exception:
                logger.exception(
                    "Failed to send follow-up response.create after tool output"
                )
            # Cleanup readiness event
            try:
                self._call_ready_events.pop(call_id, None)
            except Exception:
                pass
        except Exception:
            logger.exception(
                "Tool execution raised unexpectedly for call_id=%s", call_id
            )
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

    async def create_conversation_item(
        self, item: Dict[str, Any]
    ) -> None:  # pragma: no cover
        """Create a conversation item (e.g., for text input)."""
        payload = {"type": "conversation.item.create", "item": item}
        await self._send_tracked(payload, label="conversation.item.create")

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

    def reset_output_stream(self) -> None:  # pragma: no cover
        # No audio output stream to reset
        return

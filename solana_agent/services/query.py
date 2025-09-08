"""
Query service implementation.

This service orchestrates the processing of user queries, coordinating
other services to provide comprehensive responses while maintaining
clean separation of concerns.
"""

import logging
import asyncio
import re
import time
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    Tuple,
)

from pydantic import BaseModel

# Interface imports
from solana_agent.interfaces.services.query import QueryService as QueryServiceInterface
from solana_agent.interfaces.services.routing import (
    RoutingService as RoutingServiceInterface,
)
from solana_agent.interfaces.providers.memory import (
    MemoryProvider as MemoryProviderInterface,
)
from solana_agent.interfaces.services.knowledge_base import (
    KnowledgeBaseService as KnowledgeBaseInterface,
)
from solana_agent.interfaces.guardrails.guardrails import InputGuardrail

from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService

logger = logging.getLogger(__name__)


class QueryService(QueryServiceInterface):
    """Service for processing user queries and coordinating response generation."""

    def __init__(
        self,
        agent_service: AgentService,
        routing_service: RoutingService,
        memory_provider: Optional[MemoryProviderInterface] = None,
        knowledge_base: Optional[KnowledgeBaseInterface] = None,
        kb_results_count: int = 3,
        input_guardrails: List[InputGuardrail] = None,
    ):
        """Initialize the query service."""
        self.agent_service = agent_service
        self.routing_service = routing_service
        self.memory_provider = memory_provider
        self.knowledge_base = knowledge_base
        self.kb_results_count = kb_results_count
        self.input_guardrails = input_guardrails or []
        # Per-user sticky sessions (in-memory)
        # { user_id: { 'agent': str, 'started_at': float, 'last_updated': float, 'required_complete': bool } }
        self._sticky_sessions: Dict[str, Dict[str, Any]] = {}
        # Optional realtime service attached by factory (populated in factory)
        self.realtime = None  # type: ignore[attr-defined]

    def _get_sticky_agent(self, user_id: str) -> Optional[str]:
        sess = self._sticky_sessions.get(user_id)
        return sess.get("agent") if isinstance(sess, dict) else None

    def _set_sticky_agent(
        self, user_id: str, agent_name: str, required_complete: bool = False
    ) -> None:
        self._sticky_sessions[user_id] = {
            "agent": agent_name,
            "started_at": self._sticky_sessions.get(user_id, {}).get(
                "started_at", time.time()
            ),
            "last_updated": time.time(),
            "required_complete": required_complete,
        }

    def _update_sticky_required_complete(
        self, user_id: str, required_complete: bool
    ) -> None:
        if user_id in self._sticky_sessions:
            self._sticky_sessions[user_id]["required_complete"] = required_complete
            self._sticky_sessions[user_id]["last_updated"] = time.time()

    def _clear_sticky_agent(self, user_id: str) -> None:
        if user_id in self._sticky_sessions:
            del self._sticky_sessions[user_id]

    # LLM-backed switch intent detection (gpt-4.1-mini)
    class _SwitchIntentModel(BaseModel):
        switch: bool = False
        target_agent: Optional[str] = None
        start_new: bool = False

    async def _detect_switch_intent(
        self, text: str, available_agents: List[str]
    ) -> Tuple[bool, Optional[str], bool]:
        """Detect if the user is asking to switch agents or start a new conversation.

        Returns: (switch_requested, target_agent_name_or_none, start_new_conversation)
        Implemented as an LLM call to gpt-4.1-mini with structured output.
        """
        if not text:
            return (False, None, False)

        # Instruction and user prompt for the classifier
        instruction = (
            "You are a strict intent classifier for agent routing. "
            "Decide if the user's message requests switching to another agent or starting a new conversation. "
            "Only return JSON with keys: switch (bool), target_agent (string|null), start_new (bool). "
            "If a target agent is mentioned, it MUST be one of the provided agent names (case-insensitive). "
            "If none clearly applies, set switch=false and start_new=false and target_agent=null."
        )
        user_prompt = (
            f"Available agents (choose only from these if a target is specified): {available_agents}\n\n"
            f"User message:\n{text}\n\n"
            'Return JSON only, like: {"switch": true|false, "target_agent": "<one_of_available_or_null>", "start_new": true|false}'
        )

        # Primary: use llm_provider.parse_structured_output
        try:
            if hasattr(self.agent_service.llm_provider, "parse_structured_output"):
                try:
                    result = (
                        await self.agent_service.llm_provider.parse_structured_output(
                            prompt=user_prompt,
                            system_prompt=instruction,
                            model_class=QueryService._SwitchIntentModel,
                            model="gpt-4.1-mini",
                        )
                    )
                except TypeError:
                    # Provider may not accept 'model' kwarg
                    result = (
                        await self.agent_service.llm_provider.parse_structured_output(
                            prompt=user_prompt,
                            system_prompt=instruction,
                            model_class=QueryService._SwitchIntentModel,
                        )
                    )
                switch = bool(getattr(result, "switch", False))
                target = getattr(result, "target_agent", None)
                start_new = bool(getattr(result, "start_new", False))
                # Normalize target to available agent name
                if target:
                    target_lower = target.lower()
                    norm = None
                    for a in available_agents:
                        if a.lower() == target_lower or target_lower in a.lower():
                            norm = a
                            break
                    target = norm
                if not switch:
                    target = None
                return (switch, target, start_new)
        except Exception as e:
            logger.debug(f"LLM switch intent parse_structured_output failed: {e}")

        # Fallback: generate_response with output_model
        try:
            async for r in self.agent_service.generate_response(
                agent_name="default",
                user_id="router",
                query="",
                images=None,
                memory_context="",
                output_format="text",
                prompt=f"{instruction}\n\n{user_prompt}",
                output_model=QueryService._SwitchIntentModel,
            ):
                result = r
                switch = False
                target = None
                start_new = False
                try:
                    switch = bool(result.switch)  # type: ignore[attr-defined]
                    target = result.target_agent  # type: ignore[attr-defined]
                    start_new = bool(result.start_new)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        d = result.model_dump()
                        switch = bool(d.get("switch", False))
                        target = d.get("target_agent")
                        start_new = bool(d.get("start_new", False))
                    except Exception:
                        pass
                if target:
                    target_lower = str(target).lower()
                    norm = None
                    for a in available_agents:
                        if a.lower() == target_lower or target_lower in a.lower():
                            norm = a
                            break
                    target = norm
                if not switch:
                    target = None
                return (switch, target, start_new)
        except Exception as e:
            logger.debug(f"LLM switch intent generate_response failed: {e}")

        # Last resort: no switch
        return (False, None, False)

    async def process(
        self,
        user_id: str,
        query: Union[str, bytes],
        images: Optional[List[Union[str, bytes]]] = None,
        output_format: Literal["text", "audio"] = "text",
        realtime: bool = False,
        # Realtime minimal controls (voice/format come from audio_* args)
        vad: Optional[bool] = None,
        rt_encode_input: bool = False,
        rt_encode_output: bool = False,
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
        audio_instructions: str = "You speak in a friendly and helpful manner.",
        audio_output_format: Literal[
            "mp3", "opus", "aac", "flac", "wav", "pcm"
        ] = "aac",
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
        prompt: Optional[str] = None,
        router: Optional[RoutingServiceInterface] = None,
        output_model: Optional[Type[BaseModel]] = None,
        capture_schema: Optional[Dict[str, Any]] = None,
        capture_name: Optional[str] = None,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:  # pragma: no cover
        """Process the user request and generate a response."""
        try:
            # Shortcut: realtime path (model = gpt-realtime) with WS session per call
            if realtime:
                # Late import to avoid hard dependency when unused
                from solana_agent.adapters.openai_realtime_ws import (
                    OpenAIRealtimeWebSocketSession,
                )
                from solana_agent.interfaces.providers.realtime import (
                    RealtimeSessionOptions,
                )
                from solana_agent.services.realtime import RealtimeService
                from solana_agent.adapters.ffmpeg_transcoder import FFmpegTranscoder

                # Resolve API key from the LLM adapter
                api_key = None
                try:
                    api_key = self.agent_service.llm_provider.get_api_key()  # type: ignore[attr-defined]
                except Exception:
                    pass
                if not api_key:
                    raise ValueError("OpenAI API key is required for realtime")

                # Gather initial tool schemas from target/default agent
                agents = self.agent_service.get_all_ai_agents() or {}
                agent_name = next(iter(agents.keys())) if agents else "default"
                initial_tools: List[Dict[str, Any]] = []
                try:
                    initial_tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": t["name"],
                                "description": t.get("description", ""),
                                "parameters": t.get("parameters", {}),
                                "strict": True,
                            },
                        }
                        for t in self.agent_service.get_agent_tools(agent_name)
                    ]
                except Exception:
                    initial_tools = []

                # Map audio_* formats to MIME for client transport when encoding is requested
                def _mime_from(fmt: str) -> str:
                    f = (fmt or "").lower()
                    return {
                        "aac": "audio/aac",
                        "mp3": "audio/mpeg",
                        "mp4": "audio/mp4",
                        "m4a": "audio/mp4",
                        "mpeg": "audio/mpeg",
                        "mpga": "audio/mpeg",
                        "wav": "audio/wav",
                        "flac": "audio/flac",
                        "opus": "audio/opus",
                        "ogg": "audio/ogg",
                        "webm": "audio/webm",
                        "pcm": "audio/pcm",
                    }.get(f, "audio/pcm")

                opts = RealtimeSessionOptions(
                    model="gpt-realtime",
                    voice=audio_voice,
                    vad_enabled=True if vad is None else vad,
                    input_rate_hz=24000,
                    output_rate_hz=24000,
                    input_mime="audio/pcm",
                    output_mime="audio/pcm",
                    tools=initial_tools or None,
                    tool_choice="auto",
                )
                session = OpenAIRealtimeWebSocketSession(api_key=api_key, options=opts)
                logger.info(
                    "Realtime process: user_id=%s, voice=%s, vad=%s, rt_encode_in=%s, rt_encode_out=%s, in_fmt=%s, out_fmt=%s",
                    user_id,
                    audio_voice,
                    opts.vad_enabled,
                    rt_encode_input,
                    rt_encode_output,
                    audio_input_format,
                    audio_output_format,
                )

                # Optional transcoder sidecar for client transport
                transcoder = None
                if rt_encode_input or rt_encode_output:
                    transcoder = FFmpegTranscoder()

                rt = RealtimeService(
                    session=session,
                    options=opts,
                    transcoder=transcoder,
                    accept_compressed_input=rt_encode_input,
                    client_input_mime=_mime_from(audio_input_format),
                    encode_output=rt_encode_output,
                    client_output_mime=_mime_from(audio_output_format),
                )

                # Bind auto tool execution to agent_service
                async def _exec(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
                    try:
                        return await self.agent_service.execute_tool(
                            agent_name, tool_name, args or {}
                        )
                    except Exception as e:
                        return {"status": "error", "message": str(e)}

                session.set_tool_executor(_exec)

                # Start session
                await rt.start()
                logger.debug("Realtime process: session started")
                # Update session voice/VAD if provided
                await rt.configure(
                    voice=opts.voice,
                    vad_enabled=opts.vad_enabled,
                    instructions=audio_instructions,
                )

                # Persist a streaming turn if memory available
                turn_id = await self.realtime_begin_turn(user_id)
                logger.debug("Realtime process: began turn id=%s", turn_id)

                # Send audio
                # Keep a copy for optional fallback transcription if no realtime input transcript arrives
                original_audio_bytes: Optional[bytes] = None
                original_audio_fmt: Optional[str] = None
                if isinstance(query, (bytes, bytearray)):
                    # If rt_encode_input=True, RealtimeService will transcode using client_input_mime
                    bq = bytes(query)
                    original_audio_bytes = bq
                    original_audio_fmt = audio_input_format
                    logger.info("Realtime process: appending audio len=%d", len(bq))
                    await rt.append_audio(bq)
                    await rt.commit_input()
                    logger.debug("Realtime process: committed input")
                    # When VAD is disabled, we must explicitly request a response
                    if not opts.vad_enabled:
                        logger.info(
                            "Realtime process: sending response.create (VAD disabled)"
                        )
                        await rt.create_response(
                            {
                                "modalities": ["audio"],
                                "instructions": audio_instructions,
                            }
                        )

                # Fan-in: output audio and transcripts
                # Yield audio to caller; persist transcripts if memory configured
                async def _drain_io():
                    async for out in rt.iter_output_audio_encoded():
                        logger.debug(
                            "Realtime process: yielding audio chunk len=%d", len(out)
                        )
                        yield out

                user_tr_seen = False
                user_tr_buf = ""

                async def _drain_in_tr():
                    async for text in rt.iter_input_transcript():
                        logger.debug(
                            "Realtime process: input transcript delta %r", text[:120]
                        )
                        if turn_id and text:
                            nonlocal user_tr_seen, user_tr_buf
                            user_tr_seen = True
                            user_tr_buf += text
                            await self.realtime_update_user(user_id, turn_id, text)

                async def _drain_out_tr():
                    async for text in rt.iter_output_transcript():
                        logger.debug(
                            "Realtime process: output transcript delta %r", text[:120]
                        )
                        if turn_id and text:
                            await self.realtime_update_assistant(user_id, turn_id, text)

                # Run both transcript drains in background while yielding audio
                in_task = asyncio.create_task(_drain_in_tr())
                out_task = asyncio.create_task(_drain_out_tr())
                try:
                    async for audio_chunk in _drain_io():
                        yield audio_chunk
                finally:
                    in_task.cancel()
                    out_task.cancel()
                    if turn_id:
                        # Fallback transcription if no realtime input transcript arrived
                        if (
                            not user_tr_seen
                            and original_audio_bytes is not None
                            and original_audio_fmt is not None
                        ):
                            try:
                                logger.info(
                                    "Realtime process: no realtime input transcript; running fallback transcription"
                                )
                                fallback_text = ""
                                async for (
                                    frag
                                ) in self.agent_service.llm_provider.transcribe_audio(
                                    original_audio_bytes, original_audio_fmt
                                ):
                                    fallback_text += frag
                                if fallback_text:
                                    await self.realtime_update_user(
                                        user_id, turn_id, fallback_text
                                    )
                            except Exception as e:
                                logger.debug(
                                    "Fallback transcription failed: %s", str(e)
                                )
                        try:
                            await self.realtime_finalize_turn(user_id, turn_id)
                        except Exception:
                            pass
                    logger.debug(
                        "Realtime process: finalized turn and stopping session"
                    )
                    await rt.stop()
                return

            # 1) Transcribe audio or accept text
            user_text = ""
            if not isinstance(query, str):
                logger.info(
                    f"Received audio input, transcribing format: {audio_input_format}"
                )
                async for (
                    transcript
                ) in self.agent_service.llm_provider.transcribe_audio(
                    query, audio_input_format
                ):
                    user_text += transcript
                logger.info(f"Transcription result length: {len(user_text)}")
            else:
                user_text = query
                logger.info(f"Received text input length: {len(user_text)}")

            # 2) Input guardrails
            original_text = user_text
            for guardrail in self.input_guardrails:
                try:
                    user_text = await guardrail.process(user_text)
                except Exception as e:
                    logger.debug(f"Guardrail error: {e}")
            if user_text != original_text:
                logger.info(
                    f"Input guardrails modified user text. Original length: {len(original_text)}, New length: {len(user_text)}"
                )

            # 3) Greetings shortcut
            if not images and user_text.strip().lower() in {
                "hi",
                "hello",
                "hey",
                "ping",
                "test",
            }:
                greeting = "Hello! How can I help you today?"
                if output_format == "audio":
                    async for chunk in self.agent_service.llm_provider.tts(
                        text=greeting,
                        voice=audio_voice,
                        response_format=audio_output_format,
                        instructions=audio_instructions,
                    ):
                        yield chunk
                else:
                    yield greeting
                if self.memory_provider:
                    await self._store_conversation(user_id, original_text, greeting)
                return

            # 4) Memory context (conversation history)
            memory_context = ""
            if self.memory_provider:
                try:
                    memory_context = await self.memory_provider.retrieve(user_id)
                except Exception:
                    memory_context = ""

            # 5) Knowledge base context
            kb_context = ""
            if self.knowledge_base:
                try:
                    kb_results = await self.knowledge_base.query(
                        query_text=user_text,
                        top_k=self.kb_results_count,
                        include_content=True,
                        include_metadata=False,
                    )
                    if kb_results:
                        kb_lines = [
                            "**KNOWLEDGE BASE (CRITICAL: MAKE THIS INFORMATION THE TOP PRIORITY):**"
                        ]
                        for i, r in enumerate(kb_results, 1):
                            kb_lines.append(f"[{i}] {r.get('content', '').strip()}\n")
                        kb_context = "\n".join(kb_lines)
                except Exception:
                    kb_context = ""

            # 6) Determine agent (sticky session aware; allow explicit switch/new conversation)
            agent_name = "default"
            prev_assistant = ""
            routing_input = user_text
            if self.memory_provider:
                try:
                    prev_docs = self.memory_provider.find(
                        collection="conversations",
                        query={"user_id": user_id},
                        sort=[("timestamp", -1)],
                        limit=1,
                    )
                    if prev_docs:
                        prev_user_msg = (prev_docs[0] or {}).get(
                            "user_message", ""
                        ) or ""
                        prev_assistant = (prev_docs[0] or {}).get(
                            "assistant_message", ""
                        ) or ""
                        if prev_user_msg:
                            routing_input = f"previous_user_message: {prev_user_msg}\ncurrent_user_message: {user_text}"
                except Exception:
                    pass

            # Get available agents first so the LLM can select a valid target
            agents = self.agent_service.get_all_ai_agents() or {}
            available_agent_names = list(agents.keys())

            # LLM detects switch intent
            (
                switch_requested,
                requested_agent_raw,
                start_new,
            ) = await self._detect_switch_intent(user_text, available_agent_names)

            # Normalize requested agent to an exact available key
            requested_agent = None
            if requested_agent_raw:
                raw_lower = requested_agent_raw.lower()
                for a in available_agent_names:
                    if a.lower() == raw_lower or raw_lower in a.lower():
                        requested_agent = a
                        break

            sticky_agent = self._get_sticky_agent(user_id)

            if sticky_agent and not switch_requested:
                agent_name = sticky_agent
            else:
                try:
                    if start_new:
                        # Start fresh
                        self._clear_sticky_agent(user_id)
                    if requested_agent:
                        agent_name = requested_agent
                    else:
                        # Route if no explicit target
                        if router:
                            agent_name = await router.route_query(routing_input)
                        else:
                            agent_name = await self.routing_service.route_query(
                                routing_input
                            )
                except Exception:
                    agent_name = next(iter(agents.keys())) if agents else "default"
                self._set_sticky_agent(user_id, agent_name, required_complete=False)

            # 7) Captured data context + incremental save using previous assistant message
            capture_context = ""
            # Two completion flags:
            required_complete = False
            form_complete = False  # required + optional

            # Helpers
            def _non_empty(v: Any) -> bool:
                if v is None:
                    return False
                if isinstance(v, str):
                    s = v.strip().lower()
                    return s not in {"", "null", "none", "n/a", "na", "undefined", "."}
                if isinstance(v, (list, dict, tuple, set)):
                    return len(v) > 0
                return True

            def _parse_numbers_list(s: str) -> List[str]:
                nums = re.findall(r"\b(\d+)\b", s)
                seen, out = set(), []
                for n in nums:
                    if n not in seen:
                        seen.add(n)
                        out.append(n)
                return out

            def _extract_numbered_options(text: str) -> Dict[str, str]:
                """Parse previous assistant message for lines like:
                '1) Foo', '1. Foo', '- 1) Foo', '* 1. Foo' -> {'1': 'Foo'}"""
                options: Dict[str, str] = {}
                if not text:
                    return options
                for raw in text.splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    m = re.match(r"^(?:[-*]\s*)?(\d+)[\.)]?\s+(.*)$", line)
                    if m:
                        idx, label = m.group(1), m.group(2).strip().rstrip()
                        if len(label) >= 1:
                            options[idx] = label
                return options

            # LLM-backed field detection (gpt-4.1-mini) with graceful fallbacks
            class _FieldDetect(BaseModel):
                field: Optional[str] = None

            async def _detect_field_from_prev_question(
                prev_text: str, schema: Optional[Dict[str, Any]]
            ) -> Optional[str]:
                if not prev_text or not isinstance(schema, dict):
                    return None
                props = list((schema.get("properties") or {}).keys())
                if not props:
                    return None

                question = prev_text.strip()
                instruction = (
                    "You are a strict classifier. Given the assistant's last question and a list of "
                    "permitted schema field keys, choose exactly one key that the question is asking the user to answer. "
                    "If none apply, return null."
                )
                user_prompt = (
                    f"Schema field keys (choose exactly one of these): {props}\n"
                    f"Assistant question:\n{question}\n\n"
                    'Return strictly JSON like: {"field": "<one_of_the_keys_or_null>"}'
                )

                # Try llm_provider.parse_structured_output with mini
                try:
                    if hasattr(
                        self.agent_service.llm_provider, "parse_structured_output"
                    ):
                        try:
                            result = await self.agent_service.llm_provider.parse_structured_output(
                                prompt=user_prompt,
                                system_prompt=instruction,
                                model_class=_FieldDetect,
                                model="gpt-4.1-mini",
                            )
                        except TypeError:
                            # Provider may not accept 'model' kwarg
                            result = await self.agent_service.llm_provider.parse_structured_output(
                                prompt=user_prompt,
                                system_prompt=instruction,
                                model_class=_FieldDetect,
                            )
                        sel = None
                        try:
                            sel = getattr(result, "field", None)
                        except Exception:
                            sel = None
                        if sel is None:
                            try:
                                d = result.model_dump()
                                sel = d.get("field")
                            except Exception:
                                sel = None
                        if sel in props:
                            return sel
                except Exception as e:
                    logger.debug(
                        f"LLM parse_structured_output field detection failed: {e}"
                    )

                # Fallback: use generate_response with output_model=_FieldDetect
                try:
                    async for r in self.agent_service.generate_response(
                        agent_name=agent_name,
                        user_id=user_id,
                        query=user_text,
                        images=images,
                        memory_context="",
                        output_format="text",
                        prompt=f"{instruction}\n\n{user_prompt}",
                        output_model=_FieldDetect,
                    ):
                        fd = r
                        sel = None
                        try:
                            sel = fd.field  # type: ignore[attr-defined]
                        except Exception:
                            try:
                                d = fd.model_dump()
                                sel = d.get("field")
                            except Exception:
                                sel = None
                        if sel in props:
                            return sel
                        break
                except Exception as e:
                    logger.debug(f"LLM generate_response field detection failed: {e}")

                # Final heuristic fallback (keeps system working if LLM unavailable)
                t = question.lower()
                for key in props:
                    if key in t:
                        return key
                return None

            # Resolve active capture from args or agent config
            active_capture_name = capture_name
            active_capture_schema = capture_schema
            if not active_capture_name or not active_capture_schema:
                try:
                    cap_cfg = self.agent_service.get_agent_capture(agent_name)
                    if cap_cfg:
                        active_capture_name = active_capture_name or cap_cfg.get("name")
                        active_capture_schema = active_capture_schema or cap_cfg.get(
                            "schema"
                        )
                except Exception:
                    pass

            latest_by_name: Dict[str, Dict[str, Any]] = {}
            if self.memory_provider:
                try:
                    docs = self.memory_provider.find(
                        collection="captures",
                        query={"user_id": user_id},
                        sort=[("timestamp", -1)],
                        limit=100,
                    )
                    for d in docs or []:
                        name = (d or {}).get("capture_name")
                        if not name or name in latest_by_name:
                            continue
                        latest_by_name[name] = {
                            "data": (d or {}).get("data", {}) or {},
                            "mode": (d or {}).get("mode", "once"),
                            "agent": (d or {}).get("agent_name"),
                        }
                except Exception:
                    pass

            # Incremental save: use prev_assistant's numbered list to map numeric reply -> labels
            incremental: Dict[str, Any] = {}
            try:
                if (
                    self.memory_provider
                    and active_capture_name
                    and isinstance(active_capture_schema, dict)
                ):
                    props = (active_capture_schema or {}).get("properties", {})
                    required_fields = list(
                        (active_capture_schema or {}).get("required", []) or []
                    )
                    all_fields = list(props.keys())
                    optional_fields = [
                        f for f in all_fields if f not in set(required_fields)
                    ]

                    active_data_existing = (
                        latest_by_name.get(active_capture_name, {}) or {}
                    ).get("data", {}) or {}

                    def _missing(fields: List[str]) -> List[str]:
                        return [
                            f
                            for f in fields
                            if not _non_empty(active_data_existing.get(f))
                        ]

                    missing_required = _missing(required_fields)
                    missing_optional = _missing(optional_fields)

                    target_field: Optional[
                        str
                    ] = await _detect_field_from_prev_question(
                        prev_assistant, active_capture_schema
                    )
                    if not target_field:
                        # If exactly one required missing, target it; else if none required missing and exactly one optional missing, target it.
                        if len(missing_required) == 1:
                            target_field = missing_required[0]
                        elif len(missing_required) == 0 and len(missing_optional) == 1:
                            target_field = missing_optional[0]

                    if target_field and target_field in props:
                        f_schema = props.get(target_field, {}) or {}
                        f_type = f_schema.get("type")
                        number_to_label = _extract_numbered_options(prev_assistant)

                        if number_to_label:
                            nums = _parse_numbers_list(user_text)
                            labels = [
                                number_to_label[n] for n in nums if n in number_to_label
                            ]
                            if labels:
                                if f_type == "array":
                                    incremental[target_field] = labels
                                else:
                                    incremental[target_field] = labels[0]

                        if target_field not in incremental:
                            if f_type == "number":
                                m = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", user_text)
                                if m:
                                    try:
                                        incremental[target_field] = float(m.group(1))
                                    except Exception:
                                        pass
                            elif f_type == "array":
                                parts = [
                                    p.strip()
                                    for p in re.split(r"[,\n;]+", user_text)
                                    if p.strip()
                                ]
                                if parts:
                                    incremental[target_field] = parts
                            else:
                                if user_text.strip():
                                    incremental[target_field] = user_text.strip()

                    if incremental:
                        cleaned = {
                            k: v for k, v in incremental.items() if _non_empty(v)
                        }
                        if cleaned:
                            try:
                                await self.memory_provider.save_capture(
                                    user_id=user_id,
                                    capture_name=active_capture_name,
                                    agent_name=agent_name,
                                    data=cleaned,
                                    schema=active_capture_schema,
                                )
                            except Exception as se:
                                logger.error(f"Error saving incremental capture: {se}")

            except Exception as e:
                logger.debug(f"Incremental extraction skipped: {e}")

            # Build capture context, merging in incremental immediately (avoid read lag)
            def _get_active_data(name: Optional[str]) -> Dict[str, Any]:
                if not name:
                    return {}
                base = (latest_by_name.get(name, {}) or {}).get("data", {}) or {}
                if incremental:
                    base = {**base, **incremental}
                return base

            lines: List[str] = []
            if active_capture_name and isinstance(active_capture_schema, dict):
                props = (active_capture_schema or {}).get("properties", {})
                required_fields = list(
                    (active_capture_schema or {}).get("required", []) or []
                )
                all_fields = list(props.keys())
                optional_fields = [
                    f for f in all_fields if f not in set(required_fields)
                ]

                active_data = _get_active_data(active_capture_name)

                def _missing_from(data: Dict[str, Any], fields: List[str]) -> List[str]:
                    return [f for f in fields if not _non_empty(data.get(f))]

                missing_required = _missing_from(active_data, required_fields)
                missing_optional = _missing_from(active_data, optional_fields)

                required_complete = (
                    len(missing_required) == 0 and len(required_fields) > 0
                )
                form_complete = required_complete and len(missing_optional) == 0

                lines.append(
                    "CAPTURED FORM STATE (Authoritative; do not re-ask filled values):"
                )
                lines.append(f"- form_name: {active_capture_name}")

                if active_data:
                    pairs = [
                        f"{k}: {v}" for k, v in active_data.items() if _non_empty(v)
                    ]
                    lines.append(
                        f"- filled_fields: {', '.join(pairs) if pairs else '(none)'}"
                    )
                else:
                    lines.append("- filled_fields: (none)")

                lines.append(
                    f"- missing_required_fields: {', '.join(missing_required) if missing_required else '(none)'}"
                )
                lines.append(
                    f"- missing_optional_fields: {', '.join(missing_optional) if missing_optional else '(none)'}"
                )
                lines.append("")

            if latest_by_name:
                lines.append("OTHER CAPTURED USER DATA (for reference):")
                for cname, info in latest_by_name.items():
                    if cname == active_capture_name:
                        continue
                    data = info.get("data", {}) or {}
                    if data:
                        pairs = [f"{k}: {v}" for k, v in data.items() if _non_empty(v)]
                        lines.append(
                            f"- {cname}: {', '.join(pairs) if pairs else '(none)'}"
                        )
                    else:
                        lines.append(f"- {cname}: (none)")

            if lines:
                capture_context = "\n".join(lines) + "\n\n"
            # Update sticky session completion flag
            try:
                self._update_sticky_required_complete(user_id, required_complete)
            except Exception:
                pass

            # Merge contexts + flow rules
            combined_context = ""
            if capture_context:
                combined_context += capture_context
            if memory_context:
                combined_context += f"CONVERSATION HISTORY (Use for continuity; not authoritative for facts):\n{memory_context}\n\n"
            if kb_context:
                combined_context += kb_context + "\n"
            if combined_context:
                combined_context += (
                    "PRIORITIZATION GUIDE:\n"
                    "- Prefer Captured User Data for user-specific fields.\n"
                    "- Prefer KB/tools for facts.\n"
                    "- History is for tone and continuity.\n\n"
                    "FORM FLOW RULES:\n"
                    "- Ask exactly one field per turn.\n"
                    "- If any required fields are missing, ask the next missing required field.\n"
                    "- If all required fields are filled but optional fields are missing, ask the next missing optional field.\n"
                    "- Do NOT re-ask or verify values present in Captured User Data (auto-saved, authoritative).\n"
                    "- Do NOT provide summaries until no required or optional fields are missing.\n\n"
                )

            # 8) Generate response
            if output_format == "audio":
                async for audio_chunk in self.agent_service.generate_response(
                    agent_name=agent_name,
                    user_id=user_id,
                    query=user_text,
                    images=images,
                    memory_context=combined_context,
                    output_format="audio",
                    audio_voice=audio_voice,
                    audio_output_format=audio_output_format,
                    audio_instructions=audio_instructions,
                    prompt=prompt,
                ):
                    yield audio_chunk
                if self.memory_provider:
                    await self._store_conversation(
                        user_id=user_id,
                        user_message=user_text,
                        assistant_message=self.agent_service.last_text_response,
                    )
            else:
                full_text_response = ""
                capture_data: Optional[BaseModel] = None

                # Resolve agent capture if not provided
                if not capture_schema or not capture_name:
                    try:
                        cap = self.agent_service.get_agent_capture(agent_name)
                        if cap:
                            capture_name = cap.get("name")
                            capture_schema = cap.get("schema")
                    except Exception:
                        pass

                # Only run final structured output when no required or optional fields are missing
                if capture_schema and capture_name and form_complete:
                    try:
                        DynamicModel = self._build_model_from_json_schema(
                            capture_name, capture_schema
                        )
                        async for result in self.agent_service.generate_response(
                            agent_name=agent_name,
                            user_id=user_id,
                            query=user_text,
                            images=images,
                            memory_context=combined_context,
                            output_format="text",
                            prompt=(
                                (
                                    prompt
                                    + "\n\nUsing the captured user data above, return only the JSON for the requested schema. Do not invent values."
                                )
                                if prompt
                                else "Using the captured user data above, return only the JSON for the requested schema. Do not invent values."
                            ),
                            output_model=DynamicModel,
                        ):
                            capture_data = result  # type: ignore
                            break
                    except Exception as e:
                        logger.error(f"Error during capture structured output: {e}")

                async for chunk in self.agent_service.generate_response(
                    agent_name=agent_name,
                    user_id=user_id,
                    query=user_text,
                    images=images,
                    memory_context=combined_context,
                    output_format="text",
                    prompt=prompt,
                    output_model=output_model,
                ):
                    yield chunk
                    if output_model is None:
                        full_text_response += chunk

                if self.memory_provider and full_text_response:
                    await self._store_conversation(
                        user_id=user_id,
                        user_message=user_text,
                        assistant_message=full_text_response,
                    )

                # Save final capture data if the model returned it
                if (
                    self.memory_provider
                    and capture_schema
                    and capture_name
                    and capture_data is not None
                ):
                    try:
                        data_dict = (
                            capture_data.model_dump()
                            if hasattr(capture_data, "model_dump")
                            else capture_data.dict()
                        )
                        await self.memory_provider.save_capture(
                            user_id=user_id,
                            capture_name=capture_name,
                            agent_name=agent_name,
                            data=data_dict,
                            schema=capture_schema,
                        )
                    except Exception as e:
                        logger.error(f"Error saving capture: {e}")

        except Exception as e:
            import traceback

            error_msg = (
                "I apologize for the technical difficulty. Please try again later."
            )
            logger.error(f"Error in query processing: {e}\n{traceback.format_exc()}")

            if output_format == "audio":
                try:
                    async for chunk in self.agent_service.llm_provider.tts(
                        text=error_msg,
                        voice=audio_voice,
                        response_format=audio_output_format,
                    ):
                        yield chunk
                except Exception as tts_e:
                    logger.error(f"Error during TTS for error message: {tts_e}")
                    yield error_msg + f" (TTS Error: {tts_e})"
            else:
                yield error_msg

    async def delete_user_history(self, user_id: str) -> None:
        """Delete all conversation history for a user."""
        if self.memory_provider:
            try:
                await self.memory_provider.delete(user_id)
            except Exception as e:
                logger.error(f"Error deleting user history for {user_id}: {e}")
        else:
            logger.debug("No memory provider; skip delete_user_history")

    async def get_user_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """Get paginated message history for a user."""
        if not self.memory_provider:
            return {
                "data": [],
                "total": 0,
                "page": page_num,
                "page_size": page_size,
                "total_pages": 0,
                "error": "Memory provider not available",
            }
        try:
            skip = (page_num - 1) * page_size
            total = self.memory_provider.count_documents(
                collection="conversations", query={"user_id": user_id}
            )
            total_pages = (total + page_size - 1) // page_size if total > 0 else 0

            conversations = self.memory_provider.find(
                collection="conversations",
                query={"user_id": user_id},
                sort=[("timestamp", 1 if sort_order == "asc" else -1)],
                skip=skip,
                limit=page_size,
            )

            formatted: List[Dict[str, Any]] = []
            for conv in conversations:
                ts = conv.get("timestamp")
                ts_epoch = int(ts.timestamp()) if ts else None
                formatted.append(
                    {
                        "id": str(conv.get("_id")),
                        "user_message": conv.get("user_message"),
                        "assistant_message": conv.get("assistant_message"),
                        "timestamp": ts_epoch,
                    }
                )

            return {
                "data": formatted,
                "total": total,
                "page": page_num,
                "page_size": page_size,
                "total_pages": total_pages,
                "error": None,
            }
        except Exception as e:
            import traceback

            logger.error(
                f"Error retrieving user history for {user_id}: {e}\n{traceback.format_exc()}"
            )
            return {
                "data": [],
                "total": 0,
                "page": page_num,
                "page_size": page_size,
                "total_pages": 0,
                "error": f"Error retrieving history: {str(e)}",
            }

    async def _store_conversation(
        self, user_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Store conversation history in memory provider."""
        if not self.memory_provider:
            return
        try:
            await self.memory_provider.store(
                user_id,
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ],
            )
        except Exception as e:
            logger.error(f"Store conversation error for {user_id}: {e}")

    # --- Realtime persistence helpers (used by client/server using realtime service) ---
    async def realtime_begin_turn(
        self, user_id: str
    ) -> Optional[str]:  # pragma: no cover
        if not self.memory_provider:
            return None
        if not hasattr(self.memory_provider, "begin_stream_turn"):
            return None
        return await self.memory_provider.begin_stream_turn(user_id)  # type: ignore[attr-defined]

    async def realtime_update_user(
        self, user_id: str, turn_id: str, delta: str
    ) -> None:  # pragma: no cover
        if not self.memory_provider:
            return
        if not hasattr(self.memory_provider, "update_stream_user"):
            return
        await self.memory_provider.update_stream_user(user_id, turn_id, delta)  # type: ignore[attr-defined]

    async def realtime_update_assistant(
        self, user_id: str, turn_id: str, delta: str
    ) -> None:  # pragma: no cover
        if not self.memory_provider:
            return
        if not hasattr(self.memory_provider, "update_stream_assistant"):
            return
        await self.memory_provider.update_stream_assistant(user_id, turn_id, delta)  # type: ignore[attr-defined]

    async def realtime_finalize_turn(
        self, user_id: str, turn_id: str
    ) -> None:  # pragma: no cover
        if not self.memory_provider:
            return
        if not hasattr(self.memory_provider, "finalize_stream_turn"):
            return
        await self.memory_provider.finalize_stream_turn(user_id, turn_id)  # type: ignore[attr-defined]

    def _build_model_from_json_schema(
        self, name: str, schema: Dict[str, Any]
    ) -> Type[BaseModel]:
        """Create a Pydantic model dynamically from a JSON Schema subset."""
        from pydantic import create_model

        def py_type(js: Dict[str, Any]):
            t = js.get("type")
            if isinstance(t, list):
                non_null = [x for x in t if x != "null"]
                if not non_null:
                    return Optional[Any]
                base = py_type({"type": non_null[0]})
                return Optional[base]
            if t == "string":
                return str
            if t == "integer":
                return int
            if t == "number":
                return float
            if t == "boolean":
                return bool
            if t == "array":
                items = js.get("items", {"type": "string"})
                return List[py_type(items)]
            if t == "object":
                return Dict[str, Any]
            return Any

        properties: Dict[str, Any] = schema.get("properties", {})
        required = set(schema.get("required", []))
        fields: Dict[str, Any] = {}
        for field_name, field_schema in properties.items():
            typ = py_type(field_schema)
            default = field_schema.get("default")
            if field_name in required and default is None:
                fields[field_name] = (typ, ...)
            else:
                fields[field_name] = (typ, default)

        Model = create_model(name, **fields)  # type: ignore
        return Model

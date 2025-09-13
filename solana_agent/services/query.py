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

from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions

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
        # Persistent realtime WS pool per user for reuse across turns/devices
        # { user_id: [RealtimeService, ...] }
        self._rt_services: Dict[str, List[Any]] = {}
        # Global lock for creating/finding per-user sessions
        self._rt_lock = asyncio.Lock()

    async def _try_acquire_lock(self, lock: asyncio.Lock) -> bool:
        try:
            await asyncio.wait_for(lock.acquire(), timeout=0)
            return True
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    async def _alloc_realtime_session(
        self,
        user_id: str,
        *,
        api_key: str,
        rt_voice: str,
        final_instructions: str,
        initial_tools: Optional[List[Dict[str, Any]]],
        encode_in: bool,
        encode_out: bool,
        audio_input_format: str,
        audio_output_format: str,
        rt_output_modalities: Optional[List[Literal["audio", "text"]]] = None,
    ) -> Any:
        """Get a free (or new) realtime session for this user. Marks it busy via an internal lock.

        Returns the RealtimeService with an acquired _in_use_lock that MUST be released by caller.
        """
        from solana_agent.interfaces.providers.realtime import (
            RealtimeSessionOptions,
        )
        from solana_agent.adapters.openai_realtime_ws import (
            OpenAIRealtimeWebSocketSession,
        )
        from solana_agent.adapters.ffmpeg_transcoder import FFmpegTranscoder

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

        async with self._rt_lock:
            pool = self._rt_services.get(user_id) or []
            # Try to reuse an idle session strictly owned by this user
            for rt in pool:
                # Extra safety: never reuse a session from another user
                owner = getattr(rt, "_owner_user_id", None)
                if owner is not None and owner != user_id:
                    continue
                lock = getattr(rt, "_in_use_lock", None)
                if lock is None:
                    lock = asyncio.Lock()
                    setattr(rt, "_in_use_lock", lock)
                if not lock.locked():
                    if await self._try_acquire_lock(lock):
                        return rt
            # None free: create a new session
            opts = RealtimeSessionOptions(
                model="gpt-realtime",
                voice=rt_voice,
                vad_enabled=False,
                input_rate_hz=24000,
                output_rate_hz=24000,
                input_mime="audio/pcm",
                output_mime="audio/pcm",
                output_modalities=rt_output_modalities,
                tools=initial_tools or None,
                tool_choice="auto",
            )
            try:
                opts.instructions = final_instructions
                opts.voice = rt_voice
            except Exception:
                pass
            conv_session = OpenAIRealtimeWebSocketSession(api_key=api_key, options=opts)
            transcoder = FFmpegTranscoder() if (encode_in or encode_out) else None
            from solana_agent.services.realtime import RealtimeService

            rt = RealtimeService(
                session=conv_session,
                options=opts,
                transcoder=transcoder,
                accept_compressed_input=encode_in,
                client_input_mime=_mime_from(audio_input_format),
                encode_output=encode_out,
                client_output_mime=_mime_from(audio_output_format),
            )
            # Tag ownership to prevent any cross-user reuse
            setattr(rt, "_owner_user_id", user_id)
            setattr(rt, "_in_use_lock", asyncio.Lock())
            # Mark busy
            await getattr(rt, "_in_use_lock").acquire()
            pool.append(rt)
            self._rt_services[user_id] = pool
            return rt

    def _get_sticky_agent(self, user_id: str) -> Optional[str]:
        sess = self._sticky_sessions.get(user_id)
        return sess.get("agent") if isinstance(sess, dict) else None

    def _set_sticky_agent(
        self, user_id: str, agent_name: str, required_complete: bool = False
    ) -> None:
        self._sticky_sessions[user_id] = {
            "agent": agent_name,
            "started_at": time.time(),
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
            try:
                del self._sticky_sessions[user_id]
            except Exception:
                pass

    async def _build_combined_context(
        self,
        user_id: str,
        user_text: str,
        agent_name: str,
        capture_name: Optional[str] = None,
        capture_schema: Optional[Dict[str, Any]] = None,
        prev_assistant: str = "",
    ) -> Tuple[str, bool]:
        """Build combined context string and return required_complete flag."""
        # Memory context
        memory_context = ""
        if self.memory_provider:
            try:
                memory_context = await self.memory_provider.retrieve(user_id)
            except Exception:
                memory_context = ""

        # KB context
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

        # Capture context
        capture_context = ""
        required_complete = False

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

        if active_capture_name and isinstance(active_capture_schema, dict):
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

            active_data = (latest_by_name.get(active_capture_name, {}) or {}).get(
                "data", {}
            ) or {}

            def _non_empty(v: Any) -> bool:
                if v is None:
                    return False
                if isinstance(v, str):
                    s = v.strip().lower()
                    return s not in {"", "null", "none", "n/a", "na", "undefined", "."}
                if isinstance(v, (list, dict, tuple, set)):
                    return len(v) > 0
                return True

            props = (active_capture_schema or {}).get("properties", {})
            required_fields = list(
                (active_capture_schema or {}).get("required", []) or []
            )
            all_fields = list(props.keys())
            optional_fields = [f for f in all_fields if f not in set(required_fields)]

            def _missing_from(data: Dict[str, Any], fields: List[str]) -> List[str]:
                return [f for f in fields if not _non_empty(data.get(f))]

            missing_required = _missing_from(active_data, required_fields)
            missing_optional = _missing_from(active_data, optional_fields)

            required_complete = len(missing_required) == 0 and len(required_fields) > 0

            lines: List[str] = []
            lines.append(
                "CAPTURED FORM STATE (Authoritative; do not re-ask filled values):"
            )
            lines.append(f"- form_name: {active_capture_name}")

            if active_data:
                pairs = [f"{k}: {v}" for k, v in active_data.items() if _non_empty(v)]
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

        # Merge contexts
        combined_context = ""
        if capture_context:
            combined_context += capture_context
        if memory_context:
            combined_context += f"CONVERSATION HISTORY (Use for continuity; not authoritative for facts):\n{memory_context}\n\n"
        if kb_context:
            combined_context += kb_context + "\n"

        guide = (
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

        if combined_context:
            combined_context += guide
        else:
            # Diagnostics for why the context is empty
            try:
                logger.debug(
                    "_build_combined_context: empty sources — memory_provider=%s, knowledge_base=%s, active_capture=%s",
                    bool(self.memory_provider),
                    bool(self.knowledge_base),
                    bool(
                        active_capture_name and isinstance(active_capture_schema, dict)
                    ),
                )
            except Exception:
                pass
            # Provide minimal guide so realtime instructions are not blank
            combined_context = guide

        return combined_context, required_complete

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
        router: Optional[RoutingServiceInterface] = None,
        output_model: Optional[Type[BaseModel]] = None,
        capture_schema: Optional[Dict[str, Any]] = None,
        capture_name: Optional[str] = None,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:  # pragma: no cover
        """Process the user request and generate a response."""
        try:
            # Realtime request: HTTP STT for user + single WS for assistant audio
            if realtime:
                # 1) Determine if input is audio bytes. We now ALWAYS skip HTTP STT in realtime mode.
                #    The realtime websocket session (optionally with built-in transcription) is authoritative.
                is_audio_bytes = isinstance(query, (bytes, bytearray))
                user_text = "" if is_audio_bytes else str(query)
                # Provide a sensible default realtime transcription model when audio supplied
                if is_audio_bytes and not rt_transcription_model:
                    rt_transcription_model = "gpt-4o-mini-transcribe"

                # 2) Single agent selection (no multi-agent routing in realtime path)
                agent_name = self._get_sticky_agent(user_id)
                if not agent_name:
                    try:
                        agents = self.agent_service.get_all_ai_agents() or {}
                        agent_name = next(iter(agents.keys())) if agents else "default"
                    except Exception:
                        agent_name = "default"
                prev_assistant = ""
                if self.memory_provider:
                    try:
                        prev_docs = self.memory_provider.find(
                            collection="conversations",
                            query={"user_id": user_id},
                            sort=[("timestamp", -1)],
                            limit=1,
                        )
                        if prev_docs:
                            prev_assistant = (prev_docs[0] or {}).get(
                                "assistant_message", ""
                            ) or ""
                    except Exception:
                        pass

                # 3) Build context + tools
                combined_ctx = ""
                required_complete = False
                try:
                    (
                        combined_ctx,
                        required_complete,
                    ) = await self._build_combined_context(
                        user_id=user_id,
                        user_text=(user_text if not is_audio_bytes else ""),
                        agent_name=agent_name,
                        capture_name=capture_name,
                        capture_schema=capture_schema,
                        prev_assistant=prev_assistant,
                    )
                    try:
                        self._update_sticky_required_complete(
                            user_id, required_complete
                        )
                    except Exception:
                        pass
                except Exception:
                    combined_ctx = ""
                try:
                    # GA Realtime expects flattened tool definitions (no nested "function" object)
                    initial_tools = [
                        {
                            "type": "function",
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {}),
                            "strict": True,
                        }
                        for t in self.agent_service.get_agent_tools(agent_name)
                    ]
                except Exception:
                    initial_tools = []

                # Build realtime instructions: include full agent system prompt, context, and optional prompt (no user_text)
                system_prompt = ""
                try:
                    system_prompt = self.agent_service.get_agent_system_prompt(
                        agent_name
                    )
                except Exception:
                    system_prompt = ""

                parts: List[str] = []
                if system_prompt:
                    parts.append(system_prompt)
                if combined_ctx:
                    parts.append(combined_ctx)
                if prompt:
                    parts.append(str(prompt))
                final_instructions = "\n\n".join([p for p in parts if p])

                # 4) Open a single WS session for assistant audio
                # Realtime imports handled inside allocator helper

                api_key = None
                try:
                    api_key = self.agent_service.llm_provider.get_api_key()  # type: ignore[attr-defined]
                except Exception:
                    pass
                if not api_key:
                    raise ValueError("OpenAI API key is required for realtime")

                # Per-user persistent WS (single session)
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

                # Choose output encoding automatically when non-PCM output is requested
                encode_out = bool(
                    rt_encode_output or (audio_output_format.lower() != "pcm")
                )
                # If caller explicitly requests text-only realtime, disable output encoding entirely
                if (
                    rt_output_modalities is not None
                    and "audio" not in rt_output_modalities
                ):
                    if encode_out:
                        logger.debug(
                            "Realtime(QueryService): forcing encode_out False for text-only modalities=%s",
                            rt_output_modalities,
                        )
                    encode_out = False
                # Choose input transcoding when compressed input is provided (or explicitly requested)
                is_audio_bytes = isinstance(query, (bytes, bytearray))
                encode_in = bool(
                    rt_encode_input
                    or (is_audio_bytes and audio_input_format.lower() != "pcm")
                )

                # Allocate or reuse a realtime session for this specific request/user.
                # (Transcription options may be applied below; if they change after allocate we will reconfigure.)
                rt = await self._alloc_realtime_session(
                    user_id,
                    api_key=api_key,
                    rt_voice=rt_voice,
                    final_instructions=final_instructions,
                    initial_tools=initial_tools,
                    encode_in=encode_in,
                    encode_out=encode_out,
                    audio_input_format=audio_input_format,
                    audio_output_format=audio_output_format,
                    rt_output_modalities=rt_output_modalities,
                )
                # Ensure lock is released no matter what
                try:
                    # --- Apply realtime transcription config BEFORE connecting (new) ---
                    if rt_transcription_model and hasattr(rt, "_options"):
                        try:
                            setattr(
                                rt._options,
                                "transcription_model",
                                rt_transcription_model,
                            )
                            if rt_transcription_language is not None:
                                setattr(
                                    rt._options,
                                    "transcription_language",
                                    rt_transcription_language,
                                )
                            if rt_transcription_prompt is not None:
                                setattr(
                                    rt._options,
                                    "transcription_prompt",
                                    rt_transcription_prompt,
                                )
                            if rt_transcription_noise_reduction is not None:
                                setattr(
                                    rt._options,
                                    "transcription_noise_reduction",
                                    rt_transcription_noise_reduction,
                                )
                            if rt_transcription_include_logprobs:
                                setattr(
                                    rt._options, "transcription_include_logprobs", True
                                )
                        except Exception:
                            logger.debug(
                                "Failed pre-connect transcription option assignment",
                                exc_info=True,
                            )

                    # Tool executor
                    async def _exec(
                        tool_name: str, args: Dict[str, Any]
                    ) -> Dict[str, Any]:
                        try:
                            return await self.agent_service.execute_tool(
                                agent_name, tool_name, args or {}
                            )
                        except Exception as e:
                            return {"status": "error", "message": str(e)}

                    # If possible, set on underlying session
                    try:
                        if hasattr(rt, "_session"):
                            getattr(rt, "_session").set_tool_executor(_exec)  # type: ignore[attr-defined]
                    except Exception:
                        pass

                    # Connect/configure
                    if not getattr(rt, "_connected", False):
                        await rt.start()
                    await rt.configure(
                        voice=rt_voice,
                        vad_enabled=bool(vad) if vad is not None else False,
                        instructions=final_instructions,
                        tools=initial_tools or None,
                        tool_choice="auto",
                    )

                    # Ensure clean input buffers for this turn
                    try:
                        await rt.clear_input()
                    except Exception:
                        pass
                    # Also reset any leftover output audio so new turn doesn't replay old chunks
                    try:
                        if hasattr(rt, "reset_output_stream"):
                            rt.reset_output_stream()
                    except Exception:
                        pass

                    # Begin streaming turn (defer user transcript persistence until final to avoid duplicates)
                    turn_id = await self.realtime_begin_turn(user_id)
                    # We'll buffer the full user transcript (text input or realtime audio transcription) and persist exactly once.
                    # Initialize empty; we'll build it strictly from realtime transcript segments to avoid
                    # accidental duplication with pre-supplied user_text or prior buffers.
                    final_user_tr: str = ""
                    user_persisted = False

                    # Feed audio into WS if audio bytes provided and audio modality requested; else treat as text
                    wants_audio = (
                        (
                            getattr(rt, "_options", None)
                            and getattr(rt, "_options").output_modalities
                        )
                        and "audio" in getattr(rt, "_options").output_modalities  # type: ignore[attr-defined]
                    ) or (
                        rt_output_modalities is None
                        or (rt_output_modalities and "audio" in rt_output_modalities)
                    )
                    # Determine if realtime transcription should be enabled (always skip HTTP STT regardless)
                    # realtime_transcription_enabled now implicit (options set before connect)

                    if is_audio_bytes and not wants_audio:
                        # Feed audio solely for transcription (no audio output requested)
                        bq = bytes(query)
                        logger.info(
                            "Realtime: appending input audio for transcription only, len=%d, fmt=%s",
                            len(bq),
                            audio_input_format,
                        )
                        await rt.append_audio(bq)
                        vad_enabled_value = bool(vad) if vad is not None else False
                        if not vad_enabled_value:
                            await rt.commit_input()
                            # Request only text response
                            await rt.create_response({"modalities": ["text"]})
                        else:
                            logger.debug(
                                "Realtime: VAD enabled (text-only output) — skipping manual response.create"
                            )
                    if is_audio_bytes and wants_audio:
                        bq = bytes(query)
                        logger.info(
                            "Realtime: appending input audio to WS via FFmpeg, len=%d, fmt=%s",
                            len(bq),
                            audio_input_format,
                        )
                        await rt.append_audio(bq)
                        vad_enabled_value = bool(vad) if vad is not None else False
                        if not vad_enabled_value:
                            await rt.commit_input()
                            # Manually trigger response when VAD is disabled
                            await rt.create_response({})
                        else:
                            # With server VAD enabled, the model will auto-create a response at end of speech
                            logger.debug(
                                "Realtime: VAD enabled — skipping manual response.create"
                            )
                    else:  # Text-only path OR caller excluded audio modality
                        # For text input, create conversation item first, then response
                        await rt.create_conversation_item(
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": user_text or ""}
                                ],
                            }
                        )
                        # Determine effective modalities (fall back to provided override or text only)
                        if rt_output_modalities is not None:
                            modalities = rt_output_modalities or ["text"]
                        else:
                            mo = getattr(
                                rt, "_options", RealtimeSessionOptions()
                            ).output_modalities
                            modalities = mo if mo else ["audio"]
                        if "audio" not in modalities:
                            # Ensure we do not accidentally request audio generation
                            modalities = [m for m in modalities if m == "text"] or [
                                "text"
                            ]
                        await rt.create_response(
                            {
                                "modalities": modalities,
                            }
                        )

                    # Collect audio and transcripts
                    user_tr = ""  # Accumulates realtime input transcript segments (audio path)
                    asst_tr = ""

                    input_segments: List[str] = []

                    async def _drain_in_tr():
                        """Accumulate realtime input transcript segments, de-duplicating cumulative repeats.

                        Some realtime providers emit growing cumulative transcripts (e.g. "Hel", "Hello") or
                        may occasionally resend the full final transcript. Previous logic naively concatenated
                        every segment which could yield duplicated text ("HelloHello") if cumulative or repeated
                        finals were received. This routine keeps a canonical buffer (user_tr) and only appends
                        the non-overlapping suffix of each new segment.
                        """
                        nonlocal user_tr
                        async for t in rt.iter_input_transcript():
                            if not t:
                                continue
                            # Track raw segment for optional debugging
                            input_segments.append(t)
                            if not user_tr:
                                user_tr = t
                                continue
                            if t == user_tr:
                                # Exact duplicate of current buffer; skip
                                continue
                            if t.startswith(user_tr):
                                # Cumulative growth; append only the new suffix
                                user_tr += t[len(user_tr) :]
                                continue
                            # General case: find largest overlap between end of user_tr and start of t
                            # to avoid duplicated middle content (e.g., user_tr="My name is", t="name is John")
                            overlap = 0
                            max_check = min(len(user_tr), len(t))
                            for k in range(max_check, 0, -1):
                                if user_tr.endswith(t[:k]):
                                    overlap = k
                                    break
                            user_tr += t[overlap:]

                    # Check if we need both audio and text modalities
                    modalities = getattr(
                        rt, "_options", RealtimeSessionOptions()
                    ).output_modalities or ["audio"]
                    use_combined_stream = "audio" in modalities and "text" in modalities

                    if use_combined_stream and wants_audio:
                        # Use combined stream for both modalities
                        async def _drain_out_tr():
                            nonlocal asst_tr
                            async for t in rt.iter_output_transcript():
                                if t:
                                    asst_tr += t

                        in_task = asyncio.create_task(_drain_in_tr())
                        out_task = asyncio.create_task(_drain_out_tr())
                        try:
                            # Check if the service has iter_output_combined method
                            if hasattr(rt, "iter_output_combined"):
                                async for chunk in rt.iter_output_combined():
                                    # Adapt output based on caller's requested output_format
                                    if output_format == "text":
                                        # Only yield text modalities as plain strings
                                        if getattr(chunk, "modality", None) == "text":
                                            yield chunk.data  # type: ignore[attr-defined]
                                        continue
                                    # Audio streaming path
                                    if getattr(chunk, "modality", None) == "audio":
                                        # Yield raw bytes if data present
                                        yield getattr(chunk, "data", b"")
                                    elif (
                                        getattr(chunk, "modality", None) == "text"
                                        and output_format == "audio"
                                    ):
                                        # Optionally ignore or log text while audio requested
                                        continue
                                    else:
                                        # Fallback: ignore unknown modalities for now
                                        continue
                            else:
                                # Fallback: yield audio chunks as RealtimeChunk objects
                                async for audio_chunk in rt.iter_output_audio_encoded():
                                    if output_format == "text":
                                        # Ignore audio when text requested
                                        continue
                                    # output_format audio: provide raw bytes
                                    if hasattr(audio_chunk, "modality"):
                                        if (
                                            getattr(audio_chunk, "modality", None)
                                            == "audio"
                                        ):
                                            yield getattr(audio_chunk, "data", b"")
                                    else:
                                        yield audio_chunk
                        finally:
                            # Allow transcript drain tasks to finish to capture user/asst text before persistence
                            try:
                                await asyncio.wait_for(in_task, timeout=0.05)
                            except Exception:
                                in_task.cancel()
                            try:
                                await asyncio.wait_for(out_task, timeout=0.05)
                            except Exception:
                                out_task.cancel()
                        # HTTP STT path removed: realtime audio input transcript (if any) is authoritative
                        # Persist transcripts after combined streaming completes
                        if turn_id:
                            try:
                                effective_user_tr = user_tr or ("".join(input_segments))
                                try:
                                    setattr(
                                        self,
                                        "_last_realtime_user_transcript",
                                        effective_user_tr,
                                    )
                                except Exception:
                                    pass
                                if effective_user_tr:
                                    final_user_tr = effective_user_tr
                                elif (
                                    isinstance(query, str)
                                    and query
                                    and not input_segments
                                    and not user_tr
                                ):
                                    final_user_tr = query
                                if asst_tr:
                                    await self.realtime_update_assistant(
                                        user_id, turn_id, asst_tr
                                    )
                            except Exception:
                                pass
                            if final_user_tr and not user_persisted:
                                try:
                                    await self.realtime_update_user(
                                        user_id, turn_id, final_user_tr
                                    )
                                    user_persisted = True
                                except Exception:
                                    pass
                            try:
                                await self.realtime_finalize_turn(user_id, turn_id)
                            except Exception:
                                pass
                            if final_user_tr and not user_persisted:
                                try:
                                    await self.realtime_update_user(
                                        user_id, turn_id, final_user_tr
                                    )
                                    user_persisted = True
                                except Exception:
                                    pass
                    elif wants_audio:
                        # Use separate streams (legacy behavior)
                        async def _drain_out_tr():
                            nonlocal asst_tr
                            async for t in rt.iter_output_transcript():
                                if t:
                                    asst_tr += t

                        in_task = asyncio.create_task(_drain_in_tr())
                        out_task = asyncio.create_task(_drain_out_tr())
                        try:
                            async for audio_chunk in rt.iter_output_audio_encoded():
                                if output_format == "text":
                                    # Skip audio when caller wants text only
                                    continue
                                # output_format audio: yield raw bytes
                                if hasattr(audio_chunk, "modality"):
                                    if (
                                        getattr(audio_chunk, "modality", None)
                                        == "audio"
                                    ):
                                        yield getattr(audio_chunk, "data", b"")
                                else:
                                    yield audio_chunk
                        finally:
                            try:
                                await asyncio.wait_for(in_task, timeout=0.05)
                            except Exception:
                                in_task.cancel()
                            try:
                                await asyncio.wait_for(out_task, timeout=0.05)
                            except Exception:
                                out_task.cancel()
                        # HTTP STT path removed
                        # Persist transcripts after audio-only streaming
                        if turn_id:
                            try:
                                effective_user_tr = user_tr or ("".join(input_segments))
                                try:
                                    setattr(
                                        self,
                                        "_last_realtime_user_transcript",
                                        effective_user_tr,
                                    )
                                except Exception:
                                    pass
                                # Buffer final transcript for single persistence
                                if effective_user_tr:
                                    final_user_tr = effective_user_tr
                                elif (
                                    isinstance(query, str)
                                    and query
                                    and not input_segments
                                    and not user_tr
                                ):
                                    final_user_tr = query
                                if asst_tr:
                                    await self.realtime_update_assistant(
                                        user_id, turn_id, asst_tr
                                    )
                            except Exception:
                                pass
                            if final_user_tr and not user_persisted:
                                try:
                                    await self.realtime_update_user(
                                        user_id, turn_id, final_user_tr
                                    )
                                    user_persisted = True
                                except Exception:
                                    pass
                            try:
                                await self.realtime_finalize_turn(user_id, turn_id)
                            except Exception:
                                pass
                        # If no WS input transcript was captured, fall back to HTTP STT result
                    else:
                        # Text-only: just stream assistant transcript if available (no audio iteration)
                        # If original input was audio bytes but caller only wants text output (no audio modality),
                        # we still need to drain the input transcript stream to build user_tr.
                        in_task_audio_only = None
                        if is_audio_bytes:
                            in_task_audio_only = asyncio.create_task(_drain_in_tr())

                        async def _drain_out_tr_text():
                            nonlocal asst_tr
                            async for t in rt.iter_output_transcript():
                                if t:
                                    asst_tr += t
                                    yield t  # Yield incremental text chunks directly

                        async for t in _drain_out_tr_text():
                            # Provide plain text to caller
                            yield t
                        # Wait for input transcript (if any) before persistence
                        if "in_task_audio_only" in locals() and in_task_audio_only:
                            try:
                                await asyncio.wait_for(in_task_audio_only, timeout=0.1)
                            except Exception:
                                in_task_audio_only.cancel()
                        # No HTTP STT fallback
                        if turn_id:
                            try:
                                effective_user_tr = user_tr or ("".join(input_segments))
                                try:
                                    setattr(
                                        self,
                                        "_last_realtime_user_transcript",
                                        effective_user_tr,
                                    )
                                except Exception:
                                    pass
                                # For text-only modality but audio-origin (cumulative segments captured), persist user transcript
                                if effective_user_tr:
                                    final_user_tr = effective_user_tr
                                elif (
                                    isinstance(query, str)
                                    and query
                                    and not input_segments
                                    and not user_tr
                                ):
                                    final_user_tr = query
                                if asst_tr:
                                    await self.realtime_update_assistant(
                                        user_id, turn_id, asst_tr
                                    )
                            except Exception:
                                pass
                            if final_user_tr and not user_persisted:
                                try:
                                    await self.realtime_update_user(
                                        user_id, turn_id, final_user_tr
                                    )
                                    user_persisted = True
                                except Exception:
                                    pass
                            try:
                                await self.realtime_finalize_turn(user_id, turn_id)
                            except Exception:
                                pass
                        # Input transcript task already awaited above
                        # Clear input buffer for next turn reuse
                        try:
                            await rt.clear_input()
                        except Exception:
                            pass
                finally:
                    # Always release the session for reuse by other concurrent requests/devices
                    try:
                        lock = getattr(rt, "_in_use_lock", None)
                        if lock and lock.locked():
                            lock.release()
                    except Exception:
                        pass
                    return

            # 1) Acquire user_text (transcribe audio or direct text) for non-realtime path
            user_text = ""
            if not isinstance(query, str):
                try:
                    logger.info(
                        f"Received audio input, transcribing format: {audio_input_format}"
                    )
                    async for tpart in self.agent_service.llm_provider.transcribe_audio(  # type: ignore[attr-defined]
                        query, audio_input_format
                    ):
                        user_text += tpart
                except Exception:
                    user_text = ""
            else:
                user_text = query

            # 2) Input guardrails
            for guardrail in self.input_guardrails:
                try:
                    user_text = await guardrail.process(user_text)
                except Exception as e:
                    logger.debug(f"Guardrail error: {e}")

            # 3) Memory context (conversation history)
            memory_context = ""
            if self.memory_provider:
                try:
                    memory_context = await self.memory_provider.retrieve(user_id)
                except Exception:
                    memory_context = ""

            # 4) Knowledge base context
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

            # 5) Determine agent (sticky session aware; allow explicit switch/new conversation)
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

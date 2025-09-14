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

    def _compute_encode_in(
        self,
        *,
        is_audio_bytes: bool,
        audio_input_format: str,
        rt_encode_input: bool,
    ) -> bool:
        """Determine if input audio should be transcoded.

        Rules:
        - Must actually be raw audio bytes provided (bytes/bytearray) else False.
        - If format is already raw PCM ("pcm" or "audio/pcm") never transcode, even if rt_encode_input=True.
        - For any other declared format (e.g. mp3, wav, webm, etc.) return True because we need a
          deterministic 16‑bit PCM stream for the realtime session (channel/rate normalization) and
          to avoid depending on the upstream container specifics.

        Note: The rt_encode_input flag is currently advisory but ignored for PCM since transcoding
        would be a no‑op and wasteful. For non‑PCM formats we always transcode, so the flag does not
        change the outcome right now; it is kept for future extension (e.g. allowing direct passthrough
        of certain compressed codecs if the backend gains native support).
        """
        if not is_audio_bytes:
            return False
        fmt = (audio_input_format or "").lower()
        if fmt in {"pcm", "audio/pcm"}:
            return False
        # All other formats (including 'wav', 'mp3', etc.) require a decode/normalize step.
        return True

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
                input_rate_hz=16000,
                output_rate_hz=16000,
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
        # Prefer raw PCM passthrough for realtime output (overrides default aac when True and caller didn't request another format)
        rt_prefer_pcm: bool = False,
        rt_output_rate_hz: Optional[int] = None,
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
            # Realtime refactored branch (clean delegation)
            if realtime:
                async for chunk in self._process_realtime(
                    user_id=user_id,
                    query=query,
                    output_format=output_format,
                    vad=vad,
                    rt_encode_input=rt_encode_input,
                    rt_encode_output=rt_encode_output,
                    rt_output_modalities=rt_output_modalities,
                    rt_voice=rt_voice,
                    rt_transcription_model=rt_transcription_model,
                    rt_transcription_language=rt_transcription_language,
                    rt_transcription_prompt=rt_transcription_prompt,
                    rt_transcription_noise_reduction=rt_transcription_noise_reduction,
                    rt_transcription_include_logprobs=rt_transcription_include_logprobs,
                    audio_voice=audio_voice,
                    audio_output_format=audio_output_format,
                    audio_input_format=audio_input_format,
                    prompt=prompt,
                    capture_schema=capture_schema,
                    capture_name=capture_name,
                    rt_prefer_pcm=rt_prefer_pcm,
                    rt_output_rate_hz=rt_output_rate_hz,
                ):
                    yield chunk
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

            # For realtime shared-session concurrency error, propagate so caller sees RuntimeError
            if (
                realtime
                and isinstance(e, RuntimeError)
                and "already being consumed" in str(e)
            ):
                raise

            if output_format == "audio":
                try:
                    async for chunk in self.agent_service.llm_provider.tts(
                        text=error_msg,
                        voice=audio_voice,
                        response_format=audio_output_format,
                    ):
                        # Ensure we only yield bytes for audio path
                        if isinstance(chunk, (bytes, bytearray)):
                            yield chunk
                        else:
                            yield str(chunk).encode("utf-8")
                except Exception as tts_e:
                    logger.error(f"Error during TTS for error message: {tts_e}")
                    yield (error_msg + f" (TTS Error: {tts_e})").encode("utf-8")
            else:
                yield error_msg

    async def _process_realtime(
        self,
        *,
        user_id: str,
        query: Union[str, bytes],
        output_format: Literal["text", "audio"],
        vad: Optional[bool],
        rt_encode_input: bool,
        rt_encode_output: bool,
        rt_output_modalities: Optional[List[Literal["audio", "text"]]],
        rt_voice: str,
        rt_transcription_model: Optional[str],
        rt_transcription_language: Optional[str],
        rt_transcription_prompt: Optional[str],
        rt_transcription_noise_reduction: Optional[bool],
        rt_transcription_include_logprobs: bool,
        audio_voice: str,
        audio_output_format: str,
        audio_input_format: str,
        prompt: Optional[str],
        capture_schema: Optional[Dict[str, Any]],
        capture_name: Optional[str],
        rt_prefer_pcm: bool,
        rt_output_rate_hz: Optional[int],
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:  # pragma: no cover
        """Isolated realtime handling.

        Responsibilities (minimal for refactor completion):
        - Build context/instructions
        - Decide encode_in / encode_out
        - Allocate & configure realtime session
        - Send user input (audio or text)
        - Stream back audio or text chunks
        - Release session lock

        NOTE: Streaming transcript persistence & advanced overlap merging are handled
        by higher-level memory provider helpers (not invoked when memory_provider is None).
        """

        is_audio_bytes = isinstance(query, (bytes, bytearray))
        if is_audio_bytes and not rt_transcription_model:
            rt_transcription_model = "gpt-4o-mini-transcribe"

        # Select agent (sticky or fallback first available)
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

        combined_ctx = ""
        required_complete = False
        try:
            (
                combined_ctx,
                required_complete,
            ) = await self._build_combined_context(
                user_id=user_id,
                user_text=("" if is_audio_bytes else str(query)),
                agent_name=agent_name,
                capture_name=capture_name,
                capture_schema=capture_schema,
                prev_assistant=prev_assistant,
            )
            try:
                self._update_sticky_required_complete(user_id, required_complete)
            except Exception:
                pass
        except Exception:
            combined_ctx = ""

        try:
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

        system_prompt = ""
        try:
            system_prompt = self.agent_service.get_agent_system_prompt(agent_name)
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

        api_key = None
        try:
            api_key = self.agent_service.llm_provider.get_api_key()  # type: ignore[attr-defined]
        except Exception:
            pass
        if not api_key:
            raise ValueError("OpenAI API key is required for realtime")

        # Adjust output format preference
        if rt_prefer_pcm and audio_output_format == "aac":
            audio_output_format = "pcm"

        encode_out = bool(rt_encode_output or (audio_output_format.lower() != "pcm"))
        if rt_output_modalities is not None and "audio" not in rt_output_modalities:
            if encode_out:
                logger.debug(
                    "Realtime(QueryService): forcing encode_out False for text-only modalities=%s",
                    rt_output_modalities,
                )
            encode_out = False
        encode_in = self._compute_encode_in(
            is_audio_bytes=is_audio_bytes,
            audio_input_format=audio_input_format,
            rt_encode_input=rt_encode_input,
        )

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

        lock = getattr(rt, "_in_use_lock", None)
        try:
            # Apply transcription options *before* connect if possible
            if rt_transcription_model and hasattr(rt, "_options"):
                try:
                    setattr(rt._options, "transcription_model", rt_transcription_model)
                    if rt_transcription_language is not None:
                        setattr(
                            rt._options,
                            "transcription_language",
                            rt_transcription_language,
                        )
                    if rt_transcription_prompt is not None:
                        setattr(
                            rt._options, "transcription_prompt", rt_transcription_prompt
                        )
                    if rt_transcription_noise_reduction is not None:
                        setattr(
                            rt._options,
                            "transcription_noise_reduction",
                            rt_transcription_noise_reduction,
                        )
                    if rt_transcription_include_logprobs:
                        setattr(rt._options, "transcription_include_logprobs", True)
                except Exception:
                    logger.debug(
                        "Realtime transcription option assignment failed", exc_info=True
                    )

            async def _exec(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    return await self.agent_service.execute_tool(
                        agent_name, tool_name, args or {}
                    )
                except Exception as e:
                    return {"status": "error", "message": str(e)}

            try:
                if hasattr(rt, "_session") and hasattr(
                    rt._session, "set_tool_executor"
                ):
                    rt._session.set_tool_executor(_exec)  # type: ignore[attr-defined]
            except Exception:
                pass

            if not getattr(rt, "_connected", False):
                await rt.start()
            # Preserve existing session vad setting if caller passed None; only override when explicitly True/False
            configure_vad = None if vad is None else bool(vad)
            await rt.configure(
                voice=rt_voice,
                vad_enabled=configure_vad,
                instructions=final_instructions,
                output_rate_hz=rt_output_rate_hz,
                tools=initial_tools or None,
                tool_choice="auto",
            )

            try:
                await rt.clear_input()
            except Exception:
                pass
            try:
                if hasattr(rt, "reset_output_stream"):
                    rt.reset_output_stream()
            except Exception:
                pass

            # Send user input
            if is_audio_bytes:
                try:
                    await rt.append_audio(query)  # type: ignore[arg-type]
                    # Only commit immediately if VAD explicitly disabled. If VAD enabled or left as default, let VAD trigger commit.
                    if configure_vad is False:
                        await rt.commit_input()
                        # When VAD is disabled we must explicitly request a response.
                        if hasattr(rt, "create_response"):
                            try:
                                await rt.create_response(None)
                            except Exception:
                                logger.debug(
                                    "Failed to create_response after commit (audio, no VAD)",
                                    exc_info=True,
                                )
                except Exception:
                    logger.debug("Failed to append/commit audio input", exc_info=True)
            else:
                text_query = str(query)
                if text_query.strip():
                    try:
                        if hasattr(rt, "create_conversation_item"):
                            await rt.create_conversation_item(
                                {
                                    "type": "message",
                                    "role": "user",
                                    "content": text_query,
                                }
                            )
                        if hasattr(rt, "create_response"):
                            await rt.create_response(None)
                    except Exception:
                        logger.debug("Failed to send text user item", exc_info=True)

            # Decide streaming approach
            # Prefer combined when both modalities requested
            modalities = (
                rt_output_modalities
                if rt_output_modalities is not None
                else (["audio"] if output_format == "audio" else ["text"])
            )

            async def _yield_audio():
                async for ch in rt.iter_output_audio_encoded():
                    if getattr(ch, "modality", "audio") == "audio":
                        yield ch.data  # bytes

            async def _yield_text():
                async for t in rt.iter_output_transcript():
                    if t:
                        yield t

            # --- Streaming persistence setup ---
            turn_id: Optional[str] = None
            has_stream_hooks = False
            if self.memory_provider and (
                hasattr(self.memory_provider, "begin_stream_turn")
                and hasattr(self.memory_provider, "update_stream_user")
                and hasattr(self.memory_provider, "update_stream_assistant")
                and hasattr(self.memory_provider, "finalize_stream_turn")
            ):
                try:
                    turn_id = await self.realtime_begin_turn(user_id)
                    has_stream_hooks = True if turn_id else False
                except Exception:
                    turn_id = None
                    has_stream_hooks = False

            # Track cumulative user transcript for deduping (cumulative + duplicate finals)
            user_transcript_accum = ""
            last_persisted_user = ""

            async def _persist_user_if_needed(final_text: str):
                nonlocal last_persisted_user
                if not has_stream_hooks or not turn_id:
                    return
                # For audio inputs we delay persistence until final finalize block
                if is_audio_bytes:
                    last_persisted_user = final_text  # stash only
                    return
                if not final_text or final_text == last_persisted_user:
                    return
                try:
                    await self.realtime_update_user(user_id, turn_id, final_text)
                    last_persisted_user = final_text
                except Exception:
                    pass

            async def _persist_assistant_delta(delta: str):
                if not has_stream_hooks or not turn_id:
                    return
                if not delta:
                    return
                try:
                    await self.realtime_update_assistant(user_id, turn_id, delta)
                except Exception:
                    pass

            # Helper to merge cumulative transcript pieces
            def _merge_user_piece(piece: str) -> str:
                nonlocal user_transcript_accum
                if not piece:
                    return user_transcript_accum
                # Accept either cumulative (full) or delta; choose longest with overlap elimination
                if piece.startswith(user_transcript_accum):
                    user_transcript_accum = piece
                else:
                    # Find maximal overlap suffix of existing with prefix of new piece
                    overlap = 0
                    max_check = min(len(user_transcript_accum), len(piece))
                    for k in range(max_check, 0, -1):
                        if user_transcript_accum.endswith(piece[:k]):
                            overlap = k
                            break
                    user_transcript_accum += piece[overlap:]
                return user_transcript_accum

            async def _stream_audio_only():
                # Stream audio chunks while concurrently draining transcripts for persistence
                async for ch in rt.iter_output_audio_encoded():
                    data = getattr(ch, "data", ch)
                    if isinstance(data, (bytes, bytearray)):
                        yield data
                # After audio is done, drain user transcript (if any) then assistant transcript for persistence
                if is_audio_bytes and hasattr(rt, "iter_input_transcript"):
                    try:
                        async for u in rt.iter_input_transcript():
                            if u is None:
                                break
                            merged = _merge_user_piece(u)
                            await _persist_user_if_needed(merged)
                    except Exception:
                        pass
                # Some realtime sessions may still offer output transcript even if audio-only modality requested
                if hasattr(rt, "iter_output_transcript"):
                    try:
                        async for t in rt.iter_output_transcript():
                            if t:
                                await _persist_assistant_delta(t)
                    except Exception:
                        pass

            async def _stream_text_only():
                # If audio bytes were sent, input transcript comes from iter_input_transcript
                # Assistant transcript from iter_output_transcript
                if is_audio_bytes:
                    # Drain user transcript first (provider may finish early)
                    if hasattr(rt, "iter_input_transcript"):
                        async for u in rt.iter_input_transcript():
                            if u is None:
                                break
                            merged = _merge_user_piece(u)
                            await _persist_user_if_needed(merged)
                else:
                    # Text query provided directly: treat as complete user transcript
                    merged = _merge_user_piece(str(query))
                    await _persist_user_if_needed(merged)
                async for t in rt.iter_output_transcript():
                    if t:
                        await _persist_assistant_delta(t)
                        yield t

            async def _stream_combined():
                # Use combined if available for convenience, else synthesize
                combined_supported = hasattr(rt, "iter_output_combined")
                # Persist direct text query as user transcript if provided and not audio
                if not is_audio_bytes and str(query).strip():
                    merged = _merge_user_piece(str(query))
                    await _persist_user_if_needed(merged)
                if combined_supported:
                    async for ch in rt.iter_output_combined():
                        mod = getattr(ch, "modality", None)
                        data = getattr(ch, "data", ch)
                        if mod == "audio" and isinstance(data, (bytes, bytearray)):
                            yield data
                        elif mod == "text" and isinstance(data, str):
                            await _persist_assistant_delta(data)
                            yield data
                    # After combined finishes, drain any remaining input transcript (some stubs emit input separately)
                    if is_audio_bytes:
                        async for u in rt.iter_input_transcript():
                            if u is None:
                                break
                            merged = _merge_user_piece(u)
                            await _persist_user_if_needed(merged)
                else:
                    # Fallback: interleave manual streams (simple version)
                    audio_task = asyncio.create_task(_stream_audio_only().__anext__())
                    text_task = asyncio.create_task(_stream_text_only().__anext__())
                    # Minimal fallback; not exhaustive interleaving
                    for task in (audio_task, text_task):
                        try:
                            val = await task
                            if isinstance(val, (bytes, bytearray)):
                                yield val
                            else:
                                await _persist_assistant_delta(str(val))
                                yield str(val)
                        except Exception:
                            pass

            # Execute streaming according to modalities
            try:
                if "audio" in modalities and "text" in modalities:
                    async for out in _stream_combined():
                        yield out
                elif "audio" in modalities:
                    async for out in _stream_audio_only():
                        yield out
                else:
                    async for out in _stream_text_only():
                        yield out
            finally:
                # Persist final user transcript if we collected one and never stored
                if user_transcript_accum and has_stream_hooks and turn_id:
                    # Persist (single write) now for audio input; for text we may have already stored
                    if is_audio_bytes:
                        try:
                            await self.realtime_update_user(
                                user_id, turn_id, user_transcript_accum
                            )
                        except Exception:
                            pass
                # Expose for tests
                if user_transcript_accum:
                    setattr(
                        self, "_last_realtime_user_transcript", user_transcript_accum
                    )
                if has_stream_hooks and turn_id:
                    try:
                        await self.realtime_finalize_turn(user_id, turn_id)
                    except Exception:
                        pass
        finally:
            try:
                if lock and getattr(lock, "locked", lambda: False)():
                    lock.release()
            except Exception:
                pass

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

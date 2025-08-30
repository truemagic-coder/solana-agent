"""
Query service implementation.

This service orchestrates the processing of user queries, coordinating
other services to provide comprehensive responses while maintaining
clean separation of concerns.
"""

import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Type, Union

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
from solana_agent.interfaces.guardrails.guardrails import (
    InputGuardrail,
)

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
        """Initialize the query service.

        Args:
            agent_service: Service for AI agent management
            routing_service: Service for routing queries to appropriate agents
            memory_provider: Optional provider for memory storage and retrieval
            knowledge_base: Optional provider for knowledge base interactions
            kb_results_count: Number of results to retrieve from knowledge base
            input_guardrails: List of input guardrail instances
        """
        self.agent_service = agent_service
        self.routing_service = routing_service
        self.memory_provider = memory_provider
        self.knowledge_base = knowledge_base
        self.kb_results_count = kb_results_count
        self.input_guardrails = input_guardrails or []

    async def process(
        self,
        user_id: str,
        query: Union[str, bytes],
        images: Optional[List[Union[str, bytes]]] = None,
        output_format: Literal["text", "audio"] = "text",
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
        """Process the user request with appropriate agent and apply input guardrails.

        Args:
            user_id: User ID
            query: Text query or audio bytes
            images: Optional list of image URLs (str) or image bytes.
            output_format: Response format ("text" or "audio")
            audio_voice: Voice for TTS (text-to-speech)
            audio_instructions: Audio voice instructions
            audio_output_format: Audio output format
            audio_input_format: Audio input format
            prompt: Optional prompt for the agent
            router: Optional routing service for processing
            output_model: Optional Pydantic model for structured output

        Yields:
            Response chunks (text strings or audio bytes)
        """
        try:
            # --- 1. Handle Audio Input & Extract Text ---
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

            # --- 2. Apply Input Guardrails ---
            original_text = user_text
            processed_text = user_text
            for guardrail in self.input_guardrails:
                try:
                    processed_text = await guardrail.process(processed_text)
                    logger.debug(
                        f"Applied input guardrail: {guardrail.__class__.__name__}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error applying input guardrail {guardrail.__class__.__name__}: {e}",
                        exc_info=True,
                    )
            if processed_text != original_text:
                logger.info(
                    f"Input guardrails modified user text. Original length: {len(original_text)}, New length: {len(processed_text)}"
                )
            user_text = processed_text  # Use the processed text going forward
            # --- End Apply Input Guardrails ---

            # --- 3. Handle Simple Greetings ---
            if not images and user_text.strip().lower() in [
                "test",
                "hello",
                "hi",
                "hey",
                "ping",
            ]:
                response = "Hello! How can I help you today?"
                logger.info("Handling simple greeting.")
                if output_format == "audio":
                    async for chunk in self.agent_service.llm_provider.tts(
                        text=response,
                        voice=audio_voice,
                        response_format=audio_output_format,
                        instructions=audio_instructions,
                    ):
                        yield chunk
                else:
                    yield response

                if self.memory_provider:
                    await self._store_conversation(user_id, user_text, response)
                return

            # --- 4. Get Memory Context ---
            memory_context = ""
            if self.memory_provider:
                try:
                    memory_context = await self.memory_provider.retrieve(user_id)
                    logger.info(
                        f"Retrieved memory context length: {len(memory_context)}"
                    )
                except Exception as e:
                    logger.error(f"Error retrieving memory context: {e}", exc_info=True)

            # --- 5. Retrieve Relevant Knowledge ---
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
                        kb_context = "**KNOWLEDGE BASE (CRITICAL: MAKE THIS INFORMATION THE TOP PRIORITY):**\n"
                        for i, result in enumerate(kb_results, 1):
                            content = result.get("content", "").strip()
                            kb_context += f"[{i}] {content}\n\n"
                        logger.info(
                            f"Retrieved {len(kb_results)} results from Knowledge Base."
                        )
                    else:
                        logger.info("No relevant results found in Knowledge Base.")
                except Exception as e:
                    logger.error(f"Error retrieving knowledge: {e}", exc_info=True)

            # --- 6. Route Query ---
            agent_name = "default"
            prev_assistant = ""
            try:
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
                            prev_user = (prev_docs[0] or {}).get(
                                "user_message", ""
                            ) or ""
                            prev_assistant = (prev_docs[0] or {}).get(
                                "assistant_message", ""
                            ) or ""
                            if prev_user:
                                routing_input = (
                                    f"previous_user_message: {prev_user}\n"
                                    f"current_user_message: {user_text}"
                                )
                    except Exception as e:
                        logger.debug(f"Routing continuity lookup skipped: {e}")

                if router:
                    agent_name = await router.route_query(routing_input)
                else:
                    agent_name = await self.routing_service.route_query(routing_input)
                logger.info(f"Routed query to agent: {agent_name}")
            except Exception as e:
                logger.error(
                    f"Error during routing, falling back to default agent: {e}",
                    exc_info=True,
                )

            # --- 7. Combine Context ---
            # 7a. Build Captured User Data context from Mongo (if available)
            capture_context = ""
            form_complete = False
            if self.memory_provider:
                try:
                    docs = self.memory_provider.find(
                        collection="captures",
                        query={"user_id": user_id},
                        sort=[("timestamp", -1)],
                        limit=100,
                    )
                    latest_by_name: Dict[str, Dict[str, Any]] = {}
                    for d in docs or []:
                        name = (d or {}).get("capture_name")
                        if not name or name in latest_by_name:
                            continue
                        latest_by_name[name] = {
                            "data": (d or {}).get("data", {}) or {},
                            "mode": (d or {}).get("mode", "once"),
                            "agent": (d or {}).get("agent_name"),
                        }

                    # Determine the active capture config for this agent
                    active_capture_name = capture_name
                    active_capture_schema = capture_schema
                    if not active_capture_name or not active_capture_schema:
                        try:
                            cap_cfg = self.agent_service.get_agent_capture(agent_name)
                            if cap_cfg:
                                active_capture_name = (
                                    active_capture_name or cap_cfg.get("name")
                                )
                                active_capture_schema = (
                                    active_capture_schema or cap_cfg.get("schema")
                                )
                        except Exception:
                            pass

                    # Helpers
                    def _non_empty(v: Any) -> bool:
                        if v is None:
                            return False
                        if isinstance(v, str):
                            s = v.strip().lower()
                            return s not in {
                                "",
                                "null",
                                "none",
                                "n/a",
                                "na",
                                "undefined",
                                ".",
                            }
                        if isinstance(v, (list, dict, tuple, set)):
                            return len(v) > 0
                        return True

                    def _parse_numbers_list(s: str) -> List[str]:
                        nums = re.findall(r"\b(\d+)\b", s)
                        seen = set()
                        out: List[str] = []
                        for n in nums:
                            if n not in seen:
                                seen.add(n)
                                out.append(n)
                        return out

                    def _detect_field_from_prev_question(
                        prev_text: str, schema: Dict[str, Any]
                    ) -> Optional[str]:
                        if not prev_text or not isinstance(schema, dict):
                            return None
                        t = prev_text.lower()
                        patterns = [
                            ("ideas", ["which ideas attract you", "ideas"]),
                            (
                                "description",
                                ["please describe yourself", "describe yourself"],
                            ),
                            ("myself", ["tell us about yourself"]),
                            ("questions", ["do you have any questions"]),
                            (
                                "rating",
                                ["rating", "1 to 5", "how satisfied", "how happy"],
                            ),
                            ("email", ["email"]),
                            ("phone", ["phone"]),
                            ("name", ["name"]),
                            ("city", ["city"]),
                            ("state", ["state"]),
                        ]
                        candidates = {
                            k for k in (schema.get("properties") or {}).keys()
                        }
                        for field, keys in patterns:
                            if field in candidates and any(key in t for key in keys):
                                return field
                        for field in candidates:
                            if field in t:
                                return field
                        return None

                    def _parse_value_for_field(
                        field: str, schema: Dict[str, Any], text: str
                    ) -> Optional[Any]:
                        props = (schema or {}).get("properties", {})
                        f_schema = props.get(field, {})
                        f_type = f_schema.get("type")
                        if f_type == "array":
                            items = f_schema.get("items", {})
                            if items.get("type") == "string":
                                vals = _parse_numbers_list(text)
                                return vals if vals else None
                            parts = [
                                p.strip()
                                for p in re.split(r"[,\n;]+", text)
                                if p.strip()
                            ]
                            return parts or None
                        if f_type == "number":
                            m = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", text)
                            if m:
                                try:
                                    return float(m.group(1))
                                except Exception:
                                    return None
                            return None
                        if f_type == "string":
                            if field == "email":
                                m = re.search(
                                    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I
                                )
                                return m.group(0) if m else None
                            if field == "phone":
                                digits = re.sub(r"\D", "", text)
                                return digits if len(digits) >= 7 else None
                            nums = _parse_numbers_list(text)
                            if nums:
                                return nums[0]
                            return text.strip() or None
                        return text.strip() or None

                    # 7a.1 Incremental save based on current message and last assistant question (to avoid lag)
                    incremental: Dict[str, Any] = {}
                    try:
                        if active_capture_name and isinstance(
                            active_capture_schema, dict
                        ):
                            field = _detect_field_from_prev_question(
                                prev_assistant, active_capture_schema
                            )
                            if field:
                                val = _parse_value_for_field(
                                    field, active_capture_schema, user_text
                                )
                                if _non_empty(val):
                                    incremental[field] = val

                            if incremental:
                                try:
                                    await self.memory_provider.save_capture(
                                        user_id=user_id,
                                        capture_name=active_capture_name,
                                        agent_name=agent_name,
                                        data=incremental,
                                        schema=active_capture_schema,
                                    )
                                except Exception as se:
                                    logger.error(
                                        f"Error saving incremental capture: {se}"
                                    )
                    except Exception as e:
                        logger.debug(f"Incremental extraction skipped: {e}")

                    # 7a.2 Build prompt context using latest Mongo plus the just-saved incremental values
                    def _get_active_data(name: Optional[str]) -> Dict[str, Any]:
                        if not name:
                            return {}
                        base = (latest_by_name.get(name, {}) or {}).get(
                            "data", {}
                        ) or {}
                        if incremental:
                            base = {**base, **incremental}
                        return base

                    def _missing_required(
                        schema: Optional[Dict[str, Any]], data: Dict[str, Any]
                    ) -> List[str]:
                        if not isinstance(schema, dict):
                            return []
                        required = schema.get("required", []) or []
                        return [f for f in required if not _non_empty(data.get(f))]

                    lines: List[str] = []
                    if active_capture_name:
                        active_data = _get_active_data(active_capture_name)
                        missing_required = _missing_required(
                            active_capture_schema, active_data
                        )
                        form_complete = not missing_required

                        lines.append(
                            "CAPTURED FORM STATE (Authoritative; do not re-ask filled values):"
                        )
                        lines.append(f"- form_name: {active_capture_name}")

                        if isinstance(active_data, dict) and active_data:
                            pretty_pairs = []
                            for k, v in active_data.items():
                                if _non_empty(v):
                                    pretty_pairs.append(f"{k}: {v}")
                            if pretty_pairs:
                                lines.append(
                                    f"- filled_fields: {', '.join(pretty_pairs)}"
                                )
                            else:
                                lines.append("- filled_fields: (none)")
                        else:
                            lines.append("- filled_fields: (none)")

                        if missing_required:
                            lines.append(
                                f"- missing_required_fields: {', '.join(missing_required)}"
                            )
                        else:
                            lines.append("- missing_required_fields: (none)")

                        lines.append("")

                    if latest_by_name:
                        lines.append("OTHER CAPTURED USER DATA (for reference):")
                        for cname, info in latest_by_name.items():
                            if cname == active_capture_name:
                                continue
                            data = info.get("data", {})
                            if isinstance(data, dict) and data:
                                pairs = "; ".join(
                                    [
                                        f"{k}: {v}"
                                        for k, v in data.items()
                                        if _non_empty(v)
                                    ]
                                )
                                lines.append(
                                    f"- {cname}: {pairs if pairs else '(none)'}"
                                )
                            else:
                                lines.append(f"- {cname}: (none)")

                    if lines:
                        capture_context = "\n".join(lines) + "\n\n"
                except Exception as e:
                    logger.debug(f"Capture lookup skipped: {e}")

            # 7b. Merge contexts in priority-aware order
            combined_context = ""
            if capture_context:
                combined_context += capture_context
            if memory_context:
                combined_context += f"CONVERSATION HISTORY (Use for continuity and tone; not authoritative for factual values):\n{memory_context}\n\n"
            if kb_context:
                combined_context += f"{kb_context}\n"

            if capture_context or memory_context or kb_context:
                combined_context += (
                    "PRIORITIZATION GUIDE:\n"
                    "- For user-specific fields, prefer Captured User Data when present.\n"
                    "- For factual or current information, prioritize Knowledge Base and Tool results.\n"
                    "- Use Conversation History for style and continuity, not authoritative facts.\n\n"
                )
                combined_context += (
                    "FORM FLOW RULES:\n"
                    "- Ask exactly one missing required field per turn.\n"
                    "- Do NOT verify or re-ask any values present in Captured User Data; these are authoritative and auto-saved.\n"
                    "- If no required fields are missing, proceed without further capture questions.\n\n"
                )
            logger.debug(f"Combined context length: {len(combined_context)}")

            # --- 8. Generate Response ---
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

                # Resolve agent's default capture if not provided
                if not capture_schema or not capture_name:
                    try:
                        cap = self.agent_service.get_agent_capture(agent_name)
                        if cap:
                            capture_name = cap.get("name")
                            capture_schema = cap.get("schema")
                    except Exception:
                        pass

                # Only run full structured capture when the form is complete
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

                # Persist final capture when available
                if (
                    self.memory_provider
                    and capture_schema
                    and capture_name
                    and capture_data is not None
                ):
                    try:
                        data_dict = (
                            capture_data.model_dump()  # type: ignore[attr-defined]
                            if hasattr(capture_data, "model_dump")
                            else capture_data.dict()  # type: ignore
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
        """Delete all conversation history for a user.

        Args:
            user_id: User ID
        """
        if self.memory_provider:
            try:
                await self.memory_provider.delete(user_id)
                logger.info(f"Deleted conversation history for user: {user_id}")
            except Exception as e:
                logger.error(
                    f"Error deleting user history for {user_id}: {e}", exc_info=True
                )
        else:
            logger.warning(
                "Attempted to delete user history, but no memory provider is configured."
            )

    async def get_user_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """Get paginated message history for a user."""
        if not self.memory_provider:
            logger.warning(
                "Attempted to get user history, but no memory provider is configured."
            )
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

            formatted_conversations = []
            for conv in conversations:
                timestamp = (
                    int(conv.get("timestamp").timestamp())
                    if conv.get("timestamp")
                    else None
                )
                formatted_conversations.append(
                    {
                        "id": str(conv.get("_id")),
                        "user_message": conv.get("user_message"),
                        "assistant_message": conv.get("assistant_message"),
                        "timestamp": timestamp,
                    }
                )

            logger.info(
                f"Retrieved page {page_num}/{total_pages} of history for user {user_id}"
            )
            return {
                "data": formatted_conversations,
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
        if self.memory_provider:
            try:
                await self.memory_provider.store(
                    user_id,
                    [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_message},
                    ],
                )
                logger.info(f"Stored conversation for user {user_id}")
            except Exception as e:
                logger.error(
                    f"Error storing conversation for user {user_id}: {e}", exc_info=True
                )
        else:
            logger.debug(
                "Memory provider not configured, skipping conversation storage."
            )

    def _build_model_from_json_schema(
        self, name: str, schema: Dict[str, Any]
    ) -> Type[BaseModel]:
        """Create a Pydantic model dynamically from a JSON Schema subset.

        Supports 'type' string, integer, number, boolean, object (flat), array (of simple types),
        required fields, and default values. Nested objects/arrays can be extended later.
        """
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
        fields = {}
        for field_name, field_schema in properties.items():
            typ = py_type(field_schema)
            default = field_schema.get("default")
            if field_name in required and default is None:
                fields[field_name] = (typ, ...)
            else:
                fields[field_name] = (typ, default)

        Model = create_model(name, **fields)  # type: ignore
        return Model

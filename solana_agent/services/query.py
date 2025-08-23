"""
Query service implementation.

This service orchestrates the processing of user queries, coordinating
other services to provide comprehensive responses while maintaining
clean separation of concerns.
"""

import logging
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
            # Simple greetings typically don't involve images
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

                # Store simple interaction in memory (using processed user_text)
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
                    # Use processed user_text for KB query
                    kb_results = await self.knowledge_base.query(
                        query_text=user_text,
                        top_k=self.kb_results_count,
                        include_content=True,
                        include_metadata=False,  # Keep metadata minimal for context
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
            agent_name = "default"  # Fallback agent
            try:
                # Use processed user_text for routing (images generally don't affect routing logic here)
                if router:
                    agent_name = await router.route_query(user_text)
                else:
                    agent_name = await self.routing_service.route_query(user_text)
                logger.info(f"Routed query to agent: {agent_name}")
            except Exception as e:
                logger.error(
                    f"Error during routing, falling back to default agent: {e}",
                    exc_info=True,
                )

            # --- 7. Combine Context ---
            combined_context = ""
            if memory_context:
                combined_context += f"CONVERSATION HISTORY (Use for context, but prioritize tools/KB for facts):\n{memory_context}\n\n"
            if kb_context:
                combined_context += f"{kb_context}\n"

            if memory_context or kb_context:
                combined_context += "CRITICAL PRIORITIZATION GUIDE: For factual or current information, prioritize Knowledge Base results and Tool results (if applicable) over Conversation History.\n\n"
            logger.debug(f"Combined context length: {len(combined_context)}")

            # --- 8. Generate Response ---
            # Pass the processed user_text and images to the agent service
            if output_format == "audio":
                async for audio_chunk in self.agent_service.generate_response(
                    agent_name=agent_name,
                    user_id=user_id,
                    query=user_text,  # Pass processed text
                    images=images,
                    memory_context=combined_context,
                    output_format="audio",
                    audio_voice=audio_voice,
                    audio_output_format=audio_output_format,
                    audio_instructions=audio_instructions,
                    prompt=prompt,
                ):
                    yield audio_chunk

                # Store conversation using processed user_text
                # Note: Storing images in history is not directly supported by current memory provider interface
                if self.memory_provider:
                    await self._store_conversation(
                        user_id=user_id,
                        user_message=user_text,  # Store only text part of user query
                        assistant_message=self.agent_service.last_text_response,
                    )
            else:
                full_text_response = ""
                # If capture_schema is provided, we run a structured output pass first
                capture_data: Optional[BaseModel] = None
                # If no explicit capture provided, use the agent's configured capture
                if not capture_schema or not capture_name:
                    try:
                        cap = self.agent_service.get_agent_capture(agent_name)
                        if cap:
                            capture_name = cap.get("name")
                            capture_schema = cap.get("schema")
                    except Exception:
                        pass

                if capture_schema and capture_name:
                    try:
                        # Build a dynamic Pydantic model from JSON schema
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
                                    + "\n\nReturn only the JSON for the requested schema."
                                )
                                if prompt
                                else "Return only the JSON for the requested schema."
                            ),
                            output_model=DynamicModel,
                        ):
                            # This yields a pydantic model instance
                            capture_data = result  # type: ignore
                            break
                    except Exception as e:
                        logger.error(f"Error during capture structured output: {e}")

                async for chunk in self.agent_service.generate_response(
                    agent_name=agent_name,
                    user_id=user_id,
                    query=user_text,  # Pass processed text
                    images=images,  # <-- Pass images
                    memory_context=combined_context,
                    output_format="text",
                    prompt=prompt,
                    output_model=output_model,
                ):
                    yield chunk
                    if output_model is None:
                        full_text_response += chunk

                # Store conversation using processed user_text
                # Note: Storing images in history is not directly supported by current memory provider interface
                if self.memory_provider and full_text_response:
                    await self._store_conversation(
                        user_id=user_id,
                        user_message=user_text,  # Store only text part of user query
                        assistant_message=full_text_response,
                    )

                # Persist capture if available
                if (
                    self.memory_provider
                    and capture_schema
                    and capture_name
                    and capture_data is not None
                ):
                    try:
                        # pydantic v2: model_dump
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
                    # Fallback to yielding text error if TTS fails
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
        sort_order: str = "desc",  # "asc" for oldest-first, "desc" for newest-first
    ) -> Dict[str, Any]:
        """Get paginated message history for a user.

        Args:
            user_id: User ID
            page_num: Page number (starting from 1)
            page_size: Number of messages per page
            sort_order: Sort order ("asc" or "desc")

        Returns:
            Dictionary with paginated results and metadata.
        """
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
            # Calculate skip and limit for pagination
            skip = (page_num - 1) * page_size

            # Get total count of documents
            total = self.memory_provider.count_documents(
                collection="conversations", query={"user_id": user_id}
            )

            # Calculate total pages
            total_pages = (total + page_size - 1) // page_size if total > 0 else 0

            # Get paginated results
            conversations = self.memory_provider.find(
                collection="conversations",
                query={"user_id": user_id},
                sort=[("timestamp", 1 if sort_order == "asc" else -1)],
                skip=skip,
                limit=page_size,
            )

            # Format the results
            formatted_conversations = []
            for conv in conversations:
                timestamp = (
                    int(conv.get("timestamp").timestamp())
                    if conv.get("timestamp")
                    else None
                )
                # Assuming the stored format matches what _store_conversation saves
                # (which currently only stores text messages)
                formatted_conversations.append(
                    {
                        "id": str(conv.get("_id")),
                        "user_message": conv.get("user_message"),  # Or how it's stored
                        "assistant_message": conv.get(
                            "assistant_message"
                        ),  # Or how it's stored
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
        """Store conversation history in memory provider.

        Args:
            user_id: User ID
            user_message: User message (text part, potentially processed by input guardrails)
            assistant_message: Assistant message (potentially processed by output guardrails)
        """
        if self.memory_provider:
            try:
                # Store only the text parts for now, as memory provider interface
                # doesn't explicitly handle image data storage in history.
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
                # handle ["null", "string"] => Optional[str]
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
                # For now, represent as Dict[str, Any]
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

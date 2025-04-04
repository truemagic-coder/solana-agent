"""
Query service implementation.

This service orchestrates the processing of user queries, coordinating
other services to provide comprehensive responses while maintaining
clean separation of concerns.
"""
from typing import Any, AsyncGenerator, Dict, Literal, Optional, Union

from solana_agent.interfaces.services.query import QueryService as QueryServiceInterface
from solana_agent.interfaces.services.routing import RoutingService as RoutingServiceInterface
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.interfaces.providers.memory import MemoryProvider


class QueryService(QueryServiceInterface):
    """Service for processing user queries and coordinating response generation."""

    def __init__(
        self,
        agent_service: AgentService,
        routing_service: RoutingService,
        memory_provider: Optional[MemoryProvider] = None,
    ):
        """Initialize the query service.

        Args:
            agent_service: Service for AI agent management
            routing_service: Service for routing queries to appropriate agents
            memory_provider: Optional provider for memory storage and retrieval
        """
        self.agent_service = agent_service
        self.routing_service = routing_service
        self.memory_provider = memory_provider

    async def process(
        self,
        user_id: str,
        query: Union[str, bytes],
        output_format: Literal["text", "audio"] = "text",
        audio_voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                             "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
        audio_instructions: str = "You speak in a friendly and helpful manner.",
        audio_output_format: Literal['mp3', 'opus',
                                     'aac', 'flac', 'wav', 'pcm'] = "aac",
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
        prompt: Optional[str] = None,
        router: Optional[RoutingServiceInterface] = None,
        internet_search: bool = True,
    ) -> AsyncGenerator[Union[str, bytes], None]:  # pragma: no cover
        """Process the user request with appropriate agent.

        Args:
            user_id: User ID
            query: Text query or audio bytes
            output_format: Response format ("text" or "audio")
            audio_voice: Voice for TTS (text-to-speech)
            audio_instructions: Audio voice instructions
            audio_output_format: Audio output format
            audio_input_format: Audio input format
            prompt: Optional prompt for the agent
            router: Optional routing service for processing
            internet_search: Flag to use OpenAI Internet search

        Yields:
            Response chunks (text strings or audio bytes)
        """
        try:
            # Handle audio input if provided
            user_text = ""
            if not isinstance(query, str):
                async for transcript in self.agent_service.llm_provider.transcribe_audio(query, audio_input_format):
                    user_text += transcript
            else:
                user_text = query

            # Handle simple greetings
            if user_text.strip().lower() in ["test", "hello", "hi", "hey", "ping"]:
                response = "Hello! How can I help you today?"
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

                # Store simple interaction in memory
                if self.memory_provider:
                    await self._store_conversation(user_id, user_text, response)
                return

            # Get memory context if available
            memory_context = ""
            if self.memory_provider:
                memory_context = await self.memory_provider.retrieve(user_id)

            # Route query to appropriate agent
            if router:
                agent_name = await router.route_query(user_text)
            else:
                agent_name = await self.routing_service.route_query(user_text)

            print(f"Routed to agent: {agent_name}")

            # Generate response
            if output_format == "audio":
                async for audio_chunk in self.agent_service.generate_response(
                    agent_name=agent_name,
                    user_id=user_id,
                    query=query,
                    memory_context=memory_context,
                    output_format="audio",
                    audio_voice=audio_voice,
                    audio_input_format=audio_input_format,
                    audio_output_format=audio_output_format,
                    audio_instructions=audio_instructions,
                    prompt=prompt,
                    internet_search=internet_search,
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
                async for chunk in self.agent_service.generate_response(
                    agent_name=agent_name,
                    user_id=user_id,
                    query=user_text,
                    memory_context=memory_context,
                    output_format="text",
                    prompt=prompt,
                    internet_search=internet_search,
                ):
                    yield chunk
                    full_text_response += chunk

                if self.memory_provider and full_text_response:
                    await self._store_conversation(
                        user_id=user_id,
                        user_message=user_text,
                        assistant_message=full_text_response
                    )

        except Exception as e:
            error_msg = f"I apologize for the technical difficulty. {str(e)}"
            if output_format == "audio":
                async for chunk in self.agent_service.llm_provider.tts(
                    text=error_msg,
                    voice=audio_voice,
                    response_format=audio_output_format
                ):
                    yield chunk
            else:
                yield error_msg

            print(f"Error in query processing: {str(e)}")
            import traceback
            print(traceback.format_exc())

    async def delete_user_history(self, user_id: str) -> None:
        """Delete all conversation history for a user.

        Args:
            user_id: User ID
        """
        if self.memory_provider:
            try:
                await self.memory_provider.delete(user_id)
            except Exception as e:
                print(f"Error deleting user history: {str(e)}")

    async def get_user_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "desc"  # "asc" for oldest-first, "desc" for newest-first
    ) -> Dict[str, Any]:
        """Get paginated message history for a user.

        Args:
            user_id: User ID
            page_num: Page number (starting from 1)
            page_size: Number of messages per page
            sort_order: Sort order ("asc" or "desc")

        Returns:
            Dictionary with paginated results and metadata:
            {
                "data": List of conversation entries,
                "total": Total number of entries,
                "page": Current page number,
                "page_size": Number of items per page,
                "total_pages": Total number of pages,
                "error": Error message if any
            }
        """
        if not self.memory_provider:
            return {
                "data": [],
                "total": 0,
                "page": page_num,
                "page_size": page_size,
                "total_pages": 0,
                "error": "Memory provider not available"
            }

        try:
            # Calculate skip and limit for pagination
            skip = (page_num - 1) * page_size

            # Get total count of documents
            total = self.memory_provider.count_documents(
                collection="conversations",
                query={"user_id": user_id}
            )

            # Calculate total pages
            total_pages = (total + page_size - 1) // page_size

            # Get paginated results
            conversations = self.memory_provider.find(
                collection="conversations",
                query={"user_id": user_id},
                sort=[("timestamp", 1 if sort_order == "asc" else -1)],
                skip=skip,
                limit=page_size
            )

            # Format the results
            formatted_conversations = []
            for conv in conversations:
                # Convert datetime to Unix timestamp (seconds since epoch)
                timestamp = int(conv.get("timestamp").timestamp()
                                ) if conv.get("timestamp") else None

                formatted_conversations.append({
                    "id": str(conv.get("_id")),
                    "user_message": conv.get("user_message"),
                    "assistant_message": conv.get("assistant_message"),
                    "timestamp": timestamp,
                })

            return {
                "data": formatted_conversations,
                "total": total,
                "page": page_num,
                "page_size": page_size,
                "total_pages": total_pages,
                "error": None
            }

        except Exception as e:
            print(f"Error retrieving user history: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {
                "data": [],
                "total": 0,
                "page": page_num,
                "page_size": page_size,
                "total_pages": 0,
                "error": f"Error retrieving history: {str(e)}"
            }

    async def _store_conversation(
        self, user_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Store conversation history in memory provider.

        Args:
            user_id: User ID
            user_message: User message
            assistant_message: Assistant message
        """
        if self.memory_provider:
            try:
                await self.memory_provider.store(
                    user_id,
                    [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_message},
                    ],
                )
            except Exception as e:
                print(f"Error storing conversation: {e}")

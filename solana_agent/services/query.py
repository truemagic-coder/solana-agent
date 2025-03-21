"""
Query service implementation.

This service orchestrates the processing of user queries, coordinating
other services to provide comprehensive responses while maintaining
clean separation of concerns.
"""
from pathlib import Path
from typing import Any, AsyncGenerator, BinaryIO, Dict, Literal, Optional, Union

from solana_agent.interfaces.services.query import QueryService as QueryServiceInterface
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
        query: Union[str, Path, BinaryIO],
        output_format: Literal["text", "audio"] = "text",
        voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                       "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
        audio_instructions: Optional[str] = None,
    ) -> AsyncGenerator[Union[str, bytes], None]:  # pragma: no cover
        """Process the user request with appropriate agent.

        Args:
            user_id: User ID
            query: Text query or audio file input
            output_format: Response format ("text" or "audio")
            voice: Voice to use for audio output
            audio_instructions: Optional instructions for audio synthesis

        Yields:
            Response chunks (text strings or audio bytes)
        """
        try:
            # Handle audio input if provided
            user_text = ""
            if not isinstance(query, str):
                async for transcript in self.agent_service.llm_provider.transcribe_audio(query):
                    user_text += transcript
            else:
                user_text = query

            # Handle simple greetings
            if user_text.strip().lower() in ["test", "hello", "hi", "hey", "ping"]:
                response = "Hello! How can I help you today?"
                if output_format == "audio":
                    async for chunk in self.agent_service.llm_provider.tts(response, instructions=audio_instructions, voice=voice):
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
            agent_name = await self.routing_service.route_query(user_text)

            # Generate response using agent service
            full_response = ""
            async for chunk in self.agent_service.generate_response(
                agent_name=agent_name,
                user_id=user_id,
                query=user_text,
                memory_context=memory_context,
                output_format=output_format,
                voice=voice
            ):
                yield chunk
                if output_format == "text":
                    full_response += chunk

            # For audio responses, get transcription for storage
            if output_format == "audio":
                # Re-generate response in text format for storage
                async for chunk in self.agent_service.generate_response(
                    agent_name=agent_name,
                    user_id=user_id,
                    query=user_text,
                    memory_context=memory_context,
                    output_format="text"
                ):
                    full_response += chunk

            # Store conversation and extract insights
            if self.memory_provider:
                await self._store_conversation(user_id, user_text, full_response)

        except Exception as e:
            error_msg = f"I apologize for the technical difficulty. {str(e)}"
            if output_format == "audio":
                async for chunk in self.agent_service.llm_provider.tts(error_msg, instructions=audio_instructions, voice=voice):
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
        self, user_id: str, user_text: str, response_text: str
    ) -> None:
        """Store conversation history in memory provider.

        Args:
            user_id: User ID
            user_text: User message
            response_text: Assistant response
        """
        if self.memory_provider:
            try:
                # Truncate excessively long responses
                truncated_response = self._truncate(response_text)

                await self.memory_provider.store(
                    user_id,
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": truncated_response},
                    ],
                )
            except Exception as e:
                print(f"Error storing conversation: {e}")

    def _truncate(self, text: str, limit: int = 2500) -> str:
        """Truncate text to be within token limits.

        Args:
            text: Text to truncate
            limit: Character limit

        Returns:
            Truncated text
        """
        if len(text) <= limit:
            return text

        # Try to truncate at a sentence boundary
        truncated = text[:limit]
        last_period = truncated.rfind(".")
        if last_period > limit * 0.8:  # Only use period if reasonably close to the end
            return truncated[:last_period + 1]

        return truncated + "..."

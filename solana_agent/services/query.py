"""
Query service implementation.

This service orchestrates the processing of user queries, coordinating
other services to provide comprehensive responses while maintaining
clean separation of concerns.
"""
from typing import Any, AsyncGenerator, Dict, Optional

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
        self, user_id: str, user_text: str, timezone: str = None
    ) -> AsyncGenerator[str, None]:  # pragma: no cover
        """Process the user request with appropriate agent.

        Args:
            user_id: User ID
            user_text: User query text
            timezone: Optional user timezone

        Yields:
            Response text chunks
        """
        try:
            # Handle simple greetings
            if user_text.strip().lower() in ["test", "hello", "hi", "hey", "ping"]:
                response = f"Hello! How can I help you today?"
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
            agent_name = await self.routing_service.route_query(user_id, user_text)

            # Generate response
            full_response = ""
            async for chunk in self.agent_service.generate_response(
                agent_name=agent_name,
                user_id=user_id,
                query=user_text,
                memory_context=memory_context
            ):
                yield chunk
                full_response += chunk

            # Store conversation and extract insights
            if self.memory_provider:
                await self._store_conversation(user_id, user_text, full_response)

        except Exception as e:
            yield f"I apologize for the technical difficulty. {str(e)}"
            import traceback
            print(f"Error in query processing: {str(e)}")
            print(traceback.format_exc())

    async def get_user_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "asc"  # "asc" for oldest-first, "desc" for newest-first
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
                formatted_conversations.append({
                    "id": str(conv.get("_id")),
                    "user_message": conv.get("user_message"),
                    "assistant_message": conv.get("assistant_message"),
                    "timestamp": conv.get("timestamp").isoformat()
                    if conv.get("timestamp") else None
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

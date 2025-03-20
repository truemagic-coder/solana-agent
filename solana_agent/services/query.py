"""
Query service implementation.

This service orchestrates the processing of user queries, coordinating
other services to provide comprehensive responses while maintaining
clean separation of concerns.
"""
from typing import AsyncGenerator, Dict, Optional, Any

from solana_agent.interfaces import QueryService as QueryServiceInterface
from solana_agent.interfaces import (
    AgentService, RoutingService, MemoryService,
)
from solana_agent.interfaces import MemoryProvider


class QueryService(QueryServiceInterface):
    """Service for processing user queries and coordinating response generation."""

    def __init__(
        self,
        agent_service: AgentService,
        routing_service: RoutingService,
        memory_service: MemoryService,
        memory_provider: Optional[MemoryProvider] = None,
    ):
        """Initialize the query service.

        Args:
            agent_service: Service for AI agent management
            routing_service: Service for routing queries to appropriate agents
            memory_service: Service for memory operations
            memory_provider: Optional provider for memory storage and retrieval
        """
        self.agent_service = agent_service
        self.routing_service = routing_service
        self.memory_service = memory_service
        self.memory_provider = memory_provider

    async def process(
        self, user_id: str, user_text: str, timezone: str = None
    ) -> AsyncGenerator[str, None]:
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
                await self._extract_and_store_insights(
                    user_id,
                    {"user": user_text, "assistant": full_response}
                )

        except Exception as e:
            yield f"I apologize for the technical difficulty. {str(e)}"
            import traceback
            print(f"Error in query processing: {str(e)}")
            print(traceback.format_exc())

    async def _extract_and_store_insights(
        self, user_id: str, conversation: Dict[str, str]
    ) -> None:
        """Extract insights from conversation and store in collective memory.

        Args:
            user_id: User ID
            conversation: Conversation data
        """
        if not self.memory_service:
            return

        try:
            # Extract insights
            insights = await self.memory_service.extract_insights(conversation)

            # Store them if any found
            if insights:
                await self.memory_service.store_insights(user_id, insights)

        except Exception as e:
            print(f"Error extracting insights: {e}")

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

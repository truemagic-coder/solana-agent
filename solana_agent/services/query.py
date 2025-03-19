"""
Query service implementation.

This service orchestrates the processing of user queries, coordinating
other services to provide comprehensive responses while maintaining
clean separation of concerns.
"""
import asyncio
from typing import AsyncGenerator, Dict, Optional, Any

from solana_agent.interfaces import QueryService as QueryServiceInterface
from solana_agent.interfaces import (
    AgentService, RoutingService, TicketService,
    MemoryService, NPSService, CriticService, CommandService
)
from solana_agent.interfaces import MemoryProvider
from solana_agent.domains import TicketStatus, Ticket, TicketResolution


class QueryService(QueryServiceInterface):
    """Service for processing user queries and coordinating response generation."""

    def __init__(
        self,
        agent_service: AgentService,
        routing_service: RoutingService,
        ticket_service: TicketService,
        memory_service: MemoryService,
        nps_service: NPSService,
        command_service: CommandService,
        memory_provider: Optional[MemoryProvider] = None,
        critic_service: Optional[CriticService] = None,
        enable_critic: bool = True,
        stalled_ticket_timeout: Optional[int] = 60,
    ):
        """Initialize the query service.

        Args:
            agent_service: Service for AI and human agent management
            routing_service: Service for routing queries to appropriate agents
            ticket_service: Service for ticket operations
            memory_service: Service for memory operations
            nps_service: Service for handling NPS surveys
            command_service: Service for processing system commands
            memory_provider: Optional provider for memory storage and retrieval
            critic_service: Optional service for critiquing responses
            enable_critic: Whether to enable the critic service
            stalled_ticket_timeout: Minutes before a ticket is considered stalled
        """
        self.agent_service = agent_service
        self.routing_service = routing_service
        self.ticket_service = ticket_service
        self.memory_service = memory_service
        self.nps_service = nps_service
        self.command_service = command_service
        self.memory_provider = memory_provider
        self.critic_service = critic_service
        self.enable_critic = enable_critic

    async def process(
        self, user_id: str, user_text: str, timezone: str = None
    ) -> AsyncGenerator[str, None]:
        """Process the user request with appropriate agent and handle ticket management.

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

            # Check for system commands
            if user_text.startswith("!") and self.command_service:
                command_response = await self.command_service.process_command(
                    user_id, user_text, timezone
                )
                if command_response:
                    yield command_response
                    return

            # Check for active ticket
            active_ticket = self.ticket_service.get_active_for_user(user_id)

            if active_ticket:
                # Process existing ticket
                async for chunk in self._process_existing_ticket(user_id, user_text, active_ticket):
                    yield chunk
            else:
                async for chunk in self._process_new_ticket(user_id, user_text):
                    yield chunk

        except Exception as e:
            yield f"I apologize for the technical difficulty. {str(e)}"
            import traceback
            print(f"Error in query processing: {str(e)}")
            print(traceback.format_exc())

    async def _check_ticket_resolution(
        self, response: str, query: str
    ) -> TicketResolution:
        """Determine if a ticket can be considered resolved based on the response.

        Args:
            response: Assistant's response
            query: User's query

        Returns:
            Resolution assessment
        """
        prompt = f"""
        Analyze this conversation and determine if the user query has been fully resolved.
        
        USER QUERY: {query}
        
        ASSISTANT RESPONSE: {response}
        
        Determine if this query is:
        1. "resolved" - The user's question/request has been fully addressed
        2. "needs_followup" - The assistant couldn't fully address the issue or more information is needed
        3. "cannot_determine" - Cannot tell if the issue is resolved
        """

        try:
            # Use structured output parsing
            resolution = await self.agent_service.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="You are a resolution analysis system. Analyze conversations and determine if queries have been resolved.",
                model_class=TicketResolution,
                temperature=0.2,
            )
            return resolution
        except Exception as e:
            print(f"Exception in resolution check: {e}")

        # Default fallback
        return TicketResolution(
            status="cannot_determine",
            confidence=0.2,
            reasoning="Failed to analyze resolution status",
            suggested_actions=["Review conversation manually"]
        )

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

    async def _store_conversation(self, user_id: str, user_text: str, response_text: str) -> None:
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

    async def _process_existing_ticket(
        self, user_id: str, user_text: str, ticket: Ticket
    ) -> AsyncGenerator[str, None]:
        """Process a message for an existing ticket."""
        # Get assigned agent or re-route if needed
        agent_name = ticket.assigned_to
        if not agent_name:
            agent_name, _ = await self.routing_service.route_query(user_id, user_text)
            self.ticket_service.update_ticket(
                ticket_id=ticket.id,
                status=TicketStatus.IN_PROGRESS,
                assigned_to=agent_name
            )

        # Get memory context if available
        memory_context = ""
        if self.memory_provider:
            memory_context = await self.memory_provider.retrieve(user_id)

        # Generate response with streaming
        full_response = ""
        try:
            async for chunk in self.agent_service.generate_response(
                agent_name=agent_name,
                user_id=user_id,
                query=user_text,
                memory_context=memory_context,
            ):
                full_response += chunk
                yield chunk

            # Store conversation in memory
            if self.memory_provider:
                await self._store_conversation(user_id, user_text, full_response)

        except Exception as e:
            error_msg = f"Error processing ticket: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield f"I'm sorry, I encountered an error processing your request."

    async def _process_new_ticket(
        self,
        user_id: str,
        user_text: str,
    ) -> AsyncGenerator[str, None]:
        """Process a new ticket."""
        # Route query to appropriate agent
        agent_name, ticket = await self.routing_service.route_query(user_id, user_text)

        # Get memory context if available
        memory_context = ""
        if self.memory_provider:
            memory_context = await self.memory_provider.retrieve(user_id)

        # Generate response
        full_response = ""
        try:
            async for chunk in self.agent_service.generate_response(
                agent_name=agent_name,
                user_id=user_id,
                query=user_text,
                memory_context=memory_context,
                ticket_id=ticket.id
            ):
                yield chunk
                full_response += chunk

            # Store conversation
            if self.memory_provider:
                await self._store_conversation(user_id, user_text, full_response)

            # Check resolution
            resolution = await self._check_ticket_resolution(full_response, user_text)
            if resolution.status == "resolved" and resolution.confidence >= 0.7:
                self.ticket_service.mark_ticket_resolved(
                    ticket.id,
                    confidence=resolution.confidence,
                    reasoning=resolution.reasoning
                )

        except Exception as e:
            print(f"Error in _process_new_ticket: {str(e)}")
            yield f"I apologize, but I encountered an error: {str(e)}"

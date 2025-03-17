"""
Query service implementation.

This service orchestrates the processing of user queries, coordinating
other services to provide comprehensive responses while maintaining
clean separation of concerns.
"""
import asyncio
import datetime
from typing import AsyncGenerator, Dict, Optional, Any

from solana_agent.domain.tasks import ComplexityAssessment
from solana_agent.interfaces.services import QueryService as QueryServiceInterface
from solana_agent.interfaces.services import (
    AgentService, RoutingService, TicketService, HandoffService,
    MemoryService, NPSService, CriticService, TaskPlanningService,
    ProjectApprovalService, ProjectSimulationService, SchedulingService,
    CommandService
)
from solana_agent.interfaces.providers import MemoryProvider
from solana_agent.domain.tickets import TicketStatus, Ticket, TicketResolution


class QueryService(QueryServiceInterface):
    """Service for processing user queries and coordinating response generation."""

    def __init__(
        self,
        agent_service: AgentService,
        routing_service: RoutingService,
        ticket_service: TicketService,
        handoff_service: HandoffService,
        memory_service: MemoryService,
        nps_service: NPSService,
        command_service: CommandService,
        memory_provider: Optional[MemoryProvider] = None,
        critic_service: Optional[CriticService] = None,
        task_planning_service: Optional[TaskPlanningService] = None,
        project_approval_service: Optional[ProjectApprovalService] = None,
        project_simulation_service: Optional[ProjectSimulationService] = None,
        scheduling_service: Optional[SchedulingService] = None,
        enable_critic: bool = True,
        require_human_approval: bool = False,
        stalled_ticket_timeout: Optional[int] = 60,
    ):
        """Initialize the query service.

        Args:
            agent_service: Service for AI and human agent management
            routing_service: Service for routing queries to appropriate agents
            ticket_service: Service for ticket operations
            handoff_service: Service for handling agent handoffs
            memory_service: Service for memory operations
            nps_service: Service for handling NPS surveys
            command_service: Service for processing system commands
            memory_provider: Optional provider for memory storage and retrieval
            critic_service: Optional service for critiquing responses
            task_planning_service: Optional service for task planning
            project_approval_service: Optional service for project approval workflows
            project_simulation_service: Optional service for simulating project outcomes
            scheduling_service: Optional service for scheduling tasks
            enable_critic: Whether to enable the critic service
            require_human_approval: Whether complex tasks require human approval
            stalled_ticket_timeout: Minutes before a ticket is considered stalled
        """
        self.agent_service = agent_service
        self.routing_service = routing_service
        self.ticket_service = ticket_service
        self.handoff_service = handoff_service
        self.memory_service = memory_service
        self.nps_service = nps_service
        self.command_service = command_service
        self.memory_provider = memory_provider
        self.critic_service = critic_service
        self.task_planning_service = task_planning_service
        self.project_approval_service = project_approval_service
        self.project_simulation_service = project_simulation_service
        self.scheduling_service = scheduling_service
        self.enable_critic = enable_critic
        self.require_human_approval = require_human_approval
        self.stalled_ticket_timeout = stalled_ticket_timeout

        self._shutdown_event = asyncio.Event()
        self._stalled_ticket_task = None

        # Start background task for stalled ticket detection
        if self.stalled_ticket_timeout and self.ticket_service and not self._stalled_ticket_task:
            try:
                self._stalled_ticket_task = asyncio.create_task(
                    self._run_stalled_ticket_checks())
            except RuntimeError:
                # No running event loop - likely in test environment
                pass

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
        # Start stalled ticket task if not already running
        if self.stalled_ticket_timeout is not None and self._stalled_ticket_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._stalled_ticket_task = loop.create_task(
                    self._run_stalled_ticket_checks())
            except RuntimeError:
                pass  # No running event loop available

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
                async for chunk in self._process_existing_ticket(user_id, user_text, active_ticket, timezone):
                    yield chunk
            else:
                # Create new ticket
                complexity = await self._assess_task_complexity(user_text)

                async for chunk in self._process_new_ticket(user_id, user_text, complexity, timezone):
                    yield chunk

        except Exception as e:
            yield f"I apologize for the technical difficulty. {str(e)}"
            import traceback
            print(f"Error in query processing: {str(e)}")
            print(traceback.format_exc())

    async def shutdown(self):
        """Clean shutdown of the query processor."""
        self._shutdown_event.set()

        # Cancel stalled ticket task
        if hasattr(self, '_stalled_ticket_task') and self._stalled_ticket_task is not None:
            self._stalled_ticket_task.cancel()
            try:
                await self._stalled_ticket_task
            except (asyncio.CancelledError, TypeError):
                pass

    async def _run_stalled_ticket_checks(self):
        """Run periodic checks for stalled tickets."""
        try:
            while not self._shutdown_event.is_set():
                await self._check_for_stalled_tickets()
                # Check every 5 minutes or half the timeout period, whichever is smaller
                check_interval = min(
                    300, self.stalled_ticket_timeout * 30) if self.stalled_ticket_timeout else 300
                await asyncio.sleep(check_interval)
        except asyncio.CancelledError:
            # Task was cancelled, clean exit
            pass
        except Exception as e:
            print(f"Error in stalled ticket check: {e}")

    async def _check_for_stalled_tickets(self):
        """Check for tickets that haven't been updated in a while and reassign them."""
        if not self.stalled_ticket_timeout:
            return

        try:
            # Find tickets that haven't been updated in the configured time
            cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
                minutes=self.stalled_ticket_timeout
            )

            stalled_tickets = self.ticket_service.find_stalled_tickets(
                cutoff_time, [TicketStatus.ACTIVE, TicketStatus.TRANSFERRED]
            )

            for ticket in stalled_tickets:
                # Skip tickets without an assigned agent
                if not ticket.assigned_to:
                    continue

                # Re-route the query to see if a different agent is better
                new_agent, _ = await self.routing_service.route_query(user_id=ticket.user_id, query=ticket.description)

                # Only reassign if a different agent is suggested
                if new_agent != ticket.assigned_to:
                    await self.handoff_service.process_handoff(
                        ticket.id,
                        ticket.assigned_to,
                        new_agent,
                        f"Automatically reassigned after {self.stalled_ticket_timeout} minutes of inactivity"
                    )

        except Exception as e:
            print(f"Error in stalled ticket check: {e}")

    async def _process_existing_ticket(
        self, user_id: str, user_text: str, ticket: Ticket, timezone: str = None
    ) -> AsyncGenerator[str, None]:
        """Process a message for an existing ticket.

        Args:
            user_id: User ID
            user_text: User query text
            ticket: Existing ticket
            timezone: Optional user timezone

        Yields:
            Response text chunks
        """
        # Get assigned agent or re-route if needed
        agent_name = ticket.assigned_to
        if not agent_name:
            agent_name, _ = await self.routing_service.route_query(user_id, user_text)
            self.ticket_service.update_ticket_status(
                ticket.id, TicketStatus.IN_PROGRESS, assigned_to=agent_name
            )

        # Get memory context if available
        memory_context = ""
        if self.memory_provider:
            memory_context = await self.memory_provider.retrieve(user_id)

        # Try to generate response
        full_response = ""
        handoff_info = None
        handoff_detected = False

        try:
            # Generate response with streaming
            async for chunk in self.agent_service.generate_response(
                agent_name=agent_name,
                user_id=user_id,
                query=user_text,
                memory_context=memory_context,
            ):
                # Detect possible handoff signals
                if chunk.strip().startswith("HANDOFF:") or (
                    not full_response and chunk.strip().startswith("{")
                ):
                    handoff_detected = True
                    full_response += chunk
                    continue

                full_response += chunk
                yield chunk

            # Handle handoff if needed
            if handoff_detected or (
                not full_response.strip() and
                self.agent_service.has_pending_handoff(agent_name)
            ):
                handoff_data = self.agent_service.get_pending_handoff(
                    agent_name)
                if handoff_data:
                    target_agent = handoff_data.get("target_agent")
                    reason = handoff_data.get("reason", "No reason provided")

                    if target_agent:
                        # Process the handoff
                        await self.handoff_service.process_handoff(
                            ticket.id,
                            agent_name,
                            target_agent,
                            reason,
                        )

                        # Generate response from new agent
                        new_response_buffer = ""
                        async for chunk in self.agent_service.generate_response(
                            agent_name=target_agent,
                            user_id=user_id,
                            query=user_text,
                            memory_context=memory_context,
                        ):
                            new_response_buffer += chunk
                            yield chunk

                        # Update full response for storage
                        full_response = new_response_buffer

                # Clear pending handoff
                self.agent_service.clear_pending_handoff(agent_name)

            # Store conversation in memory
            if self.memory_provider:
                await self._store_conversation(user_id, user_text, full_response)

        except Exception as e:
            print(f"Error processing ticket: {str(e)}")
            yield f"I'm sorry, I encountered an error processing your request: {str(e)}"

    async def _process_new_ticket(
        self,
        user_id: str,
        user_text: str,
        complexity: Dict[str, Any],
        timezone: str = None,
    ) -> AsyncGenerator[str, None]:
        """Process a message creating a new ticket.

        Args:
            user_id: User ID
            user_text: User query text
            complexity: Complexity assessment of the task
            timezone: Optional user timezone

        Yields:
            Response text chunks
        """
        # Check if task planning is needed
        if self.task_planning_service:
            needs_breakdown, reasoning = await self.task_planning_service.needs_breakdown(user_text)

            if needs_breakdown:
                # Create ticket with planning status
                ticket = await self.ticket_service.get_or_create_ticket(
                    user_id, user_text, complexity
                )

                # Mark as planning
                self.ticket_service.update_ticket_status(
                    ticket.id, TicketStatus.PLANNING
                )

                # Generate subtasks
                subtasks = await self.task_planning_service.generate_subtasks(
                    ticket.id, user_text
                )

                # Generate response about the plan
                yield "I've analyzed your request and determined it's a complex task that should be broken down.\n\n"
                yield f"Task complexity assessment: {reasoning}\n\n"
                yield f"I've created a plan with {len(subtasks)} subtasks:\n\n"

                for i, subtask in enumerate(subtasks, 1):
                    yield f"{i}. {subtask.title}: {subtask.description}\n"

                yield f"\nEstimated total time: {sum(s.estimated_minutes for s in subtasks)} minutes\n"
                yield f"\nYou can check the plan status with !status {ticket.id}"
                return

        # Check if human approval is required
        is_simple_query = (
            complexity.get("t_shirt_size") in ["XS", "S"]
            and complexity.get("story_points", 3) <= 3
        )

        if self.require_human_approval and not is_simple_query:
            # Create ticket first
            ticket = await self.ticket_service.get_or_create_ticket(
                user_id, user_text, complexity
            )

            # Simulate project if service is available
            if self.project_simulation_service:
                simulation = await self.project_simulation_service.simulate_project(user_text)
                yield "Analyzing project feasibility...\n\n"
                yield "## Project Simulation Results\n\n"
                yield f"**Complexity**: {simulation['complexity']['t_shirt_size']}\n"
                yield f"**Timeline**: {simulation['timeline']['realistic']} days\n"
                yield f"**Risk Level**: {simulation['risks']['overall_risk']}\n"
                yield f"**Recommendation**: {simulation['recommendation']}\n\n"

            # Submit for approval
            if self.project_approval_service:
                await self.project_approval_service.submit_for_approval(ticket)
                yield "\nThis project has been submitted for approval. You'll be notified once it's reviewed."
                return

        # Route query to appropriate agent
        agent_name, ticket = await self.routing_service.route_query(user_id, user_text)

        # Get memory context if available
        memory_context = ""
        if self.memory_provider:
            memory_context = await self.memory_provider.retrieve(user_id)

        # Generate initial response with streaming
        full_response = ""
        handoff_detected = False

        try:
            # Generate response with streaming
            async for chunk in self.agent_service.generate_response(
                agent_name, user_id, user_text, memory_context, temperature=0.7
            ):
                # Check if this looks like a JSON handoff
                if chunk.strip().startswith("{") and not handoff_detected:
                    handoff_detected = True
                    full_response += chunk
                    continue

                # Only yield if not a JSON chunk
                if not handoff_detected:
                    yield chunk
                    full_response += chunk

            # Handle handoff if detected
            if handoff_detected or self.agent_service.has_pending_handoff(agent_name):
                handoff_data = self.agent_service.get_pending_handoff(
                    agent_name)
                if handoff_data:
                    target_agent = handoff_data.get("target_agent")
                    reason = handoff_data.get("reason", "No reason provided")

                    if target_agent:
                        # Process handoff and update ticket
                        await self.handoff_service.process_handoff(
                            ticket.id,
                            agent_name,
                            target_agent,
                            reason,
                        )

                        # Generate response from new agent
                        new_response = ""
                        async for chunk in self.agent_service.generate_response(
                            target_agent,
                            user_id,
                            user_text,
                            memory_context,
                            temperature=0.7
                        ):
                            yield chunk
                            new_response += chunk

                        # Update full response for storage
                        full_response = new_response

                # Clear pending handoff
                self.agent_service.clear_pending_handoff(agent_name)

            # Check if ticket can be considered resolved
            resolution = await self._check_ticket_resolution(full_response, user_text)

            if resolution.status == "resolved" and resolution.confidence >= 0.7:
                self.ticket_service.mark_ticket_resolved(
                    ticket.id,
                    {
                        "confidence": resolution.confidence,
                        "reasoning": resolution.reasoning,
                    },
                )

                # Create NPS survey
                self.nps_service.create_survey(user_id, ticket.id, agent_name)

            # Store in memory provider
            if self.memory_provider:
                await self._store_conversation(user_id, user_text, full_response)

            # Extract and store insights in background
            if full_response:
                asyncio.create_task(
                    self._extract_and_store_insights(
                        user_id, {"message": user_text,
                                  "response": full_response}
                    )
                )

        except Exception as e:
            print(f"Error in _process_new_ticket: {str(e)}")
            yield f"I'm sorry, I encountered an error processing your request: {str(e)}"

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
        # Get the model to use for analysis
        model = self.agent_service.get_default_model()

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

    async def _assess_task_complexity(self, query: str) -> Dict[str, Any]:
        """Assess the complexity of a task using standardized metrics.

        Args:
            query: Task description

        Returns:
            Complexity assessment
        """
        # Short-circuit for very simple messages
        simple_messages = ["test", "hello", "hi", "hey", "ping", "thanks"]
        if len(query.strip()) <= 10 and query.lower().strip() in simple_messages:
            return {
                "t_shirt_size": "XS",
                "story_points": 1,
                "estimated_minutes": 5,
                "technical_complexity": 1,
                "domain_knowledge": 1,
            }

        # Use the task planning service if available
        if self.task_planning_service:
            try:
                return await self.task_planning_service._assess_task_complexity(query)
            except Exception as e:
                print(f"Error using task planning service for complexity: {e}")

        # Fallback to LLM-based assessment
        try:
            # Use structured output parsing with the agent service
            complexity_assessment = await self.agent_service.llm_provider.parse_structured_output(
                prompt=f"Analyze this task and provide standardized complexity metrics:\n\nTASK: {query}",
                system_prompt="You're an expert at estimating task complexity. Provide t-shirt size, story points, and time estimates.",
                model_class=ComplexityAssessment,
                temperature=0.2,
            )
            return complexity_assessment.model_dump()
        except Exception as e:
            print(f"Error assessing complexity: {e}")

        # Default values if all else fails
        return {
            "t_shirt_size": "S",
            "story_points": 2,
            "estimated_minutes": 15,
            "technical_complexity": 3,
            "domain_knowledge": 2,
        }

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

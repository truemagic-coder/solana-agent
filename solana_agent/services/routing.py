"""
Routing service implementation.

This service manages query routing to appropriate agents based on
specializations, availability, and query analysis.
"""
from typing import Dict, List, Optional, Any, Tuple
import datetime

from solana_agent.interfaces import RoutingService as RoutingServiceInterface
from solana_agent.interfaces import TicketService, SchedulingService
from solana_agent.interfaces import AgentService
from solana_agent.interfaces import LLMProvider
from solana_agent.domains import Ticket
from solana_agent.domains import QueryAnalysis
from solana_agent.domains import ScheduledTask, ScheduledTaskStatus


class RoutingService(RoutingServiceInterface):
    """Service for routing queries to appropriate agents."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        agent_service: AgentService,
        ticket_service: TicketService,
        scheduling_service: SchedulingService = None,  # Add scheduling service
        default_agent: str = "general"
    ):
        """Initialize the routing service.

        Args:
            llm_provider: Provider for language model interactions
            agent_service: Service for agent management
            ticket_service: Service for ticket management
            scheduling_service: Optional service for intelligent scheduling
            default_agent: Default agent name for fallback
        """
        self.llm_provider = llm_provider
        self.agent_service = agent_service
        self.ticket_service = ticket_service
        self.scheduling_service = scheduling_service  # Store scheduling service
        self.default_agent = default_agent

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query to determine routing information.

        Args:
            query: User query to analyze

        Returns:
            Analysis results including specializations and complexity
        """
        prompt = f"""
        Analyze this user query and determine:
        1. The primary specialization needed to address it
        2. Any secondary specializations that might be helpful
        3. The complexity level (1-5, where 5 is most complex)
        4. If it might require human assistance
        5. Any key topics or technologies mentioned

        User Query: {query}

        Be objective and thorough in your analysis.
        """

        try:
            analysis = await self.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="Analyze user queries to determine appropriate routing.",
                model_class=QueryAnalysis,
                temperature=0.2
            )

            return {
                "primary_specialization": analysis.primary_specialization,
                "secondary_specializations": analysis.secondary_specializations,
                "complexity_level": analysis.complexity_level,
                "requires_human": analysis.requires_human,
                "topics": analysis.topics,
                "confidence": analysis.confidence
            }
        except Exception as e:
            print(f"Error analyzing query: {e}")
            # Return default analysis on error
            return {
                "primary_specialization": "general",
                "secondary_specializations": [],
                "complexity_level": 1,
                "requires_human": False,
                "topics": [],
                "confidence": 0.0
            }

    async def route_query(
        self, user_id: str, query: str
    ) -> Tuple[str, Ticket]:
        """Route a query to the appropriate agent.

        Args:
            user_id: User ID
            query: User query

        Returns:
            Tuple of (agent_name, ticket)
        """
        # Analyze query
        analysis = await self.analyze_query(query)

        # Create or get ticket
        ticket = await self.ticket_service.get_or_create_ticket(
            user_id=user_id,
            query=query,
            complexity={
                "level": analysis["complexity_level"],
                "requires_human": analysis["requires_human"],
                "confidence": analysis["confidence"]
            }
        )

        # Find appropriate agent based on analysis
        selected_agent = None
        is_scheduled = False
        scheduled_task = None

        # For complex queries requiring human assistance
        if analysis["complexity_level"] >= 4 and analysis["requires_human"]:
            # Try to find a human agent first
            selected_agent = await self._find_best_human_agent(
                analysis["primary_specialization"],
                analysis["secondary_specializations"],
                ticket
            )

        # If no human agent, or less complex query, find an AI agent
        if not selected_agent:
            selected_agent, is_scheduled, scheduled_task = await self._find_best_ai_agent(
                analysis["primary_specialization"],
                analysis["secondary_specializations"],
                ticket,
                analysis["complexity_level"]
            )

        # Fall back to default if needed
        if not selected_agent:
            selected_agent = self.default_agent

        # Assign ticket to selected agent
        if ticket.status == "new":
            self.ticket_service.assign_ticket(ticket.id, selected_agent)

            note_text = f"Routed based on specialization: {analysis['primary_specialization']}. " + \
                f"Complexity: {analysis['complexity_level']}/5."

            if is_scheduled and scheduled_task:
                scheduled_time = scheduled_task.scheduled_start.strftime(
                    "%Y-%m-%d %H:%M")
                note_text += f" Scheduled for {scheduled_time}."

            self.ticket_service.add_note_to_ticket(
                ticket.id,
                note_text,
                "system"
            )

        return selected_agent, ticket

    async def reroute_ticket(self, ticket_id: str, target_agent: str, reason: str) -> bool:
        """Reroute a ticket to a different agent.

        Args:
            ticket_id: Ticket ID
            target_agent: Target agent name
            reason: Reason for rerouting

        Returns:
            True if rerouting was successful
        """
        # Get current ticket
        ticket = self.ticket_service.get_ticket_by_id(ticket_id)
        if not ticket:
            return False

        # If using scheduling service, check for and update any scheduled tasks
        if self.scheduling_service:
            # Look for tasks associated with this ticket
            if hasattr(self.scheduling_service.repository, "get_tasks_by_metadata"):
                tasks = self.scheduling_service.repository.get_tasks_by_metadata(
                    {"ticket_id": ticket_id}
                )

                # Update task assignment if found
                for task in tasks:
                    if task.status in [ScheduledTaskStatus.PENDING, ScheduledTaskStatus.SCHEDULED]:
                        task.assigned_to = target_agent
                        self.scheduling_service.repository.update_scheduled_task(
                            task)

        # Update ticket assignment
        success = self.ticket_service.assign_ticket(ticket_id, target_agent)

        if success:
            self.ticket_service.add_note_to_ticket(
                ticket_id,
                f"Rerouted to {target_agent}. Reason: {reason}",
                "system"
            )

        return success

    async def _find_best_human_agent(
        self,
        primary_specialization: str,
        secondary_specializations: List[str],
        ticket: Ticket
    ) -> Optional[str]:
        """Find the best human agent for a query based on specialization and availability.

        Args:
            primary_specialization: Primary specialization needed
            secondary_specializations: Secondary specializations
            ticket: Ticket to assign

        Returns:
            Best human agent name or None if not found
        """
        human_agents = getattr(self.agent_service, "_human_agents", {})
        if not human_agents:
            return None

        # Create a list to score agents
        agent_scores = []

        for agent_id, agent in human_agents.items():
            # Skip if agent isn't available
            if not agent.availability:
                continue

            # Base score
            score = 0

            # Check primary specialization
            primary_match = any(s.lower() == primary_specialization.lower()
                                for s in agent.specializations)
            if primary_match:
                score += 10

            # Check secondary specializations
            for sec_spec in secondary_specializations:
                if any(s.lower() == sec_spec.lower() for s in agent.specializations):
                    score += 3

            # If using scheduling service, adjust score based on workload
            if self.scheduling_service:
                try:
                    # Get agent's current tasks
                    tasks = await self.scheduling_service.get_agent_tasks(
                        agent_id,
                        start_time=datetime.datetime.now(
                            datetime.timezone.utc),
                        include_completed=False
                    )

                    # Penalize score based on number of active tasks
                    score -= len(tasks) * 2
                except Exception as e:
                    print(f"Error checking agent workload: {e}")

            agent_scores.append((agent_id, score, agent))

        # Sort by score
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        # Return the highest scoring agent, if any
        if agent_scores and agent_scores[0][1] > 0:
            return agent_scores[0][0]

        return None

    async def _find_best_ai_agent(
        self,
        primary_specialization: str,
        secondary_specializations: List[str],
        ticket: Ticket,
        complexity_level: int
    ) -> Tuple[Optional[str], bool, Optional[ScheduledTask]]:
        """Find the best AI agent for a query, potentially scheduling for later.

        Args:
            primary_specialization: Primary specialization needed
            secondary_specializations: Secondary specializations
            ticket: Ticket to assign
            complexity_level: Query complexity level

        Returns:
            Tuple of (agent_name, is_scheduled, scheduled_task)
        """
        # Start with direct specialization matching
        ai_agents = getattr(self.agent_service, "_ai_agents", {})

        # If no scheduling service, fall back to simple matching
        if not self.scheduling_service:
            for name, agent in ai_agents.items():
                if agent.specialization.lower() == primary_specialization.lower():
                    return name, False, None
            return self.default_agent, False, None

        # With scheduling service, do more sophisticated matching

        # 1. Create a task for the ticket
        estimated_minutes = complexity_level * 15  # Simple estimate based on complexity

        task = ScheduledTask(
            task_id=f"ticket_{ticket.id}",
            title=f"Handle ticket: {ticket.title}",
            description=ticket.description,
            status=ScheduledTaskStatus.PENDING,
            priority=self._get_priority_from_complexity(complexity_level),
            estimated_minutes=estimated_minutes,
            metadata={
                "ticket_id": ticket.id,
                "primary_specialization": primary_specialization,
                "secondary_specializations": secondary_specializations,
                "complexity_level": complexity_level
            }
        )

        # 2. Try to find an optimal agent and time slot
        try:
            # Schedule the task using the scheduling service
            scheduled_task = await self.scheduling_service.schedule_task(task)

            # Check if it was scheduled successfully
            if scheduled_task and scheduled_task.assigned_to:
                # Task was scheduled for immediate or future handling
                is_scheduled = scheduled_task.scheduled_start is not None
                return scheduled_task.assigned_to, is_scheduled, scheduled_task

        except Exception as e:
            print(f"Error scheduling task: {e}")

        # 3. Fall back to simple matching if scheduling fails
        for name, agent in ai_agents.items():
            if agent.specialization.lower() == primary_specialization.lower():
                return name, False, None

        # Use default agent as last resort
        return self.default_agent, False, None

    def _get_priority_from_complexity(self, complexity_level: int) -> int:
        """Convert complexity level to priority.

        Args:
            complexity_level: Complexity level (1-5)

        Returns:
            Priority value
        """
        # Map complexity to priority (higher complexity = higher priority)
        # Priority scale can be adjusted as needed
        priority_map = {
            1: 1,  # Low complexity = low priority
            2: 2,
            3: 5,  # Medium complexity = medium priority
            4: 8,
            5: 10  # High complexity = high priority
        }
        # Default to medium priority
        return priority_map.get(complexity_level, 5)

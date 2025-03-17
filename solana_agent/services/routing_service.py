"""
Routing service implementation.

This service manages query routing to appropriate agents based on
specializations, availability, and query analysis.
"""
from typing import Dict, List, Optional, Any, Tuple

from solana_agent.interfaces.services import RoutingService as RoutingServiceInterface
from solana_agent.interfaces.services import TicketService
from solana_agent.interfaces.services import AgentService
from solana_agent.interfaces.providers import LLMProvider
from solana_agent.domain.agents import AIAgent, HumanAgent, AgentType
from solana_agent.domain.tickets import Ticket
from solana_agent.domain.models import QueryAnalysis


class RoutingService(RoutingServiceInterface):
    """Service for routing queries to appropriate agents."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        agent_service: AgentService,
        ticket_service: TicketService,
        default_agent: str = "general"
    ):
        """Initialize the routing service.

        Args:
            llm_provider: Provider for language model interactions
            agent_service: Service for agent management
            ticket_service: Service for ticket management
            default_agent: Default agent name for fallback
        """
        self.llm_provider = llm_provider
        self.agent_service = agent_service
        self.ticket_service = ticket_service
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

        # Find appropriate agent
        specializations = self.agent_service.get_specializations()
        primary_spec = analysis["primary_specialization"]

        # Try to match primary specialization
        if primary_spec in specializations:
            selected_agent = self._get_agent_for_specialization(primary_spec)

            # Check if this is a high complexity query that might need human help
            if analysis["complexity_level"] >= 4 and analysis["requires_human"]:
                # Try to find human agent with matching specialization
                human_agent = self._get_human_agent_for_specialization(
                    primary_spec)
                if human_agent:
                    selected_agent = human_agent.name
        else:
            # No match for primary specialization, use default
            selected_agent = self.default_agent

        # Assign ticket to selected agent
        if ticket.status == "new":
            self.ticket_service.assign_ticket(ticket.id, selected_agent)
            self.ticket_service.add_note_to_ticket(
                ticket.id,
                f"Routed based on specialization: {primary_spec}. " +
                f"Complexity: {analysis['complexity_level']}/5.",
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

        # Update ticket assignment
        success = self.ticket_service.assign_ticket(ticket_id, target_agent)

        if success:
            self.ticket_service.add_note_to_ticket(
                ticket_id,
                f"Rerouted to {target_agent}. Reason: {reason}",
                "system"
            )

        return success

    def _get_agent_for_specialization(self, specialization: str) -> str:
        """Get an agent name for a specialization.

        Args:
            specialization: Agent specialization

        Returns:
            Agent name
        """
        # In a real implementation, this would use a more sophisticated
        # lookup based on agent availability, load balancing, etc.
        # For now, we just use the specialization name as the agent name
        # or fall back to default

        # Check if there's an AI agent with this specialization
        ai_agents = getattr(self.agent_service, "_ai_agents", {})
        for name, agent in ai_agents.items():
            if agent.specialization.lower() == specialization.lower():
                return name

        return self.default_agent

    def _get_human_agent_for_specialization(self, specialization: str) -> Optional[HumanAgent]:
        """Get a human agent for a specialization.

        Args:
            specialization: Agent specialization

        Returns:
            Human agent or None if not found
        """
        # In a real implementation, this would check agent availability
        # and workload before assigning

        human_agents = getattr(self.agent_service, "_human_agents", {})
        for agent_id, agent in human_agents.items():
            if specialization.lower() in [s.lower() for s in agent.specializations] and agent.availability:
                return agent

        return None

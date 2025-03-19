"""
Handoff service implementation.

This service manages agent handoffs and escalations between AI and human agents.
"""
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

from solana_agent.interfaces import HandoffService as HandoffServiceInterface
from solana_agent.interfaces import HandoffObserver
from solana_agent.interfaces import TicketRepository, HandoffRepository
from solana_agent.services import AgentService
from solana_agent.domains import TicketStatus, TicketNote
from solana_agent.domains import HandoffEvaluation, Handoff


class HandoffService(HandoffServiceInterface, HandoffObserver):
    """Service for managing handoffs between agents."""

    def __init__(
        self,
        handoff_repository: HandoffRepository,
        ticket_repository: TicketRepository,
        agent_service: AgentService,
    ):
        """Initialize the handoff service.

        Args:
            handoff_repository: Repository for handoff records
            ticket_repository: Repository for ticket operations
            agent_service: Service for agent operations
        """
        self.handoff_repository = handoff_repository
        self.ticket_repository = ticket_repository
        self.agent_service = agent_service
        # Get the LLM provider from the agent service
        self.llm_provider = agent_service.llm_provider

    def on_handoff(self, handoff_data: Dict[str, Any]) -> None:
        """Handle handoff notifications from AgentService."""
        ticket_id = handoff_data.get("ticket_id")
        target_agent = handoff_data.get("target_agent")
        reason = handoff_data.get("reason", "No reason provided")

        if ticket_id and target_agent:
            self.handle_handoff(ticket_id, target_agent, reason)

    async def evaluate_handoff_needed(self, query: str, response: str, current_agent: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Evaluate if a handoff is needed based on query and response."""
        # Skip evaluation for very short queries - this logic should be modified
        # Currently this is skipping normal queries too
        if len(query.strip()) < 30:  # You might be using a higher threshold
            return False, None, None

        try:
            prompt = f"""
            Evaluate if this interaction requires a handoff to another agent.
            
            User query: {query}
            Current agent: {current_agent}
            Agent response: {response}
            """

            evaluation = await self.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="You're an expert at determining when a conversation should be handed off.",
                model_class=HandoffEvaluation,
                temperature=0.3
            )

            return (
                evaluation.handoff_needed,
                evaluation.target_agent,
                evaluation.reason
            )
        except Exception as e:
            print(f"Error evaluating handoff: {str(e)}")
            return False, None, None

    async def request_human_help(
        self, ticket_id: str, reason: str, specialization: Optional[str] = None
    ) -> bool:
        """Request help from a human agent.

        Args:
            ticket_id: Ticket ID
            reason: Reason for the human help request
            specialization: Optional specialization needed

        Returns:
            True if request was successful
        """
        ticket = self.ticket_repository.get_by_id(ticket_id)
        if not ticket:
            print(f"Ticket {ticket_id} not found")
            return False

        # Find an available human agent
        available_agents = []

        # Get the agent service to find matching human agents
        if specialization:
            human_agents = self.agent_service.get_all_human_agents()
            available_agents = [
                agent for agent in human_agents.values()
                if agent.availability and any(spec.lower() == specialization.lower()
                                              for spec in agent.specializations)
            ]

        if not available_agents:
            # Get any available human agent
            human_agents = self.agent_service.get_all_human_agents()
            available_agents = [
                agent for agent in human_agents.values() if agent.availability]

        if not available_agents:
            # No humans available, update ticket status
            note = TicketNote(
                content=f"Human help requested but no agents available. Reason: {reason}",
                type="system",
                timestamp=datetime.now()
            )
            self.ticket_repository.add_note(ticket_id, note)
            return False

        # Assign to the first available human
        human_agent = available_agents[0]
        self.ticket_repository.update(ticket_id, {
            "assigned_to": human_agent.id,
            "status": TicketStatus.WAITING_FOR_HUMAN,
            "updated_at": datetime.now()
        })

        # Add note about the handoff
        note = TicketNote(
            content=f"Handed off to human agent {human_agent.name}. Reason: {reason}",
            type="system",
            timestamp=datetime.now()
        )
        self.ticket_repository.add_note(ticket_id, note)

        # Record the handoff
        handoff = Handoff(
            from_agent=ticket.assigned_to or "system",
            to_agent=human_agent.id,
            ticket_id=ticket_id,
            reason=reason,
            timestamp=datetime.now(),
            successful=True,
            notes=f"Human help requested with specialization: {specialization}"
        )
        self.handoff_repository.record(handoff)

        return True

    async def handle_handoff(self, ticket_id: str, target_agent: str, reason: str) -> bool:
        """Handle a handoff to another agent."""
        ticket = self.ticket_repository.get_by_id(ticket_id)
        if not ticket:
            return False

        # Check if target is a human agent or human help request
        if target_agent.lower() in ["human", "human help", "human agent"]:
            return await self.request_human_help(ticket_id=ticket_id, reason=reason)

        # Check for human agent by ID or name
        human_agents = self.agent_service.get_all_human_agents()
        for agent_id, agent in human_agents.items():
            if agent_id == target_agent or agent.name == target_agent:
                return await self.request_human_help(ticket_id=ticket_id, reason=reason)

        # Check for AI agent
        ai_agents = self.agent_service.get_all_ai_agents()
        if target_agent not in ai_agents:
            note = TicketNote(
                content=f"Failed to handoff: Agent {target_agent} not found",
                type="system",
                timestamp=datetime.now()
            )
            self.ticket_repository.add_note(ticket_id, note)
            return False

        # Handoff to AI agent
        self.ticket_repository.update(ticket_id, {
            "assigned_to": target_agent,
            "status": TicketStatus.ASSIGNED,
            "updated_at": datetime.now()
        })

        note = TicketNote(
            content=f"Handed off to AI agent {target_agent}. Reason: {reason}",
            type="system",
            timestamp=datetime.now()
        )
        self.ticket_repository.add_note(ticket_id, note)

        # Record the handoff
        handoff = Handoff(
            from_agent=ticket.assigned_to or "system",
            to_agent=target_agent,
            ticket_id=ticket_id,
            reason=reason,
            timestamp=datetime.now(),
            successful=True
        )
        self.handoff_repository.record(handoff)

        return True

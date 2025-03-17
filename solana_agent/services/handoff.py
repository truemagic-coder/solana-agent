"""
Handoff service implementation.

This service manages agent handoffs and escalations between AI and human agents.
"""
from typing import Tuple, Optional, Any
from datetime import datetime

from solana_agent.interfaces import HandoffService as HandoffServiceInterface
from solana_agent.interfaces import TicketRepository, HandoffRepository
from solana_agent.services import AgentService
from solana_agent.domains import TicketStatus, TicketNote
from solana_agent.domains import HandoffEvaluation, Handoff


class HandoffService(HandoffServiceInterface):
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

    async def evaluate_handoff_needed(
        self, query: str, response: str, current_agent: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Evaluate if a handoff is needed based on query and response.

        Args:
            query: User query
            response: Agent response
            current_agent: Current agent name

        Returns:
            Tuple of (handoff_needed, target_agent, reason)
        """
        # Skip handoff evaluation for simple queries
        if len(query.split()) < 5 or "thank" in query.lower():
            return False, None, None

        prompt = f"""
        Evaluate if this conversation needs to be handed off to a different agent:
        
        User query: {query}
        
        Current agent ({current_agent}) response: {response}
        
        Determine if:
        1. The query is outside the current agent's expertise
        2. The response indicates inability to help or uncertainty
        3. The query requires a different specialization
        4. The query explicitly requests another agent or human help
        """

        try:
            evaluation = await self.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="Evaluate if agent handoff is needed objectively.",
                model_class=HandoffEvaluation,
                temperature=0.2
            )

            if evaluation.handoff_needed:
                return True, evaluation.target_agent, evaluation.reason
            return False, None, None
        except Exception as e:
            print(f"Error in handoff evaluation: {e}")
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

    async def handle_handoff(
        self, ticket_id: str, target_agent: str, reason: str
    ) -> bool:
        """Handle the handoff to another agent.

        Args:
            ticket_id: Ticket ID
            target_agent: Target agent name
            reason: Reason for the handoff

        Returns:
            True if handoff was successful
        """
        ticket = self.ticket_repository.get_by_id(ticket_id)
        if not ticket:
            return False

        # Check if target is a human or AI agent
        target_is_human = False

        # Use agent service to find the target agent
        ai_agents = self.agent_service.get_all_ai_agents()
        ai_agent = ai_agents.get(target_agent)

        if not ai_agent:
            # Check if it's a human agent or a special request
            if target_agent.lower() in ["human", "human agent", "support"]:
                # Generic request for human help
                return await self.request_human_help(ticket_id, reason)
            else:
                # Check for human agent by name
                human_agents = self.agent_service.get_all_human_agents()

                # Find human agent by name (case-insensitive)
                target_human = None
                for agent in human_agents.values():
                    if agent.name.lower() == target_agent.lower():
                        target_human = agent
                        break

                if target_human:
                    target_is_human = True
                    target_agent = target_human.id  # Use the ID for handoff
                else:
                    # Target agent not found
                    note = TicketNote(
                        content=f"Handoff failed: Agent '{target_agent}' not found",
                        type="system",
                        timestamp=datetime.now()
                    )
                    self.ticket_repository.add_note(ticket_id, note)
                    return False

        # Handle handoff based on agent type
        if target_is_human:
            return await self.request_human_help(ticket_id, reason)
        else:
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

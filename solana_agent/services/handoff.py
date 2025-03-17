"""
Handoff service implementation.

This service manages agent handoffs and escalations between AI and human agents.
"""
from typing import Tuple, Optional, Any
from datetime import datetime

from solana_agent.interfaces.services import HandoffService as HandoffServiceInterface
from solana_agent.interfaces.providers import LLMProvider
from solana_agent.interfaces.repositories import TicketRepository, AgentRepository
from solana_agent.domain.tickets import TicketStatus, TicketNote
from solana_agent.domain.models import HandoffEvaluation


class HandoffService(HandoffServiceInterface):
    """Service for managing handoffs between agents."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        ticket_repository: TicketRepository,
        agent_repository: AgentRepository
    ):
        """Initialize the handoff service.

        Args:
            llm_provider: Provider for language model interactions
            ticket_repository: Repository for ticket operations
            agent_repository: Repository for agent operations
        """
        self.llm_provider = llm_provider
        self.ticket_repository = ticket_repository
        self.agent_repository = agent_repository

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
        if specialization:
            available_agents = self.agent_repository.get_human_agents_by_specialization(
                specialization)

        if not available_agents:
            available_agents = self.agent_repository.get_available_human_agents()

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
        ai_agent = self.agent_repository.get_ai_agent(target_agent)

        if not ai_agent:
            # Check if it's a human agent or a special request
            if target_agent.lower() in ["human", "human agent", "support"]:
                # Generic request for human help
                return await self.request_human_help(ticket_id, reason)
            else:
                # Check for human agent by name
                human_agents = self.agent_repository.get_available_human_agents()
                target_human = next(
                    (a for a in human_agents if a.name.lower() == target_agent.lower()), None)

                if target_human:
                    target_is_human = True
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

            return True

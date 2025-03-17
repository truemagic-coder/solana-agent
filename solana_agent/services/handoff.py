class HandoffService:
    """Service for managing handoffs between agents."""

    def __init__(
        self,
        handoff_repository: HandoffRepository,
        ticket_repository: TicketRepository,
        agent_registry: AgentRegistry,
    ):
        self.handoff_repository = handoff_repository
        self.ticket_repository = ticket_repository
        self.agent_registry = agent_registry

    async def process_handoff(
        self, ticket_id: str, from_agent: str, to_agent: str, reason: str
    ) -> str:
        """Process a handoff between agents."""
        # Get ticket information
        ticket = self.ticket_repository.get_by_id(ticket_id)
        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")

        # Check if target agent exists
        if to_agent not in self.agent_registry.get_all_ai_agents() and (
            not hasattr(self.agent_registry, "get_all_human_agents")
            or to_agent not in self.agent_registry.get_all_human_agents()
        ):
            raise ValueError(f"Target agent {to_agent} not found")

        # Record the handoff
        handoff = Handoff(
            ticket_id=ticket_id,
            user_id=ticket.user_id,
            from_agent=from_agent,
            to_agent=to_agent,
            reason=reason,
            query=ticket.query,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )

        self.handoff_repository.record(handoff)

        # Update the ticket
        self.ticket_repository.update(
            ticket_id,
            {
                "assigned_to": to_agent,
                "status": TicketStatus.TRANSFERRED,
                "handoff_reason": reason,
                "updated_at": datetime.datetime.now(datetime.timezone.utc),
            },
        )

        return to_agent

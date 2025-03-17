class TicketService:
    """Service for managing tickets and their lifecycle."""

    def __init__(self, ticket_repository: TicketRepository):
        self.ticket_repository = ticket_repository

    async def get_or_create_ticket(
        self, user_id: str, query: str, complexity: Optional[Dict[str, Any]] = None
    ) -> Ticket:
        """Get active ticket for user or create a new one."""
        # Check for active ticket
        ticket = self.ticket_repository.get_active_for_user(user_id)
        if ticket:
            return ticket

        # Create new ticket
        new_ticket = Ticket(
            id=str(uuid.uuid4()),
            user_id=user_id,
            query=query,
            status=TicketStatus.NEW,
            assigned_to="",  # Will be assigned later
            created_at=datetime.datetime.now(datetime.timezone.utc),
            complexity=complexity,
        )

        ticket_id = self.ticket_repository.create(new_ticket)
        new_ticket.id = ticket_id
        return new_ticket

    def update_ticket_status(
        self, ticket_id: str, status: TicketStatus, **additional_updates
    ) -> bool:
        """Update ticket status and additional fields."""
        updates = {
            "status": status,
            "updated_at": datetime.datetime.now(datetime.timezone.utc),
        }
        updates.update(additional_updates)

        return self.ticket_repository.update(ticket_id, updates)

    def mark_ticket_resolved(
        self, ticket_id: str, resolution_data: Dict[str, Any]
    ) -> bool:
        """Mark a ticket as resolved with resolution information."""
        updates = {
            "status": TicketStatus.RESOLVED,
            "resolved_at": datetime.datetime.now(datetime.timezone.utc),
            "resolution_confidence": resolution_data.get("confidence", 0.0),
            "resolution_reasoning": resolution_data.get("reasoning", ""),
            "updated_at": datetime.datetime.now(datetime.timezone.utc),
        }

        return self.ticket_repository.update(ticket_id, updates)

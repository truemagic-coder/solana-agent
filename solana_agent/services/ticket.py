"""
Ticket service implementation.

This service manages support tickets and their lifecycle.
"""
import uuid
import datetime
from typing import Dict, List, Optional, Any

from solana_agent.interfaces import TicketService as TicketServiceInterface
from solana_agent.interfaces import TicketRepository
from solana_agent.domains import Ticket, TicketStatus, TicketNote, TicketPriority


class TicketService(TicketServiceInterface):
    """Service for managing tickets and their lifecycle."""

    def __init__(self, ticket_repository: TicketRepository):
        """Initialize the ticket service.

        Args:
            ticket_repository: Repository for ticket operations
        """
        self.ticket_repository = ticket_repository

    async def get_or_create_ticket(
        self, user_id: str, query: str, complexity: Optional[Dict[str, Any]] = None
    ) -> Ticket:
        """Get active ticket for user or create a new one.

        Args:
            user_id: User ID
            query: User query
            complexity: Optional complexity metadata

        Returns:
            Active or newly created ticket
        """
        # Check for active ticket
        ticket = self.ticket_repository.get_active_for_user(user_id)
        if ticket:
            return ticket

        # Create new ticket
        new_ticket = Ticket(
            id=str(uuid.uuid4()),
            title=query[:50] + "..." if len(query) > 50 else query,
            description=query,
            user_id=user_id,
            status=TicketStatus.NEW,
            assigned_to="",  # Will be assigned later
            created_at=datetime.datetime.now(datetime.timezone.utc),
            updated_at=datetime.datetime.now(datetime.timezone.utc),
            priority=TicketPriority.MEDIUM,
            metadata={"complexity": complexity} if complexity else {}
        )

        ticket_id = self.ticket_repository.create(new_ticket)
        new_ticket.id = ticket_id
        return new_ticket

    def update_ticket_status(
        self, ticket_id: str, status: TicketStatus, **additional_updates
    ) -> bool:
        """Update ticket status and additional fields.

        Args:
            ticket_id: Ticket ID
            status: New status
            **additional_updates: Additional fields to update

        Returns:
            True if update was successful
        """
        updates = {
            "status": status,
            "updated_at": datetime.datetime.now(datetime.timezone.utc),
        }
        updates.update(additional_updates)

        return self.ticket_repository.update(ticket_id, updates)

    def mark_ticket_resolved(
        self, ticket_id: str, resolution_data: Dict[str, Any]
    ) -> bool:
        """Mark a ticket as resolved with resolution information.

        Args:
            ticket_id: Ticket ID
            resolution_data: Resolution details

        Returns:
            True if update was successful
        """
        updates = {
            "status": TicketStatus.RESOLVED,
            "resolved_at": datetime.datetime.now(datetime.timezone.utc),
            "updated_at": datetime.datetime.now(datetime.timezone.utc),
            "metadata": {
                "resolution_confidence": resolution_data.get("confidence", 0.0),
                "resolution_reasoning": resolution_data.get("reasoning", "")
            }
        }

        return self.ticket_repository.update(ticket_id, updates)

    def add_note_to_ticket(
        self, ticket_id: str, content: str, note_type: str = "system", created_by: Optional[str] = None
    ) -> bool:
        """Add a note to a ticket.

        Args:
            ticket_id: Ticket ID
            content: Note content
            note_type: Type of note (system, agent, user)
            created_by: ID of note creator

        Returns:
            True if note was added successfully
        """
        note = TicketNote(
            id=str(uuid.uuid4()),
            content=content,
            type=note_type,
            created_by=created_by,
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )

        return self.ticket_repository.add_note(ticket_id, note)

    def get_ticket_by_id(self, ticket_id: str) -> Optional[Ticket]:
        """Get a ticket by ID.

        Args:
            ticket_id: Ticket ID

        Returns:
            Ticket or None if not found
        """
        return self.ticket_repository.get_by_id(ticket_id)

    def get_tickets_by_user(self, user_id: str, limit: int = 20) -> List[Ticket]:
        """Get tickets for a specific user.

        Args:
            user_id: User ID
            limit: Maximum number of tickets to return

        Returns:
            List of tickets
        """
        return self.ticket_repository.get_by_user(user_id, limit)

    def get_tickets_by_status(self, status: TicketStatus, limit: int = 50) -> List[Ticket]:
        """Get tickets by status.

        Args:
            status: Ticket status
            limit: Maximum number of tickets to return

        Returns:
            List of tickets
        """
        return self.ticket_repository.get_by_status(status, limit)

    def assign_ticket(self, ticket_id: str, agent_id: str) -> bool:
        """Assign a ticket to an agent.

        Args:
            ticket_id: Ticket ID
            agent_id: Agent ID

        Returns:
            True if assignment was successful
        """
        updates = {
            "assigned_to": agent_id,
            "status": TicketStatus.ASSIGNED,
            "updated_at": datetime.datetime.now(datetime.timezone.utc)
        }

        success = self.ticket_repository.update(ticket_id, updates)

        if success:
            self.add_note_to_ticket(
                ticket_id,
                f"Ticket assigned to agent {agent_id}",
                "system"
            )

        return success

    def close_ticket(self, ticket_id: str, reason: str = "") -> bool:
        """Close a ticket.

        Args:
            ticket_id: Ticket ID
            reason: Closure reason

        Returns:
            True if closure was successful
        """
        updates = {
            "status": TicketStatus.CLOSED,
            "closed_at": datetime.datetime.now(datetime.timezone.utc),
            "updated_at": datetime.datetime.now(datetime.timezone.utc),
            "metadata.closure_reason": reason
        }

        success = self.ticket_repository.update(ticket_id, updates)

        if success and reason:
            self.add_note_to_ticket(
                ticket_id,
                f"Ticket closed: {reason}",
                "system"
            )

        return success

    def find_stalled_tickets(self, timeout_minutes: int = 1440) -> List[Ticket]:
        """Find tickets that have been inactive and should be marked as stalled.

        Args:
            timeout_minutes: Number of minutes of inactivity to consider a ticket stalled
                            (default: 1440 = 24 hours)

        Returns:
            List of stalled tickets
        """
        # Calculate cutoff time for staleness
        cutoff_time = datetime.datetime.now(
            datetime.timezone.utc) - datetime.timedelta(minutes=timeout_minutes)

        # Find tickets that:
        # 1. Are in an active status
        # 2. Haven't been updated since the cutoff time
        active_statuses = [
            TicketStatus.NEW.value,
            TicketStatus.ASSIGNED.value,
            TicketStatus.IN_PROGRESS.value,
            TicketStatus.WAITING_FOR_USER.value
        ]

        # Query for potentially stalled tickets
        tickets = self.ticket_repository.find_tickets_by_criteria(
            status_in=active_statuses,
            updated_before=cutoff_time
        )

        # Additional filtering if needed
        stalled_tickets = []
        for ticket in tickets:
            # You can add more complex stalled ticket logic here if needed
            stalled_tickets.append(ticket)

        return stalled_tickets

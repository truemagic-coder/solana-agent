"""
MongoDB implementation of the ticket repository.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
import json

from solana_agent.domains import Ticket, TicketNote
from solana_agent.domains import TicketStatus
from solana_agent.interfaces import TicketRepository


class MongoTicketRepository(TicketRepository):
    """MongoDB implementation of the TicketRepository interface."""

    def __init__(self, db_adapter):
        """Initialize the repository with a database adapter.

        Args:
            db_adapter: MongoDB adapter instance
        """
        self.db = db_adapter
        self.collection = "tickets"

        # Ensure collection exists
        self.db.create_collection(self.collection)

        # Create indexes for common queries
        self.db.create_index(self.collection, [("user_id", 1)])
        self.db.create_index(self.collection, [("status", 1)])
        self.db.create_index(self.collection, [("assigned_to", 1)])
        self.db.create_index(self.collection, [("created_at", -1)])

    def create(self, ticket: Ticket) -> str:
        """Create a new ticket and return its ID."""
        # Convert domain model to document
        doc = ticket.model_dump()

        # Insert document
        return self.db.insert_one(self.collection, doc)

    def get_by_id(self, ticket_id: str) -> Optional[Ticket]:
        """Get a ticket by ID."""
        doc = self.db.find_one(self.collection, {"id": ticket_id})
        if not doc:
            return None

        return Ticket.model_validate(doc)

    def get_active_for_user(self, user_id: str) -> Optional[Ticket]:
        """Get active ticket for a user."""
        active_statuses = [
            TicketStatus.NEW.value,
            TicketStatus.ACTIVE.value,
            TicketStatus.PENDING.value
        ]

        doc = self.db.find_one(
            self.collection,
            {
                "user_id": user_id,
                "status": {"$in": active_statuses}
            }
        )

        if not doc:
            return None

        return Ticket.model_validate(doc)

    def find(self, query: Dict, sort_by: Optional[str] = None, limit: int = 0) -> List[Ticket]:
        """Find tickets matching query."""
        sort_option = None
        if sort_by:
            # Determine sort direction
            direction = -1 if sort_by.startswith("-") else 1
            field = sort_by[1:] if sort_by.startswith("-") else sort_by
            sort_option = [(field, direction)]

        docs = self.db.find(self.collection, query,
                            sort=sort_option, limit=limit)

        # Convert to domain models
        return [Ticket.model_validate(doc) for doc in docs]

    def update(self, ticket_id: str, updates: Dict[str, Any]) -> bool:
        """Update a ticket."""
        # Always update the 'updated_at' field
        updates_with_timestamp = {
            **updates,
            "updated_at": datetime.now()
        }

        return self.db.update_one(
            self.collection,
            {"id": ticket_id},
            {"$set": updates_with_timestamp}
        )

    def count(self, query: Dict) -> int:
        """Count tickets matching query."""
        return self.db.count_documents(self.collection, query)

    async def find_stalled_tickets(
        self, cutoff_time: datetime, statuses: List[TicketStatus]
    ) -> List[Ticket]:
        """Find tickets that haven't been updated since the cutoff time."""
        status_values = [status.value for status in statuses]

        query = {
            "status": {"$in": status_values},
            "updated_at": {"$lt": cutoff_time}
        }

        docs = self.db.find(self.collection, query)
        return [Ticket.model_validate(doc) for doc in docs]

    def add_note(self, ticket_id: str, note: TicketNote) -> bool:
        """Add a note to a ticket."""
        note_dict = note.model_dump()

        return self.db.update_one(
            self.collection,
            {"id": ticket_id},
            {"$push": {"notes": note_dict}}
        )

    def get_subtasks(self, parent_id: str) -> List[Ticket]:
        """Get all subtasks for a parent ticket."""
        docs = self.db.find(
            self.collection,
            {
                "parent_id": parent_id,
                "is_subtask": True
            }
        )

        return [Ticket.model_validate(doc) for doc in docs]

    def get_parent(self, subtask_id: str) -> Optional[Ticket]:
        """Get the parent ticket for a subtask."""
        # First get the subtask to find its parent_id
        subtask = self.get_by_id(subtask_id)
        if not subtask or not subtask.parent_id:
            return None

        # Then get the parent
        return self.get_by_id(subtask.parent_id)

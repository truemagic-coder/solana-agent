"""
Repository interfaces for data access.

These interfaces define the contracts for data access components,
allowing for different storage implementations (MongoDB, memory, etc.)
without changing the business logic.
"""
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from solana_agent.domains import UserFeedback
from solana_agent.domains import Ticket, TicketNote
from solana_agent.domains import AIAgent
from solana_agent.domains import TicketStatus
from solana_agent.domains import MemoryInsight


class TicketRepository(ABC):
    """Interface for ticket data access."""

    @abstractmethod
    def create(self, ticket: Ticket) -> str:
        """Create a new ticket and return its ID."""
        pass

    @abstractmethod
    def get_by_id(self, ticket_id: str) -> Optional[Ticket]:
        """Get a ticket by ID."""
        pass

    @abstractmethod
    def get_active_for_user(self, user_id: str) -> Optional[Ticket]:
        """Get active ticket for a user."""
        pass

    @abstractmethod
    def find(self, query: Dict, sort_by: Optional[str] = None, limit: int = 0) -> List[Ticket]:
        """Find tickets matching query."""
        pass

    @abstractmethod
    def update(self, ticket_id: str, updates: Dict[str, Any]) -> bool:
        """Update a ticket."""
        pass

    @abstractmethod
    def count(self, query: Dict) -> int:
        """Count tickets matching query."""
        pass

    @abstractmethod
    async def find_stalled_tickets(self, cutoff_time: datetime, statuses: List[TicketStatus]) -> List[Ticket]:
        """Find tickets that haven't been updated since the cutoff time."""
        pass

    @abstractmethod
    def add_note(self, ticket_id: str, note: TicketNote) -> bool:
        """Add a note to a ticket."""
        pass

    @abstractmethod
    def get_subtasks(self, parent_id: str) -> List[Ticket]:
        """Get all subtasks for a parent ticket."""
        pass

    @abstractmethod
    def get_parent(self, subtask_id: str) -> Optional[Ticket]:
        """Get the parent ticket for a subtask."""
        pass

    @abstractmethod
    def find_tickets_by_criteria(
        self,
        status_in: Optional[List[TicketStatus]] = None,
        updated_before: Optional[datetime] = None,
    ) -> List[Ticket]:
        """Find tickets by criteria."""
        pass


class AgentRepository(ABC):
    """Interface for agent data access."""

    @abstractmethod
    def get_ai_agent_by_name(self, name: str) -> Optional[AIAgent]:
        """Get an AI agent by name."""
        pass

    @abstractmethod
    def get_ai_agent(self, name: str) -> Optional[AIAgent]:
        """Get an AI agent by name."""
        pass

    @abstractmethod
    def get_all_ai_agents(self) -> List[AIAgent]:
        """Get all AI agents."""
        pass

    @abstractmethod
    def save_ai_agent(self, agent: AIAgent) -> bool:
        """Save an AI agent."""
        pass

    @abstractmethod
    def delete_ai_agent(self, name: str) -> bool:
        """Delete an AI agent."""
        pass


class MemoryRepository(ABC):
    """Interface for memory storage and retrieval."""

    @abstractmethod
    def store_insight(self, user_id: str, insight: MemoryInsight) -> None:
        """Store a memory insight.

        Args:
            user_id: ID of the user the insight relates to
            insight: Memory insight to store
        """
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory for insights matching a query.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of matching memory items
        """
        pass

    @abstractmethod
    def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history for a user.

        Args:
            user_id: User ID
            limit: Maximum number of items to return

        Returns:
            List of conversation history items
        """
        pass

    @abstractmethod
    def delete_user_memory(self, user_id: str) -> bool:
        """Delete all memory for a user.

        Args:
            user_id: User ID

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def get_user_history_paginated(self, user_id: str, page_num: int, page_size: int) -> List[Dict]:
        """Get paginated conversation history for a user.

        Args:
            user_id: User ID
            page_num: Page number (starting from 1)
            page_size: Number of items per page

        Returns:
            List of conversation history items
        """
        pass

    @abstractmethod
    def count_user_history(self, user_id: str) -> int:
        """Count the number of items in a user's conversation history.

        Args:
            user_id: User ID

        Returns:
            Number of items
        """
        pass


class FeedbackRepository(ABC):
    """Interface for feedback storage and retrieval."""

    @abstractmethod
    def store_feedback(self, feedback: UserFeedback) -> str:
        """Store user feedback.

        Args:
            feedback: Feedback object to store

        Returns:
            Feedback ID
        """
        pass

    @abstractmethod
    def get_user_feedback(self, user_id: str) -> List[UserFeedback]:
        """Get all feedback for a user.

        Args:
            user_id: User ID

        Returns:
            List of feedback items
        """
        pass

    @abstractmethod
    def get_average_nps(self, days: int = 30) -> float:
        """Calculate average NPS score for a time period.

        Args:
            days: Number of days to include

        Returns:
            Average NPS score
        """
        pass

    @abstractmethod
    def get_nps_distribution(self, days: int = 30) -> Dict[int, int]:
        """Get distribution of NPS scores for a time period.

        Args:
            days: Number of days to include

        Returns:
            Dictionary mapping scores to counts
        """
        pass

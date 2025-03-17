"""
Repository interfaces for data access.

These interfaces define the contracts for data access components,
allowing for different storage implementations (MongoDB, memory, etc.)
without changing the business logic.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from solana_agent.domain.feedback import UserFeedback
from solana_agent.domain.tickets import Ticket, TicketNote
from solana_agent.domain.agents import AIAgent, HumanAgent, AgentPerformance
from solana_agent.domain.resources import Resource, ResourceBooking
from solana_agent.domain.scheduling import ScheduledTask, AgentSchedule
from solana_agent.domain.enums import TicketStatus
from solana_agent.domain.memory import MemoryInsight


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


class AgentRepository(ABC):
    """Interface for agent data access."""

    @abstractmethod
    def get_ai_agent(self, name: str) -> Optional[AIAgent]:
        """Get an AI agent by name."""
        pass

    @abstractmethod
    def get_all_ai_agents(self) -> Dict[str, AIAgent]:
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

    @abstractmethod
    def get_human_agent(self, agent_id: str) -> Optional[HumanAgent]:
        """Get a human agent by ID."""
        pass

    @abstractmethod
    def get_human_agents_by_specialization(self, specialization: str) -> List[HumanAgent]:
        """Get human agents with a specific specialization."""
        pass

    @abstractmethod
    def get_available_human_agents(self) -> List[HumanAgent]:
        """Get currently available human agents."""
        pass

    @abstractmethod
    def save_human_agent(self, agent: HumanAgent) -> bool:
        """Save a human agent."""
        pass

    @abstractmethod
    def save_agent_performance(self, performance: AgentPerformance) -> bool:
        """Save agent performance metrics."""
        pass

    @abstractmethod
    def get_agent_performance(self, agent_id: str, period_start: datetime, period_end: datetime) -> Optional[AgentPerformance]:
        """Get performance metrics for an agent within a time period."""
        pass


class ResourceRepository(ABC):
    """Interface for resource data access."""

    @abstractmethod
    def create_resource(self, resource: Resource) -> str:
        """Create a new resource and return its ID."""
        pass

    @abstractmethod
    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a resource by ID."""
        pass

    @abstractmethod
    def find_resources(self, query: Dict) -> List[Resource]:
        """Find resources matching query."""
        pass

    @abstractmethod
    def update_resource(self, resource_id: str, updates: Dict[str, Any]) -> bool:
        """Update a resource."""
        pass

    @abstractmethod
    def delete_resource(self, resource_id: str) -> bool:
        """Delete a resource."""
        pass

    @abstractmethod
    def create_booking(self, booking: ResourceBooking) -> str:
        """Create a new resource booking and return its ID."""
        pass

    @abstractmethod
    def get_booking(self, booking_id: str) -> Optional[ResourceBooking]:
        """Get a booking by ID."""
        pass

    @abstractmethod
    def find_bookings(self, query: Dict) -> List[ResourceBooking]:
        """Find bookings matching query."""
        pass

    @abstractmethod
    def get_resource_bookings(self, resource_id: str, start_time: datetime, end_time: datetime) -> List[ResourceBooking]:
        """Get bookings for a resource within a time period."""
        pass

    @abstractmethod
    def update_booking(self, booking_id: str, updates: Dict[str, Any]) -> bool:
        """Update a booking."""
        pass

    @abstractmethod
    def delete_booking(self, booking_id: str) -> bool:
        """Delete a booking."""
        pass


class SchedulingRepository(ABC):
    """Interface for scheduling data access."""

    @abstractmethod
    def get_scheduled_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a scheduled task by ID."""
        pass

    @abstractmethod
    def create_scheduled_task(self, task: ScheduledTask) -> str:
        """Create a new scheduled task and return its ID."""
        pass

    @abstractmethod
    def update_scheduled_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update a scheduled task."""
        pass

    @abstractmethod
    def delete_scheduled_task(self, task_id: str) -> bool:
        """Delete a scheduled task."""
        pass

    @abstractmethod
    def get_tasks_by_status(self, status: str) -> List[ScheduledTask]:
        """Get tasks with a specific status."""
        pass

    @abstractmethod
    def get_agent_tasks(self, agent_id: str, start_date: datetime, end_date: datetime) -> List[ScheduledTask]:
        """Get tasks for an agent within a time period."""
        pass

    @abstractmethod
    def get_unscheduled_tasks(self) -> List[ScheduledTask]:
        """Get tasks that haven't been scheduled yet."""
        pass

    @abstractmethod
    def get_agent_schedule(self, agent_id: str, date: datetime.date) -> Optional[AgentSchedule]:
        """Get schedule for an agent on a specific date."""
        pass

    @abstractmethod
    def get_all_agent_schedules(self, date: datetime.date) -> List[AgentSchedule]:
        """Get schedules for all agents on a specific date."""
        pass

    @abstractmethod
    def save_agent_schedule(self, schedule: AgentSchedule) -> bool:
        """Save an agent's schedule."""
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

"""
Service interfaces for business logic components.

These interfaces define the contracts for business logic services,
ensuring proper separation of concerns and testability.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Any, Tuple
from solana_agent.domain.feedback import UserFeedback
from solana_agent.domain.tickets import Ticket, TicketResolution
from solana_agent.domain.agents import AIAgent, HumanAgent
from solana_agent.domain.resources import Resource, ResourceBooking
from solana_agent.domain.scheduling import ScheduledTask, AgentSchedule, ScheduleConflict
from solana_agent.domain.memory import MemoryInsight


class AgentService(ABC):
    """Interface for agent management and response generation."""

    @abstractmethod
    def register_ai_agent(self, name: str, instructions: str, specialization: str, model: str = "gpt-4o-mini") -> None:
        """Register an AI agent with its specialization."""
        pass

    @abstractmethod
    def register_human_agent(self, agent: HumanAgent) -> str:
        """Register a human agent and return its ID."""
        pass

    @abstractmethod
    def get_agent_by_name(self, name: str) -> Optional[Any]:
        """Get an agent by name."""
        pass

    @abstractmethod
    def get_agent_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for an agent."""
        pass

    @abstractmethod
    def get_specializations(self) -> Dict[str, str]:
        """Get all registered specializations."""
        pass

    @abstractmethod
    async def generate_response(self, agent_name: str, user_id: str, query: str, memory_context: str = "", **kwargs) -> AsyncGenerator[str, None]:
        """Generate a response from an agent."""
        pass

    @abstractmethod
    def assign_tool_to_agent(self, agent_name: str, tool_name: str) -> bool:
        """Assign a tool to an agent."""
        pass

    @abstractmethod
    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get tools available to an agent."""
        pass

    @abstractmethod
    def execute_tool(self, agent_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on behalf of an agent."""
        pass


class TicketService(ABC):
    """Interface for ticket management."""

    @abstractmethod
    def create_ticket(self, user_id: str, query: str) -> Ticket:
        """Create a new ticket."""
        pass

    @abstractmethod
    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Get a ticket by ID."""
        pass

    @abstractmethod
    def get_active_ticket(self, user_id: str) -> Optional[Ticket]:
        """Get active ticket for a user."""
        pass

    @abstractmethod
    def update_ticket_status(self, ticket_id: str, status: str, reason: Optional[str] = None) -> bool:
        """Update ticket status."""
        pass

    @abstractmethod
    def assign_ticket(self, ticket_id: str, agent_id: str, reason: Optional[str] = None) -> bool:
        """Assign ticket to an agent."""
        pass

    @abstractmethod
    def create_subtask(self, parent_id: str, title: str, description: str, estimated_minutes: int) -> Ticket:
        """Create a subtask for a ticket."""
        pass

    @abstractmethod
    def get_subtasks(self, parent_id: str) -> List[Ticket]:
        """Get all subtasks for a parent ticket."""
        pass

    @abstractmethod
    async def check_ticket_resolution(self, ticket_id: str, response: str) -> TicketResolution:
        """Check if a ticket can be resolved based on the latest response."""
        pass

    @abstractmethod
    async def process_stalled_tickets(self) -> int:
        """Process tickets that haven't been updated for a while."""
        pass

    @abstractmethod
    def get_ticket_history(self, ticket_id: str) -> List[Dict[str, Any]]:
        """Get the full history of a ticket."""
        pass


class QueryService(ABC):
    """Interface for processing user queries."""

    @abstractmethod
    async def process(self, user_id: str, query: str) -> AsyncGenerator[str, None]:
        """Process a query from a user."""
        pass

    @abstractmethod
    async def route_query(self, query: str) -> str:
        """Route a query to the appropriate agent."""
        pass

    @abstractmethod
    async def check_handoff_needed(self, agent_name: str, response: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if handoff is needed based on agent response."""
        pass

    @abstractmethod
    async def handle_system_command(self, user_id: str, command: str) -> Optional[str]:
        """Handle a system command."""
        pass

    @abstractmethod
    async def assess_task_complexity(self, query: str) -> Dict[str, Any]:
        """Assess the complexity of a task."""
        pass


class ResourceService(ABC):
    """Interface for resource management."""

    @abstractmethod
    def create_resource(self, resource: Resource) -> str:
        """Create a new resource."""
        pass

    @abstractmethod
    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a resource by ID."""
        pass

    @abstractmethod
    def find_available_resources(self, resource_type: str, start_time: datetime, end_time: datetime, required_features: List[str] = None) -> List[Resource]:
        """Find available resources for a time period."""
        pass

    @abstractmethod
    def book_resource(self, resource_id: str, user_id: str, start_time: datetime, end_time: datetime, ticket_id: Optional[str] = None) -> Optional[str]:
        """Book a resource and return booking ID."""
        pass

    @abstractmethod
    async def cancel_booking(self, booking_id: str, user_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel a resource booking."""
        pass

    @abstractmethod
    def get_bookings_for_user(self, user_id: str) -> List[ResourceBooking]:
        """Get all bookings for a user."""
        pass

    @abstractmethod
    def get_bookings_for_resource(self, resource_id: str, start_time: datetime, end_time: datetime) -> List[ResourceBooking]:
        """Get bookings for a resource within a time period."""
        pass


class SchedulingService(ABC):
    """Interface for task scheduling."""

    @abstractmethod
    def create_task(self, task: ScheduledTask) -> str:
        """Create a new scheduled task."""
        pass

    @abstractmethod
    def schedule_task(self, task_id: str, agent_id: str, start_time: datetime) -> bool:
        """Schedule a task for an agent."""
        pass

    @abstractmethod
    def get_agent_schedule(self, agent_id: str, date: datetime.date) -> Optional[AgentSchedule]:
        """Get schedule for an agent on a specific date."""
        pass

    @abstractmethod
    def get_agent_availability(self, agent_id: str, start_date: datetime.date, end_date: datetime.date) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """Get available time slots for an agent."""
        pass

    @abstractmethod
    async def optimize_schedule(self, agent_id: str, date: datetime.date) -> bool:
        """Optimize the schedule for an agent."""
        pass

    @abstractmethod
    def detect_conflicts(self, agent_id: str, date: datetime.date) -> List[ScheduleConflict]:
        """Detect scheduling conflicts for an agent."""
        pass

    @abstractmethod
    async def resolve_scheduling_conflicts(self, conflicts: List[ScheduleConflict]) -> List[ScheduleConflict]:
        """Attempt to resolve scheduling conflicts."""
        pass

    @abstractmethod
    def update_task_progress(self, task_id: str, progress: int) -> bool:
        """Update the progress of a task."""
        pass


class MemoryService(ABC):
    """Interface for memory management services."""

    @abstractmethod
    async def extract_insights(self, conversation: Dict[str, str]) -> List[MemoryInsight]:
        """Extract insights from a conversation.

        Args:
            conversation: Dictionary with 'message' and 'response' keys

        Returns:
            List of extracted memory insights
        """
        pass

    @abstractmethod
    async def store_insights(self, user_id: str, insights: List[MemoryInsight]) -> None:
        """Store multiple insights in memory.

        Args:
            user_id: ID of the user these insights relate to
            insights: List of insights to store
        """
        pass

    @abstractmethod
    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """Search collective memory for relevant insights.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of matching memory items
        """
        pass

    @abstractmethod
    async def summarize_user_history(self, user_id: str) -> str:
        """Summarize a user's conversation history.

        Args:
            user_id: User ID to summarize history for

        Returns:
            Summary text
        """
        pass


class NPSService(ABC):
    """Interface for NPS and user feedback services."""

    @abstractmethod
    def store_nps_rating(self, user_id: str, score: int, ticket_id: Optional[str] = None) -> str:
        """Store an NPS rating from a user.

        Args:
            user_id: User ID
            score: NPS score (0-10)
            ticket_id: Optional ticket ID

        Returns:
            Feedback ID
        """
        pass

    @abstractmethod
    def store_feedback(self, user_id: str, feedback_text: str, ticket_id: Optional[str] = None) -> str:
        """Store textual feedback from a user.

        Args:
            user_id: User ID
            feedback_text: Feedback text
            ticket_id: Optional ticket ID

        Returns:
            Feedback ID
        """
        pass

    @abstractmethod
    def get_user_feedback_history(self, user_id: str) -> List[UserFeedback]:
        """Get feedback history for a user.

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

    @abstractmethod
    def calculate_nps_score(self, days: int = 30) -> Dict[str, Any]:
        """Calculate full NPS metrics.

        Args:
            days: Number of days to include

        Returns:
            Dictionary with promoters, detractors, passive, and NPS score
        """
        pass


class HandoffService(ABC):
    """Interface for agent handoff services."""

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass


class CriticService(ABC):
    """Interface for response evaluation and quality assessment services."""

    @abstractmethod
    async def evaluate_response(self, query: str, response: str) -> Dict[str, Any]:
        """Evaluate the quality of a response to a user query.

        Args:
            query: User query
            response: Agent response to evaluate

        Returns:
            Evaluation results including scores and feedback
        """
        pass

    @abstractmethod
    async def needs_human_intervention(
        self, query: str, response: str, threshold: float = 5.0
    ) -> bool:
        """Determine if a response requires human intervention.

        Args:
            query: User query
            response: Agent response
            threshold: Quality threshold below which human help is needed

        Returns:
            True if human intervention is recommended
        """
        pass

    @abstractmethod
    async def suggest_improvements(self, query: str, response: str) -> str:
        """Suggest improvements for a response.

        Args:
            query: User query
            response: Agent response to improve

        Returns:
            Improvement suggestions
        """
        pass

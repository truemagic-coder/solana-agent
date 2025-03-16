"""
Service interfaces for business logic components.

These interfaces define the contracts for business logic services,
ensuring proper separation of concerns and testability.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Any, Tuple
from solana_agent.domain.tickets import Ticket, TicketResolution
from solana_agent.domain.agents import AIAgent, HumanAgent
from solana_agent.domain.resources import Resource, ResourceBooking
from solana_agent.domain.scheduling import ScheduledTask, AgentSchedule, ScheduleConflict


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

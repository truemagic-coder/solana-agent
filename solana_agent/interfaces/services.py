"""
Service interfaces for business logic components.

These interfaces define the contracts for business logic services,
ensuring proper separation of concerns and testability.
"""
from abc import ABC, abstractmethod
from datetime import date, datetime
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


class NotificationService(ABC):
    """Interface for notification management services."""

    @abstractmethod
    async def send_notification(
        self,
        user_id: str,
        message: str,
        channel: str = "email",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a notification to a user.

        Args:
            user_id: User ID or email address
            message: Notification message
            channel: Notification channel (email, sms, in_app)
            metadata: Additional metadata

        Returns:
            True if notification was sent successfully
        """
        pass

    @abstractmethod
    async def send_from_template(
        self,
        user_id: str,
        template_id: str,
        data: Dict[str, Any],
        channel: str = "email"
    ) -> bool:
        """Send a notification using a template.

        Args:
            user_id: User ID or email address
            template_id: Template identifier
            data: Template data for substitution
            channel: Notification channel

        Returns:
            True if notification was sent successfully
        """
        pass

    @abstractmethod
    def register_template(self, template_id: str, template: Any) -> None:
        """Register a notification template.

        Args:
            template_id: Template identifier
            template: Template object
        """
        pass

    @abstractmethod
    async def schedule_notification(
        self,
        user_id: str,
        message: str,
        scheduled_time: datetime,
        channel: str = "email",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Schedule a notification for future delivery.

        Args:
            user_id: User ID or email address
            message: Notification message
            scheduled_time: When to send the notification
            channel: Notification channel
            metadata: Additional metadata

        Returns:
            Scheduled notification ID
        """
        pass

    @abstractmethod
    async def cancel_scheduled_notification(self, notification_id: str) -> bool:
        """Cancel a scheduled notification.

        Args:
            notification_id: Scheduled notification ID

        Returns:
            True if cancellation was successful
        """
        pass


class ProjectApprovalService(ABC):
    """Interface for project approval services."""

    @abstractmethod
    async def submit_project(self, project: Any) -> str:
        """Submit a project for approval.

        Args:
            project: Project to submit

        Returns:
            Project ID
        """
        pass

    @abstractmethod
    async def get_project(self, project_id: str) -> Optional[Any]:
        """Get a project by ID.

        Args:
            project_id: Project ID

        Returns:
            Project or None if not found
        """
        pass

    @abstractmethod
    async def review_project(
        self, project_id: str, reviewer_id: str, is_human_reviewer: bool = True
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Review a project against approval criteria.

        Args:
            project_id: Project ID
            reviewer_id: ID of the reviewer
            is_human_reviewer: Whether the reviewer is human

        Returns:
            Tuple of (overall_score, criteria_results)
        """
        pass

    @abstractmethod
    async def submit_review(
        self,
        project_id: str,
        reviewer_id: str,
        criteria_scores: List[Dict[str, Any]],
        overall_score: float,
        comments: str = ""
    ) -> bool:
        """Submit a completed review.

        Args:
            project_id: Project ID
            reviewer_id: ID of the reviewer
            criteria_scores: Scores for each criterion
            overall_score: Overall project score
            comments: Optional review comments

        Returns:
            True if review was submitted successfully
        """
        pass

    @abstractmethod
    async def approve_project(self, project_id: str, approver_id: str, comments: str = "") -> bool:
        """Approve a project.

        Args:
            project_id: Project ID
            approver_id: ID of the approver
            comments: Optional approval comments

        Returns:
            True if approval was successful
        """
        pass

    @abstractmethod
    async def reject_project(self, project_id: str, rejector_id: str, reason: str) -> bool:
        """Reject a project.

        Args:
            project_id: Project ID
            rejector_id: ID of the rejector
            reason: Rejection reason

        Returns:
            True if rejection was successful
        """
        pass


class RoutingService(ABC):
    """Interface for query routing services."""

    @abstractmethod
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query to determine routing information.

        Args:
            query: User query to analyze

        Returns:
            Analysis results including specializations and complexity
        """
        pass

    @abstractmethod
    async def route_query(self, user_id: str, query: str) -> Tuple[str, Any]:
        """Route a query to the appropriate agent.

        Args:
            user_id: User ID
            query: User query

        Returns:
            Tuple of (agent_name, ticket)
        """
        pass

    @abstractmethod
    async def reroute_ticket(self, ticket_id: str, target_agent: str, reason: str) -> bool:
        """Reroute a ticket to a different agent.

        Args:
            ticket_id: Ticket ID
            target_agent: Target agent name
            reason: Reason for rerouting

        Returns:
            True if rerouting was successful
        """
        pass


class TicketService(ABC):
    """Interface for ticket management services."""

    @abstractmethod
    async def get_or_create_ticket(
        self, user_id: str, query: str, complexity: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get active ticket for user or create a new one.

        Args:
            user_id: User ID
            query: User query
            complexity: Optional complexity metadata

        Returns:
            Active or newly created ticket
        """
        pass

    @abstractmethod
    def update_ticket_status(self, ticket_id: str, status: Any, **additional_updates) -> bool:
        """Update ticket status and additional fields.

        Args:
            ticket_id: Ticket ID
            status: New status
            **additional_updates: Additional fields to update

        Returns:
            True if update was successful
        """
        pass

    @abstractmethod
    def mark_ticket_resolved(self, ticket_id: str, resolution_data: Dict[str, Any]) -> bool:
        """Mark a ticket as resolved with resolution information.

        Args:
            ticket_id: Ticket ID
            resolution_data: Resolution details

        Returns:
            True if update was successful
        """
        pass

    @abstractmethod
    def add_note_to_ticket(
        self, ticket_id: str, content: str, note_type: str = "system", created_by: Optional[str] = None
    ) -> bool:
        """Add a note to a ticket.

        Args:
            ticket_id: Ticket ID
            content: Note content
            note_type: Type of note
            created_by: ID of note creator

        Returns:
            True if note was added successfully
        """
        pass

    @abstractmethod
    def get_ticket_by_id(self, ticket_id: str) -> Optional[Any]:
        """Get a ticket by ID.

        Args:
            ticket_id: Ticket ID

        Returns:
            Ticket or None if not found
        """
        pass

    @abstractmethod
    def get_tickets_by_user(self, user_id: str, limit: int = 20) -> List[Any]:
        """Get tickets for a specific user.

        Args:
            user_id: User ID
            limit: Maximum number of tickets to return

        Returns:
            List of tickets
        """
        pass

    @abstractmethod
    def get_tickets_by_status(self, status: Any, limit: int = 50) -> List[Any]:
        """Get tickets by status.

        Args:
            status: Ticket status
            limit: Maximum number of tickets to return

        Returns:
            List of tickets
        """
        pass

    @abstractmethod
    def assign_ticket(self, ticket_id: str, agent_id: str) -> bool:
        """Assign a ticket to an agent.

        Args:
            ticket_id: Ticket ID
            agent_id: Agent ID

        Returns:
            True if assignment was successful
        """
        pass

    @abstractmethod
    def close_ticket(self, ticket_id: str, reason: str = "") -> bool:
        """Close a ticket.

        Args:
            ticket_id: Ticket ID
            reason: Closure reason

        Returns:
            True if closure was successful
        """
        pass


class ResourceService(ABC):
    """Interface for resource management services."""

    @abstractmethod
    async def create_resource(self, resource_data: Dict[str, Any], resource_type: str) -> str:
        """Create a new resource from dictionary data.

        Args:
            resource_data: Resource properties
            resource_type: Type of resource

        Returns:
            Resource ID
        """
        pass

    @abstractmethod
    async def get_resource(self, resource_id: str) -> Optional[Any]:
        """Get a resource by ID.

        Args:
            resource_id: Resource ID

        Returns:
            Resource or None if not found
        """
        pass

    @abstractmethod
    async def update_resource(self, resource_id: str, updates: Dict[str, Any]) -> bool:
        """Update a resource.

        Args:
            resource_id: Resource ID
            updates: Dictionary of updates to apply

        Returns:
            True if update was successful
        """
        pass

    @abstractmethod
    async def list_resources(self, resource_type: Optional[str] = None) -> List[Any]:
        """List all resources, optionally filtered by type.

        Args:
            resource_type: Optional type to filter by

        Returns:
            List of resources
        """
        pass

    @abstractmethod
    async def find_available_resources(
        self,
        start_time: datetime,
        end_time: datetime,
        capacity: Optional[int] = None,
        tags: Optional[List[str]] = None,
        resource_type: Optional[str] = None
    ) -> List[Any]:
        """Find available resources for a time period.

        Args:
            start_time: Start of time window
            end_time: End of time window
            capacity: Minimum capacity required
            tags: Required resource tags
            resource_type: Type of resource

        Returns:
            List of available resources
        """
        pass

    @abstractmethod
    async def create_booking(
        self,
        resource_id: str,
        user_id: str,
        title: str,
        start_time: datetime,
        end_time: datetime,
        description: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Create a booking for a resource.

        Args:
            resource_id: Resource ID
            user_id: User ID
            title: Booking title
            start_time: Start time
            end_time: End time
            description: Optional description
            notes: Optional notes

        Returns:
            Tuple of (success, booking_id, error_message)
        """
        pass

    @abstractmethod
    async def cancel_booking(self, booking_id: str, user_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel a booking.

        Args:
            booking_id: Booking ID
            user_id: User ID attempting to cancel

        Returns:
            Tuple of (success, error_message)
        """
        pass

    @abstractmethod
    async def get_resource_schedule(
        self,
        resource_id: str,
        start_date: date,
        end_date: date
    ) -> List[Any]:
        """Get a resource's schedule for a date range.

        Args:
            resource_id: Resource ID
            start_date: Start date
            end_date: End date

        Returns:
            List of bookings in the date range
        """
        pass

    @abstractmethod
    async def get_user_bookings(
        self, user_id: str, include_cancelled: bool = False
    ) -> List[Dict[str, Any]]:
        """Get all bookings for a user with resource details.

        Args:
            user_id: User ID
            include_cancelled: Whether to include cancelled bookings

        Returns:
            List of booking and resource information
        """
        pass


class ProjectSimulationService(ABC):
    """Interface for project simulation services."""

    @abstractmethod
    async def simulate_project(self, project_description: str) -> Dict[str, Any]:
        """Run a full simulation on a potential project using historical data when available.

        Args:
            project_description: Description of the project to simulate

        Returns:
            Simulation results including complexity, timeline, risks, etc.
        """
        pass

    @abstractmethod
    async def _assess_risks(
        self, project_description: str, similar_projects: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Assess potential risks in the project using historical data.

        Args:
            project_description: Project description
            similar_projects: Optional list of similar historical projects

        Returns:
            Risk assessment results
        """
        pass

    @abstractmethod
    async def _estimate_timeline(
        self,
        project_description: str,
        complexity: Dict[str, Any],
        similar_projects: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Estimate timeline with confidence intervals using historical data.

        Args:
            project_description: Project description
            complexity: Complexity assessment
            similar_projects: Optional list of similar historical projects

        Returns:
            Timeline estimates
        """
        pass

    @abstractmethod
    async def _assess_resource_needs(
        self, project_description: str, complexity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess resource requirements for the project.

        Args:
            project_description: Project description
            complexity: Complexity assessment

        Returns:
            Resource requirements
        """
        pass


class SchedulingService(ABC):
    """Interface for task scheduling and agent coordination services."""

    @abstractmethod
    async def schedule_task(
        self,
        task: Any,
        preferred_agent_id: str = None
    ) -> Any:
        """Schedule a task with optimal time and agent assignment.

        Args:
            task: Task to schedule
            preferred_agent_id: Preferred agent ID

        Returns:
            Scheduled task
        """
        pass

    @abstractmethod
    async def find_optimal_time_slot_with_resources(
        self,
        task: Any,
        resource_service: Any,
        agent_schedule: Optional[Any] = None
    ) -> Optional[Any]:
        """Find the optimal time slot for a task based on both agent and resource availability.

        Args:
            task: Task to schedule
            resource_service: Resource service for checking resource availability
            agent_schedule: Optional cached agent schedule

        Returns:
            Optimal time window or None if not found
        """
        pass

    @abstractmethod
    async def optimize_schedule(self) -> Dict[str, Any]:
        """Optimize the entire schedule to maximize efficiency.

        Returns:
            Optimization results
        """
        pass

    @abstractmethod
    async def register_agent_schedule(self, schedule: Any) -> bool:
        """Register or update an agent's schedule.

        Args:
            schedule: Agent schedule

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def get_agent_schedule(self, agent_id: str) -> Optional[Any]:
        """Get an agent's schedule.

        Args:
            agent_id: Agent ID

        Returns:
            Agent schedule or None if not found
        """
        pass

    @abstractmethod
    async def get_agent_tasks(
        self,
        agent_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        include_completed: bool = False
    ) -> List[Any]:
        """Get all tasks scheduled for an agent within a time range.

        Args:
            agent_id: Agent ID
            start_time: Optional start time filter
            end_time: Optional end time filter
            include_completed: Whether to include completed tasks

        Returns:
            List of tasks
        """
        pass

    @abstractmethod
    async def mark_task_started(self, task_id: str) -> bool:
        """Mark a task as started.

        Args:
            task_id: Task ID

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def mark_task_completed(self, task_id: str) -> bool:
        """Mark a task as completed.

        Args:
            task_id: Task ID

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def find_available_time_slots(
        self,
        agent_id: str,
        duration_minutes: int,
        start_after: datetime = None,
        end_before: datetime = None,
        count: int = 3,
        agent_schedule: Optional[Any] = None
    ) -> List[Any]:
        """Find available time slots for an agent.

        Args:
            agent_id: Agent ID
            duration_minutes: Task duration in minutes
            start_after: Don't look for slots before this time
            end_before: Don't look for slots after this time
            count: Maximum number of slots to return
            agent_schedule: Optional cached agent schedule

        Returns:
            List of available time windows
        """
        pass

    @abstractmethod
    async def resolve_scheduling_conflicts(self) -> Dict[str, Any]:
        """Detect and resolve scheduling conflicts.

        Returns:
            Conflict resolution results
        """
        pass

    @abstractmethod
    async def request_time_off(
        self,
        agent_id: str,
        start_time: datetime,
        end_time: datetime,
        reason: str
    ) -> Tuple[bool, str, Optional[str]]:
        """Request time off for a human agent.

        Args:
            agent_id: Agent ID
            start_time: Start time
            end_time: End time
            reason: Reason for time off

        Returns:
            Tuple of (success, status, request_id)
        """
        pass

    @abstractmethod
    async def cancel_time_off_request(
        self,
        agent_id: str,
        request_id: str
    ) -> Tuple[bool, str]:
        """Cancel a time off request.

        Args:
            agent_id: Agent ID
            request_id: Request ID

        Returns:
            Tuple of (success, status)
        """
        pass

    @abstractmethod
    async def get_agent_time_off_history(
        self,
        agent_id: str
    ) -> List[Dict[str, Any]]:
        """Get an agent's time off history.

        Args:
            agent_id: Agent ID

        Returns:
            List of time off requests
        """
        pass

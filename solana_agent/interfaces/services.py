"""
Service interfaces for business logic components.

These interfaces define the contracts for business logic services,
ensuring proper separation of concerns and testability.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional, Any, Tuple
from solana_agent.domains import UserFeedback
from solana_agent.domains import Ticket
from solana_agent.domains import MemoryInsight


class AgentService(ABC):
    """Interface for agent management and response generation."""

    @abstractmethod
    def register_ai_agent(self, name: str, instructions: str, specialization: str, model: str = "gpt-4o-mini") -> None:
        """Register an AI agent with its specialization."""
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
    def assign_tool_for_agent(self, agent_name: str, tool_name: str) -> bool:
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


class QueryService(ABC):
    """Interface for processing user queries."""

    @abstractmethod
    async def process(self, user_id: str, query: str) -> AsyncGenerator[str, None]:
        """Process a query from a user."""
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


class RoutingService(ABC):
    """Interface for query routing services."""

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

    @abstractmethod
    def find_stalled_tickets(self, timeout_minutes: int) -> List[Ticket]:
        """Find tickets that have been stalled for too long.

        Args:
            timeout_minutes: Number of minutes before a ticket is considered stalled

        Returns:
            List of stalled tickets
        """
        pass

    @abstractmethod
    def get_active_for_user(self, user_id: str) -> Optional[Ticket]:
        """Get active ticket for a user.

        Args:
            user_id: User ID

        Returns:
            Active ticket or None if not found
        """
        pass

    @abstractmethod
    def update_ticket(self, ticket_id: str, **fields_to_update) -> bool:
        """Update a ticket with new fields.

        Args:
            ticket_id: Ticket ID
            **fields_to_update: Fields to update

        Returns:
            True if update was successful
        """
        pass


class CommandService(ABC):
    """Interface for processing system commands."""

    @abstractmethod
    async def process_command(
        self, user_id: str, command_text: str, timezone: Optional[str] = None
    ) -> Optional[str]:
        """Process a system command.

        Args:
            user_id: User ID
            command_text: Command text including prefix
            timezone: Optional user timezone

        Returns:
            Command result or None if not a command
        """
        pass

    @abstractmethod
    def get_available_commands(self) -> List[Dict[str, Any]]:
        """Get information about all available commands.

        Returns:
            List of command information dictionaries
        """
        pass

    @abstractmethod
    def register_handler(self, handler: Any) -> None:
        """Register a command handler.

        Args:
            handler: Command handler to register
        """
        pass

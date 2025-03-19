"""
Command service implementation.

This service handles system commands denoted by a prefix (e.g., !) and
routes them to appropriate handlers.
"""
from abc import ABC, abstractmethod
import datetime
from typing import Dict, List, Optional, Any


from solana_agent.interfaces import CommandService as CommandServiceInterface
from solana_agent.interfaces import (
    TicketService, AgentService
)


class CommandContext:
    """Context for command execution with services and parameters."""

    def __init__(
        self,
        user_id: str,
        raw_command: str,
        args: List[str],
        timezone: Optional[str] = None,
        **services
    ):
        """Initialize the command context.

        Args:
            user_id: User ID
            raw_command: Raw command text
            args: Command arguments
            timezone: Optional user timezone
            **services: Available services for command execution
        """
        self.user_id = user_id
        self.raw_command = raw_command
        self.args = args
        self.timezone = timezone
        self.services = services

    def get_service(self, service_name: str) -> Any:
        """Get a service by name.

        Args:
            service_name: Service name

        Returns:
            Service instance or None if not found
        """
        return self.services.get(service_name)


class CommandHandler(ABC):
    """Base class for command handlers."""

    @property
    @abstractmethod
    def command(self) -> str:
        """The command trigger word."""
        pass

    @property
    def aliases(self) -> List[str]:
        """Alternative command names."""
        return []

    @property
    def description(self) -> str:
        """Command description."""
        return "No description provided"

    @property
    def usage(self) -> str:
        """Command usage instructions."""
        return f"!{self.command}"

    @property
    def examples(self) -> List[str]:
        """Examples of command usage."""
        return []

    @abstractmethod
    async def execute(self, context: CommandContext) -> str:
        """Execute the command.

        Args:
            context: Command execution context

        Returns:
            Command result
        """
        pass


class HelpCommandHandler(CommandHandler):
    """Handler for the help command."""

    def __init__(self, registry: Dict[str, CommandHandler]):
        """Initialize the help command handler.

        Args:
            registry: Command registry
        """
        self.registry = registry

    @property
    def command(self) -> str:
        return "help"

    @property
    def description(self) -> str:
        return "Show help information for available commands"

    @property
    def usage(self) -> str:
        return "!help [command]"

    @property
    def examples(self) -> List[str]:
        return ["!help", "!help status"]

    async def execute(self, context: CommandContext) -> str:
        """Show help for commands."""
        if len(context.args) > 0:
            # Show help for specific command
            cmd_name = context.args[0].lower()
            if cmd_name.startswith("!"):
                cmd_name = cmd_name[1:]

            handler = self.registry.get(cmd_name)

            if not handler:
                return f"Unknown command: {cmd_name}"

            result = [
                f"## Command: !{handler.command}",
                f"\n**Description**: {handler.description}",
                f"\n**Usage**: {handler.usage}"
            ]

            if handler.aliases:
                result.append(
                    f"\n**Aliases**: {', '.join(['!' + a for a in handler.aliases])}")

            if handler.examples:
                result.append("\n**Examples**:")
                for example in handler.examples:
                    result.append(f"- `{example}`")

            return "\n".join(result)
        else:
            # List all commands
            result = ["## Available Commands\n"]

            for cmd_name, handler in sorted(self.registry.items()):
                # Skip aliases to avoid duplicates
                primary_handlers = {
                    h.command: h for h in self.registry.values()}
                if cmd_name not in primary_handlers:
                    continue

                result.append(f"- **!{cmd_name}** - {handler.description}")

            result.append(
                "\nFor more details on a specific command, type `!help <command>`")
            return "\n".join(result)


class StatusCommandHandler(CommandHandler):
    """Handler for the status command."""

    @property
    def command(self) -> str:
        return "status"

    @property
    def aliases(self) -> List[str]:
        return ["ticket", "info"]

    @property
    def description(self) -> str:
        return "Check the status of a ticket or task"

    @property
    def usage(self) -> str:
        return "!status [ticket_id]"

    @property
    def examples(self) -> List[str]:
        return ["!status", "!status abc123"]

    async def execute(self, context: CommandContext) -> str:
        """Check status of tickets."""
        ticket_service = context.get_service("ticket_service")
        if not ticket_service:
            return "Ticket service is not available"

        # Check if ticket ID is provided
        if len(context.args) > 0:
            ticket_id = context.args[0]
            ticket = ticket_service.get_ticket_by_id(ticket_id)

            if not ticket:
                return f"Ticket {ticket_id} not found"

            # Get status info
            status_info = await self._get_ticket_details(ticket, context)
            return status_info
        else:
            # Show user's active tickets
            active_tickets = ticket_service.get_all_for_user(
                context.user_id, limit=5, include_resolved=False
            )

            if not active_tickets:
                return "You don't have any active tickets"

            result = ["## Your Active Tickets\n"]

            for ticket in active_tickets:
                result.append(
                    f"- **{ticket.id}**: {ticket.title} (Status: {ticket.status})"
                )

            result.append(
                "\nUse `!status <ticket_id>` to see detailed information")
            return "\n".join(result)

    async def _get_ticket_details(self, ticket: Any, context: CommandContext) -> str:
        """Get detailed ticket information.

        Args:
            ticket: Ticket object
            context: Command context

        Returns:
            Formatted ticket details
        """
        result = [
            f"## Ticket: {ticket.id}",
            f"\n**Title**: {ticket.title}",
            f"\n**Status**: {ticket.status}",
            f"\n**Created**: {self._format_time(ticket.created_at, context.timezone)}",
            f"\n**Updated**: {self._format_time(ticket.updated_at, context.timezone)}"
        ]

        if ticket.assigned_to:
            result.append(f"\n**Assigned to**: {ticket.assigned_to}")

        # Add subtasks if this is a parent ticket
        if hasattr(ticket, "is_parent") and ticket.is_parent:
            task_planning_service = context.get_service(
                "task_planning_service")
            if task_planning_service:
                try:
                    plan_status = await task_planning_service.get_plan_status(ticket.id)
                    result.append(
                        f"\n**Plan Progress**: {plan_status.progress}%")
                    result.append(
                        f"\n**Subtasks**: {plan_status.subtask_count}")
                    result.append(
                        f"\n**Estimated Completion**: {plan_status.estimated_completion}")
                    result.append(f"\n**Status**: {plan_status.visualization}")
                except Exception as e:
                    result.append(f"\n**Error getting plan status**: {str(e)}")

        # Add ticket history/interactions
        if hasattr(ticket, "interactions") and ticket.interactions:
            result.append("\n\n### Recent Activity")
            for interaction in sorted(ticket.interactions[-3:], key=lambda x: x.timestamp):
                result.append(
                    f"\n- **{interaction.type}** ({self._format_time(interaction.timestamp, context.timezone)})")
                if interaction.content and len(interaction.content) > 0:
                    short_content = interaction.content[:50] + (
                        "..." if len(interaction.content) > 50 else "")
                    result.append(f"  {short_content}")

        return "\n".join(result)

    def _format_time(self, dt: datetime.datetime, timezone: Optional[str] = None) -> str:
        """Format datetime with timezone adjustment if specified.

        Args:
            dt: Datetime to format
            timezone: Optional timezone

        Returns:
            Formatted datetime string
        """
        if not dt:
            return "N/A"

        if timezone:
            try:
                import pytz
                tz = pytz.timezone(timezone)
                dt = dt.astimezone(tz)
                return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except (ImportError, pytz.exceptions.UnknownTimeZoneError):
                pass

        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


class CommandService(CommandServiceInterface):
    """Service for processing system commands."""

    def __init__(
        self,
        ticket_service: Optional[TicketService] = None,
        agent_service: Optional[AgentService] = None,
        command_prefix: str = "!",
        additional_handlers: Optional[List[CommandHandler]] = None,
    ):
        """Initialize the command service.

        Args:
            ticket_service: Service for ticket operations
            agent_service: Service for agent management
            command_prefix: Prefix for commands
            additional_handlers: Additional command handlers
        """
        self.services = {
            "ticket_service": ticket_service,
            "agent_service": agent_service,
        }

        self.command_prefix = command_prefix
        self.registry = {}  # command_name -> handler

        # Register built-in handlers
        self._register_built_in_handlers()

        # Register additional handlers
        if additional_handlers:
            for handler in additional_handlers:
                self.register_handler(handler)

        # Update help handler with complete registry
        self.registry["help"] = HelpCommandHandler(self.registry)

    def _register_built_in_handlers(self) -> None:
        """Register built-in command handlers."""
        built_in = [
            StatusCommandHandler(),
        ]

        for handler in built_in:
            self.register_handler(handler)

    def register_handler(self, handler: CommandHandler) -> None:
        """Register a command handler.

        Args:
            handler: Command handler to register
        """
        # Register primary command
        self.registry[handler.command] = handler

        # Register aliases
        for alias in handler.aliases:
            self.registry[alias] = handler

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
        if not command_text.startswith(self.command_prefix):
            return None

        # Parse command
        command_parts = command_text[len(self.command_prefix):].split()
        if not command_parts:
            return "Please enter a valid command. Use !help for available commands."

        command_name = command_parts[0].lower()
        args = command_parts[1:]

        # Look up handler
        handler = self.registry.get(command_name)
        if not handler:
            return f"Unknown command: {command_name}. Use !help for available commands."

        # Create context
        context = CommandContext(
            user_id=user_id,
            raw_command=command_text,
            args=args,
            timezone=timezone,
            **self.services
        )

        # Execute command
        try:
            return await handler.execute(context)
        except Exception as e:
            import traceback
            print(f"Error executing command {command_name}: {str(e)}")
            print(traceback.format_exc())
            return f"Error executing command: {str(e)}"

    def get_available_commands(self) -> List[Dict[str, Any]]:
        """Get information about all available commands.

        Returns:
            List of command information dictionaries
        """
        unique_handlers = {}
        for name, handler in self.registry.items():
            if handler.command == name:  # Only include primary commands
                unique_handlers[name] = handler

        result = []
        for name, handler in sorted(unique_handlers.items()):
            result.append({
                "name": name,
                "aliases": handler.aliases,
                "description": handler.description,
                "usage": handler.usage,
                "examples": handler.examples
            })

        return result

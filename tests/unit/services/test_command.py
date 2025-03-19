"""
Tests for the CommandService implementation.

This module tests command processing, handler registration, and built-in commands.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import datetime
from typing import List, Any

from solana_agent.services.command import (
    CommandService, CommandHandler, CommandContext, StatusCommandHandler,
)


# ---------------------
# Fixtures
# ---------------------

@pytest.fixture
def mock_ticket_service():
    """Return a mock ticket service."""
    service = Mock()

    # Setup mock ticket
    mock_ticket = Mock()
    mock_ticket.id = "ticket123"
    mock_ticket.title = "Test Ticket"
    mock_ticket.status = "Open"
    mock_ticket.created_at = datetime.datetime(2025, 3, 15, 10, 30, 0)
    mock_ticket.updated_at = datetime.datetime(2025, 3, 16, 14, 45, 0)
    mock_ticket.assigned_to = "agent1"
    mock_ticket.interactions = []

    # Setup mock methods
    service.get_ticket_by_id = Mock(return_value=mock_ticket)
    service.get_all_for_user = Mock(return_value=[mock_ticket])

    return service


@pytest.fixture
def mock_agent_service():
    """Return a mock agent service."""
    service = Mock()
    return service


@pytest.fixture
def mock_task_planning_service():
    """Return a mock task planning service."""
    service = Mock()

    # Setup mock plan status
    mock_plan_status = Mock()
    mock_plan_status.progress = 75
    mock_plan_status.subtask_count = 4
    mock_plan_status.estimated_completion = "2025-03-20"
    mock_plan_status.visualization = "█████████▒▒▒"

    # Setup mock methods as AsyncMock for async functions
    service.get_plan_status = AsyncMock(return_value=mock_plan_status)

    return service


@pytest.fixture
def mock_project_approval_service():
    """Return a mock project approval service."""
    service = Mock()
    return service


@pytest.fixture
def custom_command_handler():
    """Return a custom command handler for testing."""
    class TestCommandHandler(CommandHandler):
        @property
        def command(self) -> str:
            return "test"

        @property
        def aliases(self) -> List[str]:
            return ["t", "tst"]

        @property
        def description(self) -> str:
            return "Test command for unit testing"

        async def execute(self, context: CommandContext) -> str:
            return f"Test command executed with args: {' '.join(context.args)}"

    return TestCommandHandler()


@pytest.fixture
def command_service(
    mock_ticket_service,
    mock_agent_service,
):
    """Return a command service with mocked dependencies."""
    return CommandService(
        ticket_service=mock_ticket_service,
        agent_service=mock_agent_service,
    )


@pytest.fixture
def command_service_with_custom_handler(command_service, custom_command_handler):
    """Return a command service with a custom handler registered."""
    command_service.register_handler(custom_command_handler)
    return command_service


# ---------------------
# Initialization Tests
# ---------------------

def test_command_service_initialization(command_service):
    """Test that the command service initializes properly."""
    # Verify that built-in handlers are registered
    assert "help" in command_service.registry
    assert "status" in command_service.registry

    # Verify aliases are registered
    assert "ticket" in command_service.registry


def test_command_service_with_custom_prefix():
    """Test initializing command service with custom prefix."""
    service = CommandService(command_prefix="/")
    assert service.command_prefix == "/"


def test_register_custom_handler(command_service, custom_command_handler):
    """Test registering a custom command handler."""
    # Register the handler
    command_service.register_handler(custom_command_handler)

    # Verify handler and its aliases are registered
    assert "test" in command_service.registry
    assert "t" in command_service.registry
    assert "tst" in command_service.registry

    # Verify all point to the same handler
    assert command_service.registry["test"] is custom_command_handler
    assert command_service.registry["t"] is custom_command_handler
    assert command_service.registry["tst"] is custom_command_handler


# ---------------------
# Command Processing Tests
# ---------------------

@pytest.mark.asyncio
async def test_process_non_command(command_service):
    """Test processing text that isn't a command."""
    result = await command_service.process_command(
        "user1", "This is not a command", None)
    assert result is None


@pytest.mark.asyncio
async def test_process_empty_command(command_service):
    """Test processing an empty command."""
    result = await command_service.process_command("user1", "!", None)
    assert "valid command" in result


@pytest.mark.asyncio
async def test_process_unknown_command(command_service):
    """Test processing an unknown command."""
    result = await command_service.process_command(
        "user1", "!unknown", None)
    assert "Unknown command" in result


@pytest.mark.asyncio
async def test_process_help_command(command_service):
    """Test processing the help command."""
    result = await command_service.process_command("user1", "!help", None)
    assert "Available Commands" in result
    assert "status" in result


@pytest.mark.asyncio
async def test_process_help_for_specific_command(command_service):
    """Test processing help for a specific command."""
    result = await command_service.process_command(
        "user1", "!help status", None)
    assert "Command: !status" in result
    assert "Description" in result
    assert "Usage" in result
    assert "Aliases" in result


@pytest.mark.asyncio
async def test_process_command_with_custom_handler(command_service_with_custom_handler):
    """Test processing a custom command."""
    result = await command_service_with_custom_handler.process_command(  # <-- Fixed variable name
        "user1", "!test arg1 arg2", None)
    assert "Test command executed with args: arg1 arg2" in result


@pytest.mark.asyncio
async def test_process_command_with_timezone(command_service):
    """Test processing a command with timezone."""
    # Create a patched version of the status handler execute method to verify timezone
    original_execute = StatusCommandHandler.execute
    execute_called_with = {}

    async def patched_execute(self, context):
        execute_called_with["timezone"] = context.timezone
        return await original_execute(self, context)

    with patch.object(StatusCommandHandler, 'execute', patched_execute):
        await command_service.process_command("user1", "!status", "America/New_York")
        assert execute_called_with["timezone"] == "America/New_York"


# ---------------------
# Help Command Handler Tests
# ---------------------

@pytest.mark.asyncio
async def test_help_command_list_all(command_service):
    """Test help command listing all available commands."""
    # Create a context for testing
    context = CommandContext(
        user_id="user1",
        raw_command="!help",
        args=[],
        timezone=None
    )

    # Get help handler and execute
    help_handler = command_service.registry["help"]
    result = await help_handler.execute(context)

    # Verify result contains commands
    assert "Available Commands" in result
    assert "status" in result


@pytest.mark.asyncio
async def test_help_command_for_specific_command(command_service):
    """Test help command for a specific command."""
    # Create a context for testing
    context = CommandContext(
        user_id="user1",
        raw_command="!help status",
        args=["status"],
        timezone=None
    )

    # Get help handler and execute
    help_handler = command_service.registry["help"]
    result = await help_handler.execute(context)

    # Verify result contains command details
    assert "Command: !status" in result
    assert "Description" in result
    assert "Check the status" in result


@pytest.mark.asyncio
async def test_help_command_unknown_command(command_service):
    """Test help command for an unknown command."""
    # Create a context for testing
    context = CommandContext(
        user_id="user1",
        raw_command="!help unknown",
        args=["unknown"],
        timezone=None
    )

    # Get help handler and execute
    help_handler = command_service.registry["help"]
    result = await help_handler.execute(context)

    # Verify result indicates unknown command
    assert "Unknown command" in result


# ---------------------
# Status Command Handler Tests
# ---------------------

@pytest.mark.asyncio
async def test_status_command_no_args(command_service, mock_ticket_service):
    """Test status command with no arguments."""
    # Create a context for testing
    context = CommandContext(
        user_id="user1",
        raw_command="!status",
        args=[],
        timezone=None,
        ticket_service=mock_ticket_service
    )

    # Get status handler and execute
    status_handler = command_service.registry["status"]
    result = await status_handler.execute(context)

    # Verify mock was called
    mock_ticket_service.get_all_for_user.assert_called_once()

    # Verify result contains ticket list
    assert "Your Active Tickets" in result
    assert "Test Ticket" in result


@pytest.mark.asyncio
async def test_status_command_with_ticket_id(command_service, mock_ticket_service):
    """Test status command with a ticket ID."""
    # Create a context for testing
    context = CommandContext(
        user_id="user1",
        raw_command="!status ticket123",
        args=["ticket123"],
        timezone=None,
        ticket_service=mock_ticket_service
    )

    # Get status handler and execute
    status_handler = command_service.registry["status"]
    result = await status_handler.execute(context)

    # Verify mock was called
    mock_ticket_service.get_ticket_by_id.assert_called_once_with("ticket123")

    # Verify result contains ticket details
    assert "Ticket: ticket123" in result
    assert "Test Ticket" in result
    assert "Open" in result


@pytest.mark.asyncio
async def test_status_command_with_missing_ticket(command_service, mock_ticket_service):
    """Test status command with a non-existent ticket ID."""
    # Setup mock to return None for this ticket ID
    mock_ticket_service.get_ticket_by_id.return_value = None

    # Create a context for testing
    context = CommandContext(
        user_id="user1",
        raw_command="!status nonexistent",
        args=["nonexistent"],
        timezone=None,
        ticket_service=mock_ticket_service
    )

    # Get status handler and execute
    status_handler = command_service.registry["status"]
    result = await status_handler.execute(context)

    # Verify result indicates ticket not found
    assert "not found" in result


@pytest.mark.asyncio
async def test_status_command_missing_service(command_service):
    """Test status command when ticket service is unavailable."""
    # Create a context without ticket service
    context = CommandContext(
        user_id="user1",
        raw_command="!status",
        args=[],
        timezone=None
    )

    # Get status handler and execute
    status_handler = command_service.registry["status"]
    result = await status_handler.execute(context)

    # Verify result indicates service unavailable
    assert "not available" in result


# ---------------------
# Command Context Tests
# ---------------------

def test_command_context_get_service():
    """Test CommandContext get_service method."""
    # Create a context with services
    mock_service = Mock()
    context = CommandContext(
        user_id="user1",
        raw_command="!test",
        args=[],
        timezone="UTC",
        test_service=mock_service
    )

    # Test getting an existing service
    assert context.get_service("test_service") is mock_service

    # Test getting a non-existent service
    assert context.get_service("nonexistent") is None


# ---------------------
# Error Handling Tests
# ---------------------

@pytest.mark.asyncio
async def test_command_execution_error(command_service, custom_command_handler):
    """Test handling errors during command execution."""
    # Register the command handler but patch it to raise an exception
    command_service.register_handler(custom_command_handler)

    with patch.object(custom_command_handler, 'execute',
                      side_effect=Exception("Command failed")):
        result = await command_service.process_command("user1", "!test", None)

        # Verify result indicates error
        assert "Error executing command" in result
        assert "Command failed" in result


# ---------------------
# Command Listing Tests
# ---------------------

def test_get_available_commands(command_service):
    """Test getting information about available commands."""
    commands = command_service.get_available_commands()

    # Verify list contains expected commands
    command_names = [cmd["name"] for cmd in commands]
    assert "help" in command_names
    assert "status" in command_names

    # Verify command details
    status_cmd = next(cmd for cmd in commands if cmd["name"] == "status")
    assert "ticket" in status_cmd["aliases"]
    assert "description" in status_cmd
    assert "usage" in status_cmd
    assert "examples" in status_cmd

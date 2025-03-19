"""
Tests for the TicketService implementation.

This module tests ticket creation, updating, and management
throughout the support ticket lifecycle.
"""
import pytest
import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional

from solana_agent.services.ticket import TicketService
from solana_agent.domains import Ticket, TicketStatus, TicketNote, TicketPriority


@pytest.fixture
def mock_repository():
    """Create a mock ticket repository."""
    repo = Mock()

    # Configure common repository method mocks
    repo.create = Mock(return_value="ticket-123")
    repo.get_by_id = Mock(return_value=None)
    repo.update = Mock(return_value=True)
    repo.add_note = Mock(return_value=True)
    repo.get_by_user = Mock(return_value=[])
    repo.get_by_status = Mock(return_value=[])
    repo.get_active_for_user = Mock(return_value=None)

    return repo


@pytest.fixture
def ticket_service(mock_repository):
    """Create a ticket service with mocked repository."""
    return TicketService(mock_repository)


@pytest.fixture
def sample_ticket():
    """Create a sample ticket for testing."""
    return Ticket(
        id="ticket-123",
        title="Help with Solana integration",
        description="I'm having trouble integrating my dApp with Solana.",
        user_id="user-456",
        status=TicketStatus.NEW,
        assigned_to="",
        created_at=datetime.datetime.now(datetime.timezone.utc),
        updated_at=datetime.datetime.now(datetime.timezone.utc),
        priority=TicketPriority.MEDIUM,
        metadata={}
    )


@pytest.fixture
def assigned_ticket():
    """Create a sample assigned ticket."""
    return Ticket(
        id="ticket-123",
        title="Help with Solana integration",
        description="I'm having trouble integrating my dApp with Solana.",
        user_id="user-456",
        status=TicketStatus.ASSIGNED,
        assigned_to="agent-789",
        created_at=datetime.datetime.now(datetime.timezone.utc),
        updated_at=datetime.datetime.now(datetime.timezone.utc),
        priority=TicketPriority.MEDIUM,
        metadata={}
    )


# --------------------------
# Ticket Creation Tests
# --------------------------

@pytest.mark.asyncio
async def test_get_or_create_ticket_new(ticket_service, mock_repository):
    """Test creating a new ticket when no active ticket exists."""
    # Configure mock
    mock_repository.get_active_for_user.return_value = None

    # Execute
    ticket = await ticket_service.get_or_create_ticket(
        user_id="user-456",
        query="How do I integrate with Solana?"
    )

    # Assertions
    assert ticket is not None
    assert ticket.user_id == "user-456"
    assert ticket.title == "How do I integrate with Solana?"
    assert ticket.status == TicketStatus.NEW
    mock_repository.create.assert_called_once()


@pytest.mark.asyncio
async def test_get_or_create_ticket_existing(ticket_service, mock_repository, sample_ticket):
    """Test retrieving an existing active ticket."""
    # Configure mock to return an existing ticket
    mock_repository.get_active_for_user.return_value = sample_ticket

    # Execute
    ticket = await ticket_service.get_or_create_ticket(
        user_id="user-456",
        query="Different query"
    )

    # Assertions
    assert ticket == sample_ticket
    mock_repository.create.assert_not_called()


@pytest.mark.asyncio
async def test_get_or_create_ticket_long_query(ticket_service, mock_repository):
    """Test creating a ticket with a long query that gets truncated."""
    # Configure mock
    mock_repository.get_active_for_user.return_value = None

    # Create a long query
    long_query = "This is a very long query that should be truncated " * 5

    # Execute
    ticket = await ticket_service.get_or_create_ticket(
        user_id="user-456",
        query=long_query
    )

    # Assertions
    assert len(ticket.title) <= 53  # 50 chars + "..."
    assert ticket.title.endswith("...")
    assert ticket.description == long_query
    mock_repository.create.assert_called_once()


@pytest.mark.asyncio
async def test_create_ticket_with_complexity(ticket_service, mock_repository):
    """Test creating a ticket with complexity metadata."""
    # Configure mock
    mock_repository.get_active_for_user.return_value = None

    # Complexity data
    complexity = {
        "level": 3,
        "topics": ["solana", "web3"],
        "estimated_time": "2 hours"
    }

    # Execute
    ticket = await ticket_service.get_or_create_ticket(
        user_id="user-456",
        query="How do I integrate with Solana?",
        complexity=complexity
    )

    # Assertions
    assert ticket.metadata.get("complexity") == complexity
    mock_repository.create.assert_called_once()
    created_ticket = mock_repository.create.call_args[0][0]
    assert created_ticket.metadata.get("complexity") == complexity


# --------------------------
# Ticket Retrieval Tests
# --------------------------

def test_get_ticket_by_id_exists(ticket_service, mock_repository, sample_ticket):
    """Test retrieving a ticket that exists."""
    # Configure mock
    mock_repository.get_by_id.return_value = sample_ticket

    # Execute
    result = ticket_service.get_ticket_by_id("ticket-123")

    # Assertions
    assert result == sample_ticket
    mock_repository.get_by_id.assert_called_once_with("ticket-123")


def test_get_ticket_by_id_not_found(ticket_service, mock_repository):
    """Test retrieving a ticket that doesn't exist."""
    # Configure mock
    mock_repository.get_by_id.return_value = None

    # Execute
    result = ticket_service.get_ticket_by_id("nonexistent-ticket")

    # Assertions
    assert result is None
    mock_repository.get_by_id.assert_called_once_with("nonexistent-ticket")


def test_get_tickets_by_user(ticket_service, mock_repository, sample_ticket):
    """Test getting tickets for a user."""
    # Configure mock
    mock_repository.get_by_user.return_value = [sample_ticket]

    # Execute
    tickets = ticket_service.get_tickets_by_user("user-456")

    # Assertions
    assert len(tickets) == 1
    assert tickets[0] == sample_ticket
    mock_repository.get_by_user.assert_called_once_with("user-456", 20)


def test_get_tickets_by_status(ticket_service, mock_repository, sample_ticket):
    """Test getting tickets by status."""
    # Configure mock
    mock_repository.get_by_status.return_value = [sample_ticket]

    # Execute
    tickets = ticket_service.get_tickets_by_status(TicketStatus.NEW)

    # Assertions
    assert len(tickets) == 1
    assert tickets[0] == sample_ticket
    mock_repository.get_by_status.assert_called_once_with(TicketStatus.NEW, 50)


# --------------------------
# Ticket Update Tests
# --------------------------

def test_update_ticket_status(ticket_service, mock_repository):
    """Test updating a ticket's status."""
    # Execute
    success = ticket_service.update_ticket_status(
        "ticket-123",
        TicketStatus.IN_PROGRESS,
        priority=TicketPriority.HIGH
    )

    # Assertions
    assert success is True
    mock_repository.update.assert_called_once()

    # Check that both status and additional updates were included
    update_args = mock_repository.update.call_args[0][1]
    assert update_args["status"] == TicketStatus.IN_PROGRESS
    assert update_args["priority"] == TicketPriority.HIGH
    assert "updated_at" in update_args


def test_mark_ticket_resolved(ticket_service, mock_repository):
    """Test marking a ticket as resolved."""
    # Setup resolution data
    resolution_data = {
        "confidence": 0.92,
        "reasoning": "Issue was fixed by updating dependencies"
    }

    # Execute
    success = ticket_service.mark_ticket_resolved(
        "ticket-123", resolution_data)

    # Assertions
    assert success is True
    mock_repository.update.assert_called_once()

    # Check resolution data was included
    update_args = mock_repository.update.call_args[0][1]
    assert update_args["status"] == TicketStatus.RESOLVED
    assert "resolved_at" in update_args
    assert "updated_at" in update_args
    assert update_args["metadata"]["resolution_confidence"] == 0.92
    assert update_args["metadata"]["resolution_reasoning"] == "Issue was fixed by updating dependencies"


# --------------------------
# Ticket Notes Tests
# --------------------------

def test_add_note_to_ticket(ticket_service, mock_repository):
    """Test adding a note to a ticket."""
    # Execute
    success = ticket_service.add_note_to_ticket(
        "ticket-123",
        "This is a test note",
        "agent",
        "agent-789"
    )

    # Assertions
    assert success is True
    mock_repository.add_note.assert_called_once()

    # Check note properties
    note = mock_repository.add_note.call_args[0][1]
    assert note.content == "This is a test note"
    assert note.type == "agent"
    assert note.created_by == "agent-789"
    assert note.id is not None
    assert note.timestamp is not None


def test_add_system_note_to_ticket(ticket_service, mock_repository):
    """Test adding a system note to a ticket."""
    # Execute
    success = ticket_service.add_note_to_ticket(
        "ticket-123",
        "System generated note"
    )

    # Assertions
    assert success is True
    note = mock_repository.add_note.call_args[0][1]
    assert note.type == "system"
    assert note.created_by is None


# --------------------------
# Ticket Assignment Tests
# --------------------------

def test_assign_ticket(ticket_service, mock_repository):
    """Test assigning a ticket to an agent."""
    # Execute
    success = ticket_service.assign_ticket("ticket-123", "agent-789")

    # Assertions
    assert success is True
    mock_repository.update.assert_called_once()

    # Check update data
    update_args = mock_repository.update.call_args[0][1]
    assert update_args["assigned_to"] == "agent-789"
    assert update_args["status"] == TicketStatus.ASSIGNED
    assert "updated_at" in update_args

    # Check that a note was added
    mock_repository.add_note.assert_called_once()
    note_args = mock_repository.add_note.call_args[0]
    # Fix: Check the content attribute of the TicketNote object
    assert "agent-789" in note_args[1].content


def test_assign_ticket_update_fails(ticket_service, mock_repository):
    """Test when ticket assignment update fails."""
    # Configure mock
    mock_repository.update.return_value = False

    # Execute
    success = ticket_service.assign_ticket("ticket-123", "agent-789")

    # Assertions
    assert success is False
    mock_repository.add_note.assert_not_called()


# --------------------------
# Ticket Closure Tests
# --------------------------

def test_close_ticket_with_reason(ticket_service, mock_repository):
    """Test closing a ticket with a reason."""
    # Execute
    success = ticket_service.close_ticket(
        "ticket-123",
        "Issue resolved after customer upgraded their SDK"
    )

    # Assertions
    assert success is True
    mock_repository.update.assert_called_once()

    # Check update data
    update_args = mock_repository.update.call_args[0][1]
    assert update_args["status"] == TicketStatus.CLOSED
    assert "closed_at" in update_args
    assert "updated_at" in update_args
    assert update_args["metadata.closure_reason"] == "Issue resolved after customer upgraded their SDK"

    # Check that a note was added
    mock_repository.add_note.assert_called_once()
    note_args = mock_repository.add_note.call_args[0]
    # Fix: Check the content attribute of the TicketNote object
    assert "Issue resolved" in note_args[1].content


def test_close_ticket_without_reason(ticket_service, mock_repository):
    """Test closing a ticket without a reason."""
    # Execute
    success = ticket_service.close_ticket("ticket-123")

    # Assertions
    assert success is True
    mock_repository.update.assert_called_once()

    # Check update data
    update_args = mock_repository.update.call_args[0][1]
    assert update_args["status"] == TicketStatus.CLOSED

    # Check that no note was added (since no reason was provided)
    mock_repository.add_note.assert_not_called()


def test_close_ticket_update_fails(ticket_service, mock_repository):
    """Test when ticket closure update fails."""
    # Configure mock
    mock_repository.update.return_value = False

    # Execute
    success = ticket_service.close_ticket("ticket-123", "Some reason")

    # Assertions
    assert success is False
    mock_repository.add_note.assert_not_called()

# --------------------------
# Stalled Ticket Tests
# --------------------------


def test_find_stalled_tickets_default_timeout(ticket_service, mock_repository, sample_ticket):
    """Test finding stalled tickets with default timeout."""
    # Create a sample stalled ticket
    stalled_ticket = sample_ticket.model_copy()
    stalled_ticket.status = TicketStatus.IN_PROGRESS
    stalled_ticket.updated_at = datetime.datetime.now(
        datetime.timezone.utc) - datetime.timedelta(days=2)  # 2 days old

    # Configure mock repository
    mock_repository.find_tickets_by_criteria.return_value = [stalled_ticket]

    # Execute
    stalled_tickets = ticket_service.find_stalled_tickets()

    # Assertions
    assert len(stalled_tickets) == 1
    assert stalled_tickets[0] == stalled_ticket

    # Verify repository was called with correct parameters
    mock_repository.find_tickets_by_criteria.assert_called_once()
    call_args = mock_repository.find_tickets_by_criteria.call_args[1]

    # Check that active statuses were included
    assert TicketStatus.NEW.value in call_args['status_in']
    assert TicketStatus.ASSIGNED.value in call_args['status_in']
    assert TicketStatus.IN_PROGRESS.value in call_args['status_in']
    assert TicketStatus.WAITING_FOR_USER.value in call_args['status_in']

    # Check that the cutoff time is approximately 24 hours ago (default timeout)
    cutoff_time = call_args['updated_before']
    time_diff = datetime.datetime.now(datetime.timezone.utc) - cutoff_time
    assert 1439 <= time_diff.total_seconds() / 60 <= 1441  # ~24 hours in minutes


def test_find_stalled_tickets_custom_timeout(ticket_service, mock_repository):
    """Test finding stalled tickets with custom timeout."""
    # Configure mock repository
    mock_repository.find_tickets_by_criteria.return_value = []

    # Execute with 30 minute timeout
    ticket_service.find_stalled_tickets(timeout_minutes=30)

    # Verify repository was called with correct parameters
    mock_repository.find_tickets_by_criteria.assert_called_once()
    call_args = mock_repository.find_tickets_by_criteria.call_args[1]

    # Check that the cutoff time is approximately 30 minutes ago
    cutoff_time = call_args['updated_before']
    time_diff = datetime.datetime.now(datetime.timezone.utc) - cutoff_time
    assert 29 <= time_diff.total_seconds() / 60 <= 31  # ~30 minutes


def test_find_stalled_tickets_none_found(ticket_service, mock_repository):
    """Test when no stalled tickets are found."""
    # Configure mock repository to return empty list
    mock_repository.find_tickets_by_criteria.return_value = []

    # Execute
    stalled_tickets = ticket_service.find_stalled_tickets()

    # Assertions
    assert len(stalled_tickets) == 0
    mock_repository.find_tickets_by_criteria.assert_called_once()


def test_find_stalled_tickets_multiple_statuses(ticket_service, mock_repository, sample_ticket):
    """Test finding stalled tickets with multiple statuses."""
    # Create sample stalled tickets with different statuses
    ticket1 = sample_ticket.model_copy()
    ticket1.id = "ticket-1"
    ticket1.status = TicketStatus.NEW
    ticket1.updated_at = datetime.datetime.now(
        datetime.timezone.utc) - datetime.timedelta(days=3)

    ticket2 = sample_ticket.model_copy()
    ticket2.id = "ticket-2"
    ticket2.status = TicketStatus.ASSIGNED
    ticket2.updated_at = datetime.datetime.now(
        datetime.timezone.utc) - datetime.timedelta(days=2)

    ticket3 = sample_ticket.model_copy()
    ticket3.id = "ticket-3"
    ticket3.status = TicketStatus.WAITING_FOR_USER
    ticket3.updated_at = datetime.datetime.now(
        datetime.timezone.utc) - datetime.timedelta(days=4)

    # Configure mock repository
    mock_repository.find_tickets_by_criteria.return_value = [
        ticket1, ticket2, ticket3]

    # Execute
    stalled_tickets = ticket_service.find_stalled_tickets()

    # Assertions
    assert len(stalled_tickets) == 3
    ticket_ids = [t.id for t in stalled_tickets]
    assert "ticket-1" in ticket_ids
    assert "ticket-2" in ticket_ids
    assert "ticket-3" in ticket_ids


def test_find_stalled_tickets_zero_timeout(ticket_service, mock_repository):
    """Test finding stalled tickets with zero timeout (edge case)."""
    # Configure mock repository to return empty list
    mock_repository.find_tickets_by_criteria.return_value = []

    # Execute with 0 minute timeout (all tickets would be considered stalled)
    stalled_tickets = ticket_service.find_stalled_tickets(timeout_minutes=0)

    # Assertions
    assert stalled_tickets == []

    # Verify repository was called with correct parameters
    mock_repository.find_tickets_by_criteria.assert_called_once()
    call_args = mock_repository.find_tickets_by_criteria.call_args[1]

    # Check that the cutoff time is approximately current time (0 minutes ago)
    cutoff_time = call_args['updated_before']
    time_diff = datetime.datetime.now(datetime.timezone.utc) - cutoff_time
    assert time_diff.total_seconds() < 2  # Should be less than 2 seconds difference

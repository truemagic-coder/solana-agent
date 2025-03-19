"""
Tests for the MongoTicketRepository implementation.

This module contains unit tests for the MongoDB-based ticket repository.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta
import uuid

from solana_agent.repositories.ticket import MongoTicketRepository
from solana_agent.domains import Ticket, TicketNote, TicketStatus, TicketPriority


@pytest.fixture
def mock_db_adapter():
    """Create a mock database adapter."""
    adapter = Mock()
    adapter.create_collection = Mock()
    adapter.create_index = Mock()
    adapter.insert_one = Mock(return_value="mock_id")
    adapter.find_one = Mock()
    adapter.find = Mock()
    adapter.update_one = Mock(return_value=True)
    adapter.count_documents = Mock(return_value=5)
    return adapter


@pytest.fixture
def ticket_repo(mock_db_adapter):
    """Create a repository with mocked database adapter."""
    return MongoTicketRepository(mock_db_adapter)


@pytest.fixture
def sample_ticket():
    """Create a sample ticket for testing."""
    return Ticket(
        id=f"ticket_{str(uuid.uuid4())[:8]}",
        title="Fix login page issue",
        description="Users are experiencing intermittent login failures",
        user_id="user123",
        status=TicketStatus.NEW,
        priority=TicketPriority.MEDIUM,
        assigned_to=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        tags=["login", "authentication", "bug"],
        metadata={"browser": "Chrome", "version": "112.0.5615.49"}
    )


@pytest.fixture
def sample_note():
    """Create a sample ticket note for testing."""
    # Updated to match the domain model
    return TicketNote(
        id=f"note_{str(uuid.uuid4())[:8]}",
        content="Investigated issue, found potential cause in auth service",
        type="agent",  # Changed from author_id
        created_by="agent456",  # Changed from author_id
        timestamp=datetime.now(),  # Changed from created_at
        metadata={"source": "investigation"}  # Added required field
    )


# Fix the sample_subtask fixture by removing setattr calls
@pytest.fixture
def sample_subtask():
    """Create a sample subtask ticket for testing."""
    # Modified to use metadata for subtask relationships
    subtask = Ticket(
        id=f"subtask_{str(uuid.uuid4())[:8]}",
        title="Check auth service logs",
        description="Analyze authentication service logs during failure periods",
        user_id="user123",
        status=TicketStatus.NEW,
        priority=TicketPriority.HIGH,
        assigned_to="agent789",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        tags=["subtask", "investigation"],
        metadata={
            "is_subtask": True,  # Values already stored in metadata
            "parent_id": "parent_ticket_123"
        }
    )
    return subtask


class TestMongoTicketRepository:
    """Tests for the MongoTicketRepository implementation."""

    def test_init(self, mock_db_adapter):
        """Test repository initialization."""
        repo = MongoTicketRepository(mock_db_adapter)

        # Verify collection is created
        mock_db_adapter.create_collection.assert_called_once_with("tickets")

        # Verify indexes are created
        assert mock_db_adapter.create_index.call_count == 4
        mock_db_adapter.create_index.assert_any_call(
            "tickets", [("user_id", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "tickets", [("status", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "tickets", [("assigned_to", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "tickets", [("created_at", -1)])

    def test_create(self, ticket_repo, mock_db_adapter, sample_ticket):
        """Test creating a ticket."""
        # Create ticket
        result_id = ticket_repo.create(sample_ticket)

        # Verify result
        assert result_id == "mock_id"

        # Verify DB operation
        mock_db_adapter.insert_one.assert_called_once()
        collection, data = mock_db_adapter.insert_one.call_args[0]
        assert collection == "tickets"
        assert data["id"] == sample_ticket.id
        assert data["title"] == "Fix login page issue"
        assert data["user_id"] == "user123"
        assert data["status"] == TicketStatus.NEW
        assert isinstance(data["created_at"], datetime)

    def test_get_by_id_found(self, ticket_repo, mock_db_adapter, sample_ticket):
        """Test retrieving an existing ticket."""
        ticket_id = sample_ticket.id

        # Configure mock to return the ticket
        mock_db_adapter.find_one.return_value = sample_ticket.model_dump()

        # Get the ticket
        result = ticket_repo.get_by_id(ticket_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once_with(
            "tickets", {"id": ticket_id})

        # Verify result
        assert result is not None
        assert result.id == ticket_id
        assert result.title == "Fix login page issue"
        assert result.status == TicketStatus.NEW
        assert result.user_id == "user123"

    def test_get_by_id_not_found(self, ticket_repo, mock_db_adapter):
        """Test retrieving a non-existent ticket."""
        ticket_id = "nonexistent"

        # Configure mock to return None
        mock_db_adapter.find_one.return_value = None

        # Get the ticket
        result = ticket_repo.get_by_id(ticket_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once_with(
            "tickets", {"id": ticket_id})

        # Verify result
        assert result is None

    def test_get_active_for_user_found(self, ticket_repo, mock_db_adapter, sample_ticket):
        """Test getting an active ticket for a user."""
        user_id = "user123"

        # Configure mock to return the ticket
        mock_db_adapter.find_one.return_value = sample_ticket.model_dump()

        # Get the active ticket
        result = ticket_repo.get_active_for_user(user_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once()
        collection, query = mock_db_adapter.find_one.call_args[0]
        assert collection == "tickets"
        assert query["user_id"] == user_id
        assert "$in" in query["status"]

        # FIX: The values are strings, not enum objects
        status_values = query["status"]["$in"]  # These are strings, not enums
        assert TicketStatus.NEW.value in status_values
        assert TicketStatus.ASSIGNED.value in status_values
        assert TicketStatus.IN_PROGRESS.value in status_values
        assert TicketStatus.WAITING_FOR_USER.value in status_values
        assert TicketStatus.WAITING_FOR_HUMAN.value in status_values
        assert TicketStatus.PLANNING.value in status_values

        # Verify result
        assert result is not None
        assert result.user_id == user_id
        assert result.id == sample_ticket.id

    def test_get_active_for_user_not_found(self, ticket_repo, mock_db_adapter):
        """Test getting an active ticket when none exists."""
        user_id = "user456"

        # Configure mock to return None
        mock_db_adapter.find_one.return_value = None

        # Get the active ticket
        result = ticket_repo.get_active_for_user(user_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once()

        # Verify result
        assert result is None

    def test_find_basic(self, ticket_repo, mock_db_adapter, sample_ticket):
        """Test finding tickets with basic query."""
        query = {"status": TicketStatus.NEW}

        # Configure mock to return list of tickets
        mock_db_adapter.find.return_value = [sample_ticket.model_dump()]

        # Find tickets
        results = ticket_repo.find(query)

        # Verify DB query
        mock_db_adapter.find.assert_called_once_with(
            "tickets", query, sort=None, limit=0)

        # Verify results
        assert len(results) == 1
        assert results[0].id == sample_ticket.id
        assert results[0].title == "Fix login page issue"

    def test_find_with_sort_and_limit(self, ticket_repo, mock_db_adapter, sample_ticket):
        """Test finding tickets with sort and limit."""
        query = {"assigned_to": "agent456"}
        sort_by = "-priority"  # Sort by priority descending
        limit = 10

        # Configure mock to return list of tickets
        mock_db_adapter.find.return_value = [sample_ticket.model_dump()]

        # Find tickets
        results = ticket_repo.find(query, sort_by=sort_by, limit=limit)

        # Verify DB query
        mock_db_adapter.find.assert_called_once_with(
            "tickets", query, sort=[("priority", -1)], limit=limit)

        # Verify results
        assert len(results) == 1
        assert results[0].id == sample_ticket.id

    def test_count(self, ticket_repo, mock_db_adapter):
        """Test counting tickets."""
        query = {"status": TicketStatus.CLOSED}

        # Count tickets
        result = ticket_repo.count(query)

        # Verify DB operation
        mock_db_adapter.count_documents.assert_called_once_with(
            "tickets", query)

        # Verify result
        assert result == 5  # Mock returns 5

    @pytest.mark.asyncio
    async def test_find_stalled_tickets(self, ticket_repo, mock_db_adapter, sample_ticket):
        """Test finding stalled tickets."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        # Use valid statuses from the domain model
        statuses = [TicketStatus.IN_PROGRESS, TicketStatus.WAITING_FOR_USER]

        # Configure mock to return list of tickets
        mock_db_adapter.find.return_value = [sample_ticket.model_dump()]

        # Find stalled tickets
        results = await ticket_repo.find_stalled_tickets(cutoff_time, statuses)

        # Verify DB query
        mock_db_adapter.find.assert_called_once()
        collection, query = mock_db_adapter.find.call_args[0]
        assert collection == "tickets"
        assert "$in" in query["status"]
        assert query["updated_at"]["$lt"] == cutoff_time

        # Verify results
        assert len(results) == 1
        assert results[0].id == sample_ticket.id

    def test_add_note(self, ticket_repo, mock_db_adapter, sample_ticket, sample_note):
        """Test adding a note to a ticket."""
        ticket_id = sample_ticket.id

        # Add note
        result = ticket_repo.add_note(ticket_id, sample_note)

        # Verify result
        assert result is True

        # Verify DB operation
        mock_db_adapter.update_one.assert_called_once()
        collection, query, update = mock_db_adapter.update_one.call_args[0]
        assert collection == "tickets"
        assert query == {"id": ticket_id}
        assert "$push" in update
        assert "notes" in update["$push"]
        assert update["$push"]["notes"]["id"] == sample_note.id
        assert update["$push"]["notes"]["content"] == "Investigated issue, found potential cause in auth service"
        assert update["$push"]["notes"]["type"] == "agent"  # Check new field
        assert "timestamp" in update["$push"]["notes"]  # Check renamed field

    def test_get_subtasks(self, ticket_repo, mock_db_adapter, sample_subtask):
        """Test getting subtasks for a parent ticket."""
        parent_id = "parent_ticket_123"

        # Configure mock to return list of subtasks
        mock_db_adapter.find.return_value = [sample_subtask.model_dump()]

        # Get subtasks
        results = ticket_repo.get_subtasks(parent_id)

        # Verify DB query - now using metadata fields
        mock_db_adapter.find.assert_called_once_with(
            "tickets",
            {"metadata.parent_id": parent_id, "metadata.is_subtask": True}
        )

        # Verify results
        assert len(results) == 1
        assert results[0].id == sample_subtask.id
        assert results[0].title == "Check auth service logs"
        assert "is_subtask" in results[0].metadata
        assert "parent_id" in results[0].metadata

    def test_get_parent_found(self, ticket_repo, mock_db_adapter, sample_ticket, sample_subtask):
        """Test getting parent ticket for a subtask."""
        subtask_id = sample_subtask.id
        parent_id = sample_subtask.metadata["parent_id"]  # Get from metadata

        # Configure mocks for the two find_one calls
        mock_db_adapter.find_one.side_effect = [
            sample_subtask.model_dump(),
            sample_ticket.model_dump()
        ]

        # Get parent
        result = ticket_repo.get_parent(subtask_id)

        # Verify DB queries
        assert mock_db_adapter.find_one.call_count == 2
        mock_db_adapter.find_one.assert_any_call("tickets", {"id": subtask_id})
        mock_db_adapter.find_one.assert_any_call("tickets", {"id": parent_id})

        # Verify result
        assert result is not None
        assert result.id == sample_ticket.id

    def test_get_parent_no_parent_id(self, ticket_repo, mock_db_adapter, sample_ticket):
        """Test getting parent when subtask has no parent_id."""
        # Create a ticket without parent_id metadata
        subtask = sample_ticket.model_copy()
        subtask.metadata = {}  # No parent_id in metadata
        subtask_id = subtask.id

        # Configure mock to return a subtask without parent_id
        mock_db_adapter.find_one.return_value = subtask.model_dump()

        # Get parent
        result = ticket_repo.get_parent(subtask_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once_with(
            "tickets", {"id": subtask_id})

        # Verify result
        assert result is None

    def test_get_parent_subtask_not_found(self, ticket_repo, mock_db_adapter):
        """Test getting parent when subtask doesn't exist."""
        subtask_id = "nonexistent"

        # Configure mock to return None
        mock_db_adapter.find_one.return_value = None

        # Get parent
        result = ticket_repo.get_parent(subtask_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once_with(
            "tickets", {"id": subtask_id})

        # Verify result
        assert result is None


def test_find_tickets_by_criteria_status_only(ticket_repo, mock_db_adapter, sample_ticket):
    """Test finding tickets by status criteria only."""
    # Setup
    statuses = [TicketStatus.NEW.value, TicketStatus.IN_PROGRESS.value]
    mock_db_adapter.find.return_value = [sample_ticket.model_dump()]

    # Execute
    results = ticket_repo.find_tickets_by_criteria(status_in=statuses)

    # Verify DB query
    mock_db_adapter.find.assert_called_once()
    collection, query = mock_db_adapter.find.call_args[0]
    assert collection == "tickets"
    assert "status" in query
    assert "$in" in query["status"]
    assert query["status"]["$in"] == statuses
    assert "updated_at" not in query

    # Verify results
    assert len(results) == 1
    assert isinstance(results[0], Ticket)
    assert results[0].id == sample_ticket.id


def test_find_tickets_by_criteria_updated_before_only(ticket_repo, mock_db_adapter, sample_ticket):
    """Test finding tickets by update time criteria only."""
    # Setup
    cutoff_time = datetime.now() - timedelta(hours=24)
    mock_db_adapter.find.return_value = [sample_ticket.model_dump()]

    # Execute
    results = ticket_repo.find_tickets_by_criteria(updated_before=cutoff_time)

    # Verify DB query
    mock_db_adapter.find.assert_called_once()
    collection, query = mock_db_adapter.find.call_args[0]
    assert collection == "tickets"
    assert "status" not in query
    assert "updated_at" in query
    assert "$lt" in query["updated_at"]
    assert query["updated_at"]["$lt"] == cutoff_time

    # Verify results
    assert len(results) == 1
    assert isinstance(results[0], Ticket)
    assert results[0].id == sample_ticket.id


def test_find_tickets_by_criteria_both_filters(ticket_repo, mock_db_adapter, sample_ticket):
    """Test finding tickets using both status and time criteria."""
    # Setup
    statuses = [TicketStatus.ASSIGNED.value,
                TicketStatus.WAITING_FOR_USER.value]
    cutoff_time = datetime.now() - timedelta(hours=48)
    mock_db_adapter.find.return_value = [sample_ticket.model_dump()]

    # Execute
    results = ticket_repo.find_tickets_by_criteria(
        status_in=statuses,
        updated_before=cutoff_time
    )

    # Verify DB query
    mock_db_adapter.find.assert_called_once()
    collection, query = mock_db_adapter.find.call_args[0]
    assert collection == "tickets"
    assert "status" in query
    assert "$in" in query["status"]
    assert query["status"]["$in"] == statuses
    assert "updated_at" in query
    assert "$lt" in query["updated_at"]
    assert query["updated_at"]["$lt"] == cutoff_time

    # Verify results
    assert len(results) == 1
    assert isinstance(results[0], Ticket)
    assert results[0].id == sample_ticket.id


def test_find_tickets_by_criteria_no_matches(ticket_repo, mock_db_adapter):
    """Test finding tickets when no tickets match criteria."""
    # Setup
    statuses = [TicketStatus.STALLED.value]
    mock_db_adapter.find.return_value = []

    # Execute
    results = ticket_repo.find_tickets_by_criteria(status_in=statuses)

    # Verify DB query
    mock_db_adapter.find.assert_called_once()

    # Verify results
    assert len(results) == 0
    assert isinstance(results, list)


def test_find_tickets_by_criteria_no_filters(ticket_repo, mock_db_adapter, sample_ticket):
    """Test finding tickets with no filter criteria (should return all)."""
    # Setup
    mock_db_adapter.find.return_value = [
        sample_ticket.model_dump(),
        sample_ticket.model_dump()
    ]

    # Execute
    results = ticket_repo.find_tickets_by_criteria()

    # Verify DB query
    mock_db_adapter.find.assert_called_once()
    collection, query = mock_db_adapter.find.call_args[0]
    assert collection == "tickets"
    assert query == {}  # Empty query should find all tickets

    # Verify results
    assert len(results) == 2
    assert all(isinstance(result, Ticket) for result in results)

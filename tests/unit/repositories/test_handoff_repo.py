"""
Tests for handoff repository implementations.

This module contains unit tests for MongoHandoffRepository.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta

from solana_agent.repositories.handoff import MongoHandoffRepository
from solana_agent.domains import Handoff


@pytest.fixture
def mock_db_adapter():
    """Create a mock database adapter."""
    adapter = Mock()
    adapter.create_collection = Mock()
    adapter.create_index = Mock()
    adapter.insert_one = Mock(return_value="mock_id")
    adapter.find = Mock()
    adapter.count_documents = Mock()
    return adapter


@pytest.fixture
def handoff_repository(mock_db_adapter):
    """Create a repository with mocked database adapter."""
    return MongoHandoffRepository(mock_db_adapter)


@pytest.fixture
def sample_handoff():
    """Create a sample handoff for testing."""
    return Handoff(
        from_agent="ai_assistant",
        to_agent="human_expert",
        ticket_id="ticket123",
        reason="Technical question requires specialist knowledge",
        timestamp=datetime.now(),
        successful=True,
        notes="Handed off to specialist for detailed database optimization advice"
    )


@pytest.fixture
def sample_handoffs():
    """Create a list of sample handoffs for testing."""
    base_time = datetime.now()
    return [
        # Handoff from AI to human expert
        {
            "id": "handoff1",
            "from_agent": "ai_assistant",
            "to_agent": "human_expert",
            "ticket_id": "ticket123",
            "reason": "Technical question",
            "timestamp": (base_time - timedelta(days=1)).isoformat(),
            "successful": True,
            "notes": "Requires database expertise"
        },
        # Handoff from AI to different human expert
        {
            "id": "handoff2",
            "from_agent": "ai_assistant",
            "to_agent": "product_manager",
            "ticket_id": "ticket456",
            "reason": "Feature request",
            "timestamp": base_time.isoformat(),
            "successful": True,
            "notes": "Customer requested roadmap information"
        },
        # Handoff from human to AI
        {
            "id": "handoff3",
            "from_agent": "human_expert",
            "to_agent": "ai_assistant",
            "ticket_id": "ticket789",
            "reason": "Question answered",
            "timestamp": (base_time + timedelta(days=1)).isoformat(),
            "successful": True,
            "notes": "Customer needs documentation links"
        }
    ]


class TestMongoHandoffRepository:
    """Tests for the MongoHandoffRepository implementation."""

    def test_init(self, mock_db_adapter):
        """Test repository initialization."""
        repo = MongoHandoffRepository(mock_db_adapter)

        # Verify collection is created
        mock_db_adapter.create_collection.assert_called_once_with("handoffs")

        # Verify indexes are created
        assert mock_db_adapter.create_index.call_count == 3
        mock_db_adapter.create_index.assert_any_call(
            "handoffs", [("from_agent", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "handoffs", [("to_agent", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "handoffs", [("timestamp", 1)])

    def test_record_handoff(self, handoff_repository, mock_db_adapter, sample_handoff):
        """Test recording a handoff."""
        # Record the handoff
        result_id = handoff_repository.record(sample_handoff)

        # Verify result
        assert result_id == "mock_id"

        # Verify DB operation
        mock_db_adapter.insert_one.assert_called_once()
        collection, data = mock_db_adapter.insert_one.call_args[0]
        assert collection == "handoffs"
        assert data["from_agent"] == "ai_assistant"
        assert data["to_agent"] == "human_expert"
        assert data["ticket_id"] == "ticket123"
        assert data["reason"] == "Technical question requires specialist knowledge"
        assert "timestamp" in data
        assert data["successful"] is True
        assert data["notes"] == "Handed off to specialist for detailed database optimization advice"

    def test_find_for_agent_no_date_filter(self, handoff_repository, mock_db_adapter, sample_handoffs):
        """Test finding handoffs for an agent with no date filter."""
        agent_name = "ai_assistant"

        # Configure mock to return filtered data
        filtered_handoffs = [
            h for h in sample_handoffs if h["from_agent"] == agent_name]
        mock_db_adapter.find.return_value = filtered_handoffs

        # Find handoffs
        results = handoff_repository.find_for_agent(agent_name)

        # Verify DB query
        mock_db_adapter.find.assert_called_once_with(
            "handoffs",
            {"from_agent": agent_name}
        )

        # Verify results
        assert len(results) == 2
        assert all(handoff.from_agent == agent_name for handoff in results)
        assert results[0].id == "handoff1"
        assert results[0].to_agent == "human_expert"
        assert results[1].id == "handoff2"
        assert results[1].to_agent == "product_manager"

    def test_find_for_agent_with_start_date(self, handoff_repository, mock_db_adapter, sample_handoffs):
        """Test finding handoffs for an agent with start date filter."""
        agent_name = "ai_assistant"
        start_date = datetime.now() - timedelta(hours=12)

        # Configure mock to return filtered data
        filtered_handoffs = [
            h for h in sample_handoffs
            if h["from_agent"] == agent_name and
            datetime.fromisoformat(h["timestamp"]) >= start_date
        ]
        mock_db_adapter.find.return_value = filtered_handoffs

        # Find handoffs
        results = handoff_repository.find_for_agent(
            agent_name, start_date=start_date)

        # Verify DB query
        mock_db_adapter.find.assert_called_once()
        collection, query = mock_db_adapter.find.call_args[0]
        assert collection == "handoffs"
        assert query["from_agent"] == agent_name
        assert query["timestamp"]["$gte"] == start_date
        assert "$lte" not in query["timestamp"]

        # Verify results
        assert len(results) == 1
        assert results[0].id == "handoff2"

    def test_find_for_agent_with_end_date(self, handoff_repository, mock_db_adapter, sample_handoffs):
        """Test finding handoffs for an agent with end date filter."""
        agent_name = "ai_assistant"
        end_date = datetime.now()

        # Configure mock to return filtered data
        filtered_handoffs = [
            h for h in sample_handoffs
            if h["from_agent"] == agent_name and
            datetime.fromisoformat(h["timestamp"]) <= end_date
        ]
        mock_db_adapter.find.return_value = filtered_handoffs

        # Find handoffs
        results = handoff_repository.find_for_agent(
            agent_name, end_date=end_date)

        # Verify DB query
        mock_db_adapter.find.assert_called_once()
        collection, query = mock_db_adapter.find.call_args[0]
        assert collection == "handoffs"
        assert query["from_agent"] == agent_name
        assert query["timestamp"]["$lte"] == end_date
        assert "$gte" not in query["timestamp"]

        # Verify results
        assert len(results) == 2
        assert results[0].id == "handoff1"
        assert results[1].id == "handoff2"

    def test_find_for_agent_with_date_range(self, handoff_repository, mock_db_adapter, sample_handoffs):
        """Test finding handoffs for an agent with both start and end date filters."""
        agent_name = "ai_assistant"
        start_date = datetime.now() - timedelta(hours=12)
        end_date = datetime.now() + timedelta(hours=12)

        # Configure mock to return filtered data
        filtered_handoffs = [
            h for h in sample_handoffs
            if h["from_agent"] == agent_name and
            datetime.fromisoformat(h["timestamp"]) >= start_date and
            datetime.fromisoformat(h["timestamp"]) <= end_date
        ]
        mock_db_adapter.find.return_value = filtered_handoffs

        # Find handoffs
        results = handoff_repository.find_for_agent(
            agent_name, start_date, end_date)

        # Verify DB query
        mock_db_adapter.find.assert_called_once()
        collection, query = mock_db_adapter.find.call_args[0]
        assert collection == "handoffs"
        assert query["from_agent"] == agent_name
        assert query["timestamp"]["$gte"] == start_date
        assert query["timestamp"]["$lte"] == end_date

        # Verify results
        assert len(results) == 1
        assert results[0].id == "handoff2"

    def test_find_for_agent_no_results(self, handoff_repository, mock_db_adapter):
        """Test finding handoffs when there are no matches."""
        agent_name = "nonexistent_agent"

        # Configure mock to return empty list
        mock_db_adapter.find.return_value = []

        # Find handoffs
        results = handoff_repository.find_for_agent(agent_name)

        # Verify DB query
        mock_db_adapter.find.assert_called_once_with(
            "handoffs",
            {"from_agent": agent_name}
        )

        # Verify empty results
        assert len(results) == 0

    def test_count_for_agent_no_date_filter(self, handoff_repository, mock_db_adapter):
        """Test counting handoffs for an agent with no date filter."""
        agent_name = "ai_assistant"

        # Configure mock to return count
        mock_db_adapter.count_documents.return_value = 2

        # Count handoffs
        count = handoff_repository.count_for_agent(agent_name)

        # Verify DB query
        mock_db_adapter.count_documents.assert_called_once_with(
            "handoffs",
            {"from_agent": agent_name}
        )

        # Verify count
        assert count == 2

    def test_count_for_agent_with_start_date(self, handoff_repository, mock_db_adapter):
        """Test counting handoffs for an agent with start date filter."""
        agent_name = "ai_assistant"
        start_date = datetime.now() - timedelta(hours=12)

        # Configure mock to return count
        mock_db_adapter.count_documents.return_value = 1

        # Count handoffs
        count = handoff_repository.count_for_agent(
            agent_name, start_date=start_date)

        # Verify DB query
        mock_db_adapter.count_documents.assert_called_once()
        collection, query = mock_db_adapter.count_documents.call_args[0]
        assert collection == "handoffs"
        assert query["from_agent"] == agent_name
        assert query["timestamp"]["$gte"] == start_date
        assert "$lte" not in query["timestamp"]

        # Verify count
        assert count == 1

    def test_count_for_agent_with_end_date(self, handoff_repository, mock_db_adapter):
        """Test counting handoffs for an agent with end date filter."""
        agent_name = "ai_assistant"
        end_date = datetime.now()

        # Configure mock to return count
        mock_db_adapter.count_documents.return_value = 2

        # Count handoffs
        count = handoff_repository.count_for_agent(
            agent_name, end_date=end_date)

        # Verify DB query
        mock_db_adapter.count_documents.assert_called_once()
        collection, query = mock_db_adapter.count_documents.call_args[0]
        assert collection == "handoffs"
        assert query["from_agent"] == agent_name
        assert query["timestamp"]["$lte"] == end_date
        assert "$gte" not in query["timestamp"]

        # Verify count
        assert count == 2

    def test_count_for_agent_with_date_range(self, handoff_repository, mock_db_adapter):
        """Test counting handoffs for an agent with both start and end date filters."""
        agent_name = "ai_assistant"
        start_date = datetime.now() - timedelta(hours=12)
        end_date = datetime.now() + timedelta(hours=12)

        # Configure mock to return count
        mock_db_adapter.count_documents.return_value = 1

        # Count handoffs
        count = handoff_repository.count_for_agent(
            agent_name, start_date, end_date)

        # Verify DB query
        mock_db_adapter.count_documents.assert_called_once()
        collection, query = mock_db_adapter.count_documents.call_args[0]
        assert collection == "handoffs"
        assert query["from_agent"] == agent_name
        assert query["timestamp"]["$gte"] == start_date
        assert query["timestamp"]["$lte"] == end_date

        # Verify count
        assert count == 1

    def test_count_for_agent_no_matches(self, handoff_repository, mock_db_adapter):
        """Test counting handoffs when there are no matches."""
        agent_name = "nonexistent_agent"

        # Configure mock to return zero
        mock_db_adapter.count_documents.return_value = 0

        # Count handoffs
        count = handoff_repository.count_for_agent(agent_name)

        # Verify DB query
        mock_db_adapter.count_documents.assert_called_once_with(
            "handoffs",
            {"from_agent": agent_name}
        )

        # Verify count
        assert count == 0

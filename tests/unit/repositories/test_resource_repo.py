"""
Tests for the MongoResourceRepository implementation.

This module contains unit tests for the MongoDB-based resource repository.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta
import uuid

from solana_agent.repositories.resource import MongoResourceRepository
from solana_agent.domains import Resource, ResourceBooking, ResourceType, ResourceStatus


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
    adapter.delete_one = Mock(return_value=True)
    return adapter


@pytest.fixture
def resource_repo(mock_db_adapter):
    """Create a repository with mocked database adapter."""
    return MongoResourceRepository(mock_db_adapter)


@pytest.fixture
def sample_resource():
    """Create a sample resource for testing."""
    return Resource(
        id=str(uuid.uuid4()),
        name="Conference Room A",
        resource_type=ResourceType.ROOM,
        description="Main conference room with video conferencing",
        capacity=10,
        location="Building 1, Floor 3",
        # Changed from "features"
        tags=["projector", "whiteboard", "video_conference"],
        # Changed from "metadata"
        attributes={"reservation_policy": "first_come_first_served"}
    )


@pytest.fixture
def sample_booking():
    """Create a sample booking for testing."""
    start_time = datetime.now() + timedelta(days=1)
    end_time = start_time + timedelta(hours=2)

    return ResourceBooking(
        id=str(uuid.uuid4()),
        resource_id="resource123",
        user_id="user456",
        title="Team Planning Session",  # Added required title field
        description="Quarterly planning meeting",
        start_time=start_time,
        end_time=end_time,
        notes="Team planning meeting",  # Changed from "purpose"
        created_at=datetime.now(),
        status="confirmed",
        purpose="Team planning meeting"
    )


class TestMongoResourceRepository:
    """Tests for the MongoResourceRepository implementation."""

    def test_init(self, mock_db_adapter):
        """Test repository initialization."""
        repo = MongoResourceRepository(mock_db_adapter)

        # Verify collections are created
        mock_db_adapter.create_collection.assert_any_call("resources")
        mock_db_adapter.create_collection.assert_any_call("resource_bookings")
        assert mock_db_adapter.create_collection.call_count == 2

        # Verify indexes are created
        assert mock_db_adapter.create_index.call_count == 9
        mock_db_adapter.create_index.assert_any_call(
            "resources", [("id", 1)], unique=True)
        mock_db_adapter.create_index.assert_any_call(
            "resources", [("type", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "resources", [("status", 1)])

        mock_db_adapter.create_index.assert_any_call(
            "resource_bookings", [("id", 1)], unique=True)
        mock_db_adapter.create_index.assert_any_call(
            "resource_bookings", [("resource_id", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "resource_bookings", [("user_id", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "resource_bookings", [("ticket_id", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "resource_bookings", [("start_time", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "resource_bookings", [("end_time", 1)])

    # Resource Tests
    def test_create_resource(self, resource_repo, mock_db_adapter, sample_resource):
        """Test creating a resource."""
        # Create resource
        result_id = resource_repo.create_resource(sample_resource)

        # Verify result
        assert result_id == "mock_id"

        # Verify DB operation
        mock_db_adapter.insert_one.assert_called_once()
        collection, data = mock_db_adapter.insert_one.call_args[0]
        assert collection == "resources"
        assert data["id"] == sample_resource.id
        assert data["name"] == "Conference Room A"
        assert data["resource_type"] == ResourceType.ROOM
        assert data["description"] == "Main conference room with video conferencing"

    def test_get_resource_found(self, resource_repo, mock_db_adapter, sample_resource):
        """Test retrieving an existing resource."""
        resource_id = sample_resource.id

        # Configure mock to return the resource
        mock_db_adapter.find_one.return_value = sample_resource.model_dump()

        # Get the resource
        result = resource_repo.get_resource(resource_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once_with(
            "resources", {"id": resource_id})

        # Verify result
        assert result is not None
        assert result.id == resource_id
        assert result.name == "Conference Room A"
        assert result.resource_type == ResourceType.ROOM
        assert result.capacity == 10

    def test_get_resource_not_found(self, resource_repo, mock_db_adapter):
        """Test retrieving a non-existent resource."""
        resource_id = "nonexistent"

        # Configure mock to return None (not found)
        mock_db_adapter.find_one.return_value = None

        # Get the resource
        result = resource_repo.get_resource(resource_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once_with(
            "resources", {"id": resource_id})

        # Verify result
        assert result is None

    def test_find_resources(self, resource_repo, mock_db_adapter, sample_resource):
        """Test finding resources by query."""
        query = {
            "resource_type": ResourceType.ROOM}  # Changed from "type" to "resource_type"

        # Configure mock to return list of resources
        mock_db_adapter.find.return_value = [sample_resource.model_dump()]

        # Find resources
        results = resource_repo.find_resources(query)

        # Verify DB query
        mock_db_adapter.find.assert_called_once_with("resources", query)

        # Verify results
        assert len(results) == 1
        assert results[0].id == sample_resource.id
        assert results[0].name == "Conference Room A"

    def test_update_resource(self, resource_repo, mock_db_adapter, sample_resource):
        """Test updating a resource."""
        resource_id = sample_resource.id
        updates = {"status": ResourceStatus.UNAVAILABLE,
                   "notes": "Under maintenance"}

        # Update the resource
        result = resource_repo.update_resource(resource_id, updates)

        # Verify result
        assert result is True

        # Verify DB operation
        mock_db_adapter.update_one.assert_called_once_with(
            "resources", {"id": resource_id}, {"$set": updates})

    def test_delete_resource(self, resource_repo, mock_db_adapter):
        """Test deleting a resource."""
        resource_id = "resource123"

        # Delete the resource
        result = resource_repo.delete_resource(resource_id)

        # Verify result
        assert result is True

        # Verify DB operation
        mock_db_adapter.delete_one.assert_called_once_with(
            "resources", {"id": resource_id})

    # Booking Tests
    def test_create_booking(self, resource_repo, mock_db_adapter, sample_booking):
        """Test creating a booking."""
        # Create booking
        result_id = resource_repo.create_booking(sample_booking)

        # Verify result
        assert result_id == "mock_id"

        # Verify DB operation
        mock_db_adapter.insert_one.assert_called_once()
        collection, data = mock_db_adapter.insert_one.call_args[0]
        assert collection == "resource_bookings"
        assert data["id"] == sample_booking.id
        assert data["resource_id"] == "resource123"
        assert data["user_id"] == "user456"
        # Should be converted to ISO string
        assert isinstance(data["start_time"], str)
        assert isinstance(data["end_time"], str)
        assert data["purpose"] == "Team planning meeting"

    def test_get_booking_found(self, resource_repo, mock_db_adapter, sample_booking):
        """Test retrieving an existing booking."""
        booking_id = sample_booking.id
        booking_dict = sample_booking.model_dump()

        # Convert datetime objects to strings to simulate MongoDB storage
        booking_dict["start_time"] = booking_dict["start_time"].isoformat()
        booking_dict["end_time"] = booking_dict["end_time"].isoformat()
        booking_dict["created_at"] = booking_dict["created_at"].isoformat()

        # Configure mock to return the booking
        mock_db_adapter.find_one.return_value = booking_dict

        # Get the booking
        result = resource_repo.get_booking(booking_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once_with(
            "resource_bookings", {"id": booking_id})

        # Verify result
        assert result is not None
        assert result.id == booking_id
        assert result.resource_id == "resource123"
        assert result.user_id == "user456"
        # Should be converted back to datetime
        assert isinstance(result.start_time, datetime)
        assert isinstance(result.end_time, datetime)

    def test_get_booking_not_found(self, resource_repo, mock_db_adapter):
        """Test retrieving a non-existent booking."""
        booking_id = "nonexistent"

        # Configure mock to return None (not found)
        mock_db_adapter.find_one.return_value = None

        # Get the booking
        result = resource_repo.get_booking(booking_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once_with(
            "resource_bookings", {"id": booking_id})

        # Verify result
        assert result is None

    def test_find_bookings(self, resource_repo, mock_db_adapter, sample_booking):
        """Test finding bookings by query."""
        query = {"resource_id": "resource123"}
        booking_dict = sample_booking.model_dump()

        # Convert datetime objects to strings to simulate MongoDB storage
        booking_dict["start_time"] = booking_dict["start_time"].isoformat()
        booking_dict["end_time"] = booking_dict["end_time"].isoformat()
        booking_dict["created_at"] = booking_dict["created_at"].isoformat()

        # Configure mock to return list of bookings
        mock_db_adapter.find.return_value = [booking_dict]

        # Find bookings
        results = resource_repo.find_bookings(query)

        # Verify DB query
        mock_db_adapter.find.assert_called_once_with(
            "resource_bookings", query)

        # Verify results
        assert len(results) == 1
        assert results[0].id == sample_booking.id
        assert results[0].resource_id == "resource123"
        assert results[0].user_id == "user456"
        assert isinstance(results[0].start_time, datetime)

    def test_get_resource_bookings(self, resource_repo, mock_db_adapter, sample_booking):
        """Test getting bookings for a resource in a time period."""
        resource_id = "resource123"
        start_time = datetime.now()
        end_time = start_time + timedelta(days=2)

        booking_dict = sample_booking.model_dump()
        booking_dict["start_time"] = booking_dict["start_time"].isoformat()
        booking_dict["end_time"] = booking_dict["end_time"].isoformat()
        booking_dict["created_at"] = booking_dict["created_at"].isoformat()

        # Configure mock to return list of bookings
        mock_db_adapter.find.return_value = [booking_dict]

        # Get resource bookings
        results = resource_repo.get_resource_bookings(
            resource_id, start_time, end_time)

        # Verify DB query was constructed correctly with time range logic
        mock_db_adapter.find.assert_called_once()
        collection, query = mock_db_adapter.find.call_args[0]

        assert collection == "resource_bookings"
        assert query["resource_id"] == resource_id
        assert "$or" in query
        assert len(query["$or"]) == 3  # Three time overlap conditions

        # Verify results
        assert len(results) == 1
        assert results[0].id == sample_booking.id
        assert results[0].resource_id == "resource123"

    def test_update_booking(self, resource_repo, mock_db_adapter):
        """Test updating a booking."""
        booking_id = "booking123"
        updates = {"purpose": "Updated meeting purpose"}

        # Update the booking
        result = resource_repo.update_booking(booking_id, updates)

        # Verify result
        assert result is True

        # Verify DB operation
        mock_db_adapter.update_one.assert_called_once()
        collection, query, update = mock_db_adapter.update_one.call_args[0]
        assert collection == "resource_bookings"
        assert query == {"id": booking_id}
        assert update["$set"]["purpose"] == "Updated meeting purpose"
        # Check that updated_at was added
        assert "updated_at" in update["$set"]

    def test_delete_booking(self, resource_repo, mock_db_adapter):
        """Test deleting a booking."""
        booking_id = "booking123"

        # Delete the booking
        result = resource_repo.delete_booking(booking_id)

        # Verify result
        assert result is True

        # Verify DB operation
        mock_db_adapter.delete_one.assert_called_once_with(
            "resource_bookings", {"id": booking_id})

"""
Tests for the ResourceService implementation.

This module tests the resource management service that handles
resource creation, booking, and availability tracking.
"""
import pytest
import uuid
import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from typing import Dict, List, Any, Optional

from solana_agent.services.resource import ResourceService
from solana_agent.domains import Resource, ResourceBooking, TimeWindow


@pytest.fixture
def mock_repository():
    """Create a mock repository for testing."""
    repo = Mock()

    # Setup common repository method mocks - make them return values directly rather than coroutines
    repo.create_resource = Mock(return_value="resource-123")
    repo.get_resource = Mock(return_value=None)
    repo.update_resource = Mock(return_value=True)
    repo.delete_resource = Mock(return_value=True)
    repo.list_resources = Mock(return_value=[])
    repo.find_resources = Mock(return_value=[])
    repo.create_booking = Mock(return_value="booking-123")
    repo.get_booking = Mock(return_value=None)
    repo.update_booking = Mock(return_value=True)
    repo.cancel_booking = Mock(return_value=True)
    # Use this instead of availability
    repo.get_resource_schedule = Mock(return_value=[])
    repo.get_resource_bookings = Mock(return_value=[])
    repo.get_user_bookings = Mock(return_value=[])

    # This is the critical fix - this private method needs to return False
    # so that booking conflicts don't prevent creation
    repo._has_conflicting_bookings = Mock(return_value=False)

    return repo


@pytest.fixture
def resource_service(mock_repository):
    """Create a resource service instance with mocked repository."""
    return ResourceService(mock_repository)


@pytest.fixture
def sample_resource_data():
    """Sample resource data for testing."""
    return {
        "name": "Conference Room A",
        "description": "Large conference room with projector",
        "location": "Floor 3, Building B",
        "capacity": 20,
        "tags": ["meeting", "presentation", "large"],
        "attributes": {
            "has_projector": True,
            "has_whiteboard": True,
            "has_videoconference": True
        },
        "availability_schedule": [
            {
                "day_of_week": "monday",
                "start_time": "09:00",
                "end_time": "17:00"
            },
            {
                "day_of_week": "tuesday",
                "start_time": "09:00",
                "end_time": "17:00"
            }
        ]
    }


@pytest.fixture
def sample_resource():
    """Create a sample resource object."""
    return Resource(
        id="resource-123",
        name="Conference Room A",
        resource_type="room",
        description="Large conference room with projector",
        location="Floor 3, Building B",
        capacity=20,
        tags=["meeting", "presentation", "large"],
        attributes={
            "has_projector": True,
            "has_whiteboard": True,
            "has_videoconference": True
        },
        availability_schedule=[
            {
                "day_of_week": "monday",
                "start_time": "09:00",
                "end_time": "17:00"
            },
            {
                "day_of_week": "tuesday",
                "start_time": "09:00",
                "end_time": "17:00"
            }
        ]
    )


@pytest.fixture
def sample_booking():
    """Create a sample booking object."""
    start_time = datetime.datetime.now(
        datetime.timezone.utc) + datetime.timedelta(days=1)
    end_time = start_time + datetime.timedelta(hours=2)

    return ResourceBooking(
        id="booking-123",
        resource_id="resource-123",
        user_id="user-456",
        title="Team Meeting",
        description="Weekly team sync",
        status="confirmed",
        start_time=start_time,
        end_time=end_time,
        notes="Please prepare your updates",
        created_at=datetime.datetime.now(datetime.timezone.utc)
    )


# --------------------------
# Resource Creation Tests
# --------------------------

@pytest.mark.asyncio
async def test_create_resource_success(resource_service, mock_repository, sample_resource_data):
    """Test successful resource creation."""
    # Execute function
    resource_id = await resource_service.create_resource(sample_resource_data, "room")

    # Assertions
    assert resource_id == "resource-123"
    mock_repository.create_resource.assert_called_once()

    # Verify created resource properties
    created_resource = mock_repository.create_resource.call_args[0][0]
    assert isinstance(created_resource, Resource)
    assert created_resource.name == "Conference Room A"
    assert created_resource.resource_type == "room"
    assert created_resource.capacity == 20
    assert len(created_resource.tags) == 3
    assert created_resource.attributes["has_projector"] is True


@pytest.mark.asyncio
async def test_create_resource_minimal_data(resource_service, mock_repository):
    """Test resource creation with minimal required data."""
    # Setup minimal data
    minimal_data = {"name": "Simple Room"}

    # Execute function
    resource_id = await resource_service.create_resource(minimal_data, "room")

    # Assertions
    assert resource_id == "resource-123"

    # Verify created resource properties
    created_resource = mock_repository.create_resource.call_args[0][0]
    assert created_resource.name == "Simple Room"
    assert created_resource.resource_type == "room"
    assert created_resource.description is None
    assert created_resource.capacity is None
    assert created_resource.tags == []
    assert created_resource.attributes == {}


@pytest.mark.asyncio
async def test_create_resource_with_empty_lists(resource_service, mock_repository):
    """Test resource creation with explicitly empty lists."""
    # Setup data with empty lists
    data_with_empty = {
        "name": "Empty Lists Room",
        "tags": [],
        "attributes": {}
    }

    # Execute function
    resource_id = await resource_service.create_resource(data_with_empty, "room")

    # Verify created resource properties
    created_resource = mock_repository.create_resource.call_args[0][0]
    assert created_resource.tags == []
    assert created_resource.attributes == {}


# --------------------------
# Resource Retrieval Tests
# --------------------------

@pytest.mark.asyncio
async def test_get_resource_exists(resource_service, mock_repository, sample_resource):
    """Test retrieving an existing resource."""
    # Configure mock
    mock_repository.get_resource.return_value = sample_resource

    # Execute function
    result = await resource_service.get_resource("resource-123")

    # Assertions
    assert result == sample_resource
    mock_repository.get_resource.assert_called_once_with("resource-123")


@pytest.mark.asyncio
async def test_get_resource_not_found(resource_service, mock_repository):
    """Test retrieving a non-existent resource."""
    # Configure mock to return None (resource not found)
    mock_repository.get_resource.return_value = None

    # Execute function
    result = await resource_service.get_resource("nonexistent-id")

    # Assertions
    assert result is None
    mock_repository.get_resource.assert_called_once_with("nonexistent-id")


@pytest.mark.asyncio
async def test_list_resources(resource_service, mock_repository, sample_resource):
    """Test retrieving all resources."""
    # Configure mock
    mock_repository.list_resources.return_value = [sample_resource]

    # Execute function
    results = await resource_service.list_resources()

    # Assertions
    assert len(results) == 1
    assert results[0] == sample_resource
    mock_repository.list_resources.assert_called_once()


@pytest.mark.asyncio
async def test_list_resources_by_type(resource_service, mock_repository, sample_resource):
    """Test finding resources by type."""
    # Configure mock
    mock_repository.list_resources.return_value = [sample_resource]

    # Execute function - passing as positional argument
    results = await resource_service.list_resources("room")

    # Assertions
    assert len(results) == 1
    assert results[0] == sample_resource
    mock_repository.list_resources.assert_called_once_with("room")


@pytest.mark.asyncio
async def test_find_available_resources(resource_service, mock_repository, sample_resource):
    """Test finding available resources by criteria."""
    # Configure mock
    mock_repository.find_resources.return_value = [sample_resource]

    # Setup test times
    start_time = datetime.datetime.now(datetime.timezone.utc)
    end_time = start_time + datetime.timedelta(hours=2)

    # Create a TimeWindow object
    time_window = TimeWindow(start=start_time, end=end_time)

    # Mock the is_available_at method on the Resource class
    with patch.object(Resource, 'is_available_at', return_value=True):
        # Execute function
        results = await resource_service.find_available_resources(
            start_time=start_time,
            end_time=end_time,
            capacity=10,
            tags=["meeting"],
            resource_type="room"
        )

    # Assertions
    assert len(results) == 1
    assert results[0] == sample_resource
    mock_repository.find_resources.assert_called_once()


# --------------------------
# Resource Update Tests
# --------------------------

@pytest.mark.asyncio
async def test_update_resource(resource_service, mock_repository, sample_resource):
    """Test updating a resource."""
    # Configure mock
    mock_repository.get_resource.return_value = sample_resource
    mock_repository.update_resource.return_value = True

    # Update data
    update_data = {
        "name": "Updated Conference Room",
        "capacity": 25,
        "tags": ["meeting", "large", "updated"]
    }

    # Execute function
    success = await resource_service.update_resource("resource-123", update_data)

    # Assertions
    assert success is True
    mock_repository.update_resource.assert_called_once()

    # Verify the updated resource
    updated_resource = mock_repository.update_resource.call_args[0][0]
    assert updated_resource.name == "Updated Conference Room"
    assert updated_resource.capacity == 25
    assert "updated" in updated_resource.tags


@pytest.mark.asyncio
async def test_update_nonexistent_resource(resource_service, mock_repository):
    """Test updating a resource that doesn't exist."""
    # Configure mock to return None (resource not found)
    mock_repository.get_resource.return_value = None

    # Update data
    update_data = {"name": "Updated Name"}

    # Execute function
    success = await resource_service.update_resource("nonexistent-id", update_data)

    # Assertions
    assert success is False
    mock_repository.update_resource.assert_not_called()


# --------------------------
# Booking Tests
# --------------------------

@pytest.mark.asyncio
async def test_create_booking(resource_service, mock_repository, sample_resource):
    """Test creating a booking for a resource."""
    # Configure mocks
    mock_repository.get_resource.return_value = sample_resource
    mock_repository.create_booking.return_value = "booking-123"
    # Make sure _has_conflicting_bookings returns False
    mock_repository._has_conflicting_bookings.return_value = False

    # Setup booking data
    start_time = datetime.datetime.now(
        datetime.timezone.utc) + datetime.timedelta(days=1)
    end_time = start_time + datetime.timedelta(hours=2)

    # Execute function - use actual method signature
    success, booking_id, error_msg = await resource_service.create_booking(
        resource_id="resource-123",
        user_id="user-456",
        title="Team Meeting",
        start_time=start_time,
        end_time=end_time,
        description="Weekly team sync",
        notes="Please prepare your updates"
    )

    # Assertions
    assert success is True
    assert booking_id == "booking-123"
    assert error_msg is None
    mock_repository.create_booking.assert_called_once()

    # Verify booking object
    created_booking = mock_repository.create_booking.call_args[0][0]
    assert created_booking.resource_id == "resource-123"
    assert created_booking.user_id == "user-456"
    assert created_booking.title == "Team Meeting"
    assert created_booking.start_time == start_time
    assert created_booking.end_time == end_time


@pytest.mark.asyncio
async def test_cancel_booking(resource_service, mock_repository, sample_booking):
    """Test canceling a booking."""
    # Configure mock
    mock_repository.get_booking.return_value = sample_booking
    mock_repository.cancel_booking.return_value = True

    # Execute function
    success, error = await resource_service.cancel_booking("booking-123", "user-456")

    # Assertions
    assert success is True
    assert error is None
    mock_repository.cancel_booking.assert_called_once_with("booking-123")


@pytest.mark.asyncio
async def test_cancel_booking_unauthorized(resource_service, mock_repository, sample_booking):
    """Test canceling a booking by unauthorized user."""
    # Configure mock - booking belongs to user-456
    mock_repository.get_booking.return_value = sample_booking

    # Execute function with different user
    success, error = await resource_service.cancel_booking("booking-123", "different-user")

    # Assertions
    assert success is False
    assert error is not None
    mock_repository.cancel_booking.assert_not_called()


@pytest.mark.asyncio
async def test_get_resource_schedule(resource_service, mock_repository):
    """Test getting resource schedule."""
    # Setup test data
    start_date = datetime.datetime.now(datetime.timezone.utc).date()
    end_date = start_date + datetime.timedelta(days=7)

    # Mock bookings
    booking = ResourceBooking(
        id="booking-123",
        resource_id="resource-123",
        user_id="user-456",
        title="Test Meeting",
        description="Test Description",
        status="confirmed",
        start_time=datetime.datetime.combine(
            start_date, datetime.time(9, 0), datetime.timezone.utc),
        end_time=datetime.datetime.combine(
            start_date, datetime.time(10, 0), datetime.timezone.utc),
        created_at=datetime.datetime.now(datetime.timezone.utc)
    )
    mock_repository.get_resource_schedule.return_value = [booking]

    # Execute function - using the correct method name
    bookings = await resource_service.get_resource_schedule(
        "resource-123", start_date, end_date
    )

    # Assertions
    assert len(bookings) == 1
    assert bookings[0] == booking
    mock_repository.get_resource_schedule.assert_called_once_with(
        "resource-123", start_date, end_date
    )


@pytest.mark.asyncio
async def test_get_user_bookings(resource_service, mock_repository):
    """Test retrieving user's bookings."""
    # Create a mock booking that won't conflict with trying to set attributes
    mock_booking = Mock()
    mock_booking.resource_id = "resource-123"
    mock_booking.model_dump.return_value = {
        "id": "booking-123", "title": "Team Meeting"}

    # Configure mocks
    mock_repository.get_user_bookings.return_value = [mock_booking]

    # Need to mock get_resource which is called within get_user_bookings
    mock_resource = Mock()
    mock_resource.model_dump.return_value = {
        "id": "resource-123", "name": "Conference Room"}
    mock_repository.get_resource.return_value = mock_resource

    # Execute function
    bookings = await resource_service.get_user_bookings("user-456")

    # Assertions
    assert len(bookings) == 1
    assert "booking" in bookings[0]
    assert "resource" in bookings[0]
    mock_repository.get_user_bookings.assert_called_once_with(
        "user-456", False)

"""
Resource service implementation.

This service manages resources and their bookings, including
scheduling, availability tracking, and reservation management.
"""
import uuid
import datetime
from typing import Dict, List, Optional, Any, Tuple

from solana_agent.interfaces.services import ResourceService as ResourceServiceInterface
from solana_agent.interfaces.repositories import ResourceRepository
from solana_agent.domain.resources import Resource, ResourceBooking, TimeWindow


class ResourceService(ResourceServiceInterface):
    """Service for managing resources and bookings."""

    def __init__(self, resource_repository: ResourceRepository):
        """Initialize the resource service.

        Args:
            resource_repository: Repository for resource operations
        """
        self.repository = resource_repository

    async def create_resource(
        self, resource_data: Dict[str, Any], resource_type: str
    ) -> str:
        """Create a new resource from dictionary data.

        Args:
            resource_data: Resource properties
            resource_type: Type of resource

        Returns:
            Resource ID
        """
        # Generate UUID for ID
        resource_id = str(uuid.uuid4())

        resource = Resource(
            id=resource_id,
            name=resource_data["name"],
            resource_type=resource_type,
            description=resource_data.get("description"),
            location=resource_data.get("location"),
            capacity=resource_data.get("capacity"),
            tags=resource_data.get("tags", []),
            attributes=resource_data.get("attributes", {}),
            availability_schedule=resource_data.get(
                "availability_schedule", [])
        )

        return self.repository.create_resource(resource)

    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a resource by ID.

        Args:
            resource_id: Resource ID

        Returns:
            Resource or None if not found
        """
        return self.repository.get_resource(resource_id)

    async def update_resource(
        self, resource_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update a resource.

        Args:
            resource_id: Resource ID
            updates: Dictionary of updates to apply

        Returns:
            True if update was successful
        """
        resource = self.repository.get_resource(resource_id)
        if not resource:
            return False

        # Apply updates
        for key, value in updates.items():
            if hasattr(resource, key):
                setattr(resource, key, value)

        return self.repository.update_resource(resource)

    async def list_resources(
        self, resource_type: Optional[str] = None
    ) -> List[Resource]:
        """List all resources, optionally filtered by type.

        Args:
            resource_type: Optional type to filter by

        Returns:
            List of resources
        """
        return self.repository.list_resources(resource_type)

    async def find_available_resources(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        capacity: Optional[int] = None,
        tags: Optional[List[str]] = None,
        resource_type: Optional[str] = None
    ) -> List[Resource]:
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
        # Get resources matching basic criteria
        resources = self.repository.find_resources(
            resource_type, capacity, tags)

        # Filter by availability
        available = []
        for resource in resources:
            time_window = TimeWindow(start=start_time, end=end_time)
            if resource.is_available_at(time_window):
                if not self.repository._has_conflicting_bookings(resource.id, start_time, end_time):
                    available.append(resource)

        return available

    async def create_booking(
        self,
        resource_id: str,
        user_id: str,
        title: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
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
        # Check if resource exists
        resource = self.repository.get_resource(resource_id)
        if not resource:
            return False, None, "Resource not found"

        # Check for conflicts
        if self.repository._has_conflicting_bookings(resource_id, start_time, end_time):
            return False, None, "Resource is already booked during the requested time"

        # Create booking
        booking_data = ResourceBooking(
            id=str(uuid.uuid4()),
            resource_id=resource_id,
            user_id=user_id,
            title=title,
            description=description,
            status="confirmed",
            start_time=start_time,
            end_time=end_time,
            notes=notes,
            created_at=datetime.datetime.now(datetime.timezone.utc)
        )

        booking_id = self.repository.create_booking(booking_data)

        # Return (success, booking_id, error)
        return True, booking_id, None

    async def cancel_booking(
        self, booking_id: str, user_id: str
    ) -> Tuple[bool, Optional[str]]:
        """Cancel a booking.

        Args:
            booking_id: Booking ID
            user_id: User ID attempting to cancel

        Returns:
            Tuple of (success, error_message)
        """
        # Verify booking exists
        booking = self.repository.get_booking(booking_id)
        if not booking:
            return False, "Booking not found"

        # Verify user owns the booking
        if booking.user_id != user_id:
            return False, "Not authorized to cancel this booking"

        # Cancel booking
        result = self.repository.cancel_booking(booking_id)
        if result:
            return True, None
        return False, "Failed to cancel booking"

    async def get_resource_schedule(
        self,
        resource_id: str,
        start_date: datetime.date,
        end_date: datetime.date
    ) -> List[ResourceBooking]:
        """Get a resource's schedule for a date range.

        Args:
            resource_id: Resource ID
            start_date: Start date
            end_date: End date

        Returns:
            List of bookings in the date range
        """
        return self.repository.get_resource_schedule(resource_id, start_date, end_date)

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
        bookings = self.repository.get_user_bookings(
            user_id,
            include_cancelled
        )

        result = []
        for booking in bookings:
            resource = self.repository.get_resource(booking.resource_id)
            result.append({
                "booking": booking.model_dump(),
                "resource": resource.model_dump() if resource else None
            })

        return result

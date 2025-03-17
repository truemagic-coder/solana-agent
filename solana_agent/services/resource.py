
class ResourceService:
    """Service for managing resources and bookings."""

    def __init__(self, resource_repository: ResourceRepository):
        """Initialize with resource repository."""
        self.repository = resource_repository

    async def create_resource(self, resource_data, resource_type):
        """Create a new resource from dictionary data."""
        # Generate UUID for ID since it can't be None
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

        # Don't use await when calling repository methods
        return self.repository.create_resource(resource)

    async def get_resource(self, resource_id):
        """Get a resource by ID."""
        # Don't use await
        return self.repository.get_resource(resource_id)

    async def update_resource(self, resource_id, updates):
        """Update a resource."""
        resource = self.repository.get_resource(resource_id)
        if not resource:
            return False

        # Apply updates
        for key, value in updates.items():
            if hasattr(resource, key):
                setattr(resource, key, value)

        # Don't use await
        return self.repository.update_resource(resource)

    async def list_resources(self, resource_type=None):
        """List all resources, optionally filtered by type."""
        # Don't use await
        return self.repository.list_resources(resource_type)

    async def find_available_resources(self, start_time, end_time, capacity=None, tags=None, resource_type=None):
        """Find available resources for a time period."""
        # Don't use await
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

    async def create_booking(self, resource_id, user_id, title, start_time, end_time, description=None, notes=None):
        """Create a booking for a resource."""
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

    async def cancel_booking(self, booking_id, user_id):
        """Cancel a booking."""
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

    async def get_resource_schedule(self, resource_id, start_date, end_date):
        """Get a resource's schedule for a date range."""
        return self.repository.get_resource_schedule(resource_id, start_date, end_date)

    async def get_user_bookings(self, user_id, include_cancelled=False):
        """Get all bookings for a user with resource details."""
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

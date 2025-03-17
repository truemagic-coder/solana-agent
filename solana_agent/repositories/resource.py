"""
MongoDB implementation of the resource repository.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any

from solana_agent.domain.resources import Resource, ResourceBooking
from solana_agent.domain.enums import ResourceStatus, BookingStatus
from solana_agent.interfaces.repositories import ResourceRepository


class MongoResourceRepository(ResourceRepository):
    """MongoDB implementation of the ResourceRepository interface."""

    def __init__(self, db_adapter):
        """Initialize the repository with a database adapter."""
        self.db = db_adapter
        self.resources_collection = "resources"
        self.bookings_collection = "resource_bookings"

        # Ensure collections exist
        self.db.create_collection(self.resources_collection)
        self.db.create_collection(self.bookings_collection)

        # Create indexes for resources
        self.db.create_index(self.resources_collection,
                             [("id", 1)], unique=True)
        self.db.create_index(self.resources_collection, [("type", 1)])
        self.db.create_index(self.resources_collection, [("status", 1)])

        # Create indexes for bookings
        self.db.create_index(self.bookings_collection,
                             [("id", 1)], unique=True)
        self.db.create_index(self.bookings_collection, [("resource_id", 1)])
        self.db.create_index(self.bookings_collection, [("user_id", 1)])
        self.db.create_index(self.bookings_collection, [("ticket_id", 1)])
        self.db.create_index(self.bookings_collection, [("start_time", 1)])
        self.db.create_index(self.bookings_collection, [("end_time", 1)])

    def create_resource(self, resource: Resource) -> str:
        """Create a new resource and return its ID."""
        doc = resource.model_dump()
        return self.db.insert_one(self.resources_collection, doc)

    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a resource by ID."""
        doc = self.db.find_one(self.resources_collection, {"id": resource_id})
        if not doc:
            return None

        return Resource.model_validate(doc)

    def find_resources(self, query: Dict) -> List[Resource]:
        """Find resources matching query."""
        docs = self.db.find(self.resources_collection, query)
        return [Resource.model_validate(doc) for doc in docs]

    def update_resource(self, resource_id: str, updates: Dict[str, Any]) -> bool:
        """Update a resource."""
        return self.db.update_one(
            self.resources_collection,
            {"id": resource_id},
            {"$set": updates}
        )

    def delete_resource(self, resource_id: str) -> bool:
        """Delete a resource."""
        return self.db.delete_one(self.resources_collection, {"id": resource_id})

    def create_booking(self, booking: ResourceBooking) -> str:
        """Create a new resource booking and return its ID."""
        doc = booking.model_dump()

        # Convert datetime to string for MongoDB
        if isinstance(doc["start_time"], datetime):
            doc["start_time"] = doc["start_time"].isoformat()

        if isinstance(doc["end_time"], datetime):
            doc["end_time"] = doc["end_time"].isoformat()

        if isinstance(doc["created_at"], datetime):
            doc["created_at"] = doc["created_at"].isoformat()

        if doc.get("updated_at") and isinstance(doc["updated_at"], datetime):
            doc["updated_at"] = doc["updated_at"].isoformat()

        return self.db.insert_one(self.bookings_collection, doc)

    def get_booking(self, booking_id: str) -> Optional[ResourceBooking]:
        """Get a booking by ID."""
        doc = self.db.find_one(self.bookings_collection, {"id": booking_id})
        if not doc:
            return None

        # Convert string dates back to datetime
        if isinstance(doc["start_time"], str):
            doc["start_time"] = datetime.fromisoformat(doc["start_time"])

        if isinstance(doc["end_time"], str):
            doc["end_time"] = datetime.fromisoformat(doc["end_time"])

        if isinstance(doc["created_at"], str):
            doc["created_at"] = datetime.fromisoformat(doc["created_at"])

        if doc.get("updated_at") and isinstance(doc["updated_at"], str):
            doc["updated_at"] = datetime.fromisoformat(doc["updated_at"])

        return ResourceBooking.model_validate(doc)

    def find_bookings(self, query: Dict) -> List[ResourceBooking]:
        """Find bookings matching query."""
        docs = self.db.find(self.bookings_collection, query)

        bookings = []
        for doc in docs:
            # Convert string dates back to datetime
            if isinstance(doc["start_time"], str):
                doc["start_time"] = datetime.fromisoformat(doc["start_time"])

            if isinstance(doc["end_time"], str):
                doc["end_time"] = datetime.fromisoformat(doc["end_time"])

            if isinstance(doc["created_at"], str):
                doc["created_at"] = datetime.fromisoformat(doc["created_at"])

            if doc.get("updated_at") and isinstance(doc["updated_at"], str):
                doc["updated_at"] = datetime.fromisoformat(doc["updated_at"])

            bookings.append(ResourceBooking.model_validate(doc))

        return bookings

    def get_resource_bookings(
        self, resource_id: str, start_time: datetime, end_time: datetime
    ) -> List[ResourceBooking]:
        """Get bookings for a resource within a time period."""
        start_time_str = start_time.isoformat()
        end_time_str = end_time.isoformat()

        query = {
            "resource_id": resource_id,
            "$or": [
                # Booking starts within the time period
                {"start_time": {"$gte": start_time_str, "$lt": end_time_str}},
                # Booking ends within the time period
                {"end_time": {"$gt": start_time_str, "$lte": end_time_str}},
                # Booking spans the entire time period
                {"$and": [
                    {"start_time": {"$lte": start_time_str}},
                    {"end_time": {"$gte": end_time_str}}
                ]}
            ]
        }

        return self.find_bookings(query)

    def update_booking(self, booking_id: str, updates: Dict[str, Any]) -> bool:
        """Update a booking."""
        # Always update the 'updated_at' field
        updates_with_timestamp = {
            **updates,
            "updated_at": datetime.now().isoformat()
        }

        # Convert any datetime objects to strings
        for key, value in updates_with_timestamp.items():
            if isinstance(value, datetime):
                updates_with_timestamp[key] = value.isoformat()

        return self.db.update_one(
            self.bookings_collection,
            {"id": booking_id},
            {"$set": updates_with_timestamp}
        )

    def delete_booking(self, booking_id: str) -> bool:
        """Delete a booking."""
        return self.db.delete_one(self.bookings_collection, {"id": booking_id})

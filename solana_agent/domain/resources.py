"""
Resource domain models for managing bookable resources.
"""
import datetime
import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from solana_agent.domain.enums import ResourceType, ResourceStatus, BookingStatus


class Resource(BaseModel):
    """A bookable resource."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: ResourceType
    status: ResourceStatus = ResourceStatus.AVAILABLE
    location: Optional[str] = None
    capacity: Optional[int] = None
    features: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_available(self, start_time: datetime.datetime, end_time: datetime.datetime) -> bool:
        """Check if resource is available during a time period."""
        return self.status == ResourceStatus.AVAILABLE


class ResourceBooking(BaseModel):
    """A booking for a resource."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resource_id: str
    user_id: str
    ticket_id: Optional[str] = None
    start_time: datetime.datetime
    end_time: datetime.datetime
    status: BookingStatus = BookingStatus.PENDING
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at: Optional[datetime.datetime] = None
    notes: Optional[str] = None

    def duration_minutes(self) -> int:
        """Calculate the duration of this booking in minutes."""
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() / 60)

    def update_status(self, new_status: BookingStatus) -> None:
        """Update the booking status."""
        self.status = new_status
        self.updated_at = datetime.datetime.now(datetime.timezone.utc)

    def overlaps_with(self, other_booking: 'ResourceBooking') -> bool:
        """Check if this booking overlaps with another booking."""
        # If different resources, no overlap
        if self.resource_id != other_booking.resource_id:
            return False

        # Check for time overlap
        return (
            (self.start_time <= other_booking.start_time < self.end_time) or
            (self.start_time < other_booking.end_time <= self.end_time) or
            (other_booking.start_time <= self.start_time and
             other_booking.end_time >= self.end_time)
        )


class ResourceRequest(BaseModel):
    """Request for resources to be allocated to a task."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticket_id: str
    resource_type: ResourceType
    quantity: int = 1
    required_features: List[str] = Field(default_factory=list)
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    priority: int = 1
    status: str = "pending"  # pending, fulfilled, rejected
    allocated_resource_ids: List[str] = Field(default_factory=list)

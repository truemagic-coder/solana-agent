"""
Resource domain models.

These models define structures for resources, bookings, and availability.
"""
from datetime import datetime, date, time
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class TimeWindow(BaseModel):
    """Time window for availability."""
    start: datetime = Field(..., description="Start time")
    end: datetime = Field(..., description="End time")


class AvailabilitySchedule(BaseModel):
    """Schedule of availability for a resource."""
    day_of_week: int = Field(...,
                             description="Day of week (0=Monday, 6=Sunday)", ge=0, le=6)
    start_time: time = Field(..., description="Start time")
    end_time: time = Field(..., description="End time")


class ResourceType(str, Enum):
    """Types of resources."""
    ROOM = "room"
    EQUIPMENT = "equipment"
    VEHICLE = "vehicle"
    PERSON = "person"
    SOFTWARE = "software"
    OTHER = "other"


class Resource(BaseModel):
    """Resource model."""
    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Resource name")
    resource_type: str = Field(..., description="Resource type")
    description: Optional[str] = Field(
        None, description="Resource description")
    location: Optional[str] = Field(None, description="Resource location")
    capacity: Optional[int] = Field(None, description="Resource capacity")
    tags: List[str] = Field(default_factory=list, description="Resource tags")
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Additional attributes")
    availability_schedule: List[Dict[str, Any]] = Field(
        default_factory=list, description="Availability schedule")

    def is_available_at(self, time_window: TimeWindow) -> bool:
        """Check if resource is available in the time window.

        Args:
            time_window: Time window to check

        Returns:
            True if resource is available
        """
        # If no schedule is defined, assume always available
        if not self.availability_schedule:
            return True

        # Get day of week for start and end times (0=Monday, 6=Sunday)
        start_day = time_window.start.weekday()
        end_day = time_window.end.weekday()

        # If spans multiple days, check each day
        if start_day != end_day:
            return False  # For simplicity, don't allow multi-day bookings

        # Check if time falls within any availability window for the day
        for window in self.availability_schedule:
            if window.get("day_of_week") == start_day:
                avail_start = datetime.combine(
                    time_window.start.date(), window.get("start_time"))
                avail_end = datetime.combine(
                    time_window.start.date(), window.get("end_time"))

                if avail_start <= time_window.start and avail_end >= time_window.end:
                    return True

        return False


class BookingStatus(str, Enum):
    """Status of a resource booking."""
    CONFIRMED = "confirmed"
    PENDING = "pending"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class ResourceBooking(BaseModel):
    """Resource booking model."""
    id: str = Field(..., description="Unique identifier")
    resource_id: str = Field(..., description="Resource ID")
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., description="Booking title")
    description: Optional[str] = Field(None, description="Booking description")
    status: str = Field("confirmed", description="Booking status")
    start_time: datetime = Field(..., description="Start time")
    end_time: datetime = Field(..., description="End time")
    notes: Optional[str] = Field(None, description="Booking notes")
    created_at: datetime = Field(...,
                                 description="When the booking was created")
    cancelled_at: Optional[datetime] = Field(
        None, description="When the booking was cancelled")

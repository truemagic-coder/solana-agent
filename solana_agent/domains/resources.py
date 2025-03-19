"""
Resource domain models.

These models define structures for resources, bookings, and availability.
"""
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Any
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
        """Check if the resource is available during the specified time window.

        Args:
            time_window: The time window to check availability for

        Returns:
            True if the resource is available during the time window, False otherwise
        """
        # If there's no availability schedule, assume always available
        if not self.availability_schedule:
            return True

        # Extract days that cover the requested time window
        start_day = time_window.start.date()
        end_day = time_window.end.date()

        # For multi-day bookings, we need to check each day
        current_day = start_day
        while current_day <= end_day:
            day_of_week = current_day.strftime("%A").lower()

            # Find schedule entry for this day
            day_schedule = next((
                schedule for schedule in self.availability_schedule
                if schedule["day_of_week"].lower() == day_of_week
            ), None)

            # If no schedule for this day, the resource is unavailable
            if not day_schedule:
                return False

            # Parse schedule times
            schedule_start = datetime.strptime(
                day_schedule["start_time"], "%H:%M"
            ).time()
            schedule_end = datetime.strptime(
                day_schedule["end_time"], "%H:%M"
            ).time()

            # For first day, check if booking starts within schedule
            if current_day == start_day:
                # If start time is outside schedule, resource unavailable
                if time_window.start.time() < schedule_start or time_window.start.time() > schedule_end:
                    return False

            # For last day, check if booking ends within schedule
            if current_day == end_day:
                # If end time is outside schedule, resource unavailable
                if time_window.end.time() < schedule_start or time_window.end.time() > schedule_end:
                    return False

            # Move to next day
            current_day += datetime.timedelta(days=1)

        # If all days passed checks, resource is available
        return True


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
    purpose: Optional[str] = Field(None, description="Booking purpose")

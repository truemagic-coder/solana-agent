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

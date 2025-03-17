"""
Scheduling domain models.

These models define structures for task scheduling, agent availability,
and time management.
"""
from datetime import datetime, time, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TimeWindow(BaseModel):
    """Time window for scheduling."""
    start: datetime = Field(..., description="Start time")
    end: datetime = Field(..., description="End time")

    def overlaps_with(self, other: 'TimeWindow') -> bool:
        """Check if this window overlaps with another.

        Args:
            other: Another time window

        Returns:
            True if windows overlap
        """
        return self.start < other.end and self.end > other.start


class ScheduledTaskStatus(str, Enum):
    """Status of a scheduled task."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ScheduledTask(BaseModel):
    """Task scheduled for execution."""
    task_id: str = Field(..., description="Unique identifier")
    title: str = Field(..., description="Task title")
    description: str = Field("", description="Task description")
    status: str = Field(ScheduledTaskStatus.PENDING, description="Task status")
    priority: int = Field(
        0, description="Task priority (higher = more important)")
    assigned_to: Optional[str] = Field(
        None, description="Agent ID assigned to the task")
    scheduled_start: Optional[datetime] = Field(
        None, description="Scheduled start time")
    scheduled_end: Optional[datetime] = Field(
        None, description="Scheduled end time")
    actual_start: Optional[datetime] = Field(
        None, description="Actual start time")
    actual_end: Optional[datetime] = Field(None, description="Actual end time")
    estimated_minutes: Optional[int] = Field(
        None, description="Estimated duration in minutes")
    depends_on: List[str] = Field(
        default_factory=list, description="IDs of tasks this depends on")
    constraints: List[Dict[str, Any]] = Field(
        default_factory=list, description="Scheduling constraints")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")


class AgentAvailabilityPattern(BaseModel):
    """Recurring availability pattern for an agent."""
    day_of_week: int = Field(
        ...,
        description="Day of week (0=Monday, 6=Sunday)",
        ge=0,
        le=6
    )
    start_time: time = Field(..., description="Start time of availability")
    end_time: time = Field(..., description="End time of availability")

    def contains(self, dt: datetime) -> bool:
        """Check if a datetime falls within this availability pattern.

        Args:
            dt: Datetime to check

        Returns:
            True if datetime is within this pattern
        """
        # Check day of week (0=Monday in Python's datetime)
        if dt.weekday() != self.day_of_week:
            return False

        # Check time range
        t = dt.time()
        return self.start_time <= t < self.end_time


class AgentSchedule(BaseModel):
    """Schedule for an agent."""
    agent_id: str = Field(..., description="Agent ID")
    availability_patterns: List[AgentAvailabilityPattern] = Field(
        default_factory=list,
        description="Recurring availability patterns"
    )
    time_off_periods: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Time off periods"
    )
    working_hours: Optional[Dict[str, Any]] = Field(
        None, description="Working hours configuration"
    )

    def is_available_at(self, dt: datetime) -> bool:
        """Check if agent is available at a specific time.

        Args:
            dt: Datetime to check

        Returns:
            True if agent is available
        """
        # Check for time off periods first
        for period in self.time_off_periods:
            if period.get("start") <= dt < period.get("end"):
                return False

        # Check against availability patterns
        for pattern in self.availability_patterns:
            if pattern.contains(dt):
                return True

        # If no patterns match or no patterns exist, agent is not available
        return False


class TimeOffStatus(str, Enum):
    """Status of a time off request."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    CANCELLED = "cancelled"


class TimeOffRequest(BaseModel):
    """Request for time off from an agent."""
    id: str = Field(..., description="Unique identifier")
    agent_id: str = Field(..., description="Agent ID")
    start_time: datetime = Field(..., description="Start time of time off")
    end_time: datetime = Field(..., description="End time of time off")
    reason: str = Field("", description="Reason for time off")
    status: TimeOffStatus = Field(
        TimeOffStatus.PENDING, description="Request status")
    created_at: datetime = Field(...,
                                 description="When the request was created")
    processed_at: Optional[datetime] = Field(
        None, description="When the request was processed")
    denial_reason: Optional[str] = Field(
        None, description="Reason for denial if denied")
    conflicts: List[str] = Field(
        default_factory=list, description="Conflicting task IDs")


class SchedulingEvent(BaseModel):
    """Event related to scheduling."""
    id: str = Field(..., description="Unique identifier")
    event_type: str = Field(..., description="Type of event")
    task_id: Optional[str] = Field(None, description="Related task ID")
    agent_id: Optional[str] = Field(None, description="Related agent ID")
    timestamp: datetime = Field(..., description="When the event occurred")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Event details")

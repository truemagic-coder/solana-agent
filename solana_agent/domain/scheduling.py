"""
Scheduling domain models for managing tasks and schedules.
"""
import datetime
import uuid
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

from solana_agent.domain.enums import Priority, ScheduleConflictType


class TimeConstraint(BaseModel):
    """Time constraint for task scheduling."""
    type: str  # must_start_after, must_end_before, fixed_start, fixed_end
    time: datetime.datetime
    is_flexible: bool = False
    flexibility_minutes: int = 0


class ScheduledTask(BaseModel):
    """A task scheduled for completion."""
    task_id: str
    title: str
    description: str
    estimated_minutes: int
    priority: int = 3  # 1-5 scale
    assigned_to: Optional[str] = None
    scheduled_start: Optional[datetime.datetime] = None
    scheduled_end: Optional[datetime.datetime] = None
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "scheduled"  # scheduled, in_progress, completed, canceled
    dependencies: List[str] = Field(default_factory=list)
    progress: int = 0  # 0-100 percentage

    def calculate_end_time(self, start_time: datetime.datetime) -> datetime.datetime:
        """Calculate end time based on start time and estimated duration."""
        return start_time + datetime.timedelta(minutes=self.estimated_minutes)

    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status == "completed" or self.progress >= 100


class AgentSchedule(BaseModel):
    """Schedule for an agent."""
    agent_id: str
    date: datetime.date
    allocated_minutes: int = 0
    max_minutes: int = 480  # Default 8 hours
    tasks: List[ScheduledTask] = Field(default_factory=list)

    def available_minutes(self) -> int:
        """Calculate remaining available minutes in schedule."""
        return max(0, self.max_minutes - self.allocated_minutes)

    def has_capacity_for(self, minutes: int) -> bool:
        """Check if schedule has capacity for task of given duration."""
        return self.available_minutes() >= minutes

    def add_task(self, task: ScheduledTask) -> bool:
        """Add task to schedule if capacity allows."""
        if not task.estimated_minutes:
            return False

        if not self.has_capacity_for(task.estimated_minutes):
            return False

        self.tasks.append(task)
        self.allocated_minutes += task.estimated_minutes
        return True


class ScheduleConflict(BaseModel):
    """Representation of a schedule conflict."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: ScheduleConflictType
    task_ids: List[str]
    resource_id: Optional[str] = None
    agent_id: Optional[str] = None
    time_period: Optional[Tuple[datetime.datetime, datetime.datetime]] = None
    description: str
    priority: int = 0  # Higher number means more important conflict
    possible_resolutions: List[str] = Field(default_factory=list)


class WorkWeek(BaseModel):
    """Work week configuration."""
    agent_id: str
    start_day: int = 0  # Monday = 0
    end_day: int = 4    # Friday = 4
    daily_start_time: str = "09:00"
    daily_end_time: str = "17:00"
    timezone: str = "UTC"
    working_days: List[int] = Field(default_factory=lambda: [
                                    0, 1, 2, 3, 4])  # Mon-Fri
    exceptions: Dict[str, List[str]] = Field(
        default_factory=dict)  # date -> [start, end]

    def is_working_time(self, dt: datetime.datetime) -> bool:
        """Check if given datetime is during working hours."""
        # Convert to agent's timezone
        local_dt = dt

        # Check for exception days
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in self.exceptions:
            # No hours specified means not working that day
            if not self.exceptions[date_str]:
                return False

            # Check exception hours
            time_str = dt.strftime("%H:%M")
            for i in range(0, len(self.exceptions[date_str]), 2):
                if i + 1 < len(self.exceptions[date_str]):
                    start = self.exceptions[date_str][i]
                    end = self.exceptions[date_str][i+1]
                    if start <= time_str <= end:
                        return True
            return False

        # Check regular schedule
        weekday = dt.weekday()
        if weekday not in self.working_days:
            return False

        time_str = dt.strftime("%H:%M")
        return self.daily_start_time <= time_str <= self.daily_end_time

"""
Task domain models.

These models define structures for tasks, work capacity, and plan status.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import uuid4


class TaskStatus(str, Enum):
    """Possible states of a task or subtask."""
    PLANNING = "planning"      # Task is being planned or defined
    BACKLOG = "backlog"        # Task is ready but not scheduled
    TODO = "todo"              # Task is scheduled but not started
    IN_PROGRESS = "in_progress"  # Task is currently being worked on
    REVIEW = "review"          # Task is complete but needs review
    BLOCKED = "blocked"        # Task is blocked by dependency or issue
    COMPLETED = "completed"    # Task is fully completed
    CANCELED = "canceled"      # Task was canceled before completion
    FAILED = "failed"          # Task could not be completed


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class WorkCapacityStatus(str, Enum):
    """Status of an agent's work capacity."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ON_BREAK = "on_break"


class WorkCapacity(BaseModel):
    """Model for agent work capacity."""
    agent_id: str = Field(..., description="Agent ID")
    agent_type: str = Field(..., description="Type of agent")
    max_concurrent_tasks: int = Field(...,
                                      description="Maximum concurrent tasks")
    active_tasks: int = Field(0, description="Currently active tasks")
    specializations: List[str] = Field(
        default_factory=list, description="Agent specializations")
    availability_status: WorkCapacityStatus = Field(
        WorkCapacityStatus.AVAILABLE, description="Agent availability")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(),
                                   description="When status was last updated")


class SubtaskModel(BaseModel):
    """Model for a subtask."""
    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: str(
        uuid4()), description="Unique identifier")
    parent_id: str = Field(..., description="Parent ticket ID")
    title: str = Field(..., description="Subtask title")
    description: str = Field(..., description="Subtask description")
    estimated_minutes: int = Field(
        30, description="Estimated duration in minutes")
    sequence: int = Field(0, description="Sequence order")
    dependencies: List[str] = Field(
        default_factory=list, description="Dependent subtask IDs")
    status: TaskStatus = Field(
        TaskStatus.PLANNING, description="Subtask status")
    priority: TaskPriority = Field(
        TaskPriority.MEDIUM, description="Subtask priority")
    assignee: Optional[str] = Field(None, description="Assigned agent")
    is_subtask: bool = Field(True, description="Whether this is a subtask")
    scheduled_start: Optional[datetime] = Field(
        None, description="Scheduled start time")
    scheduled_end: Optional[datetime] = Field(
        None, description="Scheduled end time")
    actual_start: Optional[datetime] = Field(
        None, description="Actual start time")
    actual_end: Optional[datetime] = Field(None, description="Actual end time")
    required_resources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Required resources")
    resource_assignments: List[Dict[str, Any]] = Field(
        default_factory=list, description="Resource assignments")
    query: Optional[str] = Field(None, description="Original query")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(
        None, description="Last update timestamp")


class PlanStatus(BaseModel):
    """Status of a task plan."""
    visualization: str = Field(...,
                               description="Visual representation of progress")
    progress: int = Field(..., description="Progress percentage")
    status: str = Field(..., description="Status description")
    estimated_completion: str = Field(...,
                                      description="Estimated completion time")
    subtask_count: int = Field(..., description="Number of subtasks")


class ComplexityAssessment(BaseModel):
    """Assessment of task complexity."""
    t_shirt_size: str = Field(...,
                              description="T-shirt size (XS, S, M, L, XL, XXL)")
    story_points: int = Field(...,
                              description="Story points (1, 2, 3, 5, 8, 13, 21)")
    estimated_minutes: int = Field(...,
                                   description="Estimated minutes to resolution")
    technical_complexity: int = Field(...,
                                      description="Technical complexity (1-10)")
    domain_knowledge: int = Field(...,
                                  description="Domain knowledge required (1-10)")


# Define structured output models for task breakdown
class SubtaskDefinition(BaseModel):
    """Definition of a subtask from LLM."""
    title: str = Field(..., description="Brief, descriptive title")
    description: str = Field(...,
                             description="Clear description of what needs to be done")
    estimated_minutes: int = Field(...,
                                   description="Estimated time to complete in minutes", ge=5)
    dependencies: List[str] = Field(default_factory=list,
                                    description="Titles of subtasks this depends on")
    priority: TaskPriority = Field(
        TaskPriority.MEDIUM, description="Subtask priority")


class ResourceRequirement(BaseModel):
    """Resource requirement for a task."""
    resource_type: str = Field(..., description="Type of resource needed")
    quantity: int = Field(1, description="Number of resources needed", ge=1)
    requirements: str = Field(
        "", description="Specific requirements or features")


class SubtaskWithResources(SubtaskDefinition):
    """Subtask definition with resource requirements."""
    required_resources: List[ResourceRequirement] = Field(
        default_factory=list,
        description="Resources required for this subtask"
    )


class TaskBreakdown(BaseModel):
    """Complete task breakdown."""
    subtasks: List[SubtaskDefinition] = Field(...,
                                              description="List of subtasks")


class TaskBreakdownWithResources(BaseModel):
    """Complete task breakdown with resource requirements."""
    subtasks: List[SubtaskWithResources] = Field(
        ..., description="List of subtasks with resources")


class Task(BaseModel):
    """Model for a main task."""
    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: str(
        uuid4()), description="Unique identifier")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Task description")
    status: TaskStatus = Field(TaskStatus.PLANNING, description="Task status")
    priority: TaskPriority = Field(
        TaskPriority.MEDIUM, description="Task priority")
    assignee: Optional[str] = Field(None, description="Assigned agent")
    estimated_minutes: int = Field(
        60, description="Estimated duration in minutes")
    subtasks: List[str] = Field(
        default_factory=list, description="IDs of subtasks")
    dependencies: List[str] = Field(
        default_factory=list, description="IDs of tasks this depends on")
    scheduled_start: Optional[datetime] = Field(
        None, description="Scheduled start time")
    scheduled_end: Optional[datetime] = Field(
        None, description="Scheduled end time")
    actual_start: Optional[datetime] = Field(
        None, description="Actual start time")
    actual_end: Optional[datetime] = Field(None, description="Actual end time")
    required_resources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Required resources")
    resource_assignments: List[Dict[str, Any]] = Field(
        default_factory=list, description="Resource assignments")
    query: Optional[str] = Field(None, description="Original query")
    complexity: Optional[ComplexityAssessment] = Field(
        None, description="Complexity assessment")
    progress: int = Field(0, description="Progress percentage (0-100)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(
        None, description="Last update timestamp")
    created_by: Optional[str] = Field(
        None, description="User or agent who created the task")
    tags: List[str] = Field(default_factory=list,
                            description="Task tags for categorization")

    def calculate_progress(self) -> int:
        """Calculate progress based on subtask completion."""
        # This would be implemented to check subtask status and update progress
        return self.progress

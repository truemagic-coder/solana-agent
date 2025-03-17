"""
Common enumerations used across the Solana Agent system.
"""
from enum import Enum


class TicketStatus(str, Enum):
    """Represents possible states of a support ticket."""
    NEW = "new"
    ACTIVE = "active"
    PENDING = "pending"
    TRANSFERRED = "transferred"
    RESOLVED = "resolved"
    PLANNING = "planning"
    SCHEDULED = "scheduled"
    CANCELED = "canceled"
    FAILED = "failed"


class AgentType(str, Enum):
    """Type of agent (AI or Human)."""
    AI = "ai"
    HUMAN = "human"


class ResourceType(str, Enum):
    """Types of resources that can be booked."""
    ROOM = "room"
    VEHICLE = "vehicle"
    EQUIPMENT = "equipment"
    SEAT = "seat"
    DESK = "desk"
    OTHER = "other"


class ResourceStatus(str, Enum):
    """Status of a resource."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"
    RESERVED = "reserved"


class BookingStatus(str, Enum):
    """Status of a resource booking."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELED = "canceled"
    COMPLETED = "completed"


class TaskSize(str, Enum):
    """T-shirt size estimates for tasks."""
    XS = "XS"
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"
    XXL = "XXL"


class Priority(str, Enum):
    """Priority levels for tasks and tickets."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"


class ScheduleConflictType(str, Enum):
    """Types of schedule conflicts."""
    OVERLAP = "overlap"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    AGENT_UNAVAILABLE = "agent_unavailable"
    TIME_CONSTRAINT = "time_constraint"

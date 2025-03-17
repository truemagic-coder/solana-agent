"""
Domain models for the Solana Agent system.

This package contains all the core domain models that represent the
business objects and value types in the system.
"""

# Import and re-export all models from domain files
from solana_agent.domain.agents import (
    AgentType,
    AIAgent,
    HumanAgent,
    AgentProfile,
    AgentSpecialization
)

from solana_agent.domain.tickets import (
    Ticket,
    TicketStatus,
    TicketInteraction,
    TicketPriority
)

from solana_agent.domain.tasks import (
    ComplexityAssessment,
    SubtaskModel,
    TaskBreakdown,
    TaskBreakdownWithResources,
    WorkCapacity,
    PlanStatus,
    TaskStatus,
    ResourceRequirement,
    ResourceAllocation,
    ResourceAssignment,
    SubtaskDefinition,
    SubtaskWithResources
)

from solana_agent.domain.scheduling import (
    AgentAvailabilityPattern,
    ScheduledTask,
    AgentSchedule,
    SchedulingEvent,
    TimeWindow,
    TimeOffRequest,
    TimeOffStatus,
    ScheduledTaskStatus
)

from solana_agent.domain.projects import (
    ProjectStatus,
    Project,
    ApprovalCriteria,
    ProjectReview,
    RiskLevel,
    Risk,
    TimelineEstimate,
    ResourceEstimate,
    FeasibilityAssessment,
    ProjectSimulation
)

from solana_agent.domain.resources import (
    Resource,
    ResourceType,
    ResourceBooking,
    BookingStatus,
    AvailabilitySchedule,
    TimeWindow
)

from solana_agent.domain.models import (
    QueryAnalysis
)

# Version of the domain model
__version__ = '0.1.0'

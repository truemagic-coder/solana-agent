"""
Agent domain models for both AI and human agents.
"""
import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from solana_agent.domain.enums import AgentType


class OrganizationMission(BaseModel):
    """Organization's mission statement and values to guide agent behavior."""
    mission: str
    values: List[str] = Field(default_factory=list)
    guidelines: List[str] = Field(default_factory=list)

    def format_as_directive(self) -> str:
        """Format the mission as a directive for agents."""
        values_text = "\n- ".join([""] + self.values) if self.values else ""
        guidelines_text = "\n- ".join([""] +
                                      self.guidelines) if self.guidelines else ""

        return (
            f"ORGANIZATION MISSION:\n{self.mission}\n\n"
            f"CORE VALUES:{values_text}\n\n"
            f"KEY GUIDELINES:{guidelines_text}"
        )


class AgentSpecialization(BaseModel):
    """Specialization for an agent with capabilities and routing information."""
    name: str
    description: str
    keywords: List[str] = Field(default_factory=list)
    capability_level: int = 1  # 1-5 scale

    def matches_query(self, query: str) -> bool:
        """Check if this specialization matches a user query based on keywords."""
        query_lower = query.lower()
        for keyword in self.keywords:
            if keyword.lower() in query_lower:
                return True
        return False


class AgentAvailability(BaseModel):
    """Availability window for an agent."""
    agent_id: str
    day_of_week: Optional[int] = None  # 0-6, None means all days
    start_time: Optional[str] = None  # "HH:MM" format, None means all day
    end_time: Optional[str] = None  # "HH:MM" format, None means all day
    timezone: str = "UTC"
    is_available: bool = True


class AIAgent(BaseModel):
    """AI agent configuration."""
    name: str
    instructions: str
    specialization: str
    model: str = "gpt-4o-mini"
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    description: Optional[str] = None
    type: AgentType = AgentType.AI
    available_tools: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HumanAgent(BaseModel):
    """Human agent profile."""
    id: str = Field(default_factory=str)
    name: str
    email: str
    specializations: List[str] = Field(default_factory=list)
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    type: AgentType = AgentType.HUMAN
    is_active: bool = True
    role: str = "agent"
    availability: List[AgentAvailability] = Field(default_factory=list)

    def is_available_now(self) -> bool:
        """Check if the agent is available at the current time."""
        if not self.is_active:
            return False

        now = datetime.datetime.now(datetime.timezone.utc)
        day_of_week = now.weekday()
        current_time = now.strftime("%H:%M")

        # If no availability is set, assume always available
        if not self.availability:
            return True

        for window in self.availability:
            if not window.is_available:
                continue

            # Check day match (if day is specified)
            day_matches = window.day_of_week is None or window.day_of_week == day_of_week

            # Check time in range (if times are specified)
            time_in_range = True
            if window.start_time and window.end_time:
                time_in_range = window.start_time <= current_time <= window.end_time

            if day_matches and time_in_range:
                return True

        return False


class AgentPerformance(BaseModel):
    """Performance metrics for an agent."""
    agent_id: str
    period_start: datetime.datetime
    period_end: datetime.datetime
    tickets_handled: int = 0
    avg_resolution_time: Optional[float] = None  # minutes
    satisfaction_score: Optional[float] = None  # 0-10
    handoff_rate: Optional[float] = None  # percentage


class AgentType(str, Enum):
    """Type of agent."""
    AI = "ai"
    HUMAN = "human"
    SYSTEM = "system"


class AIAgent(BaseModel):
    """AI agent model."""
    name: str = Field(..., description="Agent name")
    instructions: str = Field(..., description="Agent instructions")
    specialization: str = Field(..., description="Agent specialization area")
    model: str = Field(
        "gpt-4o-mini", description="Language model used by the agent")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the agent was created")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")


class HumanAgent(BaseModel):
    """Human agent model."""
    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Agent name")
    email: str = Field(..., description="Agent email")
    specializations: List[str] = Field(
        default_factory=list, description="Agent specialization areas")
    availability: bool = Field(
        True, description="Whether the agent is available")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the agent was created")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")

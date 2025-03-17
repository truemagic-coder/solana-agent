"""
Domain models for AI and human agents.

This module defines the core domain models for representing
AI agents, human agents, and organization mission/values.
"""
from typing import List, Optional, Dict, Any, Union
# Import the class directly, not the module
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, field_validator


class OrganizationMission(BaseModel):
    """Organization mission and values to guide agent behavior."""

    mission_statement: str = Field(...,
                                   description="Organization mission statement")
    values: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Organization values as name-description pairs"
    )
    goals: List[str] = Field(
        default_factory=list,
        description="Organization goals"
    )
    guidance: Optional[str] = Field(
        None, description="Additional guidance for agents")

    @field_validator("mission_statement")
    @classmethod
    def mission_statement_not_empty(cls, v: str) -> str:
        """Validate that mission statement is not empty."""
        if not v.strip():
            raise ValueError("Mission statement cannot be empty")
        return v

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Validate that values have proper format."""
        for value in values:
            if "name" not in value or "description" not in value:
                raise ValueError("Each value must have a name and description")
        return values


class AIAgent(BaseModel):
    """AI agent with specialized capabilities."""

    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(..., description="Unique agent identifier name")
    instructions: str = Field(...,
                              description="Base instructions for the agent")
    specialization: str = Field(..., description="Agent's specialized domain")
    model: str = Field("gpt-4o-mini", description="Language model to use")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp")

    @field_validator("name", "specialization")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Validate that string fields are not empty."""
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("instructions")
    @classmethod
    def instructions_not_empty(cls, v: str) -> str:
        """Validate that instructions are not empty."""
        if not v.strip():
            raise ValueError("Instructions cannot be empty")
        if len(v) < 10:
            raise ValueError("Instructions must be at least 10 characters")
        return v


class HumanAgent(BaseModel):
    """Human agent that can participate in the system."""

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(..., description="Unique human identifier")
    name: str = Field(..., description="Human agent name")
    specializations: List[str] = Field(..., description="Areas of expertise")
    availability: bool = Field(True, description="Whether agent is available")
    notification_handler: Optional[Any] = Field(
        None, description="Handler for agent notifications"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp")

    @field_validator("id", "name")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Validate that string fields are not empty."""
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("specializations")
    @classmethod
    def specializations_not_empty(cls, v: List[str]) -> List[str]:
        """Validate that specializations list is not empty."""
        if not v:
            raise ValueError("At least one specialization is required")
        return v


class AgentPerformance(BaseModel):
    """Performance metrics for an agent."""

    model_config = {"arbitrary_types_allowed": True}

    agent_id: str = Field(..., description="ID or name of the agent")
    agent_type: str = Field(..., description="Type of agent (AI or Human)")

    # Core performance metrics
    avg_response_time: Optional[float] = Field(
        None, description="Average response time in seconds")
    successful_interactions: int = Field(
        0, description="Number of successful interactions")
    failed_interactions: int = Field(
        0, description="Number of failed interactions")
    total_interactions: int = Field(
        0, description="Total number of interactions")

    # Handoff metrics
    handoffs_initiated: int = Field(
        0, description="Number of handoffs initiated by this agent")
    handoffs_received: int = Field(
        0, description="Number of handoffs received by this agent")

    # User satisfaction
    avg_satisfaction_score: Optional[float] = Field(
        None, description="Average user satisfaction (0-5)")
    nps_score: Optional[float] = Field(
        None, description="Net Promoter Score (-100 to 100)")

    # Task performance
    tasks_completed: int = Field(0, description="Number of tasks completed")
    tasks_failed: int = Field(0, description="Number of tasks that failed")
    avg_task_completion_time: Optional[float] = Field(
        None, description="Average task completion time in minutes")

    # Specialization effectiveness
    specialization_matches: int = Field(
        0, description="Number of queries matched to specialization")
    specialization_mismatches: int = Field(
        0, description="Number of queries mismatched with specialization")

    # Time metrics
    total_active_time: timedelta = Field(
        default_factory=lambda: timedelta(), description="Total time agent was active")
    first_interaction: Optional[datetime] = Field(
        None, description="Timestamp of first interaction")
    last_interaction: Optional[datetime] = Field(
        None, description="Timestamp of last interaction")

    # Custom metrics
    custom_metrics: Dict[str, Union[int, float, str]] = Field(
        default_factory=dict,
        description="Custom performance metrics specific to agent type"
    )

    # Update timestamp
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last time metrics were updated")

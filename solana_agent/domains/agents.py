"""
Domain models for AI and human agents.

This module defines the core domain models for representing
AI agents, human agents, and organization mission/values.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime  # Import the class directly, not the module
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

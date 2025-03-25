"""
Domain models for AI and human agents.

This module defines the core domain models for representing
AI agents, human agents, and business mission/values.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class BusinessMission(BaseModel):
    """Business mission and values to guide agent behavior."""

    mission: str = Field(...,
                         description="Business mission statement")
    values: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Business values as name-description pairs"
    )
    goals: List[str] = Field(
        default_factory=list,
        description="Business goals"
    )
    voice: str = Field(
        None, description="Business voice or tone")

    @field_validator("mission")
    @classmethod
    def mission_not_empty(cls, v: str) -> str:
        """Validate that mission is not empty."""
        if not v.strip():
            raise ValueError("Mission cannot be empty")
        return v

    @field_validator("voice")
    @classmethod
    def voice_not_empty(cls, v: str) -> str:
        """Validate that voice is not empty."""
        if not v.strip():
            raise ValueError("Voice cannot be empty")
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

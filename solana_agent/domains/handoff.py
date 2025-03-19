"""
Handoff domain models.

These models represent handoff records between agents.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Handoff(BaseModel):
    """Record of a handoff between agents."""
    id: Optional[str] = Field(None, description="Unique identifier")
    from_agent: str = Field(..., description="Agent handing off")
    to_agent: str = Field(..., description="Agent receiving handoff")
    ticket_id: str = Field(..., description="Related ticket ID")
    reason: str = Field(..., description="Reason for handoff")
    timestamp: datetime = Field(..., description="Handoff time")
    successful: bool = Field(..., description="Whether handoff was successful")


class HandoffEvaluation(BaseModel):
    """Evaluation of whether a handoff is needed."""
    handoff_needed: bool = Field(..., description="Whether handoff is needed")
    target_agent: Optional[str] = Field(
        None, description="Suggested target agent")
    reason: Optional[str] = Field(
        None, description="Reason for handoff recommendation")
    confidence: float = Field(...,
                              description="Confidence in recommendation (0-1)")

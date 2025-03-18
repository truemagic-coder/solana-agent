"""
Ticket domain models.

These models define structures for support tickets and related data.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TicketStatus(str, Enum):
    """Status of a support ticket."""
    NEW = "new"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_USER = "waiting_for_user"
    WAITING_FOR_HUMAN = "waiting_for_human"
    RESOLVED = "resolved"
    CLOSED = "closed"
    STALLED = "stalled"
    PLANNING = "planning"


class TicketPriority(str, Enum):
    """Priority of a support ticket."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketNote(BaseModel):
    """Note attached to a ticket."""
    id: str = Field("", description="Unique identifier")
    content: str = Field(..., description="Note content")
    type: str = Field("agent", description="Note type (agent, system, user)")
    created_by: Optional[str] = Field(None, description="ID of the creator")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the note was created")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")


class Ticket(BaseModel):
    """Support ticket model."""
    id: str = Field(..., description="Unique identifier")
    title: str = Field(..., description="Ticket title")
    description: str = Field(..., description="Ticket description")
    user_id: str = Field(...,
                         description="ID of the user who created the ticket")
    assigned_to: Optional[str] = Field(
        None, description="ID of the assigned agent")
    status: TicketStatus = Field(TicketStatus.NEW, description="Ticket status")
    priority: TicketPriority = Field(
        TicketPriority.MEDIUM, description="Ticket priority")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the ticket was created")
    updated_at: datetime = Field(
        default_factory=datetime.now, description="When the ticket was last updated")
    closed_at: Optional[datetime] = Field(
        None, description="When the ticket was closed")
    tags: List[str] = Field(default_factory=list, description="Ticket tags")
    notes: List[TicketNote] = Field(
        default_factory=list, description="Ticket notes")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")


class TicketResolutionStatus(str, Enum):
    """Status of a ticket resolution assessment."""
    RESOLVED = "resolved"
    NEEDS_FOLLOWUP = "needs_followup"
    CANNOT_DETERMINE = "cannot_determine"


class TicketResolution(BaseModel):
    """Assessment of whether a ticket has been resolved."""
    status: str = Field(
        ...,
        description="Resolution status (resolved, needs_followup, cannot_determine)"
    )
    confidence: float = Field(
        ...,
        description="Confidence in the resolution assessment (0-1)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        ...,
        description="Reasoning for the resolution assessment"
    )
    suggested_actions: List[str] = Field(
        default_factory=list,
        description="Suggested actions based on resolution status"
    )

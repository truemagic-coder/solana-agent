"""
Ticket domain models representing support requests and their lifecycle.
"""
import datetime
import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from solana_agent.domain.enums import TicketStatus, Priority


class TicketResolution(BaseModel):
    """Information about ticket resolution status."""
    status: str  # "resolved", "needs_followup", or "cannot_determine"
    confidence: float
    reasoning: str
    suggested_actions: List[str] = Field(default_factory=list)


class TicketHandoff(BaseModel):
    """Information about a ticket handoff between agents."""
    source_agent: str
    target_agent: str
    reason: str
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))


class TicketNote(BaseModel):
    """Note added to a ticket by an agent or system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticket_id: str
    author: str
    content: str
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    visibility: str = "internal"  # internal, user, public


class ComplexityAssessment(BaseModel):
    """Assessment of task complexity."""
    t_shirt_size: str
    story_points: Optional[int] = None
    estimated_hours: Optional[float] = None
    factors: List[str] = Field(default_factory=list)
    requires_breakdown: bool = False
    reasoning: str = ""


class Ticket(BaseModel):
    """Model for a support ticket."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    query: str
    status: TicketStatus = TicketStatus.NEW
    assigned_to: str = ""
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at: Optional[datetime.datetime] = None
    resolved_at: Optional[datetime.datetime] = None
    resolution_confidence: Optional[float] = None
    resolution_reasoning: Optional[str] = None
    handoff_reason: Optional[str] = None
    complexity: Optional[ComplexityAssessment] = None
    agent_context: Optional[Dict[str, Any]] = None
    priority: Priority = Priority.MEDIUM
    due_date: Optional[datetime.datetime] = None

    # Task/subtask relationship
    is_parent: bool = False
    is_subtask: bool = False
    parent_id: Optional[str] = None
    child_tickets: List[str] = Field(default_factory=list)

    # Task specific fields
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_start: Optional[datetime.datetime] = None
    scheduled_end: Optional[datetime.datetime] = None
    required_resources: List[Dict[str, Any]] = Field(default_factory=list)
    resource_assignments: List[Dict[str, Any]] = Field(default_factory=list)

    # Audit trail
    handoff_history: List[TicketHandoff] = Field(default_factory=list)
    notes: List[TicketNote] = Field(default_factory=list)

    def add_note(self, author: str, content: str, visibility: str = "internal") -> None:
        """Add a note to the ticket."""
        note = TicketNote(
            ticket_id=self.id,
            author=author,
            content=content,
            visibility=visibility
        )
        self.notes.append(note)

    def record_handoff(self, source_agent: str, target_agent: str, reason: str) -> None:
        """Record a handoff between agents."""
        handoff = TicketHandoff(
            source_agent=source_agent,
            target_agent=target_agent,
            reason=reason
        )
        self.handoff_history.append(handoff)

    def update_status(self, new_status: TicketStatus) -> None:
        """Update ticket status with timestamp."""
        self.status = new_status
        self.updated_at = datetime.datetime.now(datetime.timezone.utc)

        if new_status == TicketStatus.RESOLVED:
            self.resolved_at = self.updated_at


class NPSSurvey(BaseModel):
    """Net promoter score survey for ticket resolution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticket_id: str
    user_id: str
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    completed_at: Optional[datetime.datetime] = None
    score: Optional[int] = None
    feedback: Optional[str] = None

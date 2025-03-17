"""
Project domain models.

These models define structures for projects and approval workflows.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ProjectStatus(str, Enum):
    """Status of a project in the approval workflow."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ApprovalCriteria(BaseModel):
    """Criterion for project approval."""
    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Criterion name")
    description: str = Field(..., description="Criterion description")
    weight: float = Field(1.0, description="Weight for scoring")


class ProjectReview(BaseModel):
    """Review of a project."""
    id: str = Field(..., description="Unique identifier")
    project_id: str = Field(..., description="Project ID")
    reviewer_id: str = Field(..., description="Reviewer ID")
    criteria_scores: List[Dict[str, Any]] = Field(
        default_factory=list, description="Scores for criteria")
    overall_score: float = Field(..., description="Overall score")
    comments: str = Field("", description="Review comments")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the review was submitted")


class Project(BaseModel):
    """Project model."""
    id: str = Field("", description="Unique identifier")
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    submitter_id: str = Field(..., description="ID of the submitter")
    status: ProjectStatus = Field(
        ProjectStatus.DRAFT, description="Project status")
    goals: str = Field("", description="Project goals")
    scope: str = Field("", description="Project scope")
    resources_required: str = Field("", description="Required resources")
    timeline: str = Field("", description="Project timeline")
    submitted_at: Optional[datetime] = Field(
        None, description="When the project was submitted")
    approved_at: Optional[datetime] = Field(
        None, description="When the project was approved")
    approved_by: Optional[str] = Field(
        None, description="Who approved the project")
    approval_comments: Optional[str] = Field(
        None, description="Approval comments")
    rejected_at: Optional[datetime] = Field(
        None, description="When the project was rejected")
    rejected_by: Optional[str] = Field(
        None, description="Who rejected the project")
    rejection_reason: Optional[str] = Field(
        None, description="Rejection reason")
    reviews: List[ProjectReview] = Field(
        default_factory=list, description="Project reviews")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")

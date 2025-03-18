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


class RiskLevel(str, Enum):
    """Project risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Risk(BaseModel):
    """Project risk model."""
    description: str = Field(..., description="Risk description")
    probability: RiskLevel = Field(
        RiskLevel.MEDIUM, description="Probability of occurrence")
    impact: RiskLevel = Field(
        RiskLevel.MEDIUM, description="Impact if risk occurs")
    mitigation_strategies: List[str] = Field(
        default_factory=list, description="Strategies to mitigate risk")
    category: str = Field(..., description="Risk category")


class TimelineEstimate(BaseModel):
    """Project timeline estimate model."""
    optimistic_days: int = Field(...,
                                 description="Optimistic timeline in days")
    realistic_days: int = Field(..., description="Realistic timeline in days")
    pessimistic_days: int = Field(...,
                                  description="Pessimistic timeline in days")
    confidence_level: str = Field(...,
                                  description="Confidence level in estimate")
    key_factors: List[str] = Field(
        default_factory=list, description="Key factors affecting timeline")


class ResourceEstimate(BaseModel):
    """Project resource estimate model."""
    required_specializations: List[str] = Field(
        default_factory=list, description="Required specializations")
    agents_needed: int = Field(..., description="Number of agents needed")
    skillsets: List[Dict[str, Any]] = Field(
        default_factory=list, description="Required skillsets")
    external_resources: List[str] = Field(
        default_factory=list, description="External resources needed")
    knowledge_domains: List[str] = Field(
        default_factory=list, description="Knowledge domains involved")


class FeasibilityAssessment(BaseModel):
    """Project feasibility assessment model."""
    score: float = Field(...,
                         description="Feasibility score (0-100)", ge=0, le=100)
    recommendation: str = Field(..., description="Overall recommendation")
    challenges: List[str] = Field(
        default_factory=list, description="Expected challenges")
    key_success_factors: List[str] = Field(
        default_factory=list, description="Key success factors")
    alternative_approaches: List[Dict[str, Any]] = Field(
        default_factory=list, description="Alternative approaches")


class ProjectSimulation(BaseModel):
    """Project simulation results model."""
    complexity: Dict[str, Any] = Field(...,
                                       description="Complexity assessment")
    risks: List[Risk] = Field(default_factory=list,
                              description="Risk assessment")
    timeline: TimelineEstimate = Field(..., description="Timeline estimates")
    resources: ResourceEstimate = Field(...,
                                        description="Resource requirements")
    feasibility: FeasibilityAssessment = Field(
        ..., description="Feasibility assessment")
    similar_projects: List[Dict[str, Any]] = Field(
        default_factory=list, description="Similar historical projects")
    success_probability: float = Field(...,
                                       description="Probability of success", ge=0.0, le=1.0)
    recommendations: List[str] = Field(
        default_factory=list, description="Project recommendations")


class CriteriaEvaluation(BaseModel):
    """Evaluation of a single approval criterion."""
    score: float = Field(...,
                         description="Score for this criterion", ge=0, le=10)
    comments: str = Field("", description="Evaluation comments")


class ProjectApprovalResult(BaseModel):
    """Result of an AI-assisted project review."""
    criteria_evaluations: List[CriteriaEvaluation] = Field(
        default_factory=list, description="Evaluations for each criterion")
    overall_score: float = Field(...,
                                 description="Overall project score", ge=0, le=10)
    assessment: str = Field("", description="Overall assessment comments")

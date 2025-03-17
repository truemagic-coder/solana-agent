"""
General domain models.

These models define structures used across the system.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class MemoryInsightsResponse(BaseModel):
    """Response format for memory insights extraction."""
    insights: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of extracted insights"
    )


class ResponseEvaluation(BaseModel):
    """Evaluation of an AI response."""
    accuracy: float = Field(..., description="Accuracy score", ge=0.0, le=10.0)
    relevance: float = Field(...,
                             description="Relevance score", ge=0.0, le=10.0)
    completeness: float = Field(...,
                                description="Completeness score", ge=0.0, le=10.0)
    clarity: float = Field(..., description="Clarity score", ge=0.0, le=10.0)
    helpfulness: float = Field(...,
                               description="Helpfulness score", ge=0.0, le=10.0)
    overall_score: float = Field(...,
                                 description="Overall quality score", ge=0.0, le=10.0)
    feedback: str = Field(..., description="Detailed evaluation feedback")
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improving the response"
    )

    @validator('overall_score', pre=True, always=True)
    def compute_overall_score(cls, v, values):
        """Compute overall score if not provided."""
        if v is not None:
            return v

        scores = [
            values.get('accuracy', 0),
            values.get('relevance', 0),
            values.get('completeness', 0),
            values.get('clarity', 0),
            values.get('helpfulness', 0)
        ]

        if not all(score is not None for score in scores):
            return 0.0

        return sum(scores) / len(scores)


class HandoffEvaluation(BaseModel):
    """Evaluation of whether a handoff is needed."""
    handoff_needed: bool = Field(...,
                                 description="Whether a handoff is needed")
    target_agent: Optional[str] = Field(
        None, description="Agent to hand off to")
    reason: Optional[str] = Field(None, description="Reason for handoff")
    confidence: float = Field(
        0.0, description="Confidence in the recommendation", ge=0.0, le=1.0)

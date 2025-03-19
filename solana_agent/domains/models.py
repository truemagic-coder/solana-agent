"""
General domain models.

These models define structures used across the system.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class MemoryInsightsResponse(BaseModel):
    """Response format for memory insights extraction."""
    insights: List[Dict[str, Any]] = Field(...,
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
    improvement_suggestions: List[str] = Field(...,
                                               description="Suggestions for improving the response"
                                               )

    @field_validator('overall_score')
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


class QueryAnalysis(BaseModel):
    """Analysis of a user query for routing purposes."""
    primary_specialization: str = Field(...,
                                        description="Main specialization needed")
    secondary_specializations: List[str] = Field(
        ..., description="Other helpful specializations")
    complexity_level: int = Field(...,
                                  description="Complexity level (1-5)")
    requires_human: bool = Field(...,
                                 description="Whether human assistance is likely needed")
    topics: List[str] = Field(..., description="Key topics in the query")
    confidence: float = Field(..., description="Confidence in the analysis")


class ToolUsage(BaseModel):
    use_tool: bool
    tool_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

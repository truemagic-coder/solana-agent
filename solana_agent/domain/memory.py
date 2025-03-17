"""
Memory domain models.

These models define core structures for memory management and insights.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class MemoryInsight(BaseModel):
    """An insight extracted from conversation memory."""
    content: str = Field(..., description="The insight content")
    category: Optional[str] = Field(
        None, description="Category of the insight")
    confidence: float = Field(
        0.0, description="Confidence score of the insight", ge=0.0, le=1.0)
    source: Optional[str] = Field(None, description="Source of the insight")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the insight was created")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")


class MemorySearchResult(BaseModel):
    """Result from a memory search."""
    insight: MemoryInsight
    relevance_score: float = Field(...,
                                   description="Relevance to the query", ge=0.0, le=1.0)


class MemoryInsightsResponse(BaseModel):
    """Response format for memory insights extraction."""
    insights: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of extracted insights"
    )

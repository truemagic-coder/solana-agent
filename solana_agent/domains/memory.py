"""
Memory domain models.

These models define core structures for memory management and insights.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class MemoryInsight(BaseModel):
    """Individual memory insight with type and content."""
    content: str
    category: str
    confidence: float


class MemoryInsightsResponse(BaseModel):
    """Response format for memory insights extraction."""
    insights: List[MemoryInsight]

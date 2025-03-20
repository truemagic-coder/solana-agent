"""
General domain models.

These models define structures used across the system.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Analysis of a user query for routing purposes."""
    primary_specialization: str = Field(...,
                                        description="Main specialization needed")
    secondary_specializations: List[str] = Field(
        ..., description="Other helpful specializations")
    complexity_level: int = Field(...,
                                  description="Complexity level (1-5)")
    topics: List[str] = Field(..., description="Key topics in the query")
    confidence: float = Field(..., description="Confidence in the analysis")


class ToolUsage(BaseModel):
    use_tool: bool
    tool_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

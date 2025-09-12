from typing import List
from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Analysis of a user query for routing purposes."""

    primary_agent: str = Field(
        ...,
        description="Name of the primary agent that should handle this query (must be one of the available agent names)",
    )
    secondary_agents: List[str] = Field(
        default_factory=list,
        description="Names of secondary agents that might be helpful (must be from the available agent names)",
    )
    complexity_level: int = Field(
        default=1, description="Complexity level (1-5)", ge=1, le=5
    )
    topics: List[str] = Field(
        default_factory=list, description="Key topics in the query"
    )
    confidence: float = Field(
        default=0.5, description="Confidence in the analysis", ge=0.0, le=1.0
    )

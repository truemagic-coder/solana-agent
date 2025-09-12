from typing import List
from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Analysis of a user query for routing purposes."""

    primary_agent: str = Field(
        ...,
        description="Name of the primary agent that should handle this query (must be one of the available agent names)",
    )
    secondary_agents: List[str] = Field(
        ...,
        description="Names of secondary agents that might be helpful (must be from the available agent names)",
    )
    complexity_level: int = Field(..., description="Complexity level (1-5)")
    topics: List[str] = Field(..., description="Key topics in the query")
    confidence: float = Field(..., description="Confidence in the analysis")

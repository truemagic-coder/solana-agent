"""
Plugin system domain models.

These models define the core data structures for the plugin system,
aligning with the existing AutoTool implementation pattern.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ToolCallModel(BaseModel):
    """Model for tool calls in agent responses."""
    name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the tool")


class ToolResult(BaseModel):
    """Result from executing a tool."""
    status: str = "success"  # success or error
    result: Optional[Any] = None
    message: Optional[str] = None

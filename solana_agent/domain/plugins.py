"""
Plugin system domain models.

These models define the core data structures for the plugin system,
aligning with the existing AutoTool implementation pattern.
"""
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    required: bool = False
    default: Optional[Any] = None


class ToolSchema(BaseModel):
    """Schema definition for a tool."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


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


# Base models for plugins - these are NOT implementations
class PluginMetadata(BaseModel):
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: Optional[str] = None
    tools: List[str] = Field(default_factory=list)


class PluginConfig(BaseModel):
    """Configuration for a plugin."""
    enabled: bool = True
    settings: Dict[str, Any] = Field(default_factory=dict)

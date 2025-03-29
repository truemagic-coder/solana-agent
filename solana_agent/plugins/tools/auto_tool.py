"""
AutoTool implementation for the Solana Agent system.

This module provides the base AutoTool class that implements the Tool interface
and can be extended to create custom tools.
"""
from typing import Dict, Any

from solana_agent.interfaces.plugins.plugins import Tool


class AutoTool(Tool):
    """Base class for tools that automatically register with the system."""

    def __init__(self, name: str, description: str, registry=None):
        """Initialize the tool with name and description."""
        self._name = name
        self._description = description
        self._config = {}

        # Register with the provided registry if given
        if registry is not None:
            registry.register_tool(self)

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return self._name

    @property
    def description(self) -> str:
        """Get the description of the tool."""
        return self._description

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the tool with settings from config."""
        if config is None:
            raise TypeError("Config cannot be None")
        self._config = config

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool's parameters."""
        # Override in subclasses
        return {}

    async def execute(self, **params) -> Dict[str, Any]:
        """Execute the tool with the provided parameters."""
        # Override in subclasses
        raise NotImplementedError("Tool must implement execute method")

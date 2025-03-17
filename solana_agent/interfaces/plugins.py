"""
Plugin system interfaces.

These interfaces define the contracts for the plugin system,
enabling extensibility through tools and plugins.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable


class Tool(ABC):
    """Interface for tools that can be used by agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of the tool."""
        pass

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the tool with global configuration."""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool parameters."""
        pass

    @abstractmethod
    def execute(self, **params) -> Dict[str, Any]:
        """Execute the tool with the given parameters."""
        pass


class ToolRegistry(ABC):
    """Interface for the tool registry."""

    @abstractmethod
    def register_tool(self, tool: Tool) -> bool:
        """Register a tool in the registry."""
        pass

    @abstractmethod
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        pass

    @abstractmethod
    def assign_tool_to_agent(self, agent_name: str, tool_name: str) -> bool:
        """Assign a tool to an agent."""
        pass

    @abstractmethod
    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all tools available to an agent."""
        pass

    @abstractmethod
    def list_all_tools(self) -> List[str]:
        """List names of all registered tools."""
        pass

    @abstractmethod
    def configure_all_tools(self, config: Dict[str, Any]) -> None:
        """Configure all tools with global configuration."""
        pass


class PluginManager(ABC):
    """Interface for the plugin manager."""

    @abstractmethod
    def load_all_plugins(self) -> int:
        """Load all plugins using entry points and apply configuration."""
        pass

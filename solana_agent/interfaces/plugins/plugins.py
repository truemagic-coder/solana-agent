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
    async def execute(self, **params) -> Dict[str, Any]:
        """Execute the tool with the given parameters."""
        pass


class ToolRegistry(ABC):
    """Interface for the tool registry."""

    @abstractmethod
    def register_tool(self, tool: Tool) -> bool:
        """Register a tool in the registry."""
        pass

    @abstractmethod
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        pass

    @abstractmethod
    def assign_tool_to_agent(self, agent_name: str, tool_name: str) -> bool:
        """Give an agent access to a specific tool."""
        pass

    @abstractmethod
    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all tools available to an agent."""
        pass

    @abstractmethod
    def list_all_tools(self) -> List[str]:
        """List all registered tools."""
        pass

    @abstractmethod
    def configure_all_tools(self, config: Dict[str, Any]) -> None:
        """Configure all registered tools with the same config."""
        pass


class Plugin(ABC):
    """Interface for plugins that can be loaded by the system."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the plugin."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of the plugin."""
        pass

    @abstractmethod
    def initialize(self, tool_registry: ToolRegistry) -> bool:
        """Initialize the plugin and register its tools."""
        pass


class PluginManager(ABC):
    """Interface for the plugin manager."""

    @abstractmethod
    def register_plugin(self, plugin: Plugin) -> bool:
        """Register a plugin in the manager."""
        pass

    @abstractmethod
    def load_plugins(self) -> List[str]:
        """Load all registered plugins."""
        pass

    @abstractmethod
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        pass

    @abstractmethod
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins with their details."""
        pass

    @abstractmethod
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool with the given parameters."""
        pass

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin manager and all plugins."""
        pass

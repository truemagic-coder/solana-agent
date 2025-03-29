"""
Plugin manager for the Solana Agent system.

This module implements the concrete PluginManager that discovers,
loads, and manages plugins.
"""
import importlib
from typing import Dict, List, Any, Optional
import importlib.metadata

from solana_agent.interfaces.plugins.plugins import PluginManager as PluginManagerInterface
from solana_agent.interfaces.plugins.plugins import Plugin
from solana_agent.plugins.registry import ToolRegistry


class PluginManager(PluginManagerInterface):
    """Manager for discovering and loading plugins."""

    # Class variable to track loaded entry points
    _loaded_entry_points = set()

    def __init__(self, config: Optional[Dict[str, Any]] = None, tool_registry: Optional[ToolRegistry] = None):
        """Initialize with optional configuration and tool registry."""
        self.config = config or {}
        self.tool_registry = tool_registry or ToolRegistry()
        self._plugins = {}  # Changed to instance variable

    def register_plugin(self, plugin: Plugin) -> bool:
        """Register a plugin in the manager.

        Args:
            plugin: The plugin to register

        Returns:
            True if registration succeeded, False otherwise
        """
        try:
            # Initialize the plugin with the tool registry first
            plugin.initialize(self.tool_registry)

            # Then configure the plugin
            plugin.configure(self.config)

            # Only store plugin if both initialize and configure succeed
            self._plugins[plugin.name] = plugin
            print(f"Successfully registered plugin {plugin.name}")
            return True

        except Exception as e:
            print(f"Error registering plugin {plugin.name}: {e}")
            # Remove plugin from registry if it was added
            self._plugins.pop(plugin.name, None)
            return False

    def load_plugins(self) -> List[str]:
        """Load all plugins using entry points and apply configuration.

        Returns:
            List of loaded plugin names
        """
        loaded_plugins = []

        # Discover plugins through entry points
        for entry_point in importlib.metadata.entry_points(group='solana_agent.plugins'):
            # Skip if this entry point has already been loaded
            entry_point_id = f"{entry_point.name}:{entry_point.value}"
            if entry_point_id in PluginManager._loaded_entry_points:
                print(
                    f"Skipping already loaded plugin: {entry_point.name}")
                continue

            try:
                print(f"Found plugin entry point: {entry_point.name}")
                PluginManager._loaded_entry_points.add(entry_point_id)
                plugin_factory = entry_point.load()
                plugin = plugin_factory()

                # Register the plugin
                if self.register_plugin(plugin):
                    # Use entry_point.name instead of plugin.name
                    loaded_plugins.append(entry_point.name)

            except Exception as e:
                print(f"Error loading plugin {entry_point.name}: {e}")

        return loaded_plugins

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name.

        Args:
            name: Name of the plugin to retrieve

        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins with their details.

        Returns:
            List of plugin details dictionaries
        """
        return [
            {
                "name": plugin.name,
                "description": plugin.description
            }
            for plugin in self._plugins.values()
        ]

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool with the given parameters.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool

        Returns:
            Dictionary with execution results
        """
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return {"status": "error", "message": f"Tool {tool_name} not found"}

        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin manager and all plugins.

        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        self.tool_registry.configure_all_tools(config)
        print("Configuring all plugins with updated config")
        for name, plugin in self._plugins.items():
            try:
                print(f"Configuring plugin: {name}")
                plugin.configure(self.config)
            except Exception as e:
                print(f"Error configuring plugin {name}: {e}")

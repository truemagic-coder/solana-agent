"""
Plugin manager for the Solana Agent system.

This module implements the concrete PluginManager that discovers,
loads, and manages plugins.
"""
import importlib
from typing import Dict, List, Any, Optional

from solana_agent.interfaces.plugins import PluginManager as PluginManagerInterface
from solana_agent.plugins.registry import ToolRegistry


class PluginManager(PluginManagerInterface):
    """Manager for discovering and loading plugins."""

    # Class variable to track loaded entry points
    _loaded_entry_points = set()

    def __init__(self, config: Optional[Dict[str, Any]] = None, tool_registry: Optional[ToolRegistry] = None):
        """Initialize with optional configuration and tool registry."""
        self.config = config or {}
        self.tool_registry = tool_registry or ToolRegistry()

    def load_all_plugins(self) -> int:
        """Load all plugins using entry points and apply configuration."""
        loaded_count = 0
        plugins = []

        # Discover plugins through entry points
        for entry_point in importlib.metadata.entry_points(group='solana_agent.plugins'):
            # Skip if this entry point has already been loaded
            entry_point_id = f"{entry_point.name}:{entry_point.value}"
            if entry_point_id in PluginManager._loaded_entry_points:
                print(f"Skipping already loaded plugin: {entry_point.name}")
                continue

            try:
                print(f"Found plugin entry point: {entry_point.name}")
                PluginManager._loaded_entry_points.add(entry_point_id)
                plugin_factory = entry_point.load()
                plugin = plugin_factory()
                plugins.append(plugin)

                # Initialize the plugin with config
                if hasattr(plugin, 'initialize') and callable(plugin.initialize):
                    plugin.initialize(self.config)
                    print(
                        f"Initialized plugin {entry_point.name} with config keys: "
                        f"{list(self.config.keys() if self.config else [])}")

                loaded_count += 1
            except Exception as e:
                print(f"Error loading plugin {entry_point.name}: {e}")

        # After all plugins are initialized, register their tools
        for plugin in plugins:
            try:
                if hasattr(plugin, 'get_tools') and callable(plugin.get_tools):
                    tools = plugin.get_tools()
                    # Register each tool with our registry
                    if isinstance(tools, list):
                        for tool in tools:
                            self.tool_registry.register_tool(tool)
                            tool.configure(self.config)
                    else:
                        # Single tool case
                        self.tool_registry.register_tool(tools)
                        tools.configure(self.config)
            except Exception as e:
                print(f"Error registering tools from plugin: {e}")

        return loaded_count

    def register_plugin(self, name: str, plugin_info: Dict[str, Any]) -> bool:
        """Register a plugin manually."""
        try:
            if hasattr(plugin_info, 'get_tools') and callable(plugin_info.get_tools):
                tools = plugin_info.get_tools()

                # Register tools
                if isinstance(tools, list):
                    for tool in tools:
                        self.tool_registry.register_tool(tool)
                else:
                    self.tool_registry.register_tool(tools)

            return True
        except Exception as e:
            print(f"Error registering plugin {name}: {e}")
            return False

    def get_plugin(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a plugin by name."""
        # This implementation doesn't store plugins by name
        # We would need to extend the implementation to properly support this
        return None

    def list_plugins(self) -> List[str]:
        """List names of all loaded plugins."""
        # This implementation doesn't track plugin names
        # We would need to extend the implementation to properly support this
        return []

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool with the given parameters."""
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return {"status": "error", "message": f"Tool {tool_name} not found"}

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin manager and all plugins."""
        self.config.update(config)
        self.tool_registry.configure_all_tools(config)

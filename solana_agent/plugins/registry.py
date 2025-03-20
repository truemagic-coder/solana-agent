"""
Tool registry for the Solana Agent system.

This module implements the concrete ToolRegistry that manages tools 
and their access permissions.
"""
from typing import Dict, List, Any, Optional

from solana_agent.interfaces.plugins.plugins import ToolRegistry as ToolRegistryInterface
from solana_agent.interfaces.plugins.plugins import Tool


class ToolRegistry(ToolRegistryInterface):
    """Instance-based registry that manages tools and their access permissions."""

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools = {}  # name -> tool instance
        self._agent_tools = {}  # agent_name -> [tool_names]

    def register_tool(self, tool: Tool) -> bool:
        """Register a tool with this registry."""
        self._tools[tool.name] = tool
        print(f"Registered tool: {tool.name}")
        return True

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)

    def assign_tool_to_agent(self, agent_name: str, tool_name: str) -> bool:
        """Give an agent access to a specific tool."""
        if tool_name not in self._tools:
            print(f"Error: Tool {tool_name} is not registered")
            return False

        if agent_name not in self._agent_tools:
            self._agent_tools[agent_name] = []

        if tool_name not in self._agent_tools[agent_name]:
            self._agent_tools[agent_name].append(tool_name)

        return True

    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all tools available to an agent."""
        tool_names = self._agent_tools.get(agent_name, [])
        return [
            {
                "name": name,
                "description": self._tools[name].description,
                "parameters": self._tools[name].get_schema()
            }
            for name in tool_names if name in self._tools
        ]

    def list_all_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())

    def configure_all_tools(self, config: Dict[str, Any]) -> None:
        """Configure all registered tools with the same config."""
        for tool in self._tools.values():
            tool.configure(config)

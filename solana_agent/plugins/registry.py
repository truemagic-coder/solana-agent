"""
Tool registry for the Solana Agent system.

This module implements the concrete ToolRegistry that manages tools
and their access permissions.
"""

import logging  # Import logging
from typing import Dict, List, Any, Optional

from solana_agent.interfaces.plugins.plugins import (
    ToolRegistry as ToolRegistryInterface,
)
from solana_agent.interfaces.plugins.plugins import Tool

# Setup logger for this module
logger = logging.getLogger(__name__)


class ToolRegistry(ToolRegistryInterface):
    """Instance-based registry that manages tools and their access permissions."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize an empty tool registry."""
        self._tools = {}  # name -> tool instance
        self._agent_tools = {}  # agent_name -> [tool_names]
        self._config = config or {}

    def register_tool(self, tool: Tool) -> bool:
        """Register a tool with this registry."""
        try:
            tool.configure(self._config)

            self._tools[tool.name] = tool
            logger.info(
                f"Successfully registered and configured tool: {tool.name}"
            )  # Use logger.info
            return True
        except Exception as e:
            logger.error(f"Error registering tool: {str(e)}")  # Use logger.error
            return False

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)

    def assign_tool_to_agent(self, agent_name: str, tool_name: str) -> bool:
        """Give an agent access to a specific tool."""
        if tool_name not in self._tools:
            logger.error(  # Use logger.error
                f"Error: Tool {tool_name} is not registered. Available tools: {list(self._tools.keys())}"
            )
            return False

        # Initialize agent's tool list if not exists
        if agent_name not in self._agent_tools:
            self._agent_tools[agent_name] = [tool_name]
        elif tool_name not in self._agent_tools[agent_name]:
            # Add new tool to existing list
            self._agent_tools[agent_name] = [*self._agent_tools[agent_name], tool_name]

        logger.info(
            f"Successfully assigned tool {tool_name} to agent {agent_name}"
        )  # Use logger.info
        logger.info(
            f"Agent {agent_name} now has access to: {self._agent_tools[agent_name]}"
        )  # Use logger.info

        return True

    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all tools available to an agent."""
        tool_names = self._agent_tools.get(agent_name, [])
        tools = [
            {
                "name": name,
                "description": self._tools[name].description,
                "parameters": self._tools[name].get_schema(),
            }
            for name in tool_names
            if name in self._tools
        ]
        # Changed to debug level as this might be verbose during normal operation
        logger.debug(
            f"Tools available to agent {agent_name}: {[t['name'] for t in tools]}"
        )  # Use logger.debug
        return tools

    def list_all_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())

    def configure_all_tools(self, config: Dict[str, Any]) -> None:
        """Configure all registered tools with new configuration.

        Args:
            config: Configuration dictionary to apply
        """
        self._config.update(config)
        configure_errors = []

        for name, tool in self._tools.items():
            try:
                logger.info(f"Configuring tool: {name}")  # Use logger.info
                tool.configure(self._config)
            except Exception as e:
                logger.error(f"Error configuring tool {name}: {e}")  # Use logger.error
                configure_errors.append((name, str(e)))

        if configure_errors:
            logger.error("The following tools failed to configure:")  # Use logger.error
            for name, error in configure_errors:
                logger.error(f"- {name}: {error}")  # Use logger.error

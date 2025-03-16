"""
Plugin system for the Solana Agent.

This package provides plugin management, tool registration, and plugin discovery
mechanisms that extend the agent system with custom functionality.
"""

from solana_agent.plugins.registry import ToolRegistry
from solana_agent.plugins.manager import PluginManager
from solana_agent.plugins.tools import AutoTool

__all__ = ["ToolRegistry", "PluginManager", "AutoTool"]

"""
Solana Agent - An AI agent framework with routing, memory, and specialized agents.

This package provides a modular framework for building AI agent systems with
multiple specialized agents, memory management, and conversation routing.
"""

__version__ = "14.0.0"  # Update with your actual version

# Client interface (main entry point)
from solana_agent.client.solana_agent import SolanaAgent

# Factory for creating agent systems
from solana_agent.factories.agent_factory import SolanaAgentFactory

# Useful tools and utilities
from solana_agent.plugins.manager import PluginManager
from solana_agent.plugins.registry import ToolRegistry
from solana_agent.plugins.tools import AutoTool

# Package metadata
__all__ = [
    # Main client interfaces
    "SolanaAgent",

    # Factories
    "SolanaAgentFactory",

    # Tools
    "PluginManager",
    "ToolRegistry",
    "AutoTool",
]

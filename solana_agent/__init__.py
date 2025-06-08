"""
Solana Agent - An AI agent framework with routing, memory, and specialized agents.

This package provides a modular framework for building AI agent systems with
multiple specialized agents, memory management, and conversation routing.
"""

# Client interface (main entry point)
from solana_agent.client.solana_agent import SolanaAgent

# Factory for creating agent systems
from solana_agent.factories.agent_factory import SolanaAgentFactory

# Useful tools and utilities
from solana_agent.plugins.manager import PluginManager
from solana_agent.plugins.registry import ToolRegistry
from solana_agent.plugins.tools.auto_tool import AutoTool
from solana_agent.interfaces.plugins.plugins import Tool
from solana_agent.interfaces.guardrails.guardrails import (
    InputGuardrail,
    OutputGuardrail,
)

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
    "Tool",
    # Guardrails
    "InputGuardrail",
    "OutputGuardrail",
]

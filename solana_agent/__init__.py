"""Public Solana Agent SDK for hosted chat, wallets, MCP, and x402 execution."""

from solana_agent.client.solana_agent import SolanaAgent
from solana_agent.factories.agent_factory import SolanaAgentFactory
from solana_agent.plugins.manager import PluginManager
from solana_agent.plugins.registry import ToolRegistry
from solana_agent.plugins.tools.auto_tool import AutoTool
from solana_agent.interfaces.plugins.plugins import Tool

__all__ = [
    "SolanaAgent",
    "SolanaAgentFactory",
    "PluginManager",
    "ToolRegistry",
    "AutoTool",
    "Tool",
]

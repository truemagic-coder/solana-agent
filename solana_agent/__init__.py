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

# Domain models (for public API usage)
from solana_agent.domains import AIAgent, HumanAgent, OrganizationMission
from solana_agent.domains import Ticket, TicketStatus, TicketPriority
from solana_agent.domains import Handoff, HandoffEvaluation

# Core services (for advanced usage)
from solana_agent.services.agent import AgentService
from solana_agent.services.query import QueryService
from solana_agent.services.routing import RoutingService

# Useful tools and utilities
from solana_agent.plugins import PluginManager

# Package metadata
__all__ = [
    # Main client interfaces
    "SolanaAgent",

    # Factories
    "SolanaAgentFactory",

    # Domain models
    "AIAgent",
    "HumanAgent",
    "OrganizationMission",
    "Ticket",
    "TicketStatus",
    "TicketPriority",
    "Handoff",
    "HandoffEvaluation",

    # Core services
    "AgentService",
    "QueryService",
    "RoutingService",

    # Tools
    "PluginManager",
]

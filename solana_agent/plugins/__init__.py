"""
Plugin system for the Solana Agent.

This package provides plugin management, tool registration, and plugin discovery
mechanisms that extend the agent system with custom functionality.
"""

from solana_agent.plugins.manager import *
from solana_agent.plugins.registry import *
from solana_agent.plugins.tools import *

# Version of the domain model
__version__ = '14.0.0'

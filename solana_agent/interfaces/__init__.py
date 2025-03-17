"""
Abstract interfaces for the Solana Agent system.

These interfaces define the contracts that concrete implementations
must adhere to, following the Dependency Inversion Principle.

This package contains:
- Repository interfaces for data access
- Provider interfaces for external service adapters
- Service interfaces for business logic components
"""

from solana_agent.interfaces.plugins import *
from solana_agent.interfaces.providers import *
from solana_agent.interfaces.repositories import *
from solana_agent.interfaces.services import *

# Version of the domain model
__version__ = '14.0.0'

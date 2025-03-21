"""
Simplified client interface for interacting with the Solana Agent system.

This module provides a clean API for end users to interact with
the agent system without dealing with internal implementation details.
"""
import json
import importlib.util
from typing import AsyncGenerator, Dict, Any

from solana_agent.factories.agent_factory import SolanaAgentFactory
from solana_agent.interfaces.client.client import SolanaAgent


class SolanaAgent(SolanaAgent):
    """Simplified client interface for interacting with the agent system."""

    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """Initialize the agent system from config file or dictionary.

        Args:
            config_path: Path to configuration file (JSON or Python)
            config: Configuration dictionary
        """
        if not config and not config_path:
            raise ValueError("Either config or config_path must be provided")

        if config_path:
            with open(config_path, "r") as f:
                if config_path.endswith(".json"):
                    config = json.load(f)
                else:
                    # Assume it's a Python file
                    spec = importlib.util.spec_from_file_location(
                        "config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    config = config_module.config

        self.query_service = SolanaAgentFactory.create_from_config(config)

    async def process(self, user_id: str, message: str) -> AsyncGenerator[str, None]:
        """Process a user message and return the response stream.

        Args:
            user_id: User ID
            message: User message

        Returns:
            Async generator yielding response chunks
        """
        async for chunk in self.query_service.process(user_id, message):
            yield chunk

    async def get_user_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "desc"  # "asc" for oldest-first, "desc" for newest-first
    ) -> Dict[str, Any]:
        """
        Get paginated message history for a user.

        Args:
            user_id: User ID
            page_num: Page number (starting from 1)
            page_size: Number of messages per page
            sort_order: Sort order ("asc" or "desc")

        Returns:
            Dictionary with paginated results and metadata
        """
        return await self.query_service.get_user_history(
            user_id, page_num, page_size, sort_order
        )

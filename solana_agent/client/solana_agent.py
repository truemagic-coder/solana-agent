"""
Simplified client interface for interacting with the Solana Agent system.

This module provides a clean API for end users to interact with
the agent system without dealing with internal implementation details.
"""
import json
import datetime
import importlib.util
from typing import Dict, List, Any, AsyncGenerator

from solana_agent.factories.agent_factory import SolanaAgentFactory


class SolanaAgent:
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

    def register_agent(
        self,
        name: str,
        instructions: str,
        specialization: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Register a new AI agent.

        Args:
            name: Agent name
            instructions: Agent instructions
            specialization: Agent specialization
            model: LLM model to use
        """
        self.query_service.agent_service.register_ai_agent(
            name=name,
            instructions=instructions,
            specialization=specialization,
            model=model,
        )

    def register_human_agent(
        self, agent_id: str, name: str, specialization: str, notification_handler=None
    ) -> None:
        """Register a human agent.

        Args:
            agent_id: Agent ID
            name: Agent name
            specialization: Agent specialization
            notification_handler: Optional handler for notifications
        """
        self.query_service.agent_service.register_human_agent(
            agent_id=agent_id,
            name=name,
            specialization=specialization,
            notification_handler=notification_handler,
        )

    async def get_pending_surveys(self, user_id: str) -> List[Dict[str, Any]]:
        """Get pending surveys for a user.

        Args:
            user_id: User ID

        Returns:
            List of pending surveys
        """
        if not hasattr(self.query_service, "nps_service"):
            return []

        # Query for pending surveys from the NPS service
        surveys = self.query_service.nps_service.nps_repository.db.find(
            "nps_surveys",
            {
                "user_id": user_id,
                "status": "pending",
                "created_at": {
                    "$gte": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
                }
            }
        )

        return surveys

    async def submit_survey_response(self, survey_id: str, score: int, feedback: str = "") -> bool:
        """Submit a response to an NPS survey.

        Args:
            survey_id: Survey ID
            score: Survey score
            feedback: Optional feedback

        Returns:
            True if submission was successful
        """
        if not hasattr(self.query_service, "nps_service"):
            return False

        # Process the survey response
        return self.query_service.nps_service.process_response(survey_id, score, feedback)

    async def get_user_history_paginated(
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
        # Check if memory service exists
        if not hasattr(self.query_service, "memory_service"):
            return {
                "data": [],
                "total": 0,
                "page": page_num,
                "page_size": page_size,
                "total_pages": 0,
                "error": "Memory service not available"
            }

        # Use the memory service to get the paginated history
        return await self.query_service.memory_service.get_user_history_paginated(
            user_id, page_num, page_size, sort_order
        )

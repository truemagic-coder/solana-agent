from abc import ABC, abstractmethod
from typing import List, Optional

from solana_agent.domains.agent import AIAgent


class AgentRepository(ABC):
    """Interface for agent data access."""

    @abstractmethod
    def get_ai_agent_by_name(self, name: str) -> Optional[AIAgent]:
        """Get an AI agent by name."""
        pass

    @abstractmethod
    def get_ai_agent(self, name: str) -> Optional[AIAgent]:
        """Get an AI agent by name."""
        pass

    @abstractmethod
    def get_all_ai_agents(self) -> List[AIAgent]:
        """Get all AI agents."""
        pass

    @abstractmethod
    def save_ai_agent(self, agent: AIAgent) -> bool:
        """Save an AI agent."""
        pass

    @abstractmethod
    def delete_ai_agent(self, name: str) -> bool:
        """Delete an AI agent."""
        pass

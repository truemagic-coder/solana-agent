from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, BinaryIO, Dict, List, Literal, Union

from solana_agent.domains.agent import AIAgent


class AgentService(ABC):
    """Interface for agent management and response generation."""

    @abstractmethod
    def register_ai_agent(self, name: str, instructions: str, specialization: str) -> None:
        """Register an AI agent with its specialization."""
        pass

    @abstractmethod
    def get_agent_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for an agent."""
        pass

    @abstractmethod
    def get_specializations(self) -> Dict[str, str]:
        """Get all registered specializations."""
        pass

    @abstractmethod
    async def generate_response(
        self,
        agent_name: str,
        user_id: str,
        query: Union[str, Path, BinaryIO],
        memory_context: str = "",
        output_format: Literal["text", "audio"] = "text",
        voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                       "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
        audio_instructions: str = None,
    ) -> AsyncGenerator[Union[str, bytes], None]:
        """Generate a response from an agent."""
        pass

    @abstractmethod
    def assign_tool_for_agent(self, agent_name: str, tool_name: str) -> bool:
        """Assign a tool to an agent."""
        pass

    @abstractmethod
    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get tools available to an agent."""
        pass

    @abstractmethod
    def execute_tool(self, agent_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on behalf of an agent."""
        pass

    @abstractmethod
    def get_all_ai_agents(self) -> Dict[str, AIAgent]:
        """Get all registered AI agents."""
        pass

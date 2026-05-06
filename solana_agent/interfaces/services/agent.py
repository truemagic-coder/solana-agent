from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel

from solana_agent.domains.agent import AIAgent


class AgentService(ABC):
    """Interface for agent management and response generation."""

    @abstractmethod
    def register_ai_agent(
        self,
        name: str,
        instructions: str,
        specialization: str,
        capture_name: Optional[str] = None,
        capture_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an AI agent with its specialization."""

    @abstractmethod
    def get_agent_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for an agent."""

    @abstractmethod
    async def generate_response(
        self,
        agent_name: str,
        privy_user_id: str,
        query: Union[str, bytes],
        runtime_context: Optional[Dict[str, Any]] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        output_format: Literal["text", "audio"] = "text",
        audio_voice: Literal[
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "fable",
            "onyx",
            "nova",
            "sage",
            "shimmer",
        ] = "nova",
        audio_output_format: Literal[
            "mp3", "opus", "aac", "flac", "wav", "pcm"
        ] = "aac",
        prompt: Optional[str] = None,
        output_model: Optional[Type[BaseModel]] = None,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:
        """Generate a response from an agent."""

    @abstractmethod
    def assign_tool_for_agent(self, agent_name: str, tool_name: str) -> bool:
        """Assign a tool to an agent."""

    @abstractmethod
    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get tools available to an agent."""

    @abstractmethod
    async def execute_tool(
        self,
        agent_name: str,
        tool_name: str,
        parameters: Dict[str, Any],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a tool on behalf of an agent."""

    @abstractmethod
    def get_all_ai_agents(self) -> Dict[str, AIAgent]:
        """Get all registered AI agents."""

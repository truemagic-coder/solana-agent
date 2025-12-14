from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Literal, Optional, Type, Union

from pydantic import BaseModel
from solana_agent.interfaces.plugins.plugins import Tool
from solana_agent.interfaces.services.routing import RoutingService as RoutingInterface


class SolanaAgent(ABC):
    """Interface for the Solana Agent client."""

    @abstractmethod
    async def process(
        self,
        user_id: str,
        message: Union[str, bytes],
        prompt: Optional[str] = None,
        output_format: Literal["text", "audio"] = "text",
        capture_schema: Optional[Dict[str, Any]] = None,
        capture_name: Optional[str] = None,
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
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
        router: Optional[RoutingInterface] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        output_model: Optional[Type[BaseModel]] = None,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:
        """Process a user message and return the response stream."""
        pass

    @abstractmethod
    async def delete_user_history(self, user_id: str) -> None:
        """Delete the conversation history for a user."""
        pass

    @abstractmethod
    async def get_user_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """Get paginated message history for a user."""
        pass

    @abstractmethod
    def register_tool(self, agent_name: str, tool: Tool) -> bool:
        """Register a tool with the agent system."""
        pass

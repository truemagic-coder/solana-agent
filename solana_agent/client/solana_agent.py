"""
Simplified client interface for interacting with the Solana Agent system.

This module provides a clean API for end users to interact with
the agent system without dealing with internal implementation details.
"""
import json
import importlib.util
from typing import AsyncGenerator, Dict, Any, Literal, Optional, Union

from solana_agent.factories.agent_factory import SolanaAgentFactory
from solana_agent.interfaces.client.client import SolanaAgent as SolanaAgentInterface
from solana_agent.interfaces.plugins.plugins import Tool


class SolanaAgent(SolanaAgentInterface):
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

    async def process(
        self,
        user_id: str,
        message: Union[str, bytes],
        prompt: Optional[str] = None,
        output_format: Literal["text", "audio"] = "text",
        audio_voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                             "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
        audio_instructions: Optional[str] = None,
        audio_output_format: Literal['mp3', 'opus',
                                     'aac', 'flac', 'wav', 'pcm'] = "aac",
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
    ) -> AsyncGenerator[Union[str, bytes], None]:  # pragma: no cover
        """Process a user message and return the response stream.

        Args:
            user_id: User ID
            message: Text message or audio bytes
            prompt: Optional prompt for the agent
            output_format: Response format ("text" or "audio")
            audio_voice: Voice to use for audio output
            audio_instructions: Optional instructions for audio synthesis
            audio_output_format: Audio output format
            audio_input_format: Audio input format

        Returns:
            Async generator yielding response chunks (text strings or audio bytes)
        """
        async for chunk in self.query_service.process(
            user_id=user_id,
            query=message,
            output_format=output_format,
            audio_voice=audio_voice,
            audio_instructions=audio_instructions,
            audio_output_format=audio_output_format,
            audio_input_format=audio_input_format,
            prompt=prompt,
        ):
            yield chunk

    async def delete_user_history(self, user_id: str) -> None:
        """
        Delete the conversation history for a user.

        Args:
            user_id: User ID
        """
        await self.query_service.delete_user_history(user_id)

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

    def register_tool(self, tool: Tool) -> bool:
        """
        Register a tool with the agent system.

        Args:
            tool: Tool instance to register

        Returns:
            True if successful, False
        """

        try:
            print(f"Attempting to register tool: {tool.name}")
            success = self.query_service.agent_service.tool_registry.register_tool(
                tool)
            if success:
                print(f"Tool {tool.name} registered successfully")
                # Get all agents and assign the tool to them
                agents = self.query_service.agent_service.get_all_ai_agents()
                for agent_name in agents:
                    print(f"Assigning {tool.name} to agent {agent_name}")
                    self.query_service.agent_service.assign_tool_for_agent(
                        agent_name, tool.name)
            return success
        except Exception as e:
            print(f"Error in register_tool: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

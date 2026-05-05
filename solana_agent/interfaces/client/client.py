from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Literal, Optional, Type, Union

from pydantic import BaseModel

from solana_agent.interfaces.plugins.plugins import Tool


class SolanaAgent(ABC):
    """Interface for the public Solana Agent client."""

    @abstractmethod
    async def process(
        self,
        user_id: str,
        message: Union[str, bytes],
        runtime_context: Optional[Dict[str, Any]] = None,
        search_enabled: Optional[bool] = None,
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
        images: Optional[List[Union[str, bytes]]] = None,
        output_model: Optional[Type[BaseModel]] = None,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:
        """Process a user message and return the response stream."""

    @abstractmethod
    def register_tool(self, agent_name: str, tool: Tool) -> bool:
        """Register a tool with the agent system."""

    @abstractmethod
    async def create_wallet(
        self,
        user_id: str,
        chain_type: Literal["solana", "ethereum"] = "solana",
    ) -> Dict[str, Any]:
        """Create or return the hosted Privy wallet for a user."""

    @abstractmethod
    async def get_wallet_address(self, user_id: str) -> str:
        """Get the hosted Privy wallet public address for a user."""

    @abstractmethod
    async def export_wallet_private_key(
        self,
        wallet_id: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Export a Privy wallet private key for self-custody flows."""

    @abstractmethod
    async def prepare_x402_runtime_context(
        self,
        user_id: str,
        runtime_context: Optional[Dict[str, Any]] = None,
        chain_type: Literal["solana", "ethereum"] = "solana",
    ) -> Dict[str, Any]:
        """Create or fetch the hosted Privy wallet and return x402 runtime context."""

    @abstractmethod
    async def get_account_summary(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get billing and usage summary for the authenticated wallet account."""

    @abstractmethod
    async def get_usage_report(
        self,
        granularity: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        group_by: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get time-series hosted usage buckets for the authenticated wallet account."""

    @abstractmethod
    async def get_usage_forecast(
        self,
        window_days: int = 30,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get hosted usage forecast for the authenticated wallet account."""

    @abstractmethod
    async def get_pricing_info(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get effective hosted pricing details for the authenticated wallet account."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Literal, Optional, Type, Union

from pydantic import BaseModel

from solana_agent.interfaces.plugins.plugins import Tool


class SolanaAgent(ABC):
    """Interface for the public Solana Agent client."""

    @abstractmethod
    async def message(
        self,
        message: Union[str, bytes],
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
        **runtime_context: Any,
    ) -> Union[str, bytes, BaseModel, None]:
        """Process one request and collect the final non-streaming response."""

    @abstractmethod
    async def context(
        self,
        *,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        memory_ttl_tier: Optional[Literal["work", "project"]] = None,
        service_tier: Optional[Literal["standard", "priority"]] = None,
        search_enabled: Optional[bool] = None,
        chain_type: Literal["solana", "ethereum"] = "solana",
        **runtime_context: Any,
    ) -> Dict[str, Any]:
        """Build flat hosted runtime context for a message call."""

    @abstractmethod
    async def process_message(
        self,
        message: Union[str, bytes],
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
        **runtime_context: Any,
    ) -> Union[str, bytes, BaseModel, None]:
        """Process one request and collect the final non-streaming response."""

    @abstractmethod
    async def process(
        self,
        message: Union[str, bytes],
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
        **runtime_context: Any,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:
        """Low-level chunk iterator. Use process_message for hosted non-streaming requests."""

    @abstractmethod
    def register_tool(self, agent_name: str, tool: Tool) -> bool:
        """Register a tool with the agent system."""

    @abstractmethod
    async def create_privy_user(self) -> Dict[str, Any]:
        """Create a hosted Privy user and return its DID."""

    @abstractmethod
    async def create_wallet(
        self,
        privy_user_id: Optional[str] = None,
        chain_type: Literal["solana", "ethereum"] = "solana",
    ) -> Dict[str, Any]:
        """Create or return the active Privy-backed wallet for a user."""

    @abstractmethod
    async def rotate_wallet(
        self,
        privy_user_id: Optional[str] = None,
        chain_type: Literal["solana", "ethereum"] = "solana",
    ) -> Dict[str, Any]:
        """Rotate the active Privy-backed wallet for a user and return old_wallets."""

    @abstractmethod
    async def export_wallet_private_key(
        self,
        wallet_id: Optional[str] = None,
        privy_user_id: Optional[str] = None,
        chain_type: Literal["solana", "ethereum"] = "solana",
    ) -> str:
        """Export the hosted wallet private key for self-custody."""

    @abstractmethod
    async def get_wallet_address(self, wallet_id: Optional[str] = None) -> str:
        """Get the hosted wallet public address."""

    @abstractmethod
    async def prepare_x402_runtime_context(
        self,
        *,
        chain_type: Literal["solana", "ethereum"] = "solana",
        **runtime_context: Any,
    ) -> Dict[str, Any]:
        """Create or fetch the configured Privy user's wallet and return x402 runtime context."""

    @abstractmethod
    async def get_account_summary(
        self,
        **runtime_context: Any,
    ) -> Dict[str, Any]:
        """Get billing and usage summary for the authenticated wallet account."""

    @abstractmethod
    async def get_usage_report(
        self,
        granularity: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        group_by: Optional[str] = None,
        **runtime_context: Any,
    ) -> Dict[str, Any]:
        """Get time-series hosted usage buckets for the authenticated wallet account."""

    @abstractmethod
    async def get_usage_forecast(
        self,
        window_days: int = 30,
        **runtime_context: Any,
    ) -> Dict[str, Any]:
        """Get hosted usage forecast for the authenticated wallet account."""

    @abstractmethod
    async def get_pricing_info(
        self,
        **runtime_context: Any,
    ) -> Dict[str, Any]:
        """Get effective hosted pricing details for the authenticated wallet account."""

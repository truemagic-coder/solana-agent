"""
Simplified client interface for interacting with the Solana Agent system.

This module provides a clean API for end users to interact with
the agent system without dealing with internal implementation details.
"""

import json
import importlib.util
from typing import AsyncGenerator, Dict, Any, List, Literal, Optional, Type, Union

from pydantic import BaseModel

from solana_agent.factories.agent_factory import SolanaAgentFactory
from solana_agent.interfaces.client.client import SolanaAgent as SolanaAgentInterface
from solana_agent.interfaces.plugins.plugins import Tool
from solana_agent.interfaces.services.routing import RoutingService as RoutingInterface
from solana_agent.tools.utils.x402 import (
    export_privy_wallet_private_key,
    resolve_x402_privy_config,
)


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
                    spec = importlib.util.spec_from_file_location("config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    config = config_module.config

        self.config = dict(config)
        self.query_service = SolanaAgentFactory.create_from_config(config)

    def _config_section(self, key: str) -> Dict[str, Any]:
        section = self.config.get(key, {})
        return section if isinstance(section, dict) else {}

    def _provider_config(self) -> Dict[str, Any]:
        ai_config = self._config_section("ai")
        if ai_config:
            return ai_config
        return self._config_section("openai")

    @staticmethod
    def _runtime_privy_wallet_id(
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        context = dict(runtime_context or {})
        for context_key in ("privy_wallet_id", "hosted_privy_wallet_id"):
            value = str(context.get(context_key) or "").strip()
            if value:
                return value

        wallet_payload = context.get("privy_wallet")
        if isinstance(wallet_payload, dict):
            value = str(
                wallet_payload.get("wallet_id") or wallet_payload.get("id") or ""
            ).strip()
            if value:
                return value

        return ""

    def _x402_request_uses_hosted_privy_wallet(self) -> bool:
        tools_config = self._config_section("tools")
        x402_config = tools_config.get("x402_request")
        if not isinstance(x402_config, dict):
            return False

        configured_auth_mode = str(x402_config.get("auth_mode") or "").strip()
        provider_auth_mode = str(self._provider_config().get("auth_mode") or "").strip()
        return (configured_auth_mode or provider_auth_mode) == "x402_privy"

    def _uses_hosted_privy_wallet_for_x402(self) -> bool:
        provider_auth_mode = str(self._provider_config().get("auth_mode") or "").strip()
        return (
            provider_auth_mode == "x402_privy"
            or self._x402_request_uses_hosted_privy_wallet()
        )

    def _get_provider_method(self, method_name: str, capability: str):
        agent_service = getattr(self.query_service, "agent_service", None)
        llm_provider = getattr(agent_service, "llm_provider", None)
        method = getattr(llm_provider, method_name, None)
        if method is None:
            raise NotImplementedError(
                f"{capability} is not available for the configured provider"
            )
        return method

    def _merge_runtime_context(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
        *,
        user_id: Optional[str] = None,
        search_enabled: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        context = dict(runtime_context or {})
        if user_id is not None:
            context["user_id"] = user_id
        if search_enabled is not None:
            context["search_enabled"] = bool(search_enabled)
        return context or None

    async def _prepare_process_runtime_context(
        self,
        user_id: str,
        runtime_context: Optional[Dict[str, Any]] = None,
        *,
        search_enabled: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        context = self._merge_runtime_context(
            runtime_context,
            user_id=user_id,
            search_enabled=search_enabled,
        )
        if not self._uses_hosted_privy_wallet_for_x402():
            return context

        if self._runtime_privy_wallet_id(context):
            return context

        return await self.prepare_x402_runtime_context(
            user_id=user_id,
            runtime_context=context,
        )

    async def process(
        self,
        user_id: str,
        message: Union[str, bytes],
        runtime_context: Optional[Dict[str, Any]] = None,
        search_enabled: Optional[bool] = None,
        prompt: Optional[str] = None,
        capture_schema: Optional[Dict[str, Any]] = None,
        capture_name: Optional[str] = None,
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
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
        router: Optional[RoutingInterface] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        output_model: Optional[Type[BaseModel]] = None,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:  # pragma: no cover
        """Process a user message (text or audio) and optional images, returning the response stream.

        Args:
            user_id: User ID
            message: Text message or audio bytes
            runtime_context: Per-request hosted runtime metadata
            search_enabled: Enable the hosted search add-on for this request
            prompt: Optional prompt for the agent
            output_format: Response format ("text" or "audio")
            capture_schema: Optional Pydantic schema for structured output
            capture_name: Optional name for structured output capture
            audio_voice: Voice to use for audio output
            audio_output_format: Audio output format
            audio_input_format: Audio input format
            router: Optional routing service for processing
            images: Optional list of image URLs (str) or image bytes.
            output_model: Optional Pydantic model for structured output

        Returns:
            Async generator yielding response chunks (text strings or audio bytes)
        """
        prepared_runtime_context = await self._prepare_process_runtime_context(
            user_id,
            runtime_context,
            search_enabled=search_enabled,
        )

        async for chunk in self.query_service.process(
            user_id=user_id,
            query=message,
            runtime_context=prepared_runtime_context,
            images=images,
            output_format=output_format,
            audio_voice=audio_voice,
            audio_output_format=audio_output_format,
            audio_input_format=audio_input_format,
            prompt=prompt,
            router=router,
            output_model=output_model,
            capture_schema=capture_schema,
            capture_name=capture_name,
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
        sort_order: str = "desc",  # "asc" for oldest-first, "desc" for newest-first
    ) -> Dict[str, Any]:  # pragma: no cover
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

    def register_tool(self, agent_name: str, tool: Tool) -> bool:
        """
        Register a tool with the agent system.

        Args:
            agent_name: Name of the agent to register the tool with
            tool: Tool instance to register

        Returns:
            True if successful, False
        """
        success = self.query_service.agent_service.tool_registry.register_tool(tool)
        if success:
            self.query_service.agent_service.assign_tool_for_agent(
                agent_name, tool.name
            )
        return success

    async def create_wallet(
        self,
        user_id: str,
        chain_type: Literal["solana", "ethereum"] = "solana",
    ) -> Dict[str, Any]:
        """Create or return the hosted Privy wallet for a user."""
        method = self._get_provider_method("create_wallet", "Hosted wallet management")
        return await method(user_id=user_id, chain_type=chain_type)

    async def get_wallet_address(self, user_id: str) -> str:
        """Return the hosted Privy wallet public address for a user."""
        method = self._get_provider_method(
            "get_wallet_address",
            "Hosted wallet management",
        )
        payload = await method(user_id=user_id)
        address = str(
            payload.get("address") or payload.get("public_address") or ""
        ).strip()
        if not address:
            raise ValueError("Hosted wallet response is missing an address")
        return address

    async def export_wallet_private_key(
        self,
        wallet_id: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Export a Privy wallet private key for self-custody flows."""
        agent_service = getattr(self.query_service, "agent_service", None)
        llm_provider = getattr(agent_service, "llm_provider", None)
        context = dict(runtime_context or {})
        if wallet_id is not None:
            context["privy_wallet_id"] = wallet_id

        effective_wallet_id = self._runtime_privy_wallet_id(context)
        if not effective_wallet_id:
            raise ValueError(
                "wallet_id is required. Pass it explicitly or set runtime_context.privy_wallet_id, "
                "runtime_context.hosted_privy_wallet_id, or runtime_context.privy_wallet.id."
            )

        privy_config = resolve_x402_privy_config(
            auth_mode="x402_privy",
            privy_wallet_id=effective_wallet_id,
            privy_app_id=str(getattr(llm_provider, "privy_app_id", "") or "").strip()
            or None,
            privy_app_secret=str(
                getattr(llm_provider, "privy_app_secret", "") or ""
            ).strip()
            or None,
            privy_authorization_signature=str(
                getattr(llm_provider, "privy_authorization_signature", "") or ""
            ).strip()
            or None,
            privy_request_expiry=str(
                getattr(llm_provider, "privy_request_expiry", "") or ""
            ).strip()
            or None,
            privy_api_url=str(getattr(llm_provider, "privy_api_url", "") or "").strip()
            or None,
            rpc_url=str(getattr(llm_provider, "x402_rpc_url", "") or "").strip()
            or None,
        )
        if privy_config is None:
            raise NotImplementedError(
                "Privy wallet export requires configured Privy app credentials on the provider"
            )
        return await export_privy_wallet_private_key(privy_config)

    async def prepare_x402_runtime_context(
        self,
        user_id: str,
        runtime_context: Optional[Dict[str, Any]] = None,
        chain_type: Literal["solana", "ethereum"] = "solana",
    ) -> Dict[str, Any]:
        """Return runtime context populated with the hosted Privy wallet."""
        context = self._merge_runtime_context(runtime_context, user_id=user_id) or {}
        existing_wallet_id = self._runtime_privy_wallet_id(context)
        if existing_wallet_id:
            context.setdefault("privy_wallet_id", existing_wallet_id)
            context.setdefault("hosted_privy_wallet_id", existing_wallet_id)
            return context

        wallet = await self.create_wallet(user_id=user_id, chain_type=chain_type)
        wallet_id = str(wallet.get("wallet_id") or wallet.get("id") or "").strip()
        if wallet_id:
            context["privy_wallet_id"] = wallet_id
            context["hosted_privy_wallet_id"] = wallet_id

        address = str(
            wallet.get("address")
            or wallet.get("wallet_address")
            or wallet.get("public_key")
            or ""
        ).strip()
        if address:
            context["privy_wallet_address"] = address
            context["privy_wallet_public_key"] = address

        return context

    async def get_account_summary(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get hosted billing and usage summary for the authenticated wallet account."""
        method = self._get_provider_method(
            "get_account_summary",
            "Account reporting",
        )
        return await method(runtime_context=runtime_context)

    async def get_usage_report(
        self,
        granularity: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        group_by: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get hosted usage buckets for the authenticated wallet account."""
        method = self._get_provider_method(
            "get_usage_report",
            "Account reporting",
        )
        return await method(
            granularity,
            from_date=from_date,
            to_date=to_date,
            group_by=group_by,
            runtime_context=runtime_context,
        )

    async def get_usage_forecast(
        self,
        window_days: int = 30,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get hosted usage forecast for the authenticated wallet account."""
        method = self._get_provider_method(
            "get_usage_forecast",
            "Account reporting",
        )
        return await method(window_days=window_days, runtime_context=runtime_context)

    async def get_pricing_info(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get hosted pricing information for the authenticated wallet account."""
        method = self._get_provider_method(
            "get_pricing_info",
            "Account reporting",
        )
        return await method(runtime_context=runtime_context)

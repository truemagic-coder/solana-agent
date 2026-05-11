"""
Simplified client interface for interacting with the Solana Agent system.

This module provides a clean API for end users to interact with
the agent system without dealing with internal implementation details.
"""

import json
import importlib.util
import logging
import os
import time
from copy import deepcopy
from typing import AsyncGenerator, Dict, Any, List, Literal, Optional, Type, Union

from pydantic import BaseModel

from solana_agent.factories.agent_factory import SolanaAgentFactory
from solana_agent.factories.agent_factory import DEFAULT_AGI_MEMORY_MODEL
from solana_agent.factories.agent_factory import DEFAULT_AGI_STATELESS_MODEL
from solana_agent.interfaces.client.client import SolanaAgent as SolanaAgentInterface
from solana_agent.interfaces.plugins.plugins import Tool
from solana_agent.local_state import (
    load_saved_privy_user_id,
    load_saved_wallet_id,
    save_privy_user_id,
    save_wallet_id,
)


logger = logging.getLogger(__name__)


def _timing_trace_enabled() -> bool:
    return str(os.getenv("SOLANA_AGENT_TIMING_TRACE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class SolanaAgent(SolanaAgentInterface):
    """Simplified client interface for interacting with the agent system."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        *,
        instructions: Optional[str] = None,
        privy_user_id: Optional[str] = None,
        name: Optional[str] = None,
        specialization: Optional[str] = None,
        model: Optional[str] = None,
        stateless_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[List[str]] = None,
        x402_preferred_asset: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        context_window_tokens: Optional[int] = None,
        tokenizer_model: Optional[str] = None,
    ):
        """Initialize the hosted Solana Agent SDK.

        Args:
            config_path: Path to configuration file (JSON or Python)
            config: Configuration dictionary
            instructions: System instructions for the single public agent
            privy_user_id: Hosted Privy user DID owned by the app/user
        """
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

        self.config = self._build_config(
            config,
            instructions=instructions,
            privy_user_id=privy_user_id,
            name=name,
            specialization=specialization,
            model=model,
            stateless_model=stateless_model,
            base_url=base_url,
            api_key=api_key,
            tools=tools,
            x402_preferred_asset=x402_preferred_asset,
            max_output_tokens=max_output_tokens,
            context_window_tokens=context_window_tokens,
            tokenizer_model=tokenizer_model,
        )
        self._apply_saved_privy_user_id()
        self.query_service = SolanaAgentFactory.create_from_config(self.config)
        self._cached_hosted_wallet_id: str | None = None
        self._cached_hosted_wallet_address: str | None = None

    @staticmethod
    def _build_config(
        config: Optional[Dict[str, Any]],
        **public_kwargs: Any,
    ) -> Dict[str, Any]:
        normalized_config = deepcopy(config) if config is not None else {"ai": {}}
        has_public_overrides = any(
            value is not None for value in public_kwargs.values()
        )
        if "ai" not in normalized_config and not has_public_overrides:
            return normalized_config

        ai_config = normalized_config.setdefault("ai", {})
        if not isinstance(ai_config, dict):
            raise ValueError("AI config in config['ai'] must be a mapping.")

        public_to_ai_key = {
            "instructions": "instructions",
            "privy_user_id": "privy_user_id",
            "name": "name",
            "specialization": "specialization",
            "model": "model",
            "stateless_model": "stateless_model",
            "base_url": "base_url",
            "api_key": "api_key",
            "tools": "tools",
            "x402_preferred_asset": "x402_preferred_asset",
            "max_output_tokens": "max_output_tokens",
            "context_window_tokens": "context_window_tokens",
            "tokenizer_model": "tokenizer_model",
        }
        for source_key, ai_key in public_to_ai_key.items():
            value = public_kwargs.get(source_key)
            if value is not None:
                ai_config[ai_key] = value

        return normalized_config

    def _config_section(self, key: str) -> Dict[str, Any]:
        section = self.config.get(key, {})
        return section if isinstance(section, dict) else {}

    def _provider_config(self) -> Dict[str, Any]:
        ai_config = self._config_section("ai")
        if ai_config:
            return ai_config
        return self._config_section("openai")

    def _configured_base_url(self) -> str | None:
        base_url = str(self._provider_config().get("base_url") or "").strip()
        return base_url or None

    def _privy_user_id_target_section(self) -> Dict[str, Any]:
        ai_config = self.config.get("ai")
        if isinstance(ai_config, dict) and ai_config:
            return ai_config

        legacy_provider_config = self.config.get("openai")
        if isinstance(legacy_provider_config, dict) and legacy_provider_config:
            return legacy_provider_config

        if not isinstance(ai_config, dict):
            ai_config = {}
            self.config["ai"] = ai_config
        return ai_config

    def _set_configured_privy_user_id(self, privy_user_id: str) -> None:
        normalized_privy_user_id = str(privy_user_id or "").strip()
        if not normalized_privy_user_id:
            return

        self._privy_user_id_target_section()["privy_user_id"] = normalized_privy_user_id

        query_service = getattr(self, "query_service", None)
        agent_service = getattr(query_service, "agent_service", None)
        llm_provider = getattr(agent_service, "llm_provider", None)
        if llm_provider is not None:
            setattr(llm_provider, "privy_user_id", normalized_privy_user_id)

    @staticmethod
    def _shell_privy_user_id() -> str | None:
        for env_var in (
            "SOLANA_AGENT_PRIVY_USER_ID",
            "PRIVY_USER_ID",
            "privy_user_id",
        ):
            privy_user_id = str(os.getenv(env_var) or "").strip()
            if privy_user_id:
                return privy_user_id
        return None

    def _apply_saved_privy_user_id(self) -> None:
        try:
            self._configured_privy_user_id()
            return
        except ValueError:
            pass

        shell_privy_user_id = self._shell_privy_user_id()
        if shell_privy_user_id:
            self._set_configured_privy_user_id(shell_privy_user_id)
            return

        saved_privy_user_id = load_saved_privy_user_id(
            base_url=self._configured_base_url()
        )
        if saved_privy_user_id:
            self._set_configured_privy_user_id(saved_privy_user_id)

    def _saved_wallet_id(self) -> str | None:
        saved_wallet_id = load_saved_wallet_id(base_url=self._configured_base_url())
        return str(saved_wallet_id or "").strip() or None

    def _remember_wallet_id(self, payload: Any) -> str | None:
        normalized_wallet_id = ""
        if isinstance(payload, dict):
            normalized_wallet_id = str(
                payload.get("wallet_id") or payload.get("id") or ""
            ).strip()
        else:
            normalized_wallet_id = str(payload or "").strip()

        if not normalized_wallet_id:
            return None

        try:
            save_wallet_id(
                normalized_wallet_id,
                base_url=self._configured_base_url(),
            )
        except OSError:
            return normalized_wallet_id
        return normalized_wallet_id

    def _remember_wallet_address(self, payload: Any) -> str | None:
        normalized_wallet_address = ""
        if isinstance(payload, dict):
            normalized_wallet_address = str(
                payload.get("address")
                or payload.get("wallet_address")
                or payload.get("public_address")
                or payload.get("public_key")
                or ""
            ).strip()
        else:
            normalized_wallet_address = str(payload or "").strip()

        if not normalized_wallet_address:
            return None

        self._cached_hosted_wallet_address = normalized_wallet_address
        return normalized_wallet_address

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

    def _configured_privy_user_id(self) -> str:
        ai_config = self._config_section("ai")
        privy_user_id = str(ai_config.get("privy_user_id") or "").strip()
        if privy_user_id:
            return privy_user_id

        legacy_provider_config = self._config_section("openai")
        legacy_privy_user_id = str(
            legacy_provider_config.get("privy_user_id") or ""
        ).strip()
        if legacy_privy_user_id:
            return legacy_privy_user_id

        raise ValueError(
            "config.ai.privy_user_id or privy_user_id is required for hosted messages and wallet helpers"
        )

    def _configured_stateless_model(self) -> str:
        provider_config = self._provider_config()
        model = str(
            provider_config.get("stateless_model") or DEFAULT_AGI_STATELESS_MODEL
        ).strip()
        return model or DEFAULT_AGI_STATELESS_MODEL

    def _resolve_context_model(self, model: Optional[str]) -> Optional[str]:
        requested_model = str(model or "").strip()
        if not requested_model:
            return None
        normalized = requested_model.lower()
        if normalized == "chat":
            return self._configured_stateless_model()
        if normalized == "stateless":
            return self._configured_stateless_model()
        if normalized == "memory":
            return DEFAULT_AGI_MEMORY_MODEL
        return requested_model

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
        search_enabled: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        context = dict(runtime_context or {})
        if "privy_user_id" in context:
            raise ValueError(
                "privy_user_id belongs in config.ai.privy_user_id, not runtime_context"
            )
        if isinstance(context.get("runtime_context"), dict):
            raise ValueError(
                "Pass runtime context values as flat keyword arguments, not runtime_context={...}"
            )
        if search_enabled is not None:
            context["search_enabled"] = bool(search_enabled)
        return context or None

    async def _prepare_process_runtime_context(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
        *,
        search_enabled: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        started_at = time.perf_counter()
        merged_runtime_context = self._merge_runtime_context(
            runtime_context,
            search_enabled=search_enabled,
        )
        prepared_runtime_context = await self.prepare_x402_runtime_context(
            **(merged_runtime_context or {}),
        )
        if _timing_trace_enabled():
            logger.warning(
                "SDK timing: prepare_runtime_context seconds=%.3f context_keys=%s",
                time.perf_counter() - started_at,
                sorted((prepared_runtime_context or {}).keys()),
            )
        return prepared_runtime_context

    async def process(
        self,
        message: Union[str, bytes],
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
        images: Optional[List[Union[str, bytes]]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        **runtime_context: Any,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:  # pragma: no cover
        """Process a user message (text or audio) and optional images, yielding low-level chunks.

        Args:
            message: Text message or audio bytes
            **runtime_context: Per-request hosted runtime metadata
            search_enabled: Enable the hosted search add-on for this request
            prompt: Optional prompt for the agent
            output_format: Response format ("text" or "audio")
            capture_schema: Optional Pydantic schema for structured output
            capture_name: Optional name for structured output capture
            audio_voice: Voice to use for audio output
            audio_output_format: Audio output format
            audio_input_format: Audio input format
            images: Optional list of image URLs (str) or image bytes.
            output_model: Optional Pydantic model for structured output

        Returns:
            Async generator yielding response chunks. Hosted chat completions are
            executed as non-streaming requests; use process_message to collect
            the final response automatically.
        """
        privy_user_id = self._configured_privy_user_id()
        prepared_runtime_context = await self._prepare_process_runtime_context(
            runtime_context or None,
            search_enabled=search_enabled,
        )

        async for chunk in self.query_service.process(
            privy_user_id=privy_user_id,
            query=message,
            runtime_context=prepared_runtime_context,
            images=images,
            output_format=output_format,
            audio_voice=audio_voice,
            audio_output_format=audio_output_format,
            audio_input_format=audio_input_format,
            prompt=prompt,
            output_model=output_model,
            capture_schema=capture_schema,
            capture_name=capture_name,
        ):
            yield chunk

    async def process_message(
        self,
        message: Union[str, bytes],
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
        images: Optional[List[Union[str, bytes]]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        **runtime_context: Any,
    ) -> Union[str, bytes, BaseModel, None]:
        """Process one request and collect the final response payload.

        This is the default hosted-SDK entrypoint because the hosted platform
        delivers chat completions as non-streaming JSON responses.
        """

        text_parts: list[str] = []
        audio_parts: list[bytes] = []
        structured_output: BaseModel | None = None
        last_chunk: Union[str, bytes, BaseModel, None] = None

        async for chunk in self.process(
            message=message,
            search_enabled=search_enabled,
            prompt=prompt,
            capture_schema=capture_schema,
            capture_name=capture_name,
            output_format=output_format,
            audio_voice=audio_voice,
            audio_output_format=audio_output_format,
            audio_input_format=audio_input_format,
            images=images,
            output_model=output_model,
            **runtime_context,
        ):
            last_chunk = chunk
            if isinstance(chunk, str):
                text_parts.append(chunk)
            elif isinstance(chunk, bytes):
                audio_parts.append(chunk)
            elif isinstance(chunk, BaseModel):
                structured_output = chunk

        if structured_output is not None:
            return structured_output
        if audio_parts:
            return b"".join(audio_parts)
        if text_parts:
            return "".join(text_parts)
        return last_chunk

    async def message(
        self,
        message: Union[str, bytes],
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
        images: Optional[List[Union[str, bytes]]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        **runtime_context: Any,
    ) -> Union[str, bytes, BaseModel, None]:
        """Process one request using the README-friendly method name."""
        return await self.process_message(
            message=message,
            search_enabled=search_enabled,
            prompt=prompt,
            capture_schema=capture_schema,
            capture_name=capture_name,
            output_format=output_format,
            audio_voice=audio_voice,
            audio_output_format=audio_output_format,
            audio_input_format=audio_input_format,
            images=images,
            output_model=output_model,
            **runtime_context,
        )

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
        context = dict(runtime_context or {})

        if conversation_id:
            context["conversation_id"] = str(conversation_id).strip()

        resolved_model = self._resolve_context_model(model)
        if resolved_model:
            context["model"] = resolved_model

        if memory_ttl_tier:
            context["memory_ttl_tier"] = str(memory_ttl_tier).strip().lower()

        if service_tier:
            normalized_service_tier = str(service_tier).strip().lower()
            if normalized_service_tier not in {"standard", "priority"}:
                raise ValueError("service_tier must be one of: standard, priority")
            context["service_tier"] = normalized_service_tier

        if search_enabled is not None:
            context["search_enabled"] = bool(search_enabled)

        return await self.prepare_x402_runtime_context(
            chain_type=chain_type,
            **context,
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

    async def create_privy_user(self) -> Dict[str, Any]:
        """Create a hosted Privy user and return its DID."""
        method = self._get_provider_method(
            "create_privy_user",
            "Hosted wallet management",
        )
        payload = await method()
        privy_user_id = str(
            payload.get("privy_user_id") or payload.get("id") or ""
        ).strip()
        if privy_user_id:
            self._set_configured_privy_user_id(privy_user_id)
            try:
                save_privy_user_id(
                    privy_user_id,
                    base_url=self._configured_base_url(),
                )
            except OSError as exc:
                raise RuntimeError(
                    "Hosted Privy user was created but could not be saved locally. "
                    f"Copy this DID now: {privy_user_id}. {exc}"
                ) from exc
        return payload

    async def create_wallet(
        self,
        privy_user_id: Optional[str] = None,
        chain_type: Literal["solana", "ethereum"] = "solana",
    ) -> Dict[str, Any]:
        """Create or return the active Privy-backed wallet for a user."""
        method = self._get_provider_method("create_wallet", "Hosted wallet management")
        resolved_privy_user_id = str(privy_user_id or "").strip()
        if not resolved_privy_user_id:
            resolved_privy_user_id = self._configured_privy_user_id()
        payload = await method(
            privy_user_id=resolved_privy_user_id,
            chain_type=chain_type,
        )
        remembered_wallet_id = self._remember_wallet_id(payload)
        if remembered_wallet_id:
            self._cached_hosted_wallet_id = remembered_wallet_id
        self._remember_wallet_address(payload)
        return payload

    async def rotate_wallet(
        self,
        privy_user_id: Optional[str] = None,
        chain_type: Literal["solana", "ethereum"] = "solana",
    ) -> Dict[str, Any]:
        """Rotate the active Privy-backed wallet for a user."""
        method = self._get_provider_method("rotate_wallet", "Hosted wallet management")
        resolved_privy_user_id = str(privy_user_id or "").strip()
        if not resolved_privy_user_id:
            resolved_privy_user_id = self._configured_privy_user_id()
        payload = await method(
            privy_user_id=resolved_privy_user_id,
            chain_type=chain_type,
        )
        remembered_wallet_id = self._remember_wallet_id(payload)
        if remembered_wallet_id:
            self._cached_hosted_wallet_id = remembered_wallet_id
        self._remember_wallet_address(payload)
        return payload

    async def export_wallet_private_key(
        self,
        wallet_id: Optional[str] = None,
        privy_user_id: Optional[str] = None,
        chain_type: Literal["solana", "ethereum"] = "solana",
    ) -> str:
        """Export the hosted wallet private key for self-custody."""
        method = self._get_provider_method(
            "export_wallet_private_key",
            "Hosted wallet management",
        )
        resolved_privy_user_id = str(privy_user_id or "").strip()
        if not resolved_privy_user_id:
            resolved_privy_user_id = self._configured_privy_user_id()
        resolved_wallet_id = str(wallet_id or "").strip()
        if not resolved_wallet_id:
            resolved_wallet_id = self._saved_wallet_id() or ""
        payload = await method(
            privy_user_id=resolved_privy_user_id,
            wallet_id=resolved_wallet_id or None,
            chain_type=chain_type,
        )
        remembered_wallet_id = self._remember_wallet_id(
            payload if isinstance(payload, dict) else resolved_wallet_id
        )
        if remembered_wallet_id:
            self._cached_hosted_wallet_id = remembered_wallet_id
        private_key = str(payload.get("private_key") or "").strip()
        if not private_key:
            raise ValueError("Hosted wallet export response is missing a private_key")
        return private_key

    async def get_wallet_address(self, wallet_id: Optional[str] = None) -> str:
        """Return the hosted wallet public address.

        When ``wallet_id`` is omitted, the active wallet for ``config.ai.privy_user_id``
        is created or fetched first.
        """
        normalized_wallet_id = str(wallet_id or "").strip()
        if not normalized_wallet_id:
            wallet = await self.create_wallet()
            address = str(
                wallet.get("address")
                or wallet.get("wallet_address")
                or wallet.get("public_address")
                or wallet.get("public_key")
                or ""
            ).strip()
            if address:
                return address
            normalized_wallet_id = str(
                wallet.get("wallet_id") or wallet.get("id") or ""
            ).strip()

        method = self._get_provider_method(
            "get_wallet_address",
            "Hosted wallet management",
        )
        payload = await method(wallet_id=normalized_wallet_id)
        remembered_wallet_id = self._remember_wallet_id(
            payload if isinstance(payload, dict) else normalized_wallet_id
        )
        if remembered_wallet_id:
            self._cached_hosted_wallet_id = remembered_wallet_id
        address = str(
            payload.get("address") or payload.get("public_address") or ""
        ).strip()
        if not address:
            raise ValueError("Hosted wallet response is missing an address")
        self._cached_hosted_wallet_address = address
        return address

    async def prepare_x402_runtime_context(
        self,
        *,
        chain_type: Literal["solana", "ethereum"] = "solana",
        **runtime_context: Any,
    ) -> Dict[str, Any]:
        """Return runtime context populated with the Privy-backed wallet."""
        context = self._merge_runtime_context(runtime_context or None) or {}
        privy_user_id = self._configured_privy_user_id()
        existing_wallet_id = self._runtime_privy_wallet_id(context)
        if not existing_wallet_id:
            existing_wallet_id = self._cached_hosted_wallet_id or self._saved_wallet_id() or ""
        if existing_wallet_id:
            context.setdefault("privy_wallet_id", existing_wallet_id)
            context.setdefault("hosted_privy_wallet_id", existing_wallet_id)
            existing_wallet_address = str(
                context.get("privy_wallet_address")
                or context.get("privy_wallet_public_key")
                or ""
            ).strip()
            if not existing_wallet_address:
                existing_wallet_address = str(
                    self._cached_hosted_wallet_address or ""
                ).strip()
            if not existing_wallet_address:
                existing_wallet_address = await self.get_wallet_address(
                    existing_wallet_id
                )
            self._cached_hosted_wallet_id = existing_wallet_id
            context.setdefault("privy_wallet_address", existing_wallet_address)
            context.setdefault(
                "privy_wallet_public_key",
                existing_wallet_address,
            )
            return context

        wallet = await self.create_wallet(
            privy_user_id=privy_user_id,
            chain_type=chain_type,
        )

        wallet_id = str(wallet.get("wallet_id") or wallet.get("id") or "").strip()
        if wallet_id:
            self._cached_hosted_wallet_id = wallet_id
            context["privy_wallet_id"] = wallet_id
            context["hosted_privy_wallet_id"] = wallet_id

        address = str(
            wallet.get("address")
            or wallet.get("wallet_address")
            or wallet.get("public_key")
            or ""
        ).strip()
        if address:
            self._cached_hosted_wallet_address = address
            context["privy_wallet_address"] = address
            context["privy_wallet_public_key"] = address

        return context

    async def get_account_summary(
        self,
        **runtime_context: Any,
    ) -> Dict[str, Any]:
        """Get hosted billing and usage summary for the authenticated wallet account."""
        method = self._get_provider_method(
            "get_account_summary",
            "Account reporting",
        )
        return await method(
            runtime_context=self._merge_runtime_context(runtime_context)
        )

    async def get_usage_report(
        self,
        granularity: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        group_by: Optional[str] = None,
        **runtime_context: Any,
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
            runtime_context=self._merge_runtime_context(runtime_context),
        )

    async def get_usage_forecast(
        self,
        window_days: int = 30,
        **runtime_context: Any,
    ) -> Dict[str, Any]:
        """Get hosted usage forecast for the authenticated wallet account."""
        method = self._get_provider_method(
            "get_usage_forecast",
            "Account reporting",
        )
        return await method(
            window_days=window_days,
            runtime_context=self._merge_runtime_context(runtime_context),
        )

    async def get_pricing_info(
        self,
        **runtime_context: Any,
    ) -> Dict[str, Any]:
        """Get hosted pricing information for the authenticated wallet account."""
        method = self._get_provider_method(
            "get_pricing_info",
            "Account reporting",
        )
        return await method(
            runtime_context=self._merge_runtime_context(runtime_context)
        )

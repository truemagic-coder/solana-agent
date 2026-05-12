"""Factory for creating the thin public Solana Agent SDK runtime."""

import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv

from solana_agent.adapters.openai_adapter import OpenAIAdapter
from solana_agent.domains.agent import BusinessMission
from solana_agent.plugins.manager import PluginManager
from solana_agent.services.agent import AgentService
from solana_agent.services.query import QueryService

logger = logging.getLogger(__name__)

DEFAULT_AGI_BASE_URL = "https://ai.solana-agent.com/v1"
DEFAULT_AGI_MEMORY_MODEL = "solana-agent-memory"
DEFAULT_AGI_STATELESS_MODEL = "solana-agent-chat"
PRIMARY_AI_CONFIG_KEY = "ai"
LEGACY_AI_CONFIG_KEY = "openai"
LOCAL_MEMORY_CONFIG_ERROR = (
    "Local mongo/zep memory configuration is not supported in the public SDK. "
    "Use the hosted service for conversation memory."
)
GUARDRAILS_CONFIG_ERROR = (
    "Guardrails are not supported in the public SDK. "
    "Run those policies in your hosted service instead."
)
MULTI_AGENT_CONFIG_ERROR = "The public SDK supports exactly one agent. Remove routing and extra agents from the config."
LEGACY_AGENTS_CONFIG_ERROR = (
    "The public SDK no longer supports config['agents'] or config['agent_tools']. "
    "Move single-agent fields into config['ai'] instead."
)
UNSUPPORTED_PUBLIC_TOOL_ERROR = (
    "x402_request is not supported in the public SDK. "
    "Use MCP for external tools and hosted wallet APIs for Solana Agent flows."
)
UNSUPPORTED_PUBLIC_TOOL_NAMES = frozenset({"x402_request"})


class SolanaAgentFactory:
    """Factory for building the public single-agent SDK runtime."""

    @staticmethod
    def _load_optional_dotenv_file() -> None:
        dotenv_path = str(
            os.getenv("SOLANA_AGENT_DOTENV_PATH")
            or os.getenv("OPENAI_API_DOTENV_PATH")
            or ""
        ).strip()
        if not dotenv_path:
            return
        load_dotenv(dotenv_path=dotenv_path, override=False)

    @staticmethod
    def _provider_config(config: Dict[str, Any]) -> Dict[str, Any]:
        if PRIMARY_AI_CONFIG_KEY in config:
            provider_config = config.get(PRIMARY_AI_CONFIG_KEY)
            if not isinstance(provider_config, dict):
                raise ValueError("AI config in config['ai'] must be a mapping.")
            return provider_config

        if LEGACY_AI_CONFIG_KEY in config:
            provider_config = config.get(LEGACY_AI_CONFIG_KEY)
            if not isinstance(provider_config, dict):
                raise ValueError("AI config in config['openai'] must be a mapping.")
            logger.warning("config['openai'] is deprecated; use config['ai'] instead.")
            return provider_config

        raise ValueError("AI config is required in config['ai'].")

    @staticmethod
    def _business_mission(config: Dict[str, Any]) -> BusinessMission | None:
        org_config = config.get("business")
        if not isinstance(org_config, dict):
            return None

        return BusinessMission(
            mission=org_config.get("mission", ""),
            values=[
                {"name": key, "description": value}
                for key, value in org_config.get("values", {}).items()
            ],
            goals=org_config.get("goals", []),
            voice=org_config.get("voice", ""),
        )

    @staticmethod
    def _single_agent_config(config: Dict[str, Any]) -> Dict[str, Any]:
        if "agents" in config or "agent_tools" in config:
            raise ValueError(LEGACY_AGENTS_CONFIG_ERROR)

        agent_config = SolanaAgentFactory._provider_config(config)
        instructions = str(agent_config.get("instructions") or "").strip()

        name = str(agent_config.get("name") or "default").strip() or "default"
        specialization = (
            str(agent_config.get("specialization") or "general").strip() or "general"
        )

        tools = agent_config.get("tools", [])
        if tools is None:
            tools = []
        if not isinstance(tools, list):
            raise ValueError("AI config in config['ai']['tools'] must be a list.")
        normalized_tools = [
            tool_name
            for tool_name in (str(tool or "").strip() for tool in tools)
            if tool_name
        ]
        SolanaAgentFactory._reject_unsupported_public_tools(config, normalized_tools)

        normalized = {
            "name": name,
            "instructions": instructions,
            "specialization": specialization,
            "tools": normalized_tools,
        }
        for optional_key in ("capture_name", "capture_schema"):
            if optional_key in agent_config:
                normalized[optional_key] = agent_config[optional_key]
        normalized["name"] = name
        normalized["instructions"] = instructions
        normalized["specialization"] = specialization
        return normalized

    @staticmethod
    def _reject_unsupported_public_tools(
        config: Dict[str, Any],
        tool_names: list[str],
    ) -> None:
        configured_tools = set(tool_names)
        tools_config = config.get("tools")
        if isinstance(tools_config, dict):
            configured_tools.update(
                str(tool_name or "").strip() for tool_name in tools_config
            )
        if configured_tools & UNSUPPORTED_PUBLIC_TOOL_NAMES:
            raise ValueError(UNSUPPORTED_PUBLIC_TOOL_ERROR)

    @staticmethod
    def _assign_agent_tools(
        agent_service: AgentService,
        config: Dict[str, Any],
        agent_name: str,
        agent_config: Dict[str, Any],
    ) -> None:
        for tool_name in agent_config.get("tools", []) or []:
            agent_service.assign_tool_for_agent(agent_name, tool_name)

        SolanaAgentFactory._validate_agent_tools_config(config, agent_name)
        configured_agent_tools = config.get("agent_tools", {})
        if not configured_agent_tools:
            return

        for tool_name in configured_agent_tools.get(agent_name, []) or []:
            agent_service.assign_tool_for_agent(agent_name, tool_name)

    @staticmethod
    def _validate_agent_tools_config(config: Dict[str, Any], agent_name: str) -> None:
        configured_agent_tools = config.get("agent_tools", {})
        if not configured_agent_tools:
            return
        del agent_name
        raise ValueError(LEGACY_AGENTS_CONFIG_ERROR)

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> QueryService:  # pragma: no cover
        legacy_provider_keys = [
            provider_name
            for provider_name in ("groq", "cerebras", "grok")
            if provider_name in config
        ]
        if legacy_provider_keys:
            joined = ", ".join(sorted(legacy_provider_keys))
            raise ValueError(
                f"Legacy provider sections are no longer supported: {joined}. "
                "Use config['ai'] with the AGI transport instead."
            )

        if "mongo" in config or "zep" in config:
            raise ValueError(LOCAL_MEMORY_CONFIG_ERROR)
        if config.get("guardrails"):
            raise ValueError(GUARDRAILS_CONFIG_ERROR)

        SolanaAgentFactory._load_optional_dotenv_file()
        agent_config = SolanaAgentFactory._single_agent_config(config)
        SolanaAgentFactory._validate_agent_tools_config(config, agent_config["name"])

        provider_config = SolanaAgentFactory._provider_config(config)
        auth_mode = str(provider_config.get("auth_mode") or "").strip()
        if auth_mode:
            raise ValueError(
                "Public SDK auth is managed by the hosted service. Remove auth_mode from config['ai']."
            )
        auth_mode = "hosted_managed"

        llm_x402_preferred_asset = provider_config.get("x402_preferred_asset")
        llm_privy_user_id = str(provider_config.get("privy_user_id") or "").strip()
        llm_private_key = str(
            provider_config.get("private_key") or os.getenv("SOLANA_PRIVATE_KEY") or ""
        ).strip()
        llm_x402_rpc_url = str(
            provider_config.get("x402_rpc_url")
            or os.getenv("HELIUS_RPC_URL")
            or os.getenv("SOLANA_RPC_URL")
            or os.getenv("OPENAI_API_SOLANA_RPC_URL")
            or ""
        ).strip()

        llm_api_key = provider_config.get("api_key")
        requested_model = str(provider_config.get("model") or "").strip() or None
        stateless_model = (
            str(
                provider_config.get("stateless_model") or DEFAULT_AGI_STATELESS_MODEL
            ).strip()
            or DEFAULT_AGI_STATELESS_MODEL
        )
        llm_model = requested_model
        llm_base_url = provider_config.get("base_url")
        llm_reasoning_effort = provider_config.get("reasoning_effort")
        llm_context_window_tokens = provider_config.get("context_window_tokens")
        llm_max_output_tokens = provider_config.get("max_output_tokens")
        llm_tokenizer_model = provider_config.get("tokenizer_model")

        llm_api_key = llm_api_key or "x402"
        if requested_model in {None, "memory"}:
            llm_model = DEFAULT_AGI_MEMORY_MODEL
        elif requested_model in {"chat", "stateless"}:
            llm_model = stateless_model
        else:
            llm_model = requested_model
        llm_base_url = llm_base_url or DEFAULT_AGI_BASE_URL

        llm_adapter_kwargs: Dict[str, Any] = {
            "api_key": llm_api_key,
            "model": llm_model,
        }
        if llm_base_url:
            llm_adapter_kwargs["base_url"] = llm_base_url
        if llm_reasoning_effort:
            llm_adapter_kwargs["reasoning_effort"] = llm_reasoning_effort
        if llm_context_window_tokens is not None:
            llm_adapter_kwargs["context_window_tokens"] = llm_context_window_tokens
        if llm_max_output_tokens is not None:
            llm_adapter_kwargs["max_output_tokens"] = llm_max_output_tokens
        if llm_tokenizer_model:
            llm_adapter_kwargs["tokenizer_model"] = llm_tokenizer_model
        llm_adapter_kwargs["auth_mode"] = auth_mode
        if llm_x402_preferred_asset:
            llm_adapter_kwargs["x402_preferred_asset"] = llm_x402_preferred_asset
        if llm_privy_user_id:
            llm_adapter_kwargs["privy_user_id"] = llm_privy_user_id
        if llm_private_key:
            llm_adapter_kwargs["private_key"] = llm_private_key
        if llm_x402_rpc_url:
            llm_adapter_kwargs["x402_rpc_url"] = llm_x402_rpc_url

        logfire_config = config.get("logfire")
        if isinstance(logfire_config, dict):
            logfire_api_key = logfire_config.get("api_key")
            if not logfire_api_key:
                raise ValueError("Pydantic Logfire API key is required.")
            llm_adapter_kwargs["logfire_api_key"] = logfire_api_key

        llm_adapter = OpenAIAdapter(**llm_adapter_kwargs)
        agent_service = AgentService(
            llm_provider=llm_adapter,
            business_mission=SolanaAgentFactory._business_mission(config),
            config=config,
            model=llm_model,
        )

        agent_service.plugin_manager = PluginManager(
            config=config,
            tool_registry=agent_service.tool_registry,
        )
        try:
            loaded_plugins = agent_service.plugin_manager.load_plugins()
            logger.info("Loaded plugins: %s", loaded_plugins)
        except Exception as error:
            logger.error("Error loading plugins: %s", error)

        agent_service.register_ai_agent(
            name=agent_config["name"],
            instructions=agent_config["instructions"],
            specialization=agent_config["specialization"],
            capture_name=agent_config.get("capture_name"),
            capture_schema=agent_config.get("capture_schema"),
        )
        SolanaAgentFactory._assign_agent_tools(
            agent_service,
            config,
            agent_config["name"],
            agent_config,
        )

        return QueryService(agent_service=agent_service)

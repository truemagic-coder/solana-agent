"""Factory for creating the thin public Solana Agent SDK runtime."""

import logging
from typing import Any, Dict

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


class SolanaAgentFactory:
    """Factory for building the public single-agent SDK runtime."""

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
        agents = config.get("agents", [])
        if agents is None:
            agents = []
        if not isinstance(agents, list):
            raise ValueError("config['agents'] must be a list.")
        if len(agents) > 1:
            raise ValueError(MULTI_AGENT_CONFIG_ERROR)

        if not agents:
            return {
                "name": "default",
                "instructions": "You are a helpful Solana AI assistant for hosted wallet and x402 workflows.",
                "specialization": "general",
            }

        agent_config = agents[0]
        if not isinstance(agent_config, dict):
            raise ValueError("Each agent config must be a mapping.")

        instructions = str(agent_config.get("instructions") or "").strip()
        if not instructions:
            raise ValueError("The configured agent must include instructions.")

        name = str(agent_config.get("name") or "default").strip() or "default"
        specialization = (
            str(agent_config.get("specialization") or "general").strip() or "general"
        )

        normalized = dict(agent_config)
        normalized["name"] = name
        normalized["instructions"] = instructions
        normalized["specialization"] = specialization
        return normalized

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
        if not isinstance(configured_agent_tools, dict):
            raise ValueError("config['agent_tools'] must be a mapping.")

        unexpected_agents = [
            configured_name
            for configured_name in configured_agent_tools
            if configured_name != agent_name
        ]
        if unexpected_agents:
            raise ValueError(MULTI_AGENT_CONFIG_ERROR)

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

        agent_config = SolanaAgentFactory._single_agent_config(config)
        SolanaAgentFactory._validate_agent_tools_config(config, agent_config["name"])

        provider_config = SolanaAgentFactory._provider_config(config)
        auth_mode = provider_config.get("auth_mode", "api_key")
        if auth_mode not in {"api_key", "x402_private_key", "x402_privy"}:
            raise ValueError(
                "Unsupported auth_mode. Supported values are: api_key, x402_private_key, x402_privy."
            )

        use_hosted_transport = auth_mode in {"x402_private_key", "x402_privy"}
        llm_private_key = provider_config.get("private_key")
        llm_privy_app_id = provider_config.get("privy_app_id") or provider_config.get(
            "app_id"
        )
        llm_privy_app_secret = provider_config.get(
            "privy_app_secret"
        ) or provider_config.get("app_secret")
        llm_privy_authorization_signature = provider_config.get(
            "privy_authorization_signature"
        )
        llm_privy_request_expiry = provider_config.get("privy_request_expiry")
        llm_privy_api_url = provider_config.get("privy_api_url")
        llm_x402_rpc_url = provider_config.get("x402_rpc_url")
        llm_x402_preferred_asset = provider_config.get("x402_preferred_asset")

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

        if use_hosted_transport:
            if auth_mode == "x402_private_key":
                if not llm_private_key:
                    raise ValueError(
                        "AI x402 signing key is required when auth_mode is x402_private_key."
                    )
            elif not (llm_privy_app_id and llm_privy_app_secret):
                raise ValueError(
                    "Privy app credentials are required when auth_mode is x402_privy. "
                    "Set privy_app_id and privy_app_secret; pass privy_wallet_id at runtime."
                )

            llm_api_key = llm_api_key or "x402"
            if requested_model in {None, "memory"}:
                llm_model = DEFAULT_AGI_MEMORY_MODEL
            elif requested_model == "stateless":
                llm_model = stateless_model
            else:
                llm_model = requested_model
            llm_base_url = llm_base_url or DEFAULT_AGI_BASE_URL
        elif not llm_api_key:
            raise ValueError(
                "AI API key is required unless auth_mode is x402_private_key or x402_privy."
            )

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
        if use_hosted_transport:
            llm_adapter_kwargs["auth_mode"] = auth_mode
            llm_adapter_kwargs["private_key"] = llm_private_key
            if llm_privy_app_id:
                llm_adapter_kwargs["privy_app_id"] = llm_privy_app_id
            if llm_privy_app_secret:
                llm_adapter_kwargs["privy_app_secret"] = llm_privy_app_secret
            if llm_privy_authorization_signature:
                llm_adapter_kwargs["privy_authorization_signature"] = (
                    llm_privy_authorization_signature
                )
            if llm_privy_request_expiry:
                llm_adapter_kwargs["privy_request_expiry"] = llm_privy_request_expiry
            if llm_privy_api_url:
                llm_adapter_kwargs["privy_api_url"] = llm_privy_api_url
            if llm_x402_rpc_url:
                llm_adapter_kwargs["x402_rpc_url"] = llm_x402_rpc_url
            if llm_x402_preferred_asset:
                llm_adapter_kwargs["x402_preferred_asset"] = llm_x402_preferred_asset

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

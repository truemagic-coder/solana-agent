"""
Factory for creating and wiring components of the Solana Agent system.

This module handles the creation and dependency injection for all
services and components used in the system.
"""

import importlib
import logging
from typing import Dict, Any, List

# Service imports
from solana_agent.interfaces.guardrails.guardrails import (
    InputGuardrail,
    OutputGuardrail,
)
from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService

# Adapter imports
from solana_agent.adapters.openai_adapter import OpenAIAdapter
from solana_agent.domains.agent import BusinessMission
from solana_agent.plugins.manager import PluginManager

# Deprecated local-memory hooks remain as sentinels so legacy tests can patch
# them while the v34 runtime rejects those config paths explicitly.
MongoDBAdapter = None
MemoryRepository = None

# Setup logger for this module
logger = logging.getLogger(__name__)

DEFAULT_AGI_BASE_URL = "https://ai.solana-agent.com/v1"
DEFAULT_AGI_MEMORY_MODEL = "solana-agent-memory"
DEFAULT_AGI_STATELESS_MODEL = "solana-agent-chat"
PRIMARY_AI_CONFIG_KEY = "ai"
LEGACY_AI_CONFIG_KEY = "openai"
LOCAL_MEMORY_CONFIG_ERROR = (
    "Local mongo/zep memory configuration is no longer supported in the v34 AGI runtime. "
    "Use config['ai'] with remote AGI memory instead."
)


class SolanaAgentFactory:
    """Factory for creating and wiring components of the Solana Agent system."""

    @staticmethod
    def _create_guardrails(guardrail_configs: List[Dict[str, Any]]) -> List[Any]:
        """Instantiates guardrails from configuration."""
        guardrails = []
        if not guardrail_configs:
            return guardrails

        for config in guardrail_configs:
            class_path = config.get("class")
            guardrail_config = config.get("config", {})
            if not class_path:
                logger.warning(
                    f"Guardrail config missing 'class': {config}"
                )  # Use logger.warning
                continue
            try:
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                guardrail_class = getattr(module, class_name)
                # Instantiate the guardrail, handling potential errors during init
                try:
                    guardrails.append(guardrail_class(config=guardrail_config))
                    logger.info(
                        f"Successfully loaded guardrail: {class_path}"
                    )  # Use logger.info
                except Exception as init_e:
                    logger.error(
                        f"Error initializing guardrail '{class_path}': {init_e}"
                    )  # Use logger.error
                    # Optionally re-raise or just skip this guardrail

            except (ImportError, AttributeError, ValueError) as e:
                logger.error(
                    f"Error loading guardrail class '{class_path}': {e}"
                )  # Use logger.error
            except Exception as e:  # Catch unexpected errors during import/getattr
                logger.exception(
                    f"Unexpected error loading guardrail '{class_path}': {e}"
                )  # Use logger.exception
        return guardrails

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> QueryService:  # pragma: no cover
        """Create the agent system from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configured QueryService instance
        """
        legacy_provider_keys = [
            provider_name
            for provider_name in ("groq", "cerebras", "grok")
            if provider_name in config
        ]
        if legacy_provider_keys:
            joined = ", ".join(sorted(legacy_provider_keys))
            raise ValueError(
                f"Legacy provider sections are no longer supported: {joined}. "
                "Use config['ai'] with the AGI x402 transport instead."
            )

        if "mongo" in config or "zep" in config:
            raise ValueError(LOCAL_MEMORY_CONFIG_ERROR)

        # AI runtime config for the AGI transport path.
        provider_label = "AI"
        provider_config_key = None
        provider_config = None
        if PRIMARY_AI_CONFIG_KEY in config:
            provider_config_key = PRIMARY_AI_CONFIG_KEY
            provider_config = config.get(PRIMARY_AI_CONFIG_KEY)
        elif LEGACY_AI_CONFIG_KEY in config:
            provider_config_key = LEGACY_AI_CONFIG_KEY
            provider_config = config.get(LEGACY_AI_CONFIG_KEY)
            logger.warning("config['openai'] is deprecated; use config['ai'] instead.")

        if provider_config_key is None:
            raise ValueError("AI config is required in config['ai'].")
        if not isinstance(provider_config, dict):
            raise ValueError("AI config in config['ai'] must be a mapping.")

        auth_mode = provider_config.get("auth_mode", "api_key")
        if auth_mode not in {"api_key", "x402_private_key", "x402_privy"}:
            raise ValueError(
                "Unsupported auth_mode. Supported values are: api_key, x402_private_key, x402_privy."
            )

        use_remote_memory = auth_mode in {"x402_private_key", "x402_privy"}
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
        llm_reasoning_effort = provider_config.get(
            "reasoning_effort"
        )  # Optional: "low", "medium", or "high"
        llm_context_window_tokens = provider_config.get("context_window_tokens")
        llm_max_output_tokens = provider_config.get("max_output_tokens")
        llm_tokenizer_model = provider_config.get("tokenizer_model")

        if use_remote_memory:
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
            provider_label = "Solana Agent AGI"
        elif not llm_api_key:
            raise ValueError(
                "AI API key is required unless auth_mode is x402_private_key or x402_privy."
            )

        if llm_model:
            logger.info(
                f"Using {provider_label} as LLM provider with model: {llm_model}"
            )
        else:
            logger.info(f"Using {provider_label} as LLM provider")

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
        if use_remote_memory:
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

        if "logfire" in config:
            if "api_key" not in config["logfire"]:
                raise ValueError("Pydantic Logfire API key is required.")
            llm_adapter_kwargs["logfire_api_key"] = config["logfire"].get("api_key")

        llm_adapter = OpenAIAdapter(**llm_adapter_kwargs)

        # Create business mission if specified in config
        business_mission = None
        if "business" in config:
            org_config = config["business"]
            business_mission = BusinessMission(
                mission=org_config.get("mission", ""),
                values=[
                    {"name": k, "description": v}
                    for k, v in org_config.get("values", {}).items()
                ],
                goals=org_config.get("goals", []),
                voice=org_config.get("voice", ""),
            )

        # capture_mode removed: repository now always upserts/merges per capture

        # v34 runtime delegates conversation memory to the AGI service.
        memory_provider = None

        guardrail_config = config.get("guardrails", {})
        input_guardrails: List[InputGuardrail] = SolanaAgentFactory._create_guardrails(
            guardrail_config.get("input", [])
        )
        output_guardrails: List[OutputGuardrail] = (
            SolanaAgentFactory._create_guardrails(guardrail_config.get("output", []))
        )
        logger.info(  # Use logger.info
            f"Loaded {len(input_guardrails)} input guardrails and {len(output_guardrails)} output guardrails."
        )

        # Create primary services
        agent_service = AgentService(
            llm_provider=llm_adapter,
            business_mission=business_mission,
            config=config,
            model=llm_model,
            output_guardrails=output_guardrails,
        )

        # Create routing service
        routing_model = llm_model  # Use the same model as the main LLM by default
        if not routing_model:
            # Fall back to OpenAI routing_model config
            routing_model = (
                provider_config.get("routing_model")
                if isinstance(provider_config, dict)
                else None
            )
        routing_service = RoutingService(
            llm_provider=llm_adapter,
            agent_service=agent_service,
            model=routing_model,
        )

        # Debug the agent service tool registry
        logger.debug(  # Use logger.debug
            f"Agent service tools after initialization: {agent_service.tool_registry.list_all_tools()}"
        )

        # Initialize plugin system
        agent_service.plugin_manager = PluginManager(
            config=config, tool_registry=agent_service.tool_registry
        )
        try:
            loaded_plugins = agent_service.plugin_manager.load_plugins()
            logger.info(f"Loaded {loaded_plugins} plugins")  # Use logger.info
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")  # Use logger.error
            loaded_plugins = 0

        # Register predefined agents
        for agent_config in config.get("agents", []):  # pragma: no cover
            extra_kwargs = {}
            if "capture_name" in agent_config:
                extra_kwargs["capture_name"] = agent_config.get("capture_name")
            if "capture_schema" in agent_config:
                extra_kwargs["capture_schema"] = agent_config.get("capture_schema")

            agent_service.register_ai_agent(
                name=agent_config["name"],
                instructions=agent_config["instructions"],
                specialization=agent_config["specialization"],
                **extra_kwargs,
            )

            # Register tools for this agent
            if "tools" in agent_config:
                for tool_name in agent_config["tools"]:
                    logger.debug(  # Use logger.debug
                        f"Available tools before registering {tool_name}: {agent_service.tool_registry.list_all_tools()}"
                    )
                    agent_service.assign_tool_for_agent(agent_config["name"], tool_name)
                    logger.info(  # Use logger.info
                        f"Successfully registered {tool_name} for agent {agent_config['name']}"
                    )

        # Global tool registrations
        if "agent_tools" in config:
            for agent_name, tools in config["agent_tools"].items():
                for tool_name in tools:
                    agent_service.assign_tool_for_agent(agent_name, tool_name)

        # Create and return the query service
        query_service = QueryService(
            agent_service=agent_service,
            routing_service=routing_service,
            memory_provider=memory_provider,
            input_guardrails=input_guardrails,
        )

        return query_service

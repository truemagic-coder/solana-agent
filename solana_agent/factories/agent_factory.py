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

# Repository imports
from solana_agent.repositories.memory import MemoryRepository

# Adapter imports
from solana_agent.adapters.openai_adapter import OpenAIAdapter
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter

# Domain and plugin imports
from solana_agent.domains.agent import BusinessMission
from solana_agent.plugins.manager import PluginManager

# Setup logger for this module
logger = logging.getLogger(__name__)


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
        # Create adapters

        if "mongo" in config:
            if "connection_string" not in config["mongo"]:
                raise ValueError("MongoDB connection string is required.")
            if "database" not in config["mongo"]:
                raise ValueError("MongoDB database name is required.")
            db_adapter = MongoDBAdapter(
                connection_string=config["mongo"]["connection_string"],
                database_name=config["mongo"]["database"],
            )
        else:
            db_adapter = None

        # OpenAI is the only supported LLM provider
        if "openai" not in config or "api_key" not in config["openai"]:
            raise ValueError("OpenAI API key is required in config.")

        llm_api_key = config["openai"]["api_key"]
        llm_model = config["openai"].get("model")  # Optional model override
        if llm_model:
            logger.info(f"Using OpenAI as LLM provider with model: {llm_model}")
        else:
            logger.info("Using OpenAI as LLM provider")

        if "logfire" in config:
            if "api_key" not in config["logfire"]:
                raise ValueError("Pydantic Logfire API key is required.")
            llm_adapter = OpenAIAdapter(
                api_key=llm_api_key,
                model=llm_model,
                logfire_api_key=config["logfire"].get("api_key"),
            )
        else:
            llm_adapter = OpenAIAdapter(
                api_key=llm_api_key,
                model=llm_model,
            )

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

        # Create repositories
        memory_provider = None

        if "zep" in config and "mongo" in config:
            mem_kwargs: Dict[str, Any] = {
                "mongo_adapter": db_adapter,
                "zep_api_key": config["zep"].get("api_key"),
            }
            memory_provider = MemoryRepository(**mem_kwargs)

        if "mongo" in config and "zep" not in config:
            mem_kwargs = {"mongo_adapter": db_adapter}
            memory_provider = MemoryRepository(**mem_kwargs)

        if "zep" in config and "mongo" not in config:
            if "api_key" not in config["zep"]:
                raise ValueError("Zep API key is required.")
            mem_kwargs = {"zep_api_key": config["zep"].get("api_key")}
            memory_provider = MemoryRepository(**mem_kwargs)

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
                config.get("openai", {}).get("routing_model")
                if isinstance(config.get("openai"), dict)
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

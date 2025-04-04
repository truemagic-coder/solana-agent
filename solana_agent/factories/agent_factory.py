"""
Factory for creating and wiring components of the Solana Agent system.

This module handles the creation and dependency injection for all
services and components used in the system.
"""
from typing import Dict, Any

# Service imports
from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService

# Repository imports
from solana_agent.repositories.memory import MemoryRepository

# Adapter imports
from solana_agent.adapters.llm_adapter import OpenAIAdapter
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter

# Domain and plugin imports
from solana_agent.domains.agent import BusinessMission
from solana_agent.plugins.manager import PluginManager


class SolanaAgentFactory:
    """Factory for creating and wiring components of the Solana Agent system."""

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> QueryService:
        """Create the agent system from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configured QueryService instance
        """
        # Create adapters

        if "mongo" in config:
            # MongoDB connection string and database name
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

        llm_adapter = OpenAIAdapter(
            api_key=config["openai"]["api_key"],
        )

        # Create business mission if specified in config
        business_mission = None
        if "business" in config:
            org_config = config["business"]
            business_mission = BusinessMission(
                mission=org_config.get("mission", ""),
                values=[{"name": k, "description": v}
                        for k, v in org_config.get("values", {}).items()],
                goals=org_config.get("goals", []),
                voice=org_config.get("voice", "")
            )

        # Create repositories
        memory_provider = None

        if "zep" in config and "mongo" in config:
            if "api_key" not in config["zep"]:
                raise ValueError("Zep API key is required.")
            memory_provider = MemoryRepository(
                mongo_adapter=db_adapter, zep_api_key=config["zep"].get("api_key"))

        if "mongo" in config and not "zep" in config:
            memory_provider = MemoryRepository(mongo_adapter=db_adapter)

        if "zep" in config and not "mongo" in config:
            if "api_key" not in config["zep"]:
                raise ValueError("Zep API key is required.")
            memory_provider = MemoryRepository(
                zep_api_key=config["zep"].get("api_key")
            )

        # Create primary services
        agent_service = AgentService(
            llm_provider=llm_adapter,
            business_mission=business_mission,
            config=config,
        )

        # Debug the agent service tool registry
        print(
            f"Agent service tools after initialization: {agent_service.tool_registry.list_all_tools()}")

        # Create routing service
        routing_service = RoutingService(
            llm_provider=llm_adapter,
            agent_service=agent_service,
        )

        # Initialize plugin system
        agent_service.plugin_manager = PluginManager(
            config=config,
            tool_registry=agent_service.tool_registry
        )
        try:
            loaded_plugins = agent_service.plugin_manager.load_plugins()
            print(f"Loaded {loaded_plugins} plugins")
        except Exception as e:
            print(f"Error loading plugins: {e}")
            loaded_plugins = 0

        # Register predefined agents
        for agent_config in config.get("agents", []):
            agent_service.register_ai_agent(
                name=agent_config["name"],
                instructions=agent_config["instructions"],
                specialization=agent_config["specialization"],
            )

            # Register tools for this agent
            if "tools" in agent_config:
                for tool_name in agent_config["tools"]:
                    print(
                        f"Available tools before registering {tool_name}: {agent_service.tool_registry.list_all_tools()}")
                    agent_service.assign_tool_for_agent(
                        agent_config["name"], tool_name
                    )
                    print(
                        f"Successfully registered {tool_name} for agent {agent_config['name']}")

        # Global tool registrations
        if "agent_tools" in config:
            for agent_name, tools in config["agent_tools"].items():
                for tool_name in tools:
                    agent_service.assign_tool_for_agent(
                        agent_name, tool_name)

        # Create and return the query service
        query_service = QueryService(
            agent_service=agent_service,
            routing_service=routing_service,
            memory_provider=memory_provider,
        )

        return query_service

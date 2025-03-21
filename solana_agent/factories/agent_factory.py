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
from solana_agent.repositories.agent import MongoAgentRepository

# Adapter imports
from solana_agent.adapters.llm_adapter import OpenAIAdapter
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter

# Domain and plugin imports
from solana_agent.domains.agent import OrganizationMission
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
        db_adapter = MongoDBAdapter(
            connection_string=config["mongo"]["connection_string"],
            database_name=config["mongo"]["database"],
        )

        llm_adapter = OpenAIAdapter(
            api_key=config["openai"]["api_key"],
        )

        # Create organization mission if specified in config
        organization_mission = None
        if "organization" in config:
            org_config = config["organization"]
            organization_mission = OrganizationMission(
                mission_statement=org_config.get("mission_statement", ""),
                values=[{"name": k, "description": v}
                        for k, v in org_config.get("values", {}).items()],
                goals=org_config.get("goals", []),
                guidance=org_config.get("guidance", "")
            )

        # Create repositories
        memory_provider = MemoryRepository(
            db_adapter, config["zep"].get("api_key"), config["zep"].get("base_url"))
        agent_repo = MongoAgentRepository(db_adapter)

        # Create primary services
        agent_service = AgentService(
            agent_repository=agent_repo,
            llm_provider=llm_adapter,
            organization_mission=organization_mission,
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
        loaded_plugins = agent_service.plugin_manager.load_plugins()
        print(f"Loaded {loaded_plugins} plugins")

        # Sync MongoDB with config-defined agents
        config_defined_agents = [agent["name"]
                                 for agent in config.get("agents", [])]
        all_db_agents = agent_repo.get_all_ai_agents()
        db_agent_names = [agent.name for agent in all_db_agents]

        # Delete agents not in config
        agents_to_delete = [
            name for name in db_agent_names if name not in config_defined_agents]
        for agent_name in agents_to_delete:
            print(
                f"Deleting agent '{agent_name}' from MongoDB - no longer defined in config")
            agent_repo.delete_ai_agent(agent_name)

        # Register predefined agents
        for agent_config in config.get("ai_agents", []):
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
                    try:
                        agent_service.assign_tool_for_agent(
                            agent_config["name"], tool_name
                        )
                        print(
                            f"Successfully registered {tool_name} for agent {agent_config['name']}")
                    except ValueError as e:
                        print(
                            f"Error registering tool {tool_name} for agent {agent_config['name']}: {e}")

        # Global tool registrations
        if "agent_tools" in config:
            for agent_name, tools in config["agent_tools"].items():
                for tool_name in tools:
                    try:
                        agent_service.assign_tool_for_agent(
                            agent_name, tool_name)
                    except ValueError as e:
                        print(f"Error registering tool: {e}")

        # Create and return the query service
        query_service = QueryService(
            agent_service=agent_service,
            routing_service=routing_service,
            memory_provider=memory_provider,
        )

        return query_service

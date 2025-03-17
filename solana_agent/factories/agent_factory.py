"""
Factory for creating and wiring components of the Solana Agent system.

This module handles the creation and dependency injection for all
services and components used in the system.
"""
from typing import Dict, Any

from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.services.ticket import TicketService
from solana_agent.services.handoff import HandoffService
from solana_agent.services.memory import MemoryService
from solana_agent.services.nps import NPSService
from solana_agent.services.critic import CriticService
from solana_agent.services.task_planning import TaskPlanningService
from solana_agent.services.project_approval import ProjectApprovalService
from solana_agent.services.project_simulation import ProjectSimulationService
from solana_agent.services.notification import NotificationService
from solana_agent.services.scheduling import SchedulingService
from solana_agent.services.command import CommandService

from solana_agent.repositories.ticket import MongoTicketRepository
from solana_agent.repositories.feedback import MongoFeedbackRepository
from solana_agent.repositories.mongo_memory import MongoMemoryRepository
from solana_agent.repositories.agent import MongoAgentRepository
from solana_agent.repositories.scheduling import SchedulingRepository
from solana_agent.repositories.handoff import MongoHandoffRepository

from solana_agent.adapters.llm_adapter import OpenAIAdapter
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
from solana_agent.adapters.memory_adapter import MongoMemoryProvider, ZepMemoryAdapter, DualMemoryProvider
from solana_agent.adapters.vector_adapter import QdrantAdapter, PineconeAdapter

from solana_agent.domain.agents import OrganizationMission
from solana_agent.plugins import PluginManager


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
            model=config.get("openai", {}).get("default_model", "gpt-4o-mini"),
        )

        mongo_memory = MongoMemoryProvider(db_adapter)

        zep_memory = None
        if "zep" in config:
            zep_memory = ZepMemoryAdapter(
                api_key=config["zep"].get("api_key"),
                base_url=config["zep"].get("base_url"),
            )

        memory_provider = DualMemoryProvider(mongo_memory, zep_memory)

        # Create vector store provider if configured
        vector_provider = None
        if "qdrant" in config:
            vector_provider = QdrantAdapter(
                url=config["qdrant"].get("url", "http://localhost:6333"),
                api_key=config["qdrant"].get("api_key"),
                collection_name=config["qdrant"].get(
                    "collection", "solana_agent"),
                embedding_model=config["qdrant"].get(
                    "embedding_model", "text-embedding-3-small"),
            )
        if "pinecone" in config:
            vector_provider = PineconeAdapter(
                api_key=config["pinecone"]["api_key"],
                index_name=config["pinecone"]["index"],
                embedding_model=config["pinecone"].get(
                    "embedding_model", "text-embedding-3-small"),
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
        ticket_repo = MongoTicketRepository(db_adapter)
        nps_repo = MongoFeedbackRepository(db_adapter)
        memory_repo = MongoMemoryRepository(db_adapter, vector_provider)
        agent_repo = MongoAgentRepository(db_adapter)
        handoff_repo = MongoHandoffRepository(db_adapter)

        # Create services
        agent_service = AgentService(
            llm_provider=llm_adapter,
            agent_repository=agent_repo,
            organization_mission=organization_mission,
            config=config
        )

        # Debug the agent service tool registry to confirm tools were registered
        print(
            f"Agent service tools after initialization: {agent_service.tool_registry.list_all_tools()}")

        routing_service = RoutingService(
            llm_provider=llm_adapter,
            agent_service=agent_service,
            router_model=config.get("router_model", "gpt-4o-mini"),
        )

        ticket_service = TicketService(ticket_repo)
        handoff_service = HandoffService(
            handoff_repository=handoff_repo,
            ticket_repository=ticket_repo,
            agent_service=agent_service
        )
        memory_service = MemoryService(memory_repo, llm_adapter)
        nps_service = NPSService(nps_repo, ticket_repo)

        # Create command service
        command_service = CommandService(
            ticket_service=ticket_service,
            agent_service=agent_service
        )

        # Create critic service if enabled
        critic_service = None
        if config.get("enable_critic", True):
            critic_service = CriticService(llm_adapter)

        # Create task planning service
        task_planning_service = TaskPlanningService(
            ticket_repository=ticket_repo,
            llm_provider=llm_adapter,
            agent_service=agent_service
        )

        notification_service = NotificationService(
            agent_repository=agent_repo,  # Use single agent repository
            tool_registry=agent_service.tool_registry
        )

        project_approval_service = ProjectApprovalService(
            ticket_repository=ticket_repo,
            agent_repository=agent_repo,  # Use single agent repository
            notification_service=notification_service
        )
        project_simulation_service = ProjectSimulationService(
            llm_provider=llm_adapter,
            task_planning_service=task_planning_service
        )

        # Create scheduling repository and service
        scheduling_repository = SchedulingRepository(db_adapter)

        scheduling_service = SchedulingService(
            scheduling_repository=scheduling_repository,
            task_planning_service=task_planning_service,
            agent_service=agent_service
        )

        # Update task_planning_service with scheduling_service if needed
        if task_planning_service:
            task_planning_service.scheduling_service = scheduling_service

        # Initialize plugin system if plugins directory is configured
        agent_service.plugin_manager = PluginManager()
        loaded_plugins = agent_service.plugin_manager.load_all_plugins()
        print(f"Loaded {loaded_plugins} plugins")

        # Get list of all agents defined in config
        config_defined_agents = [agent["name"]
                                 for agent in config.get("ai_agents", [])]

        # Sync MongoDB with config-defined agents (delete any agents not in config)
        # Fixed method call on unified repo
        all_db_agents = agent_repo.get_all_ai_agents()
        # Access name property on agent objects
        db_agent_names = [agent.name for agent in all_db_agents]

        # Find agents that exist in DB but not in config
        agents_to_delete = [
            name for name in db_agent_names if name not in config_defined_agents]

        # Delete those agents
        for agent_name in agents_to_delete:
            print(
                f"Deleting agent '{agent_name}' from MongoDB - no longer defined in config")
            # Use repository method to delete
            agent_repo.delete_ai_agent(agent_name)

        # Register predefined agents if any
        for agent_config in config.get("ai_agents", []):
            agent_service.register_ai_agent(
                name=agent_config["name"],
                instructions=agent_config["instructions"],
                specialization=agent_config["specialization"],
                model=agent_config.get("model", "gpt-4o-mini"),
            )

            # Register tools for this agent if specified
            if "tools" in agent_config:
                for tool_name in agent_config["tools"]:
                    # Print available tools before registering
                    print(
                        f"Available tools before registering {tool_name}: {agent_service.tool_registry.list_all_tools()}")
                    try:
                        agent_service.register_tool_for_agent(
                            agent_config["name"], tool_name
                        )
                        print(
                            f"Successfully registered {tool_name} for agent {agent_config['name']}")
                    except ValueError as e:
                        print(
                            f"Error registering tool {tool_name} for agent {agent_config['name']}: {e}")

        # Also support global tool registrations
        if "agent_tools" in config:
            for agent_name, tools in config["agent_tools"].items():
                for tool_name in tools:
                    try:
                        agent_service.register_tool_for_agent(
                            agent_name, tool_name)
                    except ValueError as e:
                        print(f"Error registering tool: {e}")

        # Create the query service
        query_service = QueryService(
            agent_service=agent_service,
            routing_service=routing_service,
            ticket_service=ticket_service,
            handoff_service=handoff_service,
            memory_service=memory_service,
            nps_service=nps_service,
            command_service=command_service,
            critic_service=critic_service,
            memory_provider=memory_provider,
            task_planning_service=task_planning_service,
            project_approval_service=project_approval_service,
            project_simulation_service=project_simulation_service,
            scheduling_service=scheduling_service,
            enable_critic=config.get("enable_critic", True),
            require_human_approval=config.get("require_human_approval", False),
            stalled_ticket_timeout=config.get("stalled_ticket_timeout", 60),
        )

        return query_service

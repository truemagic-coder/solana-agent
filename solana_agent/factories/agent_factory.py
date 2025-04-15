"""
Factory for creating and wiring components of the Solana Agent system.

This module handles the creation and dependency injection for all
services and components used in the system.
"""
from typing import Dict, Any

# Service imports
from solana_agent.adapters.pinecone_adapter import PineconeAdapter
from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.services.knowledge_base import KnowledgeBaseService

# Repository imports
from solana_agent.repositories.memory import MemoryRepository

# Adapter imports
from solana_agent.adapters.openai_adapter import OpenAIAdapter
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

        if "gemini" in config and "api_key" in config["gemini"] and not "grok" in config:
            # Create primary services
            agent_service = AgentService(
                llm_provider=llm_adapter,
                business_mission=business_mission,
                config=config,
                api_key=config["gemini"]["api_key"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                model="gemini-2.0-flash",
            )

            # Create routing service
            routing_service = RoutingService(
                llm_provider=llm_adapter,
                agent_service=agent_service,
                api_key=config["gemini"]["api_key"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                model="gemini-2.0-flash",
            )

        elif "gemini" in config and "api_key" in config["gemini"] and "grok" in config and "api_key" in config["grok"]:
            # Create primary services
            agent_service = AgentService(
                llm_provider=llm_adapter,
                business_mission=business_mission,
                config=config,
                api_key=config["grok"]["api_key"],
                base_url="https://api.x.ai/v1",
                model="grok-3-mini-fast-beta",
            )
            # Create routing service
            routing_service = RoutingService(
                llm_provider=llm_adapter,
                agent_service=agent_service,
                api_key=config["gemini"]["api_key"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                model="gemini-2.0-flash",
            )

        elif "grok" in config and "api_key" in config["grok"] and not "gemini" in config:
            # Create primary services
            agent_service = AgentService(
                llm_provider=llm_adapter,
                business_mission=business_mission,
                config=config,
                api_key=config["grok"]["api_key"],
                base_url="https://api.x.ai/v1",
                model="grok-3-mini-fast-beta",
            )

            # Create routing service
            routing_service = RoutingService(
                llm_provider=llm_adapter,
                agent_service=agent_service,
            )

        else:
            # Create primary services
            agent_service = AgentService(
                llm_provider=llm_adapter,
                business_mission=business_mission,
                config=config,
            )

            # Create routing service
            routing_service = RoutingService(
                llm_provider=llm_adapter,
                agent_service=agent_service,
            )

        # Debug the agent service tool registry
        print(
            f"Agent service tools after initialization: {agent_service.tool_registry.list_all_tools()}")

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

        # Initialize Knowledge Base if configured
        knowledge_base = None
        if "knowledge_base" in config and "mongo" in config:
            try:
                pinecone_config = config["knowledge_base"].get("pinecone", {})

                # Create Pinecone adapter for KB
                pinecone_adapter = PineconeAdapter(
                    api_key=pinecone_config.get("api_key"),
                    index_name=pinecone_config.get("index_name"),
                    llm_provider=llm_adapter,  # Reuse the LLM adapter
                    embedding_dimensions=pinecone_config.get(
                        "embedding_dimensions", 3072),
                    create_index_if_not_exists=pinecone_config.get(
                        "create_index", True),
                    use_pinecone_embeddings=pinecone_config.get(
                        "use_pinecone_embeddings", False),
                    pinecone_embedding_model=pinecone_config.get(
                        "embedding_model"),
                    use_reranking=pinecone_config.get("use_reranking", False),
                    rerank_model=pinecone_config.get("rerank_model")
                )

                # Create the KB service
                knowledge_base = KnowledgeBaseService(
                    pinecone_adapter=pinecone_adapter,
                    mongodb_adapter=db_adapter,
                    llm_provider=llm_adapter,
                    collection_name=config["knowledge_base"].get(
                        "collection", "knowledge_documents"),
                    rerank_results=pinecone_config.get("use_reranking", False),
                    rerank_top_k=config["knowledge_base"].get(
                        "results_count", 3)
                )

            except Exception as e:
                print(f"Failed to initialize Knowledge Base: {e}")
                import traceback
                print(traceback.format_exc())

        # Create and return the query service
        query_service = QueryService(
            agent_service=agent_service,
            routing_service=routing_service,
            memory_provider=memory_provider,
            knowledge_base=knowledge_base,
            kb_results_count=config.get(
                "knowledge_base", {}).get("results_count", 3)
        )

        return query_service

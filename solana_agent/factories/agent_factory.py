"""
Factory for creating and wiring components of the Solana Agent system.

This module handles the creation and dependency injection for all
services and components used in the system.
"""

import importlib
import logging
from typing import Dict, Any, List

# Service imports
from solana_agent.adapters.pinecone_adapter import PineconeAdapter
from solana_agent.interfaces.guardrails.guardrails import (
    InputGuardrail,
    OutputGuardrail,
)
from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.services.knowledge_base import KnowledgeBaseService
from solana_agent.services.realtime import RealtimeService
from solana_agent.adapters.openai_realtime_ws import OpenAIRealtimeWebSocketSession
from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions
from solana_agent.adapters.ffmpeg_transcoder import FFmpegTranscoder

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
    def create_from_config(config: Dict[str, Any]) -> QueryService:
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

        if "logfire" in config:
            if "api_key" not in config["logfire"]:
                raise ValueError("Pydantic Logfire API key is required.")
            if "openai" not in config or "api_key" not in config["openai"]:
                raise ValueError("OpenAI API key is required.")
            llm_adapter = OpenAIAdapter(
                api_key=config["openai"]["api_key"],
                logfire_api_key=config["logfire"].get("api_key"),
            )
        else:
            if "openai" not in config or "api_key" not in config["openai"]:
                raise ValueError("OpenAI API key is required.")
            llm_adapter = OpenAIAdapter(
                api_key=config["openai"].get("api_key"),
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
            output_guardrails=output_guardrails,
        )

        # Create routing service
        # Optional routing model override (use small, cheap model by default in service)
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

        # Initialize Knowledge Base if configured
        knowledge_base = None
        kb_config = config.get("knowledge_base")
        # Requires both KB config section and MongoDB adapter
        if kb_config and db_adapter:
            try:
                pinecone_config = kb_config.get("pinecone", {})
                splitter_config = kb_config.get("splitter", {})
                # Get OpenAI embedding config (used by KBService)
                openai_embed_config = kb_config.get("openai_embeddings", {})

                # Determine OpenAI model and dimensions for KBService
                openai_model_name = openai_embed_config.get(
                    "model_name", "text-embedding-3-large"
                )
                if openai_model_name == "text-embedding-3-large":
                    openai_dimensions = 3072
                elif openai_model_name == "text-embedding-3-small":  # pragma: no cover
                    openai_dimensions = 1536  # pragma: no cover
                else:  # pragma: no cover
                    logger.warning(
                        f"Unknown OpenAI embedding model '{openai_model_name}' specified for KB. Defaulting dimensions to 3072."
                    )  # pragma: no cover
                    openai_dimensions = 3072  # pragma: no cover

                # Create Pinecone adapter for KB
                # It now relies on external embeddings, so dimension MUST match OpenAI model
                pinecone_adapter = PineconeAdapter(
                    api_key=pinecone_config.get("api_key"),
                    index_name=pinecone_config.get("index_name"),
                    # This dimension MUST match the OpenAI model used by KBService
                    embedding_dimensions=openai_dimensions,
                    cloud_provider=pinecone_config.get("cloud_provider", "aws"),
                    region=pinecone_config.get("region", "us-east-1"),
                    metric=pinecone_config.get("metric", "cosine"),
                    create_index_if_not_exists=pinecone_config.get(
                        "create_index", True
                    ),
                    # Reranking config
                    use_reranking=pinecone_config.get("use_reranking", False),
                    rerank_model=pinecone_config.get("rerank_model"),
                    rerank_top_k=pinecone_config.get("rerank_top_k", 3),
                    initial_query_top_k_multiplier=pinecone_config.get(
                        "initial_query_top_k_multiplier", 5
                    ),
                    rerank_text_field=pinecone_config.get("rerank_text_field", "text"),
                )

                # Create the KB service using OpenAI embeddings
                knowledge_base = KnowledgeBaseService(
                    pinecone_adapter=pinecone_adapter,
                    mongodb_adapter=db_adapter,
                    # Pass OpenAI config directly
                    openai_api_key=openai_embed_config.get("api_key")
                    or config.get("openai", {}).get("api_key"),
                    openai_model_name=openai_model_name,
                    collection_name=kb_config.get(
                        "collection_name", "knowledge_documents"
                    ),
                    # Pass rerank config (though PineconeAdapter handles the logic)
                    rerank_results=pinecone_config.get("use_reranking", False),
                    rerank_top_k=pinecone_config.get("rerank_top_k", 3),
                    # Pass splitter config
                    splitter_buffer_size=splitter_config.get("buffer_size", 1),
                    splitter_breakpoint_percentile=splitter_config.get(
                        "breakpoint_percentile", 95
                    ),
                )
                logger.info(
                    "Knowledge Base Service initialized successfully."
                )  # Use logger.info

            except Exception as e:
                # Use logger.exception to include traceback automatically
                logger.exception(f"Failed to initialize Knowledge Base: {e}")
                knowledge_base = None  # Ensure KB is None if init fails

        # Optional: Realtime setup
        realtime_service = None
        rt_cfg = config.get("realtime") or {}
        if isinstance(rt_cfg, dict) and rt_cfg.get("enabled"):
            # Validate voice support when realtime enabled
            supported_voices = {
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
            }
            desired_voice = (
                config.get("business", {}).get("voice")
                or config.get("openai", {}).get("voice")
                or "nova"
            )
            if desired_voice not in supported_voices:
                raise ValueError(
                    f"Realtime enabled but unsupported voice '{desired_voice}'. Supported: {sorted(supported_voices)}"
                )

            api_key = config.get("openai", {}).get("api_key")
            if not api_key:
                raise ValueError(
                    "Realtime requires OpenAI API key in config.openai.api_key"
                )

            # Build initial tools for the default agent (if present)
            default_agent_name = None
            try:
                default_agent_name = next(
                    (a.get("name") for a in config.get("agents", []) if a.get("name")),
                    None,
                )
            except Exception:
                default_agent_name = None

            initial_tools = []
            if default_agent_name:
                try:
                    initial_tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": t["name"],
                                "description": t.get("description", ""),
                                "parameters": t.get("parameters", {}),
                                "strict": True,
                            },
                        }
                        for t in agent_service.get_agent_tools(default_agent_name)
                    ]
                except Exception:
                    initial_tools = []

            session_opts = RealtimeSessionOptions(
                model=rt_cfg.get("model") or "gpt-4o-realtime-preview-2024-12-17",
                voice=desired_voice,
                vad_enabled=bool(rt_cfg.get("vad", True)),
                input_rate_hz=int(rt_cfg.get("input_rate_hz", 24000)),
                output_rate_hz=int(rt_cfg.get("output_rate_hz", 24000)),
                input_mime=rt_cfg.get("input_mime", "audio/pcm"),
                output_mime=rt_cfg.get("output_mime", "audio/pcm"),
                instructions=config.get("business", {}).get("mission") or None,
                tools=initial_tools or None,
                tool_choice="auto",
            )
            session = OpenAIRealtimeWebSocketSession(
                api_key=api_key, options=session_opts
            )
            # Optional transcoder for mobile input/output
            transcoder = None
            accept_compressed_input = bool(rt_cfg.get("accept_compressed_input", False))
            client_input_mime = rt_cfg.get("client_input_mime", "audio/mp4")
            encode_output = bool(rt_cfg.get("encode_output", False))
            client_output_mime = rt_cfg.get("client_output_mime", "audio/aac")
            if accept_compressed_input or encode_output:
                transcoder = FFmpegTranscoder()

            realtime_service = RealtimeService(
                session=session,
                options=session_opts,
                transcoder=transcoder,
                accept_compressed_input=accept_compressed_input,
                client_input_mime=client_input_mime,
                encode_output=encode_output,
                client_output_mime=client_output_mime,
            )

            # Auto-wire tool executor to run mid-stream function calls
            async def _rt_tool_exec(
                tool_name: str, args: Dict[str, Any]
            ) -> Dict[str, Any]:
                # Use default agent for permissions; fall back to first or 'default'
                agent_name = default_agent_name or "default"
                try:
                    return await agent_service.execute_tool(
                        agent_name, tool_name, args or {}
                    )
                except Exception as e:
                    return {"status": "error", "message": str(e)}

            try:
                session.set_tool_executor(_rt_tool_exec)
            except Exception:
                pass

        # Create and return the query service
        query_service = QueryService(
            agent_service=agent_service,
            routing_service=routing_service,
            memory_provider=memory_provider,
            knowledge_base=knowledge_base,  # Pass the potentially created KB
            kb_results_count=kb_config.get("results_count", 3) if kb_config else 3,
            input_guardrails=input_guardrails,
        )

        # Attach realtime service if created
        if realtime_service:
            setattr(query_service, "realtime", realtime_service)

        return query_service

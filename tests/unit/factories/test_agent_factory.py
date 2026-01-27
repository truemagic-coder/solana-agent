"""
Tests for the SolanaAgentFactory class.

This module provides comprehensive test coverage for the factory that
creates and initializes all components of the Solana Agent system.
"""

import pytest
from unittest.mock import patch, MagicMock
from copy import deepcopy

from solana_agent.factories.agent_factory import SolanaAgentFactory
from solana_agent.interfaces.guardrails.guardrails import (
    InputGuardrail,
    OutputGuardrail,
)
from unittest.mock import mock_open


class MockInputGuardrail(InputGuardrail):
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config or {}
        print(
            f"MockInputGuardrail initialized with config: {self.config}"
        )  # Debug print

    async def process(self, text: str) -> str:
        return f"input_processed_{text}"


class MockOutputGuardrail(OutputGuardrail):
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config or {}
        print(
            f"MockOutputGuardrail initialized with config: {self.config}"
        )  # Debug print

    async def process(self, text: str) -> str:
        return f"output_processed_{text}"


class MockErrorGuardrail:  # Doesn't inherit, simulates init error
    def __init__(self, config=None):
        raise ValueError("Initialization failed")


@pytest.fixture
def guardrails_config(base_config):
    """Config with guardrails section."""
    config = deepcopy(base_config)
    config["guardrails"] = {
        "input": [
            {
                "class": "tests.unit.factories.test_agent_factory.MockInputGuardrail",
                "config": {"input_setting": "value1"},
            }
        ],
        "output": [
            {
                "class": "tests.unit.factories.test_agent_factory.MockOutputGuardrail",
                "config": {"output_setting": "value2"},
            }
        ],
    }
    return config


# capture_mode removed from framework; no fixture needed


# Base configuration for testing
@pytest.fixture
def base_config():
    return {"openai": {"api_key": "test-openai-key"}}


@pytest.fixture
def mongo_config(base_config):
    config = deepcopy(base_config)
    config["mongo"] = {
        "connection_string": "mongodb://localhost:27017",
        "database": "test_db",
    }
    return config


@pytest.fixture
def business_config(base_config):
    config = deepcopy(base_config)
    config["business"] = {
        "mission": "Test mission",
        "values": {"quality": "We value quality", "integrity": "We value integrity"},
        "goals": ["Goal 1", "Goal 2"],
        "voice": "Professional and friendly",
    }
    return config


@pytest.fixture
def agent_config(base_config):
    config = deepcopy(base_config)
    config["agents"] = [
        {
            "name": "test_agent",
            "instructions": "Test instructions",
            "specialization": "Test specialization",
            "tools": ["test_tool"],
        }
    ]
    return config

    # capture_mode removed from framework; no fixture needed


@pytest.fixture
def agent_tools_config(base_config):
    config = deepcopy(base_config)
    config["agent_tools"] = {"test_agent": ["tool1", "tool2"]}
    return config


@pytest.fixture
def zep_config(mongo_config):
    config = deepcopy(mongo_config)
    config["zep"] = {"api_key": "test-zep-key"}
    return config


@pytest.fixture
def zep_only_config(base_config):
    config = deepcopy(base_config)
    config["zep"] = {"api_key": "test-zep-key"}
    return config


@pytest.fixture
def invalid_zep_config(base_config):
    config = deepcopy(base_config)
    config["zep"] = {}  # Missing API key
    return config


@pytest.fixture
def invalid_mongo_config_missing_conn(base_config):
    """Config with mongo section missing connection_string."""
    config = deepcopy(base_config)
    config["mongo"] = {"database": "test_db"}
    return config


@pytest.fixture
def invalid_mongo_config_missing_db(base_config):
    """Config with mongo section missing database."""
    config = deepcopy(base_config)
    config["mongo"] = {"connection_string": "mongodb://localhost:27017"}
    return config


@pytest.fixture
def logfire_config(base_config):
    """Config with valid logfire section."""
    config = deepcopy(base_config)
    config["logfire"] = {"api_key": "test-logfire-key"}
    return config


@pytest.fixture
def openai_with_model_config():
    """Config with OpenAI and a custom model specified."""
    return {
        "openai": {
            "api_key": "test-openai-key",
            "model": "gpt-5.2",
        },
        "agents": [
            {
                "name": "test_agent",
                "instructions": "You are a test agent.",
                "specialization": "Testing",
            }
        ],
    }


@pytest.fixture
def groq_with_model_config():
    """Config with Groq and a custom model specified."""
    return {
        "groq": {
            "api_key": "test-groq-key",
            "model": "openai/gpt-oss-120b",
            "base_url": "https://api.groq.com/openai/v1",
        },
        "agents": [
            {
                "name": "test_agent",
                "instructions": "You are a test agent.",
                "specialization": "Testing",
            }
        ],
    }


@pytest.fixture
def invalid_logfire_config_missing_key(base_config):
    """Config with logfire section missing api_key."""
    config = deepcopy(base_config)
    config["logfire"] = {}  # Empty logfire section
    return config


@pytest.fixture
def logfire_config_missing_openai(base_config):
    """Config with logfire but missing openai api_key."""
    config = {"logfire": {"api_key": "test-logfire-key"}}  # No openai section
    return config


@pytest.fixture
def config_missing_openai_section():
    """Config completely missing the openai section."""
    return {"some_other_key": "value"}  # No openai section at all


class TestSolanaAgentFactory:
    """Test suite for the SolanaAgentFactory."""

    @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_from_config_minimal(
        self,
        mock_query_service,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        mock_mongo_adapter,
        base_config,
    ):
        """Test creating services with minimal configuration."""
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance

        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = []

        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance

        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory
        result = SolanaAgentFactory.create_from_config(base_config)

        # Verify calls
        mock_openai_adapter.assert_called_once_with(
            api_key="test-openai-key",
            model=None,
        )
        mock_agent_service.assert_called_once()
        mock_routing_service.assert_called_once()
        mock_query_service.assert_called_once()

        # Verify MongoDB was not used
        mock_mongo_adapter.assert_not_called()

        assert result == mock_query_instance

    def test_missing_openai_section(self, config_missing_openai_section):
        """Test factory creation when the entire openai section is missing."""
        # This should raise ValueError since OpenAI API key is required
        with pytest.raises(
            ValueError,
            match="OpenAI or Groq API key is required in config.",
        ):
            SolanaAgentFactory.create_from_config(config_missing_openai_section)

    @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.MemoryRepository")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_from_config_with_mongo(
        self,
        mock_query_service,
        mock_memory_repo,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        mock_mongo_adapter,
        mongo_config,
    ):
        """Test creating services with MongoDB configuration."""
        # Setup mocks
        mock_mongo_instance = MagicMock()
        mock_mongo_adapter.return_value = mock_mongo_instance

        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance

        mock_memory_instance = MagicMock()
        mock_memory_repo.return_value = mock_memory_instance

        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = []

        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance

        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory
        result = SolanaAgentFactory.create_from_config(mongo_config)

        # Verify calls
        mock_mongo_adapter.assert_called_once_with(
            connection_string="mongodb://localhost:27017", database_name="test_db"
        )
        mock_memory_repo.assert_called_once_with(mongo_adapter=mock_mongo_instance)

        assert result == mock_query_instance

    def test_invalid_mongo_config_missing_connection(
        self, invalid_mongo_config_missing_conn
    ):
        """Test handling of invalid MongoDB config (missing connection string)."""
        with pytest.raises(ValueError, match="MongoDB connection string is required."):
            SolanaAgentFactory.create_from_config(invalid_mongo_config_missing_conn)

    def test_invalid_mongo_config_missing_database(
        self, invalid_mongo_config_missing_db
    ):
        """Test handling of invalid MongoDB config (missing database name)."""
        with pytest.raises(ValueError, match="MongoDB database name is required."):
            SolanaAgentFactory.create_from_config(invalid_mongo_config_missing_db)

    @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.BusinessMission")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_from_config_with_business_mission(
        self,
        mock_query_service,
        mock_business_mission,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        mock_mongo_adapter,
        business_config,
    ):
        """Test creating services with business mission configuration."""
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance

        mock_business_instance = MagicMock()
        mock_business_mission.return_value = mock_business_instance

        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = []

        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance

        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory
        result = SolanaAgentFactory.create_from_config(business_config)

        # Verify calls
        mock_business_mission.assert_called_once_with(
            mission="Test mission",
            values=[
                {"name": "quality", "description": "We value quality"},
                {"name": "integrity", "description": "We value integrity"},
            ],
            goals=["Goal 1", "Goal 2"],
            voice="Professional and friendly",
        )

        mock_agent_service.assert_called_once_with(
            llm_provider=mock_openai_instance,
            business_mission=mock_business_instance,
            config=business_config,
            model=None,
            output_guardrails=[],
        )

        assert result == mock_query_instance

    @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.MemoryRepository")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_from_config_with_zep_and_mongo(
        self,
        mock_query_service,
        mock_memory_repo,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        mock_mongo_adapter,
        zep_config,
    ):
        """Test creating services with Zep and MongoDB configuration."""
        # Setup mocks
        mock_mongo_instance = MagicMock()
        mock_mongo_adapter.return_value = mock_mongo_instance

        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance

        mock_memory_instance = MagicMock()
        mock_memory_repo.return_value = mock_memory_instance

        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = []

        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance

        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory
        result = SolanaAgentFactory.create_from_config(zep_config)

        # Verify calls
        mock_memory_repo.assert_called_once_with(
            mongo_adapter=mock_mongo_instance, zep_api_key="test-zep-key"
        )

        assert result == mock_query_instance

    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.MemoryRepository")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_from_config_with_zep_only(
        self,
        mock_query_service,
        mock_memory_repo,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        zep_only_config,
    ):
        """Test creating services with Zep only configuration."""
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance

        mock_memory_instance = MagicMock()
        mock_memory_repo.return_value = mock_memory_instance

        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = []

        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance

        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory
        result = SolanaAgentFactory.create_from_config(zep_only_config)

        # Verify calls
        mock_memory_repo.assert_called_once_with(zep_api_key="test-zep-key")

        assert result == mock_query_instance

    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_from_config_with_invalid_zep(
        self,
        mock_query_service,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        invalid_zep_config,
    ):
        """Test creating services with invalid Zep configuration."""
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance

        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = []

        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance

        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory - should raise ValueError
        with pytest.raises(ValueError, match="Zep API key is required"):
            SolanaAgentFactory.create_from_config(invalid_zep_config)

    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.PluginManager")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_from_config_with_plugin_error(
        self,
        mock_query_service,
        mock_plugin_manager,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        base_config,
    ):
        """Test creating services when plugin loading fails."""
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance

        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = []

        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance

        mock_plugin_instance = MagicMock()
        mock_plugin_manager.return_value = mock_plugin_instance
        mock_plugin_instance.load_plugins.side_effect = Exception("Plugin error")

        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory - should handle plugin error gracefully
        result = SolanaAgentFactory.create_from_config(base_config)

        # Verify calls
        mock_plugin_instance.load_plugins.assert_called_once()
        assert result == mock_query_instance

    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_from_config_with_agent_registration(
        self,
        mock_query_service,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        agent_config,
    ):
        """Test creating services with agent registration."""
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance

        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = ["test_tool"]

        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance

        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory
        result = SolanaAgentFactory.create_from_config(agent_config)

        # Verify calls
        mock_agent_instance.register_ai_agent.assert_called_once_with(
            name="test_agent",
            instructions="Test instructions",
            specialization="Test specialization",
        )

        mock_agent_instance.assign_tool_for_agent.assert_called_once_with(
            "test_agent", "test_tool"
        )

        assert result == mock_query_instance

    # capture_mode removed: test no longer applicable

    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_from_config_with_global_tools(
        self,
        mock_query_service,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        agent_tools_config,
    ):
        """Test creating services with global tool assignments."""
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance

        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = [
            "tool1",
            "tool2",
        ]

        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance

        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory
        result = SolanaAgentFactory.create_from_config(agent_tools_config)

        # Verify calls
        assert mock_agent_instance.assign_tool_for_agent.call_count == 2
        mock_agent_instance.assign_tool_for_agent.assert_any_call("test_agent", "tool1")
        mock_agent_instance.assign_tool_for_agent.assert_any_call("test_agent", "tool2")

        assert result == mock_query_instance

    def test_invalid_mongo_config(self):
        """Test handling of invalid MongoDB configuration."""
        """
        Additional tests for the SolanaAgentFactory class to improve coverage.
        """

        # Additional fixtures for new test scenarios
        @pytest.fixture
        def minimal_config():
            """Most minimal valid configuration."""
            return {"openai": {"api_key": "test-openai-key"}}

        @pytest.fixture
        def invalid_openai_config():
            """Configuration with missing OpenAI API key."""
            return {"openai": {}}

        @pytest.fixture
        def empty_agents_config(base_config):
            """Configuration with empty agents list."""
            config = deepcopy(base_config)
            config["agents"] = []
            return config

        @pytest.fixture
        def multiple_agents_config(base_config):
            """Configuration with multiple agents."""
            config = deepcopy(base_config)
            config["agents"] = [
                {
                    "name": "agent1",
                    "instructions": "Instructions for agent 1",
                    "specialization": "Specialization 1",
                    "tools": ["tool1", "tool2"],
                },
                {
                    "name": "agent2",
                    "instructions": "Instructions for agent 2",
                    "specialization": "Specialization 2",
                },
            ]
            return config

        @pytest.fixture
        def python_config_content():
            """Python configuration file content."""
            return """
        config = {
            "openai": {
                "api_key": "test-openai-key-from-python"
            },
            "agents": [
                {
                    "name": "python_agent",
                    "instructions": "Python config agent",
                    "specialization": "Python"
                }
            ]
        }
        """

        class TestSolanaAgentFactoryAdditional:
            """Additional tests for the SolanaAgentFactory."""

            def test_missing_openai_api_key(self, invalid_openai_config):
                """Test factory creation with missing OpenAI API key."""
                # This should raise ValueError since OpenAI API key is required
                with pytest.raises(ValueError, match="OpenAI API key is required"):
                    SolanaAgentFactory.create_from_config(invalid_openai_config)

            @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
            @patch("solana_agent.factories.agent_factory.AgentService")
            @patch("solana_agent.factories.agent_factory.RoutingService")
            @patch("solana_agent.factories.agent_factory.QueryService")
            def test_empty_agents_list(
                self,
                mock_query_service,
                mock_routing_service,
                mock_agent_service,
                mock_openai_adapter,
                empty_agents_config,
            ):
                """Test creating services with empty agents list."""
                # Setup mocks
                mock_openai_instance = MagicMock()
                mock_openai_adapter.return_value = mock_openai_instance

                mock_agent_instance = MagicMock()
                mock_agent_service.return_value = mock_agent_instance
                mock_agent_instance.tool_registry.list_all_tools.return_value = []

                mock_routing_instance = MagicMock()
                mock_routing_service.return_value = mock_routing_instance

                mock_query_instance = MagicMock()
                mock_query_service.return_value = mock_query_instance

                # Call the factory
                result = SolanaAgentFactory.create_from_config(empty_agents_config)

                # Verify calls - no agent registration should occur
                mock_agent_instance.register_ai_agent.assert_not_called()
                mock_agent_instance.assign_tool_for_agent.assert_not_called()
                assert result == mock_query_instance

            @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
            @patch("solana_agent.factories.agent_factory.AgentService")
            @patch("solana_agent.factories.agent_factory.RoutingService")
            @patch("solana_agent.factories.agent_factory.QueryService")
            def test_multiple_agents_registration(
                self,
                mock_query_service,
                mock_routing_service,
                mock_agent_service,
                mock_openai_adapter,
                multiple_agents_config,
            ):
                """Test creating services with multiple agents."""
                # Setup mocks
                mock_openai_instance = MagicMock()
                mock_openai_adapter.return_value = mock_openai_instance

                mock_agent_instance = MagicMock()
                mock_agent_service.return_value = mock_agent_instance
                mock_agent_instance.tool_registry.list_all_tools.return_value = [
                    "tool1",
                    "tool2",
                ]

                mock_routing_instance = MagicMock()
                mock_routing_service.return_value = mock_routing_instance

                mock_query_instance = MagicMock()
                mock_query_service.return_value = mock_query_instance

                # Call the factory
                SolanaAgentFactory.create_from_config(multiple_agents_config)

                # Verify calls for first agent
                mock_agent_instance.register_ai_agent.assert_any_call(
                    name="agent1",
                    instructions="Instructions for agent 1",
                    specialization="Specialization 1",
                )
                mock_agent_instance.assign_tool_for_agent.assert_any_call(
                    "agent1", "tool1"
                )
                mock_agent_instance.assign_tool_for_agent.assert_any_call(
                    "agent1", "tool2"
                )

                # Verify calls for second agent
                mock_agent_instance.register_ai_agent.assert_any_call(
                    name="agent2",
                    instructions="Instructions for agent 2",
                    specialization="Specialization 2",
                )

                # Ensure register_ai_agent was called exactly twice
                assert mock_agent_instance.register_ai_agent.call_count == 2

                # Ensure assign_tool_for_agent was called exactly twice
                assert mock_agent_instance.assign_tool_for_agent.call_count == 2

            @patch("builtins.open", new_callable=mock_open)
            @patch("importlib.util.spec_from_file_location")
            @patch("importlib.util.module_from_spec")
            @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
            @patch("solana_agent.factories.agent_factory.AgentService")
            @patch("solana_agent.factories.agent_factory.RoutingService")
            @patch("solana_agent.factories.agent_factory.QueryService")
            def test_python_config_file_loading(
                self,
                mock_query_service,
                mock_routing_service,
                mock_agent_service,
                mock_openai_adapter,
                mock_module_from_spec,
                mock_spec_from_file_location,
                mock_file,
                python_config_content,
            ):
                """Test loading configuration from a Python file."""
                # Setup mock for file opening
                mock_file.return_value.read.return_value = python_config_content

                # Setup mock for Python module loading
                mock_spec = MagicMock()
                mock_spec_from_file_location.return_value = mock_spec

                mock_module = MagicMock()
                mock_module_from_spec.return_value = mock_module
                mock_module.config = {
                    "openai": {"api_key": "test-openai-key-from-python"},
                    "agents": [
                        {
                            "name": "python_agent",
                            "instructions": "Python config agent",
                            "specialization": "Python",
                        }
                    ],
                }

                # Setup other mocks
                mock_openai_instance = MagicMock()
                mock_openai_adapter.return_value = mock_openai_instance

                mock_agent_instance = MagicMock()
                mock_agent_service.return_value = mock_agent_instance
                mock_agent_instance.tool_registry.list_all_tools.return_value = []

                mock_routing_instance = MagicMock()
                mock_routing_service.return_value = mock_routing_instance

                mock_query_instance = MagicMock()
                mock_query_service.return_value = mock_query_instance

                # Call the factory
                result = SolanaAgentFactory.create_from_config(
                    config_path="config.py"  # Non-JSON file to trigger Python module loading
                )

                # Verify Python module loading was used
                mock_spec_from_file_location.assert_called_once()
                mock_module_from_spec.assert_called_once()
                mock_spec.loader.exec_module.assert_called_once_with(mock_module)

                # Verify agent was registered from Python config
                mock_agent_instance.register_ai_agent.assert_called_once_with(
                    name="python_agent",
                    instructions="Python config agent",
                    specialization="Python",
                )

                assert result == mock_query_instance

            @patch("builtins.open", new_callable=mock_open)
            @patch("solana_agent.factories.agent_factory.json.load")
            @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
            @patch("solana_agent.factories.agent_factory.AgentService")
            @patch("solana_agent.factories.agent_factory.RoutingService")
            @patch("solana_agent.factories.agent_factory.QueryService")
            def test_json_config_file_loading(
                self,
                mock_query_service,
                mock_routing_service,
                mock_agent_service,
                mock_openai_adapter,
                mock_json_load,
                mock_file,
                minimal_config,
            ):
                """Test loading configuration from a JSON file."""
                # Setup mocks
                mock_json_load.return_value = minimal_config

                mock_openai_instance = MagicMock()
                mock_openai_adapter.return_value = mock_openai_instance

                mock_agent_instance = MagicMock()
                mock_agent_service.return_value = mock_agent_instance
                mock_agent_instance.tool_registry.list_all_tools.return_value = []

                mock_routing_instance = MagicMock()
                mock_routing_service.return_value = mock_routing_instance

                mock_query_instance = MagicMock()
                mock_query_service.return_value = mock_query_instance

                # Call the factory
                result = SolanaAgentFactory.create_from_config(
                    config_path="config.json"  # JSON file
                )

                # Verify JSON loading was used
                mock_file.assert_called_once_with("config.json", "r")
                mock_json_load.assert_called_once()

                assert result == mock_query_instance

            @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
            def test_no_config_provided(self, mock_openai_adapter):
                """Test factory creation with no configuration."""
                with pytest.raises(
                    ValueError, match="Either config or config_path must be provided"
                ):
                    SolanaAgentFactory.create_from_config(None)

            # Fixes for previously failing tests

            # 1. Fix knowledge base test by properly mocking the required imports
            @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
            @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
            @patch("solana_agent.factories.agent_factory.AgentService")
            @patch("solana_agent.factories.agent_factory.RoutingService")
            @patch("solana_agent.factories.agent_factory.PineconeAdapter")
            # Added create=True
            @patch(
                "solana_agent.factories.agent_factory.KnowledgeBaseService", create=True
            )
            @patch("solana_agent.factories.agent_factory.QueryService")
            def test_create_from_config_with_knowledge_base_fixed(
                self,
                mock_query_service,
                mock_knowledge_base,
                mock_pinecone_adapter,
                mock_routing_service,
                mock_agent_service,
                mock_openai_adapter,
                mock_mongo_adapter,
                knowledge_base_config,
            ):
                """Test creating services with knowledge base configuration (fixed)."""
                # Setup mocks
                mock_mongo_instance = MagicMock()
                mock_mongo_adapter.return_value = mock_mongo_instance

                mock_openai_instance = MagicMock()
                mock_openai_adapter.return_value = mock_openai_instance

                mock_pinecone_instance = MagicMock()
                mock_pinecone_adapter.return_value = mock_pinecone_instance

                mock_kb_instance = MagicMock()
                mock_knowledge_base.return_value = mock_kb_instance

                mock_agent_instance = MagicMock()
                mock_agent_service.return_value = mock_agent_instance
                mock_agent_instance.tool_registry.list_all_tools.return_value = []

                mock_routing_instance = MagicMock()
                mock_routing_service.return_value = mock_routing_instance

                mock_query_instance = MagicMock()
                mock_query_service.return_value = mock_query_instance

                # Call the factory
                result = SolanaAgentFactory.create_from_config(knowledge_base_config)

                # Verify calls
                mock_pinecone_adapter.assert_called_once()
                mock_knowledge_base.assert_called_once()

                # We don't check the exact parameters since there are many, but we check the result
                assert result == mock_query_instance

            # 2. Fix KB error test by correctly mocking the memory repository and accepting it's being created
            @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
            @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
            @patch("solana_agent.factories.agent_factory.AgentService")
            @patch("solana_agent.factories.agent_factory.RoutingService")
            @patch("solana_agent.factories.agent_factory.PineconeAdapter")
            @patch("solana_agent.factories.agent_factory.MemoryRepository")
            @patch("solana_agent.factories.agent_factory.QueryService")
            def test_create_from_config_with_kb_error_fixed(
                self,
                mock_query_service,
                mock_memory_repository,
                mock_pinecone_adapter,
                mock_routing_service,
                mock_agent_service,
                mock_openai_adapter,
                mock_mongo_adapter,
                knowledge_base_config,
            ):
                """Test creating services when knowledge base initialization fails (fixed)."""
                # Setup mocks
                mock_mongo_instance = MagicMock()
                mock_mongo_adapter.return_value = mock_mongo_instance

                mock_openai_instance = MagicMock()
                mock_openai_adapter.return_value = mock_openai_instance

                # Simulate Pinecone error
                mock_pinecone_adapter.side_effect = Exception("Pinecone error")

                mock_memory_instance = MagicMock()
                mock_memory_repository.return_value = mock_memory_instance

                mock_agent_instance = MagicMock()
                mock_agent_service.return_value = mock_agent_instance
                mock_agent_instance.tool_registry.list_all_tools.return_value = []

                mock_routing_instance = MagicMock()
                mock_routing_service.return_value = mock_routing_instance

                mock_query_instance = MagicMock()
                mock_query_service.return_value = mock_query_instance

                # Call the factory - should handle knowledge base error gracefully
                result = SolanaAgentFactory.create_from_config(knowledge_base_config)

                # Verify that QueryService is called with memory provider but no knowledge base
                mock_query_service.assert_called_once_with(
                    agent_service=mock_agent_instance,
                    routing_service=mock_routing_instance,
                    # Memory provider is created even when KB fails
                    memory_provider=mock_memory_instance,
                    knowledge_base=None,
                    kb_results_count=5,
                )

                assert result == mock_query_instance

            # Additional test for invalid Pinecone config
            # Fix for test_create_from_config_with_knowledge_base
            @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
            @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
            @patch("solana_agent.factories.agent_factory.AgentService")
            @patch("solana_agent.factories.agent_factory.RoutingService")
            @patch("solana_agent.factories.agent_factory.PineconeAdapter")
            # Added create=True to handle the AttributeError
            @patch(
                "solana_agent.factories.agent_factory.KnowledgeBaseService", create=True
            )
            # Added MemoryRepository mock
            @patch("solana_agent.factories.agent_factory.MemoryRepository")
            @patch("solana_agent.factories.agent_factory.QueryService")
            # Fix for test_create_from_config_with_kb_error (Corrected Assertion)
            # Removed duplicate patches
            @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
            @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
            @patch("solana_agent.factories.agent_factory.AgentService")
            @patch("solana_agent.factories.agent_factory.RoutingService")
            @patch("solana_agent.factories.agent_factory.PineconeAdapter")
            # Patches are applied bottom-up, arguments should match this order
            def test_create_from_config_with_kb_error(
                self,
                mock_pinecone_adapter,
                mock_routing_service,
                mock_agent_service,
                mock_openai_adapter,
                mock_mongo_adapter,
                mock_query_service,
                mock_memory_repo,
                mock_knowledge_base,  # Corrected argument order
                knowledge_base_config,
            ):
                """Test creating services when knowledge base initialization fails."""
                # Setup mocks
                mock_mongo_instance = MagicMock()
                mock_mongo_adapter.return_value = mock_mongo_instance

                mock_openai_instance = MagicMock()
                mock_openai_adapter.return_value = mock_openai_instance

                # Simulate Pinecone error causing KB init failure
                # No need to mock pinecone instance if side_effect is Exception
                mock_pinecone_adapter.side_effect = Exception("Pinecone error")

                # Mock MemoryRepository instance - should still be created
                mock_memory_instance = MagicMock()
                mock_memory_repo.return_value = mock_memory_instance

                mock_agent_instance = MagicMock()
                mock_agent_service.return_value = mock_agent_instance
                mock_agent_instance.tool_registry.list_all_tools.return_value = []

                mock_routing_instance = MagicMock()
                mock_routing_service.return_value = mock_routing_instance

                mock_query_instance = MagicMock()
                mock_query_service.return_value = mock_query_instance

                # Call the factory - should handle knowledge base error gracefully
                result = SolanaAgentFactory.create_from_config(knowledge_base_config)

                # Verify MemoryRepository was still called
                mock_memory_repo.assert_called_once_with(
                    mongo_adapter=mock_mongo_instance
                )

                # Verify PineconeAdapter was called (attempted)
                mock_pinecone_adapter.assert_called_once()
                # Verify KnowledgeBaseService constructor was NOT called because Pinecone failed first
                mock_knowledge_base.assert_not_called()

                # Verify QueryService call includes the memory provider even if KB failed
                mock_query_service.assert_called_once_with(
                    agent_service=mock_agent_instance,
                    routing_service=mock_routing_instance,
                    memory_provider=mock_memory_instance,  # Expect the memory instance
                    knowledge_base=None,
                    kb_results_count=5,  # Comes from knowledge_base_config
                )

                assert result == mock_query_instance

            # Test default kb_results_count when not specified
            @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
            @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
            @patch("solana_agent.factories.agent_factory.AgentService")
            @patch("solana_agent.factories.agent_factory.RoutingService")
            @patch("solana_agent.factories.agent_factory.PineconeAdapter")
            @patch(
                "solana_agent.factories.agent_factory.KnowledgeBaseService", create=True
            )
            @patch("solana_agent.factories.agent_factory.MemoryRepository")
            @patch("solana_agent.factories.agent_factory.QueryService")
            def test_kb_default_results_count(
                self,
                mock_query_service,
                mock_memory_repo,
                mock_knowledge_base,
                mock_pinecone_adapter,
                mock_routing_service,
                mock_agent_service,
                mock_openai_adapter,
                mock_mongo_adapter,
                knowledge_base_config,  # Use existing fixture but modify it
            ):
                """Test KB uses default results_count if not specified."""
                # Modify config to remove results_count
                config = deepcopy(knowledge_base_config)
                del config["knowledge_base"]["results_count"]

                # Setup mocks
                mock_mongo_instance = MagicMock()
                mock_mongo_adapter.return_value = mock_mongo_instance
                mock_openai_instance = MagicMock()
                mock_openai_adapter.return_value = mock_openai_instance
                mock_pinecone_instance = MagicMock()
                mock_pinecone_adapter.return_value = mock_pinecone_instance
                mock_kb_instance = MagicMock()
                mock_knowledge_base.return_value = mock_kb_instance
                mock_memory_instance = MagicMock()
                mock_memory_repo.return_value = mock_memory_instance
                mock_agent_instance = MagicMock()
                mock_agent_service.return_value = mock_agent_instance
                mock_agent_instance.tool_registry.list_all_tools.return_value = []
                mock_routing_instance = MagicMock()
                mock_routing_service.return_value = mock_routing_instance
                mock_query_instance = MagicMock()
                mock_query_service.return_value = mock_query_instance

                # Call the factory
                result = SolanaAgentFactory.create_from_config(config)

                # Verify KnowledgeBaseService was called with default rerank_top_k (which seems to be 3 based on factory code)
                # Note: The factory code uses results_count for rerank_top_k in KB init
                # And uses results_count (default 3) for kb_results_count in QueryService init
                mock_knowledge_base.assert_called_once()
                kb_call_args, kb_call_kwargs = mock_knowledge_base.call_args
                # Check default used for KB init
                assert kb_call_kwargs.get("rerank_top_k") == 3

                # Verify QueryService call uses the default kb_results_count (3)
                mock_query_service.assert_called_once_with(
                    agent_service=mock_agent_instance,
                    routing_service=mock_routing_instance,
                    memory_provider=mock_memory_instance,
                    knowledge_base=mock_kb_instance,
                    kb_results_count=3,  # Check default used for QueryService init
                )

                assert result == mock_query_instance

                # Verify KnowledgeBaseService constructor was not called successfully due to the error
                mock_knowledge_base.assert_not_called()

                # Verify QueryService call includes the memory provider even if KB failed
                mock_query_service.assert_called_once_with(
                    agent_service=mock_agent_instance,
                    routing_service=mock_routing_instance,
                    memory_provider=mock_memory_instance,  # FIX: Expect the memory instance
                    knowledge_base=None,
                    kb_results_count=5,  # Comes from knowledge_base_config even if KB fails
                )

                assert result == mock_query_instance

            # New test: Verify KB requires MongoDB
            @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
            @patch("solana_agent.factories.agent_factory.AgentService")
            @patch("solana_agent.factories.agent_factory.RoutingService")
            @patch("solana_agent.factories.agent_factory.PineconeAdapter")
            # Patch KB anyway
            @patch(
                "solana_agent.factories.agent_factory.KnowledgeBaseService", create=True
            )
            @patch("solana_agent.factories.agent_factory.MemoryRepository")
            @patch("solana_agent.factories.agent_factory.QueryService")
            def test_create_from_config_kb_requires_mongo(
                self,
                mock_query_service,
                mock_memory_repo,
                mock_knowledge_base,
                mock_pinecone_adapter,
                mock_routing_service,
                mock_agent_service,
                mock_openai_adapter,
                base_config,  # Use base_config (no mongo)
            ):
                """Test KB is not initialized if mongo config is missing."""
                # Add knowledge_base section to base_config
                config = deepcopy(base_config)
                config["knowledge_base"] = {
                    "collection": "test_kb",
                    "results_count": 7,  # Use a different count to test default override
                    # Minimal KB config
                    "pinecone": {"api_key": "test-pinecone-key"},
                }

                # Setup mocks
                mock_openai_instance = MagicMock()
                mock_openai_adapter.return_value = mock_openai_instance

                mock_agent_instance = MagicMock()
                mock_agent_service.return_value = mock_agent_instance
                mock_agent_instance.tool_registry.list_all_tools.return_value = []

                mock_routing_instance = MagicMock()
                mock_routing_service.return_value = mock_routing_instance

                mock_query_instance = MagicMock()
                mock_query_service.return_value = mock_query_instance

                # Call the factory
                result = SolanaAgentFactory.create_from_config(config)

                # Verify PineconeAdapter and KnowledgeBaseService were NOT called due to missing mongo
                mock_pinecone_adapter.assert_not_called()
                mock_knowledge_base.assert_not_called()

                # Verify MemoryRepository was not called (as neither mongo nor zep is configured)
                mock_memory_repo.assert_not_called()

                # Verify QueryService call has no memory provider and no knowledge base
                # kb_results_count should take the value from config even if KB fails init
                mock_query_service.assert_called_once_with(
                    agent_service=mock_agent_instance,
                    routing_service=mock_routing_instance,
                    memory_provider=None,
                    knowledge_base=None,
                    kb_results_count=7,  # Should use value from config
                )

                assert result == mock_query_instance

                assert result == mock_query_instance

        # --- Tests for _create_guardrails ---

        @patch("importlib.import_module")
        def test_create_guardrails_success(self, mock_import_module):
            """Test successful loading of guardrails."""
            # Mock importlib to return modules containing mock guardrail classes
            mock_module = MagicMock()
            mock_module.MockInputGuardrail = MockInputGuardrail
            mock_module.MockOutputGuardrail = MockOutputGuardrail
            mock_import_module.return_value = mock_module

            guardrail_configs = [
                {"class": "path.to.MockInputGuardrail", "config": {"p1": "v1"}},
                {"class": "path.to.MockOutputGuardrail", "config": {"p2": "v2"}},
            ]

            guardrails = SolanaAgentFactory._create_guardrails(guardrail_configs)

            assert len(guardrails) == 2
            assert isinstance(guardrails[0], MockInputGuardrail)
            assert guardrails[0].config == {"p1": "v1"}
            assert isinstance(guardrails[1], MockOutputGuardrail)
            assert guardrails[1].config == {"p2": "v2"}
            mock_import_module.assert_any_call("path.to")

        def test_create_guardrails_empty_config(self):
            """Test loading with empty guardrail configuration list."""
            guardrails = SolanaAgentFactory._create_guardrails([])
            assert len(guardrails) == 0

        def test_create_guardrails_missing_class(self, caplog):
            """Test loading when 'class' key is missing."""
            guardrail_configs = [{"config": {"p1": "v1"}}]
            guardrails = SolanaAgentFactory._create_guardrails(guardrail_configs)
            assert len(guardrails) == 0
            assert "Guardrail config missing 'class'" in caplog.text

        @patch("importlib.import_module", side_effect=ImportError("Module not found"))
        def test_create_guardrails_import_error(self, mock_import_module, caplog):
            """Test loading when module import fails."""
            guardrail_configs = [{"class": "invalid.path.NonExistentGuardrail"}]
            guardrails = SolanaAgentFactory._create_guardrails(guardrail_configs)
            assert len(guardrails) == 0
            assert (
                "Error loading guardrail class 'invalid.path.NonExistentGuardrail'"
                in caplog.text
            )
            assert "Module not found" in caplog.text

        @patch("importlib.import_module")
        def test_create_guardrails_attribute_error(self, mock_import_module, caplog):
            """Test loading when class is not found in the module."""
            mock_module = MagicMock(spec=[])  # Module exists but class doesn't
            mock_import_module.return_value = mock_module

            guardrail_configs = [{"class": "path.to.MissingGuardrail"}]
            guardrails = SolanaAgentFactory._create_guardrails(guardrail_configs)
            assert len(guardrails) == 0
            assert (
                "Error loading guardrail class 'path.to.MissingGuardrail'"
                in caplog.text
            )
            assert (
                "object has no attribute 'MissingGuardrail'" in caplog.text
            )  # Check specific error if possible

        @patch("importlib.import_module")
        def test_create_guardrails_init_error(self, mock_import_module, caplog):
            """Test loading when guardrail initialization fails."""
            mock_module = MagicMock()
            mock_module.MockErrorGuardrail = (
                MockErrorGuardrail  # Use the error-raising mock
            )
            mock_import_module.return_value = mock_module

            guardrail_configs = [{"class": "path.to.MockErrorGuardrail"}]
            guardrails = SolanaAgentFactory._create_guardrails(guardrail_configs)
            assert len(guardrails) == 0
            assert (
                "Error initializing guardrail 'path.to.MockErrorGuardrail'"
                in caplog.text
            )
            assert "Initialization failed" in caplog.text  # Check specific error

    @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_with_logfire_success(
        self,
        mock_query_service,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        mock_mongo_adapter,
        logfire_config,  # Use the logfire fixture
    ):
        """Test creating services with valid Logfire configuration."""
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance
        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = []
        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance
        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory
        result = SolanaAgentFactory.create_from_config(logfire_config)

        # Verify OpenAIAdapter was called with both keys
        mock_openai_adapter.assert_called_once_with(
            api_key="test-openai-key",
            model=None,
            logfire_api_key="test-logfire-key",
        )
        # Verify other services were called
        mock_agent_service.assert_called_once()
        mock_routing_service.assert_called_once()
        mock_query_service.assert_called_once()
        assert result == mock_query_instance

    def test_invalid_logfire_config_missing_api_key(
        self, invalid_logfire_config_missing_key
    ):
        """Test handling of invalid Logfire config (missing api_key)."""
        # Based on the current factory code, this should raise a ValueError
        with pytest.raises(ValueError, match="Pydantic Logfire API key is required."):
            SolanaAgentFactory.create_from_config(invalid_logfire_config_missing_key)

    def test_logfire_config_missing_openai_key(self, logfire_config_missing_openai):
        """Test handling of Logfire config when OpenAI key is missing."""
        # Based on the current factory code, this should raise a ValueError
        with pytest.raises(
            ValueError,
            match="OpenAI or Groq API key is required in config.",
        ):
            SolanaAgentFactory.create_from_config(logfire_config_missing_openai)

    @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_without_logfire_section(
        self,
        mock_query_service,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        mock_mongo_adapter,
        base_config,  # Use base config (no logfire section)
    ):
        """Test creating services when logfire section is absent."""
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance
        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = []
        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance
        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory
        result = SolanaAgentFactory.create_from_config(base_config)

        # Verify OpenAIAdapter was called only with OpenAI key
        mock_openai_adapter.assert_called_once_with(
            api_key="test-openai-key",
            model=None,
        )
        # Verify other services were called
        mock_agent_service.assert_called_once()
        mock_routing_service.assert_called_once()
        mock_query_service.assert_called_once()
        assert result == mock_query_instance

    @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_openai_with_model(
        self,
        mock_query_service,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        mock_mongo_adapter,
        openai_with_model_config,
    ):
        """Test creating services with OpenAI and a custom model specified."""
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance
        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = []
        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance
        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory
        result = SolanaAgentFactory.create_from_config(openai_with_model_config)

        # Verify OpenAIAdapter was called with OpenAI config and custom model
        mock_openai_adapter.assert_called_once_with(
            api_key="test-openai-key",
            model="gpt-5.2",
        )
        # Verify other services were called
        mock_agent_service.assert_called_once()
        mock_routing_service.assert_called_once()
        mock_query_service.assert_called_once()
        assert result == mock_query_instance

    @patch("solana_agent.factories.agent_factory.MongoDBAdapter")
    @patch("solana_agent.factories.agent_factory.OpenAIAdapter")
    @patch("solana_agent.factories.agent_factory.AgentService")
    @patch("solana_agent.factories.agent_factory.RoutingService")
    @patch("solana_agent.factories.agent_factory.QueryService")
    def test_create_groq_with_model(
        self,
        mock_query_service,
        mock_routing_service,
        mock_agent_service,
        mock_openai_adapter,
        mock_mongo_adapter,
        groq_with_model_config,
    ):
        """Test creating services with Groq and a custom model specified."""
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai_adapter.return_value = mock_openai_instance
        mock_agent_instance = MagicMock()
        mock_agent_service.return_value = mock_agent_instance
        mock_agent_instance.tool_registry.list_all_tools.return_value = []
        mock_routing_instance = MagicMock()
        mock_routing_service.return_value = mock_routing_instance
        mock_query_instance = MagicMock()
        mock_query_service.return_value = mock_query_instance

        # Call the factory
        result = SolanaAgentFactory.create_from_config(groq_with_model_config)

        # Verify OpenAIAdapter was called with Groq config and custom model
        mock_openai_adapter.assert_called_once_with(
            api_key="test-groq-key",
            model="openai/gpt-oss-120b",
            base_url="https://api.groq.com/openai/v1",
        )
        # Verify other services were called
        mock_agent_service.assert_called_once()
        mock_routing_service.assert_called_once()
        mock_query_service.assert_called_once()
        assert result == mock_query_instance

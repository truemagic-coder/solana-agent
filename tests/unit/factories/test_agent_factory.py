"""
Tests for the SolanaAgentFactory implementation.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any

from solana_agent.factories.agent_factory import SolanaAgentFactory
from solana_agent.services.query import QueryService
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
from solana_agent.adapters.llm_adapter import OpenAIAdapter
from solana_agent.plugins.manager import PluginManager


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return {
        "openai": {
            "api_key": "test_key"
        }
    }


@pytest.fixture
def full_config():
    """Full configuration with all options."""
    return {
        "openai": {
            "api_key": "test_key"
        },
        "mongo": {
            "connection_string": "mongodb://localhost:27017",
            "database": "test_db"
        },
        "zep": {
            "api_key": "zep_key",
        },
        "business": {
            "mission": "Test mission",
            "values": {
                "quality": "High standards",
                "innovation": "Forward thinking"
            },
            "goals": ["Goal 1", "Goal 2"],
            "voice": "Professional"
        },
        "agents": [
            {
                "name": "test_agent",
                "instructions": "Test instructions",
                "specialization": "Testing",
                "tools": ["tool1", "tool2"]
            }
        ],
        "agent_tools": {
            "test_agent": ["tool3", "tool4"]
        }
    }


class TestSolanaAgentFactory:
    """Test suite for SolanaAgentFactory."""

    def test_create_minimal_config(self, basic_config):
        """Test creation with minimal configuration."""
        with patch('solana_agent.factories.agent_factory.OpenAIAdapter') as mock_llm:
            service = SolanaAgentFactory.create_from_config(basic_config)
            assert isinstance(service, QueryService)
            mock_llm.assert_called_once_with(api_key="test_key")

    def test_create_with_mongo_only(self):
        """Test creation with MongoDB only."""
        config = {
            "openai": {"api_key": "test_key"},
            "mongo": {
                "connection_string": "mongodb://localhost:27017",
                "database": "test_db"
            }
        }

        with patch('solana_agent.factories.agent_factory.MongoDBAdapter') as mock_mongo:
            with patch('solana_agent.factories.agent_factory.OpenAIAdapter'):
                service = SolanaAgentFactory.create_from_config(config)
                mock_mongo.assert_called_once_with(
                    connection_string="mongodb://localhost:27017",
                    database_name="test_db"
                )
                assert isinstance(service, QueryService)

    def test_create_full_config(self, full_config):
        """Test creation with full configuration."""
        with patch('solana_agent.factories.agent_factory.MongoDBAdapter') as mock_mongo:
            with patch('solana_agent.factories.agent_factory.OpenAIAdapter'):
                with patch('solana_agent.factories.agent_factory.PluginManager') as mock_plugin_manager:
                    # Configure mock plugin manager
                    mock_manager = MagicMock()
                    mock_manager.load_plugins.return_value = 2
                    mock_plugin_manager.return_value = mock_manager

                    service = SolanaAgentFactory.create_from_config(
                        full_config)

                    assert isinstance(service, QueryService)
                    mock_mongo.assert_called_once()
                    mock_plugin_manager.assert_called_once()

    def test_plugin_loading_error(self, basic_config):
        """Test error handling for plugin loading."""
        with patch('solana_agent.factories.agent_factory.OpenAIAdapter'):
            with patch('solana_agent.factories.agent_factory.PluginManager') as mock_plugin_manager:
                # Configure mock to raise exception
                mock_plugin_manager.return_value.load_plugins.side_effect = Exception(
                    "Plugin error")

                # Test that factory continues without plugins
                service = SolanaAgentFactory.create_from_config(basic_config)
                assert isinstance(service, QueryService)

                # Verify plugin manager was called but error was handled
                mock_plugin_manager.assert_called_once()
                mock_plugin_manager.return_value.load_plugins.assert_called_once()

    def test_tool_registration_error(self):
        """Test error handling for tool registration."""
        config = {
            "openai": {"api_key": "test_key"},
            "agents": [{
                "name": "test_agent",
                "instructions": "Test instructions",
                "specialization": "Testing",
                "tools": ["nonexistent_tool"]
            }]
        }

        with patch('solana_agent.factories.agent_factory.OpenAIAdapter'):
            with patch('solana_agent.factories.agent_factory.PluginManager') as mock_plugin_manager:
                # Configure mock plugin manager
                mock_manager = MagicMock()
                mock_manager.load_plugins.return_value = 0
                mock_plugin_manager.return_value = mock_manager

                service = SolanaAgentFactory.create_from_config(config)
                assert isinstance(service, QueryService)

    def test_create_with_zep_no_key_no_mongo(self):
        """Test creation with Zep config missing API key and no MongoDB."""
        config = {
            "openai": {"api_key": "test_key"},
            "zep": {
                # Missing api_key
            }
        }

        with patch('solana_agent.factories.agent_factory.OpenAIAdapter'):
            with pytest.raises(ValueError) as exc_info:
                SolanaAgentFactory.create_from_config(config)
            assert "Zep API key is required" in str(exc_info.value)

    def test_create_with_zep_no_key_with_mongo(self):
        """Test creation with Zep config missing API key but with MongoDB."""
        config = {
            "openai": {"api_key": "test_key"},
            "zep": {
                # Missing api_key
            },
            "mongo": {
                "connection_string": "mongodb://localhost:27017",
                "database": "test_db"
            }
        }

        with patch('solana_agent.factories.agent_factory.OpenAIAdapter'):
            with patch('solana_agent.factories.agent_factory.MongoDBAdapter') as mock_mongo:
                with pytest.raises(ValueError) as exc_info:
                    service = SolanaAgentFactory.create_from_config(config)
                assert "Zep API key is required" in str(exc_info.value)

    def test_create_with_zep_and_mongo(self):
        """Test creation with both valid Zep and MongoDB configs."""
        config = {
            "openai": {"api_key": "test_key"},
            "zep": {
                "api_key": "zep_test_key",
            },
            "mongo": {
                "connection_string": "mongodb://localhost:27017",
                "database": "test_db"
            }
        }

        with patch('solana_agent.factories.agent_factory.OpenAIAdapter'):
            with patch('solana_agent.factories.agent_factory.MongoDBAdapter') as mock_mongo:
                service = SolanaAgentFactory.create_from_config(config)
                assert isinstance(service, QueryService)
                assert service.memory_provider is not None
                mock_mongo.assert_called_once()

    def test_create_with_zep_only(self):
        """Test creation with only valid Zep config."""
        config = {
            "openai": {"api_key": "test_key"},
            "zep": {
                "api_key": "zep_test_key",
            }
        }

        with patch('solana_agent.factories.agent_factory.OpenAIAdapter'):
            service = SolanaAgentFactory.create_from_config(config)
            assert isinstance(service, QueryService)
            assert service.memory_provider is not None
            assert service.memory_provider.mongo is None

    def test_create_with_mongo_missing_connection_string(self):
        """Test creation with MongoDB missing connection string."""
        config = {
            "openai": {"api_key": "test_key"},
            "mongo": {
                "database": "test_db"
                # Missing connection_string
            }
        }

        with patch('solana_agent.factories.agent_factory.OpenAIAdapter'):
            with pytest.raises(ValueError) as exc_info:
                SolanaAgentFactory.create_from_config(config)
            assert "MongoDB connection string is required" in str(
                exc_info.value)

    def test_create_with_mongo_missing_database(self):
        """Test creation with MongoDB missing database name."""
        config = {
            "openai": {"api_key": "test_key"},
            "mongo": {
                "connection_string": "mongodb://localhost:27017"
                # Missing database
            }
        }

        with patch('solana_agent.factories.agent_factory.OpenAIAdapter'):
            with pytest.raises(ValueError) as exc_info:
                SolanaAgentFactory.create_from_config(config)
            assert "MongoDB database name is required" in str(exc_info.value)

    def test_create_with_mongo_missing_both(self):
        """Test creation with MongoDB missing both connection string and database."""
        config = {
            "openai": {"api_key": "test_key"},
            "mongo": {}  # Empty mongo config
        }

        with patch('solana_agent.factories.agent_factory.OpenAIAdapter'):
            with pytest.raises(ValueError) as exc_info:
                SolanaAgentFactory.create_from_config(config)
            assert "MongoDB connection string is required" in str(
                exc_info.value)

    def test_create_with_gemini_config(self):
        """Test creation with Gemini API configuration."""
        config = {
            "openai": {"api_key": "openai_test_key"},
            "gemini": {"api_key": "gemini_test_key"}
        }

        with patch('solana_agent.factories.agent_factory.OpenAIAdapter') as mock_llm:
            with patch('solana_agent.factories.agent_factory.AgentService') as mock_agent_service:
                with patch('solana_agent.factories.agent_factory.RoutingService') as mock_routing_service:
                    # Configure mocks
                    mock_llm.return_value = MagicMock()
                    mock_agent_service.return_value = MagicMock()
                    mock_routing_service.return_value = MagicMock()

                    # Call the factory method
                    service = SolanaAgentFactory.create_from_config(config)

                    # Verify AgentService was created with correct Gemini parameters
                    mock_agent_service.assert_called_once()
                    agent_service_kwargs = mock_agent_service.call_args.kwargs
                    assert agent_service_kwargs["api_key"] == "gemini_test_key"
                    assert agent_service_kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
                    assert agent_service_kwargs["model"] == "gemini-2.0-flash"

                    # Verify RoutingService was created with correct Gemini parameters
                    mock_routing_service.assert_called_once()
                    routing_service_kwargs = mock_routing_service.call_args.kwargs
                    assert routing_service_kwargs["api_key"] == "gemini_test_key"
                    assert routing_service_kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
                    assert routing_service_kwargs["model"] == "gemini-2.0-flash"

                    # Verify the QueryService was returned
                    assert isinstance(service, QueryService)

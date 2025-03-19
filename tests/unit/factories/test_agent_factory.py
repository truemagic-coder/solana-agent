"""
Tests for the SolanaAgentFactory.

This module tests the factory's ability to create and wire together
the components of the Solana Agent system.
"""
import pytest
from unittest.mock import patch, MagicMock
import json
import os
import tempfile

from solana_agent.factories.agent_factory import SolanaAgentFactory


@pytest.fixture
def minimal_config():
    """Create a minimal configuration for testing."""
    return {
        "mongo": {
            "connection_string": "mongodb://localhost:27017",
            "database": "solana_agent_test"
        },
        "openai": {
            "api_key": "dummy-api-key"
        }
    }


@pytest.fixture
def full_config():
    """Create a comprehensive configuration for testing."""
    return {
        "mongo": {
            "connection_string": "mongodb://localhost:27017",
            "database": "solana_agent_test"
        },
        "openai": {
            "api_key": "dummy-api-key",
            "default_model": "gpt-4o"
        },
        "zep": {
            "api_key": "zep-api-key",
            "base_url": "http://localhost:8000"
        },
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "qdrant-api-key",
            "collection": "test_collection",
            "embedding_model": "text-embedding-3-large"
        },
        "organization": {
            "mission_statement": "Help developers build on Solana",
            "values": {
                "speed": "We value fast execution and iteration",
                "security": "We prioritize secure code"
            },
            "goals": ["Increase Solana adoption", "Support developers"],
            "guidance": "Always be helpful and accurate"
        },
        "ai_agents": [
            {
                "name": "SolanaHelper",
                "instructions": "You help with Solana development",
                "specialization": "blockchain",
                "model": "gpt-4o",
                "tools": ["websearch", "documentation"]
            },
            {
                "name": "CryptoExpert",
                "instructions": "You're an expert in crypto protocols",
                "specialization": "crypto",
                "tools": ["calculator"]
            }
        ],
        "agent_tools": {
            "SolanaHelper": ["code_executor", "diagram_generator"]
        },
        "enable_critic": True,
        "require_human_approval": True,
        "stalled_ticket_timeout": 120
    }


@pytest.fixture
def json_config_file():
    """Create a temporary JSON config file."""
    config = {
        "mongo": {
            "connection_string": "mongodb://localhost:27017",
            "database": "solana_agent_test"
        },
        "openai": {
            "api_key": "dummy-api-key"
        }
    }

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
        temp.write(json.dumps(config).encode('utf-8'))
        temp_path = temp.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


# Mock classes for testing
class MockPluginManager:
    def load_plugins(self):
        return 3


# --------------------------
# Factory Creation Tests
# --------------------------

@patch('solana_agent.factories.agent_factory.MongoDBAdapter')
@patch('solana_agent.factories.agent_factory.OpenAIAdapter')
@patch('solana_agent.factories.agent_factory.QueryService')
# Mock HandoffService
@patch('solana_agent.factories.agent_factory.HandoffService')
@patch('solana_agent.factories.agent_factory.MongoTicketRepository')
@patch('solana_agent.factories.agent_factory.MongoHandoffRepository')
@patch('solana_agent.factories.agent_factory.AgentService')  # Add AgentService
def test_create_from_minimal_config(mock_agent_service, mock_handoff_repo, mock_ticket_repo,
                                    mock_handoff_service, mock_query_service, mock_openai,
                                    mock_mongo, minimal_config):
    """Test factory creates components with minimal config."""
    # Configure mocks
    mock_mongo.return_value = MagicMock()
    mock_openai.return_value = MagicMock()
    mock_query_service.return_value = MagicMock()
    mock_handoff_service.return_value = MagicMock()
    mock_ticket_repo.return_value = MagicMock()
    mock_handoff_repo.return_value = MagicMock()
    mock_agent_service.return_value = MagicMock()
    mock_agent_service.return_value.tool_registry = MagicMock()
    mock_agent_service.return_value.tool_registry.list_all_tools.return_value = []

    # Execute
    with patch('solana_agent.factories.agent_factory.PluginManager', return_value=MockPluginManager()), \
            patch('solana_agent.factories.agent_factory.MemoryService'), \
            patch('solana_agent.factories.agent_factory.NPSService'), \
            patch('solana_agent.factories.agent_factory.CommandService'), \
            patch('solana_agent.factories.agent_factory.CriticService'), \
            patch('solana_agent.factories.agent_factory.TaskPlanningService'), \
            patch('solana_agent.factories.agent_factory.ProjectApprovalService'), \
            patch('solana_agent.factories.agent_factory.ProjectSimulationService'), \
            patch('solana_agent.factories.agent_factory.SchedulingService'), \
            patch('solana_agent.factories.agent_factory.RoutingService'):
        result = SolanaAgentFactory.create_from_config(minimal_config)

    # Assertions
    mock_mongo.assert_called_once_with(
        connection_string="mongodb://localhost:27017",
        database_name="solana_agent_test"
    )

    mock_openai.assert_called_once()
    mock_query_service.assert_called_once()
    mock_handoff_service.assert_called_once()  # Verify HandoffService was called

    assert result is not None
    assert result == mock_query_service.return_value


@patch('solana_agent.factories.agent_factory.MongoDBAdapter')
@patch('solana_agent.factories.agent_factory.OpenAIAdapter')
@patch('solana_agent.factories.agent_factory.ZepMemoryAdapter')
@patch('solana_agent.factories.agent_factory.QdrantAdapter')
@patch('solana_agent.factories.agent_factory.AgentService')
@patch('solana_agent.factories.agent_factory.QueryService')
@patch('solana_agent.factories.agent_factory.HandoffService')
def test_create_from_full_config(mock_handoff, mock_query_service, mock_agent_service,
                                 mock_qdrant, mock_zep, mock_openai, mock_mongo, full_config):
    """Test factory creates all components with full config."""
    # Configure mocks
    mock_mongo.return_value = MagicMock()
    mock_openai.return_value = MagicMock()
    mock_zep.return_value = MagicMock()
    mock_qdrant.return_value = MagicMock()
    mock_agent_service.return_value = MagicMock()
    mock_agent_service.return_value.tool_registry = MagicMock()
    mock_agent_service.return_value.tool_registry.list_all_tools.return_value = [
        "websearch", "documentation", "calculator", "code_executor", "diagram_generator"]
    mock_query_service.return_value = MagicMock()
    mock_handoff.return_value = MagicMock()

    # Execute
    with patch('solana_agent.factories.agent_factory.PluginManager', return_value=MockPluginManager()), \
            patch('solana_agent.factories.agent_factory.MongoTicketRepository'), \
            patch('solana_agent.factories.agent_factory.MongoFeedbackRepository'), \
            patch('solana_agent.factories.agent_factory.MongoMemoryRepository'), \
            patch('solana_agent.factories.agent_factory.MongoAgentRepository'), \
            patch('solana_agent.factories.agent_factory.MongoHandoffRepository'), \
            patch('solana_agent.factories.agent_factory.MongoSchedulingRepository'), \
            patch('solana_agent.factories.agent_factory.MemoryService'), \
            patch('solana_agent.factories.agent_factory.NPSService'), \
            patch('solana_agent.factories.agent_factory.CommandService'), \
            patch('solana_agent.factories.agent_factory.CriticService'), \
            patch('solana_agent.factories.agent_factory.TaskPlanningService'), \
            patch('solana_agent.factories.agent_factory.ProjectApprovalService'), \
            patch('solana_agent.factories.agent_factory.ProjectSimulationService'), \
            patch('solana_agent.factories.agent_factory.SchedulingService'), \
            patch('solana_agent.factories.agent_factory.RoutingService'):
        result = SolanaAgentFactory.create_from_config(full_config)

    # Assertions
    mock_mongo.assert_called_once()
    mock_openai.assert_called_once_with(
        api_key="dummy-api-key",
        model="gpt-4o"
    )
    mock_zep.assert_called_once()
    mock_qdrant.assert_called_once()
    mock_agent_service.assert_called_once()
    mock_handoff.assert_called_once()  # Verify HandoffService was called
    mock_query_service.assert_called_once()

    # Check that agent registration was called
    assert mock_agent_service.return_value.register_ai_agent.call_count == 2

    # Check that the result is the query service
    assert result == mock_query_service.return_value


@patch('solana_agent.factories.agent_factory.MongoDBAdapter')
@patch('solana_agent.factories.agent_factory.OpenAIAdapter')
@patch('solana_agent.factories.agent_factory.PineconeAdapter')
@patch('solana_agent.factories.agent_factory.QueryService')
@patch('solana_agent.factories.agent_factory.HandoffService')
@patch('solana_agent.factories.agent_factory.AgentService')
def test_create_with_pinecone(mock_agent_service, mock_handoff, mock_query_service,
                              mock_pinecone, mock_openai, mock_mongo):
    """Test factory creates with Pinecone vector store."""
    # Create config with Pinecone
    config = {
        "mongo": {
            "connection_string": "mongodb://localhost:27017",
            "database": "solana_agent_test"
        },
        "openai": {
            "api_key": "dummy-api-key"
        },
        "pinecone": {
            "api_key": "pinecone-api-key",
            "index": "solana-index",
            "embedding_model": "text-embedding-3-large"
        }
    }

    # Configure mocks
    mock_mongo.return_value = MagicMock()
    mock_openai.return_value = MagicMock()
    mock_pinecone.return_value = MagicMock()
    mock_query_service.return_value = MagicMock()
    mock_handoff.return_value = MagicMock()
    mock_agent_service.return_value = MagicMock()
    mock_agent_service.return_value.tool_registry = MagicMock()
    mock_agent_service.return_value.tool_registry.list_all_tools.return_value = []

    # Execute
    with patch('solana_agent.factories.agent_factory.PluginManager', return_value=MockPluginManager()), \
            patch('solana_agent.factories.agent_factory.MongoTicketRepository'), \
            patch('solana_agent.factories.agent_factory.MongoHandoffRepository'), \
            patch('solana_agent.factories.agent_factory.MemoryService'), \
            patch('solana_agent.factories.agent_factory.NPSService'), \
            patch('solana_agent.factories.agent_factory.CommandService'), \
            patch('solana_agent.factories.agent_factory.CriticService'), \
            patch('solana_agent.factories.agent_factory.TaskPlanningService'), \
            patch('solana_agent.factories.agent_factory.ProjectApprovalService'), \
            patch('solana_agent.factories.agent_factory.ProjectSimulationService'), \
            patch('solana_agent.factories.agent_factory.SchedulingService'), \
            patch('solana_agent.factories.agent_factory.RoutingService'):
        result = SolanaAgentFactory.create_from_config(config)

    # Assertions
    mock_pinecone.assert_called_once_with(
        api_key="pinecone-api-key",
        index_name="solana-index",
        embedding_model="text-embedding-3-large"
    )
    mock_query_service.assert_called_once()


@patch('solana_agent.factories.agent_factory.MongoDBAdapter')
@patch('solana_agent.factories.agent_factory.OpenAIAdapter')
@patch('solana_agent.factories.agent_factory.AgentService')
@patch('solana_agent.factories.agent_factory.QueryService')
@patch('solana_agent.factories.agent_factory.HandoffService')
def test_organization_mission_creation(mock_handoff, mock_query_service, mock_agent_service,
                                       mock_openai, mock_mongo):
    """Test organization mission is correctly created from config."""
    # Create config with organization mission
    config = {
        "mongo": {
            "connection_string": "mongodb://localhost:27017",
            "database": "solana_agent_test"
        },
        "openai": {
            "api_key": "dummy-api-key"
        },
        "organization": {
            "mission_statement": "Help developers build on Solana",
            "values": {
                "speed": "We value fast execution and iteration"
            },
            "goals": ["Increase Solana adoption"],
            "guidance": "Always be helpful"
        }
    }

    # Configure mocks
    mock_mongo.return_value = MagicMock()
    mock_openai.return_value = MagicMock()
    mock_agent_service.return_value = MagicMock()
    mock_agent_service.return_value.tool_registry = MagicMock()
    mock_query_service.return_value = MagicMock()
    mock_handoff.return_value = MagicMock()

    # Execute
    with patch('solana_agent.factories.agent_factory.OrganizationMission') as mock_mission, \
            patch('solana_agent.factories.agent_factory.PluginManager', return_value=MockPluginManager()), \
            patch('solana_agent.factories.agent_factory.MongoTicketRepository'), \
            patch('solana_agent.factories.agent_factory.MongoHandoffRepository'), \
            patch('solana_agent.factories.agent_factory.MemoryService'), \
            patch('solana_agent.factories.agent_factory.NPSService'), \
            patch('solana_agent.factories.agent_factory.CommandService'), \
            patch('solana_agent.factories.agent_factory.CriticService'), \
            patch('solana_agent.factories.agent_factory.TaskPlanningService'), \
            patch('solana_agent.factories.agent_factory.ProjectApprovalService'), \
            patch('solana_agent.factories.agent_factory.ProjectSimulationService'), \
            patch('solana_agent.factories.agent_factory.SchedulingService'), \
            patch('solana_agent.factories.agent_factory.RoutingService'):
        result = SolanaAgentFactory.create_from_config(config)

    # Assertions
    mock_mission.assert_called_once_with(
        mission_statement="Help developers build on Solana",
        values=[
            {"name": "speed", "description": "We value fast execution and iteration"}],
        goals=["Increase Solana adoption"],
        guidance="Always be helpful"
    )

    # Check that agent service was created with the mission
    mock_agent_service.assert_called_once()
    assert "organization_mission" in mock_agent_service.call_args.kwargs


@patch('solana_agent.factories.agent_factory.MongoDBAdapter')
@patch('solana_agent.factories.agent_factory.OpenAIAdapter')
@patch('solana_agent.factories.agent_factory.AgentService')
@patch('solana_agent.factories.agent_factory.QueryService')
@patch('solana_agent.factories.agent_factory.HandoffService')
def test_agent_tool_registration(mock_handoff, mock_query_service, mock_agent_service,
                                 mock_openai, mock_mongo, full_config):
    """Test agent tools are correctly registered."""
    # Configure mocks
    mock_mongo.return_value = MagicMock()
    mock_openai.return_value = MagicMock()
    mock_agent_service.return_value = MagicMock()
    mock_agent_service.return_value.tool_registry = MagicMock()
    mock_agent_service.return_value.tool_registry.list_all_tools.return_value = [
        "websearch", "documentation", "calculator", "code_executor", "diagram_generator"
    ]
    mock_query_service.return_value = MagicMock()
    mock_handoff.return_value = MagicMock()

    # Execute
    with patch('solana_agent.factories.agent_factory.PluginManager', return_value=MockPluginManager()), \
            patch('solana_agent.factories.agent_factory.MongoTicketRepository'), \
            patch('solana_agent.factories.agent_factory.MongoFeedbackRepository'), \
            patch('solana_agent.factories.agent_factory.MongoMemoryRepository'), \
            patch('solana_agent.factories.agent_factory.MongoAgentRepository'), \
            patch('solana_agent.factories.agent_factory.MongoHandoffRepository'), \
            patch('solana_agent.factories.agent_factory.MongoSchedulingRepository'), \
            patch('solana_agent.factories.agent_factory.MemoryService'), \
            patch('solana_agent.factories.agent_factory.NPSService'), \
            patch('solana_agent.factories.agent_factory.CommandService'), \
            patch('solana_agent.factories.agent_factory.CriticService'), \
            patch('solana_agent.factories.agent_factory.TaskPlanningService'), \
            patch('solana_agent.factories.agent_factory.ProjectApprovalService'), \
            patch('solana_agent.factories.agent_factory.ProjectSimulationService'), \
            patch('solana_agent.factories.agent_factory.SchedulingService'), \
            patch('solana_agent.factories.agent_factory.RoutingService'):
        result = SolanaAgentFactory.create_from_config(full_config)

    # Assertions
    register_tool_calls = mock_agent_service.return_value.assign_tool_for_agent.call_count

    # Should have 2 from agent1 + 1 from agent2 + 2 from agent_tools = 5 calls
    assert register_tool_calls == 5


# Fix test_agent_deletion_sync function signature
@patch('solana_agent.factories.agent_factory.MongoDBAdapter')
@patch('solana_agent.factories.agent_factory.OpenAIAdapter')
@patch('solana_agent.factories.agent_factory.AgentService')
@patch('solana_agent.factories.agent_factory.QueryService')
@patch('solana_agent.factories.agent_factory.HandoffService')
def test_agent_deletion_sync(mock_handoff, mock_query_service, mock_agent_service,
                             mock_openai, mock_mongo):
    """Test agents not in config are deleted from repository."""
    config = {
        "mongo": {
            "connection_string": "mongodb://localhost:27017",
            "database": "solana_agent_test"
        },
        "openai": {
            "api_key": "dummy-api-key"
        },
        "ai_agents": [
            {
                "name": "Agent1",
                "instructions": "You are Agent1",
                "specialization": "general"
            }
        ]
    }

    # Configure mocks
    mock_mongo.return_value = MagicMock()
    mock_openai.return_value = MagicMock()
    mock_agent_service.return_value = MagicMock()
    mock_agent_service.return_value.tool_registry = MagicMock()
    mock_handoff.return_value = MagicMock()

    # Create mock agent repository with extra agents
    mock_agent_repo = MagicMock()

    # Create mock agents to return
    mock_agent1 = MagicMock()
    mock_agent1.name = "Agent1"

    mock_agent2 = MagicMock()
    mock_agent2.name = "Agent2"  # This one should be deleted

    mock_agent_repo.get_all_ai_agents.return_value = [mock_agent1, mock_agent2]

    # Patch the agent repository creation
    with patch('solana_agent.factories.agent_factory.MongoAgentRepository', return_value=mock_agent_repo), \
            patch('solana_agent.factories.agent_factory.PluginManager', return_value=MockPluginManager()), \
            patch('solana_agent.factories.agent_factory.MongoTicketRepository'), \
            patch('solana_agent.factories.agent_factory.MongoFeedbackRepository'), \
            patch('solana_agent.factories.agent_factory.MongoMemoryRepository'), \
            patch('solana_agent.factories.agent_factory.MongoHandoffRepository'), \
            patch('solana_agent.factories.agent_factory.MongoSchedulingRepository'), \
            patch('solana_agent.factories.agent_factory.MemoryService'), \
            patch('solana_agent.factories.agent_factory.NPSService'), \
            patch('solana_agent.factories.agent_factory.CommandService'), \
            patch('solana_agent.factories.agent_factory.CriticService'), \
            patch('solana_agent.factories.agent_factory.TaskPlanningService'), \
            patch('solana_agent.factories.agent_factory.ProjectApprovalService'), \
            patch('solana_agent.factories.agent_factory.ProjectSimulationService'), \
            patch('solana_agent.factories.agent_factory.SchedulingService'), \
            patch('solana_agent.factories.agent_factory.RoutingService'):
        # Execute
        result = SolanaAgentFactory.create_from_config(config)

    # Assertions - should delete Agent2
    mock_agent_repo.delete_ai_agent.assert_called_once_with("Agent2")


def test_missing_required_config():
    """Test error handling for missing required configuration."""
    # Config missing mongo connection
    invalid_config = {
        "openai": {
            "api_key": "dummy-api-key"
        }
    }

    # Should raise KeyError for missing mongo config
    with pytest.raises(KeyError):
        SolanaAgentFactory.create_from_config(invalid_config)

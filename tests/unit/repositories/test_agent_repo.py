"""
Tests for agent repository implementations.

This module contains unit tests for MongoAgentRepository.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime

from solana_agent.repositories.agent import MongoAgentRepository
from solana_agent.domains.agents import AIAgent


@pytest.fixture
def mock_db_adapter():
    """Create a mock database adapter."""
    adapter = Mock()
    adapter.create_collection = Mock()
    adapter.create_index = Mock()
    adapter.find_one = Mock()
    adapter.find = Mock()
    adapter.insert_one = Mock()
    adapter.update_one = Mock()
    adapter.delete_one = Mock()
    return adapter


@pytest.fixture
def agent_repository(mock_db_adapter):
    """Create a repository with mocked database adapter."""
    return MongoAgentRepository(mock_db_adapter)


@pytest.fixture
def sample_ai_agent():
    """Create a sample AI agent for testing."""
    return AIAgent(
        name="codebot",
        agent_type='ai',
        description="An AI coding assistant",
        model="gpt-4",
        system_prompt="You are a helpful coding assistant.",
        instructions="Help with coding questions and debugging issues.",
        specialization="python",  # This should be a string, not a list
        created_at=datetime.now(),
        is_active=True
    )


class TestMongoAgentRepository:
    """Tests for the MongoAgentRepository implementation."""

    def test_init(self, mock_db_adapter):
        """Test repository initialization."""
        repo = MongoAgentRepository(mock_db_adapter)

        # Verify collections are created
        assert mock_db_adapter.create_collection.call_count == 1

        # Verify indexes are created
        assert mock_db_adapter.create_index.call_count == 1
        mock_db_adapter.create_index.assert_any_call(
            "ai_agents", [("name", 1)], unique=True)

    def test_get_ai_agent_found(self, agent_repository, mock_db_adapter, sample_ai_agent):
        """Test getting an existing AI agent."""
        # Configure mock to return agent data
        mock_db_adapter.find_one.return_value = sample_ai_agent.model_dump()

        # Get the agent
        agent = agent_repository.get_ai_agent("codebot")

        # Verify database was queried correctly
        mock_db_adapter.find_one.assert_called_once_with(
            "ai_agents", {"name": "codebot"})

        # Verify result
        assert agent is not None
        assert agent.name == "codebot"
        assert agent.model == "gpt-4"
        assert "python" in agent.specialization

    def test_get_ai_agent_not_found(self, agent_repository, mock_db_adapter):
        """Test getting a non-existent AI agent."""
        # Configure mock to return None (not found)
        mock_db_adapter.find_one.return_value = None

        # Get the agent
        agent = agent_repository.get_ai_agent("nonexistent")

        # Verify database was queried correctly
        mock_db_adapter.find_one.assert_called_once_with(
            "ai_agents", {"name": "nonexistent"})

        # Verify result
        assert agent is None

    def test_get_all_ai_agents(self, agent_repository, mock_db_adapter, sample_ai_agent):
        """Test getting all AI agents."""
        # Configure mock to return list of agents
        agent1 = sample_ai_agent.model_dump()
        agent2 = sample_ai_agent.model_dump()
        agent2["name"] = "databot"
        agent2["description"] = "A data analysis assistant"
        mock_db_adapter.find.return_value = [agent1, agent2]

        # Get all agents
        agents = agent_repository.get_all_ai_agents()

        # Verify database was queried correctly
        mock_db_adapter.find.assert_called_once_with("ai_agents", {})

        # Verify results
        assert len(agents) == 2
        assert agents[0].name == "codebot"
        assert agents[1].name == "databot"
        assert agents[1].description == "A data analysis assistant"

    def test_save_ai_agent_new(self, agent_repository, mock_db_adapter, sample_ai_agent):
        """Test saving a new AI agent."""
        # Configure mock to return None (agent doesn't exist yet)
        mock_db_adapter.find_one.return_value = None

        # Save the agent
        result = agent_repository.save_ai_agent(sample_ai_agent)

        # Verify database operations
        mock_db_adapter.find_one.assert_called_once_with(
            "ai_agents", {"name": "codebot"})
        mock_db_adapter.insert_one.assert_called_once_with(
            "ai_agents", sample_ai_agent.model_dump())

        # Verify result
        assert result is True

    def test_save_ai_agent_update(self, agent_repository, mock_db_adapter, sample_ai_agent):
        """Test updating an existing AI agent."""
        # Configure mock to return existing agent
        mock_db_adapter.find_one.return_value = {
            "name": "codebot", "model": "gpt-3.5"}
        mock_db_adapter.update_one.return_value = True

        # Save the agent
        result = agent_repository.save_ai_agent(sample_ai_agent)

        # Verify database operations
        mock_db_adapter.find_one.assert_called_once_with(
            "ai_agents", {"name": "codebot"})
        mock_db_adapter.update_one.assert_called_once_with(
            "ai_agents",
            {"name": "codebot"},
            {"$set": sample_ai_agent.model_dump()}
        )

        # Verify result
        assert result is True

    def test_delete_ai_agent(self, agent_repository, mock_db_adapter):
        """Test deleting an AI agent."""
        # Configure mock to return success
        mock_db_adapter.delete_one.return_value = True

        # Delete the agent
        result = agent_repository.delete_ai_agent("codebot")

        # Verify database operation
        mock_db_adapter.delete_one.assert_called_once_with(
            "ai_agents", {"name": "codebot"})

        # Verify result
        assert result is True

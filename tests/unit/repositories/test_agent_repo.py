"""
Tests for agent repository implementations.

This module contains unit tests for MongoAgentRepository.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from solana_agent.repositories.agent import MongoAgentRepository
from solana_agent.domains import AIAgent, HumanAgent, AgentPerformance, AgentType


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
        agent_type=AgentType.AI,
        description="An AI coding assistant",
        model="gpt-4",
        system_prompt="You are a helpful coding assistant.",
        instructions="Help with coding questions and debugging issues.",
        specialization="python",  # This should be a string, not a list
        created_at=datetime.now(),
        is_active=True
    )


@pytest.fixture
def sample_human_agent():
    """Create a sample human agent for testing."""
    return HumanAgent(
        id="user123",
        name="Jane Doe",
        agent_type=AgentType.HUMAN,
        email="jane@example.com",
        specializations=["project_management", "design"],
        availability=True,  # Should be boolean, not dict
        schedule={  # Move schedule information to appropriate field
            "monday": ["09:00-17:00"],
            "tuesday": ["09:00-17:00"],
            "wednesday": ["09:00-17:00"],
            "thursday": ["09:00-17:00"],
            "friday": ["09:00-13:00"]
        },
        created_at=datetime.now(),
        is_active=True
    )


@pytest.fixture
def sample_performance():
    """Create sample performance metrics for testing."""
    return AgentPerformance(
        agent_id="codebot",
        agent_type="AI",
        # Remove period_start and period_end as they don't exist
        tasks_completed=15,
        avg_response_time=3.5,
        avg_satisfaction_score=4.8,
        custom_metrics={
            "code_quality": 4.7,
            "bug_rate": 0.02
        },
        # Add a last_updated field which is required
        last_updated=datetime.now()
    )


class TestMongoAgentRepository:
    """Tests for the MongoAgentRepository implementation."""

    def test_init(self, mock_db_adapter):
        """Test repository initialization."""
        repo = MongoAgentRepository(mock_db_adapter)

        # Verify collections are created
        assert mock_db_adapter.create_collection.call_count == 3
        mock_db_adapter.create_collection.assert_any_call("ai_agents")
        mock_db_adapter.create_collection.assert_any_call("human_agents")
        mock_db_adapter.create_collection.assert_any_call("agent_performance")

        # Verify indexes are created
        assert mock_db_adapter.create_index.call_count == 5
        mock_db_adapter.create_index.assert_any_call(
            "ai_agents", [("name", 1)], unique=True)
        mock_db_adapter.create_index.assert_any_call(
            "human_agents", [("id", 1)], unique=True)
        mock_db_adapter.create_index.assert_any_call(
            "human_agents", [("email", 1)], unique=True)
        mock_db_adapter.create_index.assert_any_call(
            "agent_performance", [("agent_id", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "agent_performance", [("period_start", 1)])

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

    def test_get_human_agent_found(self, agent_repository, mock_db_adapter, sample_human_agent):
        """Test getting an existing human agent."""
        # Configure mock to return agent data
        mock_db_adapter.find_one.return_value = sample_human_agent.model_dump()

        # Get the agent
        agent = agent_repository.get_human_agent("user123")

        # Verify database was queried correctly
        mock_db_adapter.find_one.assert_called_once_with(
            "human_agents", {"id": "user123"})

        # Verify result
        assert agent is not None
        assert agent.id == "user123"
        assert agent.name == "Jane Doe"
        assert "project_management" in agent.specializations

    def test_get_human_agent_not_found(self, agent_repository, mock_db_adapter):
        """Test getting a non-existent human agent."""
        # Configure mock to return None (not found)
        mock_db_adapter.find_one.return_value = None

        # Get the agent
        agent = agent_repository.get_human_agent("nonexistent")

        # Verify database was queried correctly
        mock_db_adapter.find_one.assert_called_once_with(
            "human_agents", {"id": "nonexistent"})

        # Verify result
        assert agent is None

    def test_get_human_agents_by_specialization(self, agent_repository, mock_db_adapter, sample_human_agent):
        """Test getting human agents by specialization."""
        # Configure mock to return matching agents
        mock_db_adapter.find.return_value = [sample_human_agent.model_dump()]

        # Get agents by specialization
        agents = agent_repository.get_human_agents_by_specialization("design")

        # Verify database was queried correctly
        mock_db_adapter.find.assert_called_once_with(
            "human_agents",
            {"specializations": "design", "is_active": True}
        )

        # Verify results
        assert len(agents) == 1
        assert agents[0].id == "user123"
        assert "design" in agents[0].specializations

    def test_get_available_human_agents(self, agent_repository, mock_db_adapter, sample_human_agent):
        """Test getting available human agents."""
        # Configure mock to return active agents
        mock_db_adapter.find.return_value = [sample_human_agent.model_dump()]

        # Mock the is_available_now method to always return True
        with patch('solana_agent.domains.HumanAgent.is_available_now', return_value=True):
            # Get available agents
            agents = agent_repository.get_available_human_agents()

            # Verify database was queried correctly
            mock_db_adapter.find.assert_called_once_with(
                "human_agents",
                {"is_active": True}
            )

            # Verify results
            assert len(agents) == 1
            assert agents[0].id == "user123"

    def test_get_available_human_agents_none_available(self, agent_repository, mock_db_adapter, sample_human_agent):
        """Test getting available human agents when none are available."""
        # Configure mock to return active agents
        mock_db_adapter.find.return_value = [sample_human_agent.model_dump()]

        # Mock the is_available_now method to always return False
        with patch('solana_agent.domains.HumanAgent.is_available_now', return_value=False):
            # Get available agents
            agents = agent_repository.get_available_human_agents()

            # Verify database was queried correctly
            mock_db_adapter.find.assert_called_once_with(
                "human_agents",
                {"is_active": True}
            )

            # Verify no agents are available
            assert len(agents) == 0

    def test_save_human_agent_new(self, agent_repository, mock_db_adapter, sample_human_agent):
        """Test saving a new human agent."""
        # Configure mock to return None (agent doesn't exist yet)
        mock_db_adapter.find_one.return_value = None

        # Save the agent
        result = agent_repository.save_human_agent(sample_human_agent)

        # Verify database operations
        mock_db_adapter.find_one.assert_called_once_with(
            "human_agents", {"id": "user123"})
        mock_db_adapter.insert_one.assert_called_once_with(
            "human_agents", sample_human_agent.model_dump())

        # Verify result
        assert result is True

    def test_save_human_agent_update(self, agent_repository, mock_db_adapter, sample_human_agent):
        """Test updating an existing human agent."""
        # Configure mock to return existing agent
        mock_db_adapter.find_one.return_value = {
            "id": "user123", "name": "Old Name"}
        mock_db_adapter.update_one.return_value = True

        # Save the agent
        result = agent_repository.save_human_agent(sample_human_agent)

        # Verify database operations
        mock_db_adapter.find_one.assert_called_once_with(
            "human_agents", {"id": "user123"})
        mock_db_adapter.update_one.assert_called_once_with(
            "human_agents",
            {"id": "user123"},
            {"$set": sample_human_agent.model_dump()}
        )

        # Verify result
        assert result is True

    def test_save_agent_performance_new(self, agent_repository, mock_db_adapter, sample_performance):
        """Test saving new performance metrics."""
        # Configure mock to return None (record doesn't exist yet)
        mock_db_adapter.find_one.return_value = None

        # Save the performance metrics
        result = agent_repository.save_agent_performance(sample_performance)

        # Verify database operations
        mock_db_adapter.find_one.assert_called_once()
        mock_db_adapter.insert_one.assert_called_once()

        # Verify result
        assert result is True

        # Verify result
        assert result is True

    def test_save_agent_performance_update(self, agent_repository, mock_db_adapter, sample_performance):
        """Test updating existing performance metrics."""
        # Configure mock to return existing record
        mock_db_adapter.find_one.return_value = {
            "_id": "perf1",
            "agent_id": sample_performance.agent_id
        }
        mock_db_adapter.update_one.return_value = True

        # Save the performance metrics
        result = agent_repository.save_agent_performance(sample_performance)

        # Verify database operations
        mock_db_adapter.find_one.assert_called_once()
        mock_db_adapter.update_one.assert_called_once()

        # Verify update parameters - modified assertion
        collection, query, update = mock_db_adapter.update_one.call_args[0]
        assert collection == "agent_performance"
        assert query == {"_id": "perf1"}

        # Verify result
        assert result is True

    def test_get_agent_performance(self, agent_repository, mock_db_adapter):
        """Test getting performance metrics."""
        # Sample agent ID
        agent_id = "codebot"
        # Define required dates
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        # Mock performance data with correct structure
        perf_data = {
            "agent_id": "codebot",
            "agent_type": "AI",
            "period_start": start_date.isoformat(),  # Add this field
            "period_end": end_date.isoformat(),     # Add this field
            "tasks_completed": 15,
            "avg_response_time": 3.5,
            "avg_satisfaction_score": 4.8,
            "custom_metrics": {
                "code_quality": 4.7,
                "bug_rate": 0.02
            },
            "last_updated": datetime.now().isoformat(),
            # Default fields with zero values
            "successful_interactions": 0,
            "failed_interactions": 0,
            "total_interactions": 0,
            "handoffs_initiated": 0,
            "handoffs_received": 0,
            "tasks_failed": 0,
            "specialization_matches": 0,
            "specialization_mismatches": 0,
            "total_active_time": "P0DT0H0M0S"
        }

        # Configure mock to return performance data
        mock_db_adapter.find_one.return_value = perf_data

        # Get the performance with required period_start and period_end arguments
        performance = agent_repository.get_agent_performance(
            agent_id, start_date, end_date)

        # Verify database was queried correctly
        mock_db_adapter.find_one.assert_called_once_with(
            "agent_performance",
            {
                "agent_id": "codebot",
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat()
            }
        )

        # Verify result
        assert performance is not None
        assert performance.agent_id == "codebot"
        assert performance.agent_type == "AI"
        assert performance.tasks_completed == 15
        assert performance.custom_metrics["code_quality"] == 4.7

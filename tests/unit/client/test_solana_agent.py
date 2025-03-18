"""
Tests for the SolanaAgent client interface.

This module tests the client API for interacting with the Solana Agent system.
"""
import os
import json
import pytest
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from solana_agent.client.solana_agent import SolanaAgent


class AsyncIteratorMock:
    """Mock class that will be used to return an async iterator."""

    def __init__(self, items):
        self.items = items.copy()  # Create a copy to avoid modifying the original list

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


@pytest.fixture
def mock_agent_factory():
    """Mock the SolanaAgentFactory."""
    with patch('solana_agent.client.solana_agent.SolanaAgentFactory') as mock_factory:
        # Create a mock query service
        mock_query_service = MagicMock()

        # Setup the process_query method as a Mock that returns an async generator
        async def mock_process_query(user_id, message):
            yield "Response chunk 1"
            yield "Response chunk 2"
            yield "Response chunk 3"

        # Create an AsyncMock for process_query that we can assert on
        process_query_mock = AsyncMock()

        # Set up the mock to return our generator when called
        process_query_mock.side_effect = lambda user_id, message: mock_process_query(
            user_id, message)

        # Assign the mock to the query service
        mock_query_service.process_query = process_query_mock

        # Setup the agent service
        mock_query_service.agent_service = MagicMock()
        mock_query_service.agent_service.register_ai_agent = MagicMock()
        mock_query_service.agent_service.register_human_agent = MagicMock()

        # Setup NPS service
        mock_query_service.nps_service = MagicMock()
        mock_query_service.nps_service.process_response = MagicMock(
            return_value=True)
        mock_query_service.nps_service.nps_repository = MagicMock()
        mock_query_service.nps_service.nps_repository.db = MagicMock()
        mock_query_service.nps_service.nps_repository.db.find = MagicMock(return_value=[
            {"id": "survey-1", "user_id": "user-123", "status": "pending"},
            {"id": "survey-2", "user_id": "user-123", "status": "pending"}
        ])

        # Setup memory service
        mock_query_service.memory_service = MagicMock()
        mock_query_service.memory_service.get_paginated_history = AsyncMock(return_value={
            "data": [{"message": "Hello"}, {"message": "World"}],
            "total": 2,
            "page": 1,
            "page_size": 20,
            "total_pages": 1
        })

        # Configure the factory to return our mock query service
        mock_factory.create_from_config.return_value = mock_query_service

        yield mock_factory


@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        "llm_provider": "openai",
        "llm_models": {
            "default": "gpt-4o-mini"
        },
        "services": {
            "agent": True,
            "query": True,
            "memory": True,
            "nps": True
        }
    }


@pytest.fixture
def json_config_path():
    """Create a temporary JSON configuration file."""
    config = {
        "llm_provider": "openai",
        "llm_models": {
            "default": "gpt-4o-mini"
        },
        "services": {
            "agent": True,
            "query": True,
            "memory": True,
            "nps": True
        }
    }

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
        temp.write(json.dumps(config).encode('utf-8'))
        temp_path = temp.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def python_config_path():
    """Create a temporary Python configuration file."""
    config_content = """
config = {
    "llm_provider": "openai",
    "llm_models": {
        "default": "gpt-4o-mini"
    },
    "services": {
        "agent": True,
        "query": True,
        "memory": True,
        "nps": True
    }
}
"""

    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
        temp.write(config_content.encode('utf-8'))
        temp_path = temp.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


# --------------------------
# Initialization Tests
# --------------------------

def test_init_with_config_dict(mock_agent_factory, sample_config):
    """Test initialization with a config dictionary."""
    agent = SolanaAgent(config=sample_config)

    # Check that the factory was called with the config
    mock_agent_factory.create_from_config.assert_called_once_with(
        sample_config)

    # Check that the query service was set
    assert agent.query_service == mock_agent_factory.create_from_config.return_value


def test_init_with_json_config_path(mock_agent_factory, json_config_path):
    """Test initialization with a JSON config file path."""
    agent = SolanaAgent(config_path=json_config_path)

    # Check that the factory was called
    mock_agent_factory.create_from_config.assert_called_once()

    # Check that the query service was set
    assert agent.query_service == mock_agent_factory.create_from_config.return_value


def test_init_with_python_config_path(mock_agent_factory, python_config_path):
    """Test initialization with a Python config file path."""
    agent = SolanaAgent(config_path=python_config_path)

    # Check that the factory was called
    mock_agent_factory.create_from_config.assert_called_once()

    # Check that the query service was set
    assert agent.query_service == mock_agent_factory.create_from_config.return_value


def test_init_no_config():
    """Test initialization with no config raises an error."""
    with pytest.raises(ValueError, match="Either config or config_path must be provided"):
        SolanaAgent()


# --------------------------
# Process Method Tests
# --------------------------

@pytest.mark.asyncio
async def test_process(mock_agent_factory, sample_config):
    """Test processing a user message."""
    agent = SolanaAgent(config=sample_config)

    # Create response chunks
    response_chunks = ["Response chunk 1",
                       "Response chunk 2", "Response chunk 3"]

    # Create a mock that returns an async iterator
    process_mock = MagicMock()
    process_mock.return_value = AsyncIteratorMock(response_chunks)

    # Patch the process method directly on the query service
    with patch.object(agent.query_service, 'process', process_mock):
        # Collect the response chunks
        chunks = []
        async for chunk in agent.process("user-123", "Hello, Solana!"):
            chunks.append(chunk)

        # Check that process was called with the right arguments
        process_mock.assert_called_once_with("user-123", "Hello, Solana!")

        # Check that all chunks were collected
        assert chunks == response_chunks


# --------------------------
# Agent Registration Tests
# --------------------------

def test_register_agent(mock_agent_factory, sample_config):
    """Test registering an AI agent."""
    agent = SolanaAgent(config=sample_config)

    # Register an AI agent
    agent.register_agent(
        name="TestBot",
        instructions="You are a helpful assistant.",
        specialization="general",
        model="gpt-4o"
    )

    # Check that register_ai_agent was called with the right arguments
    agent.query_service.agent_service.register_ai_agent.assert_called_once_with(
        name="TestBot",
        instructions="You are a helpful assistant.",
        specialization="general",
        model="gpt-4o"
    )


def test_register_human_agent(mock_agent_factory, sample_config):
    """Test registering a human agent."""
    agent = SolanaAgent(config=sample_config)

    # Create a mock notification handler
    mock_handler = MagicMock()

    # Register a human agent
    agent.register_human_agent(
        agent_id="human-1",
        name="John Doe",
        specialization="support",
        notification_handler=mock_handler
    )

    # Check that register_human_agent was called with the right arguments
    agent.query_service.agent_service.register_human_agent.assert_called_once_with(
        agent_id="human-1",
        name="John Doe",
        specialization="support",
        notification_handler=mock_handler
    )


# --------------------------
# Survey Tests
# --------------------------

@pytest.mark.asyncio
async def test_get_pending_surveys(mock_agent_factory, sample_config):
    """Test getting pending surveys."""
    agent = SolanaAgent(config=sample_config)

    # Get pending surveys
    surveys = await agent.get_pending_surveys("user-123")

    # Check that find was called with the right query
    agent.query_service.nps_service.nps_repository.db.find.assert_called_once()

    # Check the query parameters
    call_args = agent.query_service.nps_service.nps_repository.db.find.call_args[0]
    assert call_args[0] == "nps_surveys"
    assert call_args[1]["user_id"] == "user-123"
    assert call_args[1]["status"] == "pending"
    assert "$gte" in call_args[1]["created_at"]

    # Check that the surveys were returned
    assert len(surveys) == 2
    assert surveys[0]["id"] == "survey-1"
    assert surveys[1]["id"] == "survey-2"


@pytest.mark.asyncio
async def test_get_pending_surveys_no_nps_service(mock_agent_factory, sample_config):
    """Test getting pending surveys when NPS service is not available."""
    agent = SolanaAgent(config=sample_config)

    # Remove the NPS service
    delattr(agent.query_service, "nps_service")

    # Get pending surveys should return empty list
    surveys = await agent.get_pending_surveys("user-123")

    # Check that the surveys are empty
    assert surveys == []


@pytest.mark.asyncio
async def test_submit_survey_response(mock_agent_factory, sample_config):
    """Test submitting a survey response."""
    agent = SolanaAgent(config=sample_config)

    # Submit a survey response
    result = await agent.submit_survey_response("survey-1", 9, "Great service!")

    # Check that process_response was called with the right arguments
    agent.query_service.nps_service.process_response.assert_called_once_with(
        "survey-1", 9, "Great service!"
    )

    # Check that the result was returned
    assert result is True


@pytest.mark.asyncio
async def test_submit_survey_response_no_nps_service(mock_agent_factory, sample_config):
    """Test submitting a survey response when NPS service is not available."""
    agent = SolanaAgent(config=sample_config)

    # Remove the NPS service
    delattr(agent.query_service, "nps_service")

    # Submit a survey response should return False
    result = await agent.submit_survey_response("survey-1", 9, "Great service!")

    # Check that the result is False
    assert result is False


# --------------------------
# History Retrieval Tests
# --------------------------

@pytest.mark.asyncio
async def test_get_paginated_history(mock_agent_factory, sample_config):
    """Test getting paginated history."""
    agent = SolanaAgent(config=sample_config)

    # Get paginated history
    history = await agent.get_paginated_history(
        "user-123",
        page_num=2,
        page_size=10,
        sort_order="desc"
    )

    # Check that get_paginated_history was called with the right arguments
    agent.query_service.memory_service.get_paginated_history.assert_called_once_with(
        "user-123", 2, 10, "desc"
    )

    # Check that the history was returned
    assert history == {
        "data": [{"message": "Hello"}, {"message": "World"}],
        "total": 2,
        "page": 1,
        "page_size": 20,
        "total_pages": 1
    }


@pytest.mark.asyncio
async def test_get_paginated_history_no_memory_service(mock_agent_factory, sample_config):
    """Test getting paginated history when memory service is not available."""
    agent = SolanaAgent(config=sample_config)

    # Remove the memory service
    delattr(agent.query_service, "memory_service")

    # Get paginated history should return an error message
    history = await agent.get_paginated_history("user-123")

    # Check that the history contains an error
    assert "error" in history
    assert history["error"] == "Memory service not available"
    assert history["data"] == []
    assert history["total"] == 0

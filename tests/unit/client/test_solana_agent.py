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

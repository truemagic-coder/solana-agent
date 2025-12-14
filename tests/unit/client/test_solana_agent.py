"""
Tests for the SolanaAgent client interface.

This module provides comprehensive test coverage for the SolanaAgent client
including initialization, message processing, history management, and tool registration.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from solana_agent.client.solana_agent import SolanaAgent
from solana_agent.interfaces.plugins.plugins import Tool


@pytest.fixture
def config_dict():
    """Fixture providing test configuration."""
    return {
        "mongo": {
            "connection_string": "mongodb://localhost:27017",
            "database": "test_db",
        },
        "openai": {"api_key": "test_key"},
        "agents": [
            {
                "name": "test_agent",
                "instructions": "Test agent instructions",
                "specialization": "Testing",
            }
        ],
    }


@pytest.fixture
def mock_query_service():
    """Create a mock query service with all required methods."""
    mock = AsyncMock()

    async def mock_process(*args, **kwargs):
        yield "Test response"

    mock.process.side_effect = mock_process
    mock.delete_user_history = AsyncMock()
    mock.get_user_history = AsyncMock(return_value={"messages": [], "total": 0})

    # Configure agent service
    mock.agent_service = MagicMock()
    mock.agent_service.tool_registry = MagicMock()
    mock.agent_service.get_all_ai_agents = MagicMock(return_value=["test_agent"])
    mock.agent_service.assign_tool_for_agent = MagicMock()

    return mock


class TestSolanaAgent:
    """Test suite for SolanaAgent client."""

    @patch("solana_agent.client.solana_agent.SolanaAgentFactory")
    def test_init_with_config(self, mock_factory, config_dict, mock_query_service):
        """Test initialization with configuration dictionary."""
        mock_factory.create_from_config.return_value = mock_query_service
        agent = SolanaAgent(config=config_dict)
        mock_factory.create_from_config.assert_called_once_with(config_dict)
        assert agent.query_service == mock_query_service

    def test_init_without_config(self):
        """Test initialization fails without configuration."""
        with pytest.raises(
            ValueError, match="Either config or config_path must be provided"
        ):
            SolanaAgent()

    @pytest.mark.asyncio
    async def test_get_user_history(self, config_dict, mock_query_service):
        """Test retrieving user message history."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {"messages": [], "total": 0}
            mock_query_service.get_user_history.return_value = expected

            result = await agent.get_user_history(
                user_id="test_user", page_num=1, page_size=20, sort_order="desc"
            )

            assert result == expected
            mock_query_service.get_user_history.assert_called_once_with(
                "test_user", 1, 20, "desc"
            )

    @pytest.mark.asyncio
    async def test_delete_user_history(self, config_dict, mock_query_service):
        """Test deleting user message history."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            await agent.delete_user_history("test_user")
            mock_query_service.delete_user_history.assert_called_once_with("test_user")

    def test_register_tool_success(self, config_dict, mock_query_service):
        """Test successful tool registration."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            # Setup mock tool and registry
            mock_tool = MagicMock(spec=Tool)
            mock_tool.name = "test_tool"
            mock_query_service.agent_service.tool_registry.register_tool.return_value = True

            # Test registration
            result = agent.register_tool("test_agent", mock_tool)

            # Verify results
            assert result is True
            mock_query_service.agent_service.tool_registry.register_tool.assert_called_once_with(
                mock_tool
            )
            mock_query_service.agent_service.assign_tool_for_agent.assert_called_once_with(
                "test_agent", "test_tool"
            )

    def test_register_tool_failure(self, config_dict, mock_query_service):
        """Test failed tool registration."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            # Setup mock tool and registry
            mock_tool = MagicMock(spec=Tool)
            mock_tool.name = "test_tool"
            mock_query_service.agent_service.tool_registry.register_tool.return_value = False

            # Test registration
            result = agent.register_tool("test_agent", mock_tool)

            # Verify results
            assert result is False
            mock_query_service.agent_service.tool_registry.register_tool.assert_called_once_with(
                mock_tool
            )
            # Verify assign_tool_for_agent was not called
            mock_query_service.agent_service.assign_tool_for_agent.assert_not_called()

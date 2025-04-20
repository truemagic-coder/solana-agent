"""
Tests for the ToolRegistry implementation.

This module provides comprehensive test coverage for tool registration,
agent permissions, and tool configuration management.
"""

import pytest
from unittest.mock import MagicMock

from solana_agent.plugins.registry import ToolRegistry
from solana_agent.interfaces.plugins.plugins import Tool


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    tool = MagicMock(spec=Tool)
    tool.name = "test_tool"
    tool.description = "Test tool description"
    tool.get_schema.return_value = {
        "type": "object",
        "properties": {"param1": {"type": "string"}},
    }
    return tool


@pytest.fixture
def config():
    """Sample configuration for testing."""
    return {"api_key": "test_key", "endpoint": "https://api.test.com"}


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry."""
    registry = MagicMock(spec=ToolRegistry)
    registry.get_tool = MagicMock(return_value=None)
    # Make configure_all_tools a MagicMock to support assert_called_once_with
    registry.configure_all_tools = MagicMock()
    return registry


class TestToolRegistry:
    """Test suite for ToolRegistry."""

    def test_init_default(self):
        """Test initialization with default values."""
        registry = ToolRegistry()
        assert registry._tools == {}
        assert registry._agent_tools == {}
        assert registry._config == {}

    def test_init_with_config(self, config):
        """Test initialization with configuration."""
        registry = ToolRegistry(config)
        assert registry._config == config

    def test_register_tool_success(self, mock_tool, config):
        """Test successful tool registration."""
        registry = ToolRegistry(config)
        success = registry.register_tool(mock_tool)

        assert success is True
        assert registry._tools[mock_tool.name] == mock_tool
        mock_tool.configure.assert_called_once_with(config)

    def test_register_tool_failure(self, mock_tool):
        """Test tool registration failure."""
        mock_tool.configure.side_effect = Exception("Config failed")
        registry = ToolRegistry()

        success = registry.register_tool(mock_tool)
        assert success is False
        assert mock_tool.name not in registry._tools

    def test_get_tool_existing(self, mock_tool):
        """Test retrieving an existing tool."""
        registry = ToolRegistry()
        registry._tools[mock_tool.name] = mock_tool

        tool = registry.get_tool(mock_tool.name)
        assert tool == mock_tool

    def test_get_tool_non_existing(self):
        """Test retrieving a non-existing tool."""
        registry = ToolRegistry()
        tool = registry.get_tool("non_existing")
        assert tool is None

    def test_assign_tool_to_agent_new_agent(self, mock_tool):
        """Test assigning tool to new agent."""
        registry = ToolRegistry()
        registry._tools[mock_tool.name] = mock_tool

        success = registry.assign_tool_to_agent("test_agent", mock_tool.name)
        assert success is True
        assert registry._agent_tools["test_agent"] == [mock_tool.name]

    def test_assign_tool_to_existing_agent(self, mock_tool):
        """Test assigning additional tool to existing agent."""
        registry = ToolRegistry()
        registry._tools[mock_tool.name] = mock_tool
        registry._agent_tools["test_agent"] = ["existing_tool"]

        success = registry.assign_tool_to_agent("test_agent", mock_tool.name)
        assert success is True
        assert set(registry._agent_tools["test_agent"]) == {
            "existing_tool",
            mock_tool.name,
        }

    def test_assign_tool_already_assigned(self, mock_tool):
        """Test assigning already assigned tool."""
        registry = ToolRegistry()
        registry._tools[mock_tool.name] = mock_tool
        registry._agent_tools["test_agent"] = [mock_tool.name]

        success = registry.assign_tool_to_agent("test_agent", mock_tool.name)
        assert success is True
        assert registry._agent_tools["test_agent"] == [mock_tool.name]

    def test_assign_non_existing_tool(self):
        """Test assigning non-existing tool."""
        registry = ToolRegistry()
        success = registry.assign_tool_to_agent("test_agent", "non_existing")
        assert success is False
        assert "test_agent" not in registry._agent_tools

    def test_get_agent_tools_existing(self, mock_tool):
        """Test getting tools for existing agent."""
        registry = ToolRegistry()
        registry._tools[mock_tool.name] = mock_tool
        registry._agent_tools["test_agent"] = [mock_tool.name]

        tools = registry.get_agent_tools("test_agent")
        assert len(tools) == 1
        assert tools[0]["name"] == mock_tool.name
        assert tools[0]["description"] == mock_tool.description
        assert tools[0]["parameters"] == mock_tool.get_schema.return_value

    def test_get_agent_tools_non_existing(self):
        """Test getting tools for non-existing agent."""
        registry = ToolRegistry()
        tools = registry.get_agent_tools("non_existing")
        assert tools == []

    def test_get_agent_tools_with_missing_tool(self, mock_tool):
        """Test getting agent tools when some tools are missing."""
        registry = ToolRegistry()
        registry._tools[mock_tool.name] = mock_tool
        registry._agent_tools["test_agent"] = [mock_tool.name, "missing_tool"]

        tools = registry.get_agent_tools("test_agent")
        assert len(tools) == 1
        assert tools[0]["name"] == mock_tool.name

    def test_list_all_tools(self, mock_tool):
        """Test listing all registered tools."""
        registry = ToolRegistry()
        registry._tools[mock_tool.name] = mock_tool

        tools = registry.list_all_tools()
        assert tools == [mock_tool.name]

    def test_configure_all_tools(self, mock_tool, config):
        """Test configuring all tools."""
        registry = ToolRegistry()
        registry._tools[mock_tool.name] = mock_tool

        registry.configure_all_tools(config)
        mock_tool.configure.assert_called_once_with(config)

    def test_configure_all_tools_with_error(self, mock_tool, config):
        """Test configuring tools when one fails."""
        registry = ToolRegistry()
        mock_tool2 = MagicMock(spec=Tool)
        mock_tool2.name = "test_tool2"
        mock_tool2.configure.side_effect = Exception("Config failed")

        registry._tools = {mock_tool.name: mock_tool, mock_tool2.name: mock_tool2}

        # Should not raise exception
        registry.configure_all_tools(config)

        # Verify both tools were attempted
        mock_tool.configure.assert_called_once_with(config)
        mock_tool2.configure.assert_called_once_with(config)
        # Verify config was updated
        assert registry._config == config

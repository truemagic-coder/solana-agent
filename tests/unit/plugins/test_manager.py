"""
Tests for the PluginManager implementation.

This module provides comprehensive test coverage for plugin management,
including plugin loading, registration, configuration, and tool execution.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from solana_agent.plugins.manager import PluginManager
from solana_agent.interfaces.plugins.plugins import Plugin
from solana_agent.plugins.registry import ToolRegistry


@pytest.fixture
def mock_plugin():
    """Create a mock plugin."""
    plugin = MagicMock(spec=Plugin)
    plugin.name = "test_plugin"
    plugin.description = "Test plugin description"
    plugin.initialize = MagicMock()
    plugin.configure = MagicMock()
    return plugin


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry with proper mock methods."""
    registry = MagicMock(spec=ToolRegistry)
    registry.get_tool = MagicMock(return_value=None)
    # Ensure configure_all_tools is a MagicMock
    registry.configure_all_tools = MagicMock()
    return registry


@pytest.fixture
def config():
    """Sample configuration for testing."""
    return {"plugin_config": {"key": "value"}}


class TestPluginManager:
    """Test suite for PluginManager."""

    def test_init_default(self):
        """Test initialization with default values."""
        manager = PluginManager()
        assert isinstance(manager.tool_registry, ToolRegistry)
        assert manager.config == {}
        assert manager._plugins == {}

    def test_init_with_config_and_registry(self, config, mock_tool_registry):
        """Test initialization with config and registry."""
        manager = PluginManager(config=config, tool_registry=mock_tool_registry)
        assert manager.config == config
        assert manager.tool_registry == mock_tool_registry

    def test_register_plugin_success(self, mock_plugin, mock_tool_registry):
        """Test successful plugin registration."""
        manager = PluginManager(tool_registry=mock_tool_registry)
        success = manager.register_plugin(mock_plugin)

        assert success is True
        assert manager._plugins[mock_plugin.name] == mock_plugin
        mock_plugin.initialize.assert_called_once_with(mock_tool_registry)
        mock_plugin.configure.assert_called_once_with(manager.config)

    def test_register_plugin_failure_initialize(self, mock_plugin, mock_tool_registry):
        """Test plugin registration failure during initialization."""
        mock_plugin.initialize.side_effect = Exception("Init failed")
        manager = PluginManager(tool_registry=mock_tool_registry)

        success = manager.register_plugin(mock_plugin)
        assert success is False
        assert mock_plugin.name not in manager._plugins

    def test_register_plugin_failure_configure(self, mock_plugin, mock_tool_registry):
        """Test plugin registration failure during configuration."""
        mock_plugin.configure.side_effect = Exception("Config failed")
        manager = PluginManager(tool_registry=mock_tool_registry)

        success = manager.register_plugin(mock_plugin)
        assert success is False
        assert mock_plugin.name not in manager._plugins

    @patch("importlib.metadata.entry_points")
    def test_load_plugins_success(self, mock_entry_points, mock_plugin):
        """Test successful plugin loading from entry points."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "test_plugin"
        mock_entry_point.value = "test.plugin:factory"
        mock_entry_point.load.return_value = lambda: mock_plugin

        mock_entry_points.return_value = [mock_entry_point]

        manager = PluginManager()
        loaded = manager.load_plugins()

        assert loaded == ["test_plugin"]
        assert manager._plugins[mock_plugin.name] == mock_plugin

    @patch("importlib.metadata.entry_points")
    def test_load_plugins_skip_duplicate(self, mock_entry_points, mock_plugin):
        """Test skipping already loaded plugins."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "test_plugin"
        mock_entry_point.value = "test.plugin:factory"

        mock_entry_points.return_value = [mock_entry_point]

        manager = PluginManager()
        # Load plugins twice
        manager.load_plugins()
        loaded = manager.load_plugins()

        assert loaded == []  # Second load should skip

    @patch("importlib.metadata.entry_points")
    def test_load_plugins_entry_point_error(self, mock_entry_points):
        """Test handling entry point loading errors."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "test_plugin"
        mock_entry_point.load.side_effect = Exception("Load failed")

        mock_entry_points.return_value = [mock_entry_point]

        manager = PluginManager()
        loaded = manager.load_plugins()

        assert loaded == []
        assert len(manager._plugins) == 0

    def test_get_plugin_existing(self, mock_plugin):
        """Test retrieving an existing plugin."""
        manager = PluginManager()
        manager._plugins[mock_plugin.name] = mock_plugin

        plugin = manager.get_plugin(mock_plugin.name)
        assert plugin == mock_plugin

    def test_get_plugin_non_existing(self):
        """Test retrieving a non-existing plugin."""
        manager = PluginManager()
        plugin = manager.get_plugin("non_existing")
        assert plugin is None

    def test_list_plugins(self, mock_plugin):
        """Test listing registered plugins."""
        manager = PluginManager()
        manager._plugins[mock_plugin.name] = mock_plugin

        plugins = manager.list_plugins()
        assert len(plugins) == 1
        assert plugins[0]["name"] == mock_plugin.name
        assert plugins[0]["description"] == mock_plugin.description

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test executing a non-existing tool."""
        manager = PluginManager()
        result = await manager.execute_tool("non_existing")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_tool_registry):
        """Test successful tool execution."""
        mock_tool = AsyncMock()
        mock_tool.execute.return_value = {"status": "success", "result": "test"}
        mock_tool_registry.get_tool.return_value = mock_tool

        manager = PluginManager(tool_registry=mock_tool_registry)
        result = await manager.execute_tool("test_tool", param="value")

        assert result["status"] == "success"
        assert result["result"] == "test"
        mock_tool.execute.assert_called_once_with(param="value")

    @pytest.mark.asyncio
    async def test_execute_tool_error(self, mock_tool_registry):
        """Test tool execution error handling."""
        mock_tool = AsyncMock()
        mock_tool.execute.side_effect = Exception("Execution failed")
        mock_tool_registry.get_tool.return_value = mock_tool

        manager = PluginManager(tool_registry=mock_tool_registry)
        result = await manager.execute_tool("test_tool")

        assert result["status"] == "error"
        assert "Execution failed" in result["message"]

    def test_configure(self, mock_plugin, mock_tool_registry, config):
        """Test configuring manager and plugins."""
        # Initialize manager with mock registry
        manager = PluginManager(tool_registry=mock_tool_registry)
        manager._plugins[mock_plugin.name] = mock_plugin

        # Perform configuration
        manager.configure(config)

        # Verify configuration was applied correctly
        assert manager.config == config
        mock_tool_registry.configure_all_tools.assert_called_once_with(config)
        mock_plugin.configure.assert_called_once_with(config)

    def test_configure_plugin_error(self, mock_plugin, config):
        """Test handling plugin configuration errors."""
        mock_plugin.configure.side_effect = Exception("Config failed")

        manager = PluginManager()
        manager._plugins[mock_plugin.name] = mock_plugin

        # Should not raise exception
        manager.configure(config)

        assert manager.config == config
        mock_plugin.configure.assert_called_once_with(config)

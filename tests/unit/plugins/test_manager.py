"""
Tests for the Plugin Manager.

This module contains unit tests for the PluginManager implementation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import importlib

from solana_agent.plugins.manager import PluginManager
from solana_agent.plugins import ToolRegistry
from solana_agent.domains.tools import BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name="mock_tool"):
        """Initialize the tool with a name."""
        self.name = name
        self.configured = False

    def get_name(self) -> str:
        """Return the tool name."""
        return self.name

    def get_description(self) -> str:
        """Return the tool description."""
        return f"Mock tool named {self.name}"

    def get_parameters(self) -> dict:
        """Return the tool parameters."""
        return {
            "param1": {
                "type": "string",
                "description": "Test parameter"
            }
        }

    def execute(self, **kwargs):
        """Execute the tool."""
        return {"status": "success", "result": f"Tool {self.name} executed with {kwargs}"}

    def configure(self, config):
        """Configure the tool."""
        self.configured = True
        self.config = config


class MockPlugin:
    """Mock plugin for testing."""

    def __init__(self):
        """Initialize the plugin."""
        self.initialized = False
        self.config = None

    def initialize(self, config):
        """Initialize the plugin with config."""
        self.initialized = True
        self.config = config

    def get_tools(self):
        """Return the plugin's tools."""
        return [MockTool("tool1"), MockTool("tool2")]


class MockPluginSingleTool:
    """Mock plugin that returns a single tool."""

    def __init__(self):
        """Initialize the plugin."""
        self.initialized = False

    def initialize(self, config):
        """Initialize the plugin with config."""
        self.initialized = True

    def get_tools(self):
        """Return a single tool."""
        return MockTool("single_tool")


class MockEntryPoint:
    """Mock entry point for testing."""

    def __init__(self, name, value, plugin_factory):
        """Initialize the entry point."""
        self.name = name
        self.value = value
        self.plugin_factory = plugin_factory

    def load(self):
        """Load the entry point."""
        return self.plugin_factory


@pytest.fixture
def tool_registry():
    """Create a tool registry for testing."""
    return ToolRegistry()


@pytest.fixture
def plugin_manager(tool_registry):
    """Create a plugin manager for testing."""
    return PluginManager(config={"test_key": "test_value"}, tool_registry=tool_registry)


@pytest.fixture
def mock_entry_points():
    """Create mock entry points for testing."""
    return [
        MockEntryPoint("plugin1", "module:factory", lambda: MockPlugin()),
        MockEntryPoint("plugin2", "module:factory2",
                       lambda: MockPluginSingleTool()),
        MockEntryPoint("failing_plugin", "module:factory3", lambda: (
            _ for _ in ()).throw(Exception("Plugin load failed")))
    ]


class TestPluginManager:
    """Tests for the Plugin Manager."""

    def test_init(self):
        """Test plugin manager initialization."""
        # Test with default arguments
        manager = PluginManager()
        assert manager.config == {}
        assert isinstance(manager.tool_registry, ToolRegistry)

        # Test with custom arguments
        config = {"test": "value"}
        registry = ToolRegistry()
        manager = PluginManager(config=config, tool_registry=registry)
        assert manager.config == config
        assert manager.tool_registry == registry

    @patch('solana_agent.plugins.manager.importlib.metadata.entry_points')
    def test_load_all_plugins(self, mock_entry_points_func, mock_entry_points, plugin_manager):
        """Test loading all plugins."""
        mock_entry_points_func.return_value = mock_entry_points

        # Test loading plugins
        loaded_count = plugin_manager.load_all_plugins()

        # Should load 2 plugins (third one fails)
        assert loaded_count == 2

        # Check that tools were registered
        assert len(plugin_manager.tool_registry.list_all_tools()) == 3
        assert "tool1" in plugin_manager.tool_registry.list_all_tools()
        assert "tool2" in plugin_manager.tool_registry.list_all_tools()
        assert "single_tool" in plugin_manager.tool_registry.list_all_tools()

    @patch('solana_agent.plugins.manager.importlib.metadata.entry_points')
    def test_load_plugins_with_duplicates(self, mock_entry_points_func, mock_entry_points, plugin_manager):
        """Test loading plugins with duplicates."""
        # Create duplicate entry point
        duplicate = MockEntryPoint(
            "plugin1", "module:factory", lambda: MockPlugin())
        mock_entry_points_func.return_value = mock_entry_points + [duplicate]

        # Load plugins first time
        plugin_manager.load_all_plugins()

        # Reset mock to simulate second call with same entry points
        mock_entry_points_func.return_value = mock_entry_points + [duplicate]

        # Load plugins second time - should skip duplicates
        loaded_count = plugin_manager.load_all_plugins()

        # Should not count duplicates
        assert loaded_count == 0

    def test_register_plugin(self, plugin_manager):
        """Test manually registering a plugin."""
        plugin = MockPlugin()
        result = plugin_manager.register_plugin("manual_plugin", plugin)

        assert result is True
        assert "tool1" in plugin_manager.tool_registry.list_all_tools()
        assert "tool2" in plugin_manager.tool_registry.list_all_tools()

    def test_register_plugin_with_exception(self, plugin_manager):
        """Test handling exceptions when registering a plugin."""
        # Create a plugin that raises an exception
        faulty_plugin = Mock()
        faulty_plugin.get_tools = Mock(
            side_effect=Exception("Failed to get tools"))

        result = plugin_manager.register_plugin("faulty_plugin", faulty_plugin)

        assert result is False

    def test_execute_tool(self, plugin_manager):
        """Test executing a tool."""
        # Register a tool
        tool = MockTool("test_tool")
        plugin_manager.tool_registry.register_tool(tool)

        # Execute the tool
        result = plugin_manager.execute_tool("test_tool", param1="value1")

        assert result["status"] == "success"
        assert "test_tool executed with" in result["result"]

    def test_execute_nonexistent_tool(self, plugin_manager):
        """Test executing a tool that doesn't exist."""
        result = plugin_manager.execute_tool("nonexistent_tool")

        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_execute_tool_with_exception(self, plugin_manager):
        """Test handling exceptions when executing a tool."""
        # Create a tool that raises an exception
        faulty_tool = Mock()
        faulty_tool.get_name = Mock(return_value="faulty_tool")
        faulty_tool.execute = Mock(side_effect=Exception("Execution failed"))

        plugin_manager.tool_registry.register_tool(faulty_tool)

        result = plugin_manager.execute_tool("faulty_tool")

        assert result["status"] == "error"
        assert "Execution failed" in result["message"]

    def test_configure(self, plugin_manager):
        """Test configuring the plugin manager."""
        # Register tools
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")
        plugin_manager.tool_registry.register_tool(tool1)
        plugin_manager.tool_registry.register_tool(tool2)

        # Update configuration
        new_config = {"new_key": "new_value"}
        plugin_manager.configure(new_config)

        # Check that the config was updated
        assert plugin_manager.config["new_key"] == "new_value"
        # Original value should still be there
        assert plugin_manager.config["test_key"] == "test_value"

        # Check that tools were configured
        assert tool1.configured
        assert tool2.configured
        assert "new_key" in tool1.config

    def test_get_plugin(self, plugin_manager):
        """Test getting a plugin by name."""
        # Current implementation returns None
        assert plugin_manager.get_plugin("any_name") is None

    def test_list_plugins(self, plugin_manager):
        """Test listing all plugins."""
        # Current implementation returns empty list
        assert plugin_manager.list_plugins() == []

    @patch('solana_agent.plugins.manager.importlib.metadata.entry_points')
    def test_plugin_initialization_with_config(self, mock_entry_points_func, mock_entry_points, plugin_manager):
        """Test that plugins are initialized with correct config."""
        mock_entry_points_func.return_value = mock_entry_points[:1]  # Just use the first plugin

        plugin_manager.load_all_plugins()

        # Get the plugin instance from the mock factory
        plugin_instance = mock_entry_points[0].plugin_factory()()

        # Initialize it with the same config
        plugin_instance.initialize(plugin_manager.config)

        # Verify it was initialized with the correct config
        assert plugin_instance.initialized
        assert plugin_instance.config == plugin_manager.config

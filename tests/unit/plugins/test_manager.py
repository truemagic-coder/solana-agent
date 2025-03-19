"""
Tests for the Plugin Manager.

This module contains unit tests for the PluginManager implementation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import importlib

from solana_agent.plugins.manager import PluginManager
from solana_agent.plugins import ToolRegistry
from solana_agent.interfaces import Tool, Plugin


class MockTool(Tool):
    """Mock tool for testing the Tool interface."""

    def __init__(self, name_value="mock_tool", description_value="Mock tool description"):
        """Initialize the tool with a name."""
        self._name = name_value
        self._description = description_value
        self.configured = False
        self.config = {}

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return self._name

    @property
    def description(self) -> str:
        """Get the description of the tool."""
        return self._description

    def configure(self, config):
        """Configure the tool."""
        self.configured = True
        self.config = config

    def get_schema(self) -> dict:
        """Get the schema for the tool parameters."""
        return {
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Test parameter"
                }
            },
            "required": ["param1"]
        }

    def execute(self, **kwargs):
        """Execute the tool with the given parameters."""
        return {"status": "success", "result": f"Tool {self.name} executed with {kwargs}"}


class MockPlugin(Plugin):
    """Mock plugin for testing the Plugin interface."""

    def __init__(self, name_value="mock_plugin", description_value="Mock plugin description"):
        """Initialize the plugin."""
        self._name = name_value
        self._description = description_value
        self.initialized = False
        self._tools = [MockTool("tool1"), MockTool("tool2")]

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return self._name

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return self._description

    def initialize(self, tool_registry):
        """Initialize the plugin and register its tools."""
        self.initialized = True
        self.registry = tool_registry

        # Register all tools with the registry
        for tool in self._tools:
            tool_registry.register_tool(tool)

        return True


class MockSingleToolPlugin(Plugin):
    """Mock plugin that provides a single tool."""

    def __init__(self):
        """Initialize the plugin."""
        self._name = "single_tool_plugin"
        self._description = "A plugin with a single tool"
        self.initialized = False
        self._tool = MockTool("single_tool")

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return self._name

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return self._description

    def initialize(self, tool_registry):
        """Initialize the plugin and register its tool."""
        self.initialized = True
        self.registry = tool_registry
        tool_registry.register_tool(self._tool)
        return True


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
    registry = ToolRegistry()
    return registry


@pytest.fixture
def plugin_manager(tool_registry):
    """Create a plugin manager for testing."""
    return PluginManager(config={"test_key": "test_value"}, tool_registry=tool_registry)


@pytest.fixture
def mock_entry_points():
    """Create mock entry points for testing."""
    return [
        MockEntryPoint("mock_plugin", "module:factory", lambda: MockPlugin()),
        MockEntryPoint("single_tool_plugin", "module:factory2",
                       lambda: MockSingleToolPlugin()),
        MockEntryPoint("failing_plugin", "module:factory3", lambda: (
            _ for _ in ()).throw(Exception("Plugin load failed")))
    ]


class TestPluginManager:
    """Tests for the Plugin Manager."""

    # Add this fixture to the TestPluginManager class
    @pytest.fixture(autouse=True)
    def reset_plugin_manager_state(self):
        """Reset the PluginManager's class variables before each test."""
        PluginManager._loaded_entry_points = set()
        # No need to reset _plugins since it's now an instance variable

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
    def test_load_plugins(self, mock_entry_points_func, mock_entry_points, plugin_manager):
        """Test loading all plugins."""
        mock_entry_points_func.return_value = mock_entry_points

        # Test loading plugins - now using load_plugins instead of load_all_plugins
        loaded_plugins = plugin_manager.load_plugins()

        # Should load 2 plugins (third one fails)
        assert len(loaded_plugins) == 2
        assert "mock_plugin" in loaded_plugins
        assert "single_tool_plugin" in loaded_plugins

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
        plugin_manager.load_plugins()

        # Reset mock to simulate second call with same entry points
        mock_entry_points_func.return_value = mock_entry_points + [duplicate]

        # Load plugins second time - should skip duplicates
        loaded_plugins = plugin_manager.load_plugins()

        # Should only include non-duplicated plugins
        assert len(loaded_plugins) == 0

    def test_register_plugin(self, plugin_manager):
        """Test manually registering a plugin."""
        plugin = MockPlugin()
        result = plugin_manager.register_plugin(plugin)

        assert result is True
        assert plugin.initialized
        assert "tool1" in plugin_manager.tool_registry.list_all_tools()
        assert "tool2" in plugin_manager.tool_registry.list_all_tools()

    def test_register_plugin_with_exception(self, plugin_manager):
        """Test handling exceptions when registering a plugin."""
        # Create a plugin that raises an exception during initialization
        faulty_plugin = Mock(spec=Plugin)
        faulty_plugin.name = "faulty_plugin"
        faulty_plugin.description = "A plugin that fails to initialize"
        faulty_plugin.initialize.side_effect = Exception(
            "Failed to initialize")

        result = plugin_manager.register_plugin(faulty_plugin)

        assert result is False

    def test_get_plugin(self, plugin_manager):
        """Test getting a plugin by name."""
        # Register a plugin
        plugin = MockPlugin(name_value="test_plugin")
        plugin_manager.register_plugin(plugin)

        # Get the plugin
        retrieved_plugin = plugin_manager.get_plugin("test_plugin")

        # Should return the registered plugin
        assert retrieved_plugin is plugin
        assert retrieved_plugin.name == "test_plugin"

    def test_list_plugins(self, plugin_manager):
        """Test listing all plugins."""
        # Register plugins
        plugin1 = MockPlugin(name_value="plugin1",
                             description_value="First plugin")
        plugin2 = MockSingleToolPlugin()
        plugin_manager.register_plugin(plugin1)
        plugin_manager.register_plugin(plugin2)

        # List plugins
        plugins = plugin_manager.list_plugins()

        # Should return detailed info for all plugins
        assert len(plugins) == 2
        assert any(p["name"] == "plugin1" for p in plugins)
        assert any(p["name"] == "single_tool_plugin" for p in plugins)
        assert any(p["description"] == "First plugin" for p in plugins)

    def test_execute_tool(self, plugin_manager):
        """Test executing a tool."""
        # Register a plugin with tools
        plugin = MockPlugin()
        plugin_manager.register_plugin(plugin)

        # Execute one of the tools
        result = plugin_manager.tool_registry.get_tool(
            "tool1").execute(param1="value1")

        assert result["status"] == "success"
        assert "tool1 executed with" in result["result"]

    def test_assign_tool_to_agent(self, plugin_manager):
        """Test assigning tools to agents."""
        # Register a plugin with tools
        plugin = MockPlugin()
        plugin_manager.register_plugin(plugin)

        # Assign tools to an agent
        result1 = plugin_manager.tool_registry.assign_tool_to_agent(
            "agent1", "tool1")
        result2 = plugin_manager.tool_registry.assign_tool_to_agent(
            "agent1", "tool2")

        assert result1 is True
        assert result2 is True

        # Get tools for the agent
        agent_tools = plugin_manager.tool_registry.get_agent_tools("agent1")

        # Should have both tools
        assert len(agent_tools) == 2
        assert any(t["name"] == "tool1" for t in agent_tools)
        assert any(t["name"] == "tool2" for t in agent_tools)

    def test_configure_all_tools(self, plugin_manager):
        """Test configuring all tools."""
        # Register a plugin with tools
        plugin = MockPlugin()
        plugin_manager.register_plugin(plugin)

        # Configure all tools
        config = {"new_key": "new_value"}
        plugin_manager.tool_registry.configure_all_tools(config)

        # Get tools and check if they were configured
        tool1 = plugin_manager.tool_registry.get_tool("tool1")
        tool2 = plugin_manager.tool_registry.get_tool("tool2")

        assert tool1.configured
        assert tool2.configured
        assert tool1.config["new_key"] == "new_value"
        assert tool2.config["new_key"] == "new_value"

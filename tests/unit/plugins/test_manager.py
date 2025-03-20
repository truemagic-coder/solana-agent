import pytest
from unittest.mock import Mock, patch

from solana_agent.plugins.manager import PluginManager
from solana_agent.interfaces.plugins.plugins import Plugin
from solana_agent.plugins.registry import ToolRegistry

# Test configuration
TEST_CONFIG = {
    "plugin_settings": {
        "test_plugin": {
            "setting1": "value1",
            "setting2": "value2"
        }
    },
    "tool_settings": {
        "test_tool": {
            "api_key": "test_key"
        }
    }
}


class MockPlugin(Plugin):
    """Mock plugin for testing."""

    def __init__(self, plugin_name="test_plugin", plugin_description="Test plugin"):
        self._name = plugin_name
        self._description = plugin_description
        self.initialized = False
        self.configured = False
        self.config = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def initialize(self, tool_registry):
        self.initialized = True
        self.tool_registry = tool_registry

    def configure(self, config):
        self.configured = True
        self.config = config


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name="test_tool"):
        self.name = name
        self.execute_result = {"status": "success", "result": "test"}

    def execute(self, **kwargs):
        return self.execute_result


@pytest.fixture
def mock_tool_registry():
    registry = Mock(spec=ToolRegistry)
    registry.get_tool.return_value = MockTool()
    return registry


@pytest.fixture
def plugin_manager(mock_tool_registry):
    return PluginManager(config=TEST_CONFIG, tool_registry=mock_tool_registry)


@pytest.fixture
def mock_entry_point():
    entry_point = Mock()
    entry_point.name = "test_plugin"
    entry_point.value = "test.plugin:TestPlugin"
    plugin = MockPlugin()
    entry_point.load.return_value = lambda: plugin
    return entry_point


def test_register_plugin(plugin_manager):
    """Test plugin registration."""
    plugin = MockPlugin()
    success = plugin_manager.register_plugin(plugin)

    assert success
    assert plugin.initialized
    assert plugin.configured
    assert plugin.config == TEST_CONFIG
    assert plugin_manager.get_plugin("test_plugin") == plugin


@patch('importlib.metadata.entry_points')
def test_load_plugins(mock_entry_points, plugin_manager, mock_entry_point):
    """Test plugin loading from entry points."""
    mock_entry_points.return_value = [mock_entry_point]

    loaded_plugins = plugin_manager.load_plugins()

    assert "test_plugin" in loaded_plugins
    assert len(plugin_manager.list_plugins()) == 1


@patch('importlib.metadata.entry_points')
def test_load_plugins_duplicate(mock_entry_points, plugin_manager, mock_entry_point):
    """Test handling of duplicate plugin loading."""
    mock_entry_points.return_value = [mock_entry_point]

    # Load plugins twice
    plugin_manager.load_plugins()
    loaded_plugins = plugin_manager.load_plugins()

    # Should only be loaded once
    assert len(loaded_plugins) == 0


def test_list_plugins(plugin_manager):
    """Test listing of registered plugins."""
    plugin1 = MockPlugin("plugin1", "First plugin")
    plugin2 = MockPlugin("plugin2", "Second plugin")

    plugin_manager.register_plugin(plugin1)
    plugin_manager.register_plugin(plugin2)

    plugins = plugin_manager.list_plugins()
    assert len(plugins) == 2
    assert plugins[0]["name"] == "plugin1"
    assert plugins[1]["name"] == "plugin2"


def test_execute_tool_success(plugin_manager):
    """Test successful tool execution."""
    result = plugin_manager.execute_tool("test_tool", param1="value1")

    assert result["status"] == "success"
    assert result["result"] == "test"


def test_execute_tool_not_found(plugin_manager, mock_tool_registry):
    """Test tool execution when tool is not found."""
    mock_tool_registry.get_tool.return_value = None

    result = plugin_manager.execute_tool("nonexistent_tool")

    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_execute_tool_error(plugin_manager, mock_tool_registry):
    """Test tool execution error handling."""
    tool = MockTool()
    tool.execute = Mock(side_effect=Exception("Tool error"))
    mock_tool_registry.get_tool.return_value = tool

    result = plugin_manager.execute_tool("test_tool")

    assert result["status"] == "error"
    assert "Tool error" in result["message"]


def test_configure(plugin_manager):
    """Test configuration of plugins."""
    plugin = MockPlugin()
    plugin_manager.register_plugin(plugin)

    new_config = {"new_setting": "new_value"}
    plugin_manager.configure(new_config)

    assert plugin.configured
    assert "new_setting" in plugin.config
    assert plugin_manager.config["new_setting"] == "new_value"


def test_configure_error(plugin_manager):
    """Test error handling during plugin configuration."""
    plugin = MockPlugin()
    plugin.configure = Mock(side_effect=Exception("Config error"))
    plugin_manager.register_plugin(plugin)

    # Should not raise exception
    plugin_manager.configure({"new_setting": "value"})

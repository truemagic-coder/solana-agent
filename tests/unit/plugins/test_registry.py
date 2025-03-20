import pytest
from unittest.mock import Mock

from solana_agent.plugins.registry import ToolRegistry
from solana_agent.interfaces.plugins.plugins import Tool


class MockTool(Tool):
    """Mock tool implementation for testing."""

    def __init__(self, name: str, description: str = "Test tool"):
        self._name = name
        self._description = description
        self.configured = False
        self._schema = {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, **kwargs):
        return {"status": "success"}

    def configure(self, config):
        self.configured = True
        self.config = config

    def get_schema(self):
        return self._schema


@pytest.fixture
def tool_registry():
    """Create a fresh tool registry for each test."""
    return ToolRegistry()


@pytest.fixture
def mock_tool():
    """Create a mock tool instance."""
    return MockTool("test_tool")


def test_register_tool(tool_registry, mock_tool):
    """Test tool registration."""
    success = tool_registry.register_tool(mock_tool)

    assert success
    assert tool_registry.get_tool("test_tool") == mock_tool
    assert "test_tool" in tool_registry.list_all_tools()


def test_register_duplicate_tool(tool_registry, mock_tool):
    """Test registering the same tool twice."""
    tool_registry.register_tool(mock_tool)
    success = tool_registry.register_tool(mock_tool)

    assert success  # Should overwrite existing tool
    assert len(tool_registry.list_all_tools()) == 1


def test_get_nonexistent_tool(tool_registry):
    """Test getting a tool that doesn't exist."""
    tool = tool_registry.get_tool("nonexistent")
    assert tool is None


def test_assign_tool_to_agent(tool_registry, mock_tool):
    """Test assigning a tool to an agent."""
    tool_registry.register_tool(mock_tool)
    success = tool_registry.assign_tool_to_agent("test_agent", "test_tool")

    assert success
    tools = tool_registry.get_agent_tools("test_agent")
    assert len(tools) == 1
    assert tools[0]["name"] == "test_tool"


def test_assign_nonexistent_tool_to_agent(tool_registry):
    """Test assigning a non-existent tool to an agent."""
    success = tool_registry.assign_tool_to_agent("test_agent", "nonexistent")

    assert not success
    assert len(tool_registry.get_agent_tools("test_agent")) == 0


def test_get_agent_tools_nonexistent_agent(tool_registry):
    """Test getting tools for an agent that doesn't exist."""
    tools = tool_registry.get_agent_tools("nonexistent")
    assert len(tools) == 0


def test_get_agent_tools_with_multiple_tools(tool_registry):
    """Test getting multiple tools assigned to an agent."""
    tool1 = MockTool("tool1", "First tool")
    tool2 = MockTool("tool2", "Second tool")

    tool_registry.register_tool(tool1)
    tool_registry.register_tool(tool2)
    tool_registry.assign_tool_to_agent("test_agent", "tool1")
    tool_registry.assign_tool_to_agent("test_agent", "tool2")

    tools = tool_registry.get_agent_tools("test_agent")
    assert len(tools) == 2
    assert {t["name"] for t in tools} == {"tool1", "tool2"}
    assert all("description" in t for t in tools)
    assert all("parameters" in t for t in tools)


def test_list_all_tools(tool_registry):
    """Test listing all registered tools."""
    tools = [
        MockTool("tool1"),
        MockTool("tool2"),
        MockTool("tool3")
    ]

    for tool in tools:
        tool_registry.register_tool(tool)

    all_tools = tool_registry.list_all_tools()
    assert len(all_tools) == 3
    assert set(all_tools) == {"tool1", "tool2", "tool3"}


def test_configure_all_tools(tool_registry):
    """Test configuring all tools."""
    tools = [
        MockTool("tool1"),
        MockTool("tool2")
    ]

    for tool in tools:
        tool_registry.register_tool(tool)

    config = {"api_key": "test_key"}
    tool_registry.configure_all_tools(config)

    for tool_name in tool_registry.list_all_tools():
        tool = tool_registry.get_tool(tool_name)
        assert tool.configured
        assert tool.config == config


def test_tool_schema_in_agent_tools(tool_registry):
    """Test that tool schema is included in agent tools list."""
    tool = MockTool("schema_tool")
    tool._schema = {
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
            "param2": {"type": "integer"}
        }
    }

    tool_registry.register_tool(tool)
    tool_registry.assign_tool_to_agent("test_agent", "schema_tool")

    tools = tool_registry.get_agent_tools("test_agent")
    assert len(tools) == 1
    assert tools[0]["parameters"] == tool._schema

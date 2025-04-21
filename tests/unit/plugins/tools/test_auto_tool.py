"""
Tests for the AutoTool base class.

This module provides test coverage for the AutoTool implementation,
including initialization, configuration, and execution.
"""

import pytest
from unittest.mock import MagicMock

from solana_agent.plugins.tools.auto_tool import AutoTool
from solana_agent.interfaces.plugins.plugins import Tool


class TestTool(AutoTool):
    """Concrete implementation of AutoTool for testing."""

    __test__ = False  # Prevent pytest from collecting this helper class

    def get_schema(self):
        """Test schema implementation."""
        return {
            "type": "object",
            "properties": {
                "test_param": {"type": "string", "description": "Test parameter"}
            },
        }

    async def execute(self, **params):
        """Test execution implementation."""
        return {"status": "success", "params": params}


@pytest.fixture
def mock_registry():
    """Create a mock registry."""
    registry = MagicMock()
    registry.register_tool = MagicMock(return_value=True)
    return registry


class TestAutoTool:
    """Test suite for AutoTool base class."""

    def test_init_basic(self):
        """Test basic initialization without registry."""
        tool = TestTool("test_tool", "Test description")

        assert isinstance(tool, Tool)
        assert tool.name == "test_tool"
        assert tool.description == "Test description"
        assert tool._config == {}

    def test_init_with_registry(self, mock_registry):
        """Test initialization with registry."""
        tool = TestTool("test_tool", "Test description", registry=mock_registry)

        mock_registry.register_tool.assert_called_once_with(tool)
        assert tool.name == "test_tool"
        assert tool.description == "Test description"

    def test_configure(self):
        """Test tool configuration."""
        tool = TestTool("test_tool", "Test description")
        config = {"api_key": "test_key", "endpoint": "https://test.com"}

        tool.configure(config)
        assert tool._config == config

    def test_configure_empty(self):
        """Test configuration with empty dict."""
        tool = TestTool("test_tool", "Test description")
        tool.configure({})
        assert tool._config == {}

    def test_configure_none(self):
        """Test configuration with None."""
        tool = TestTool("test_tool", "Test description")
        with pytest.raises(TypeError):
            tool.configure(None)

    def test_get_schema_base(self):
        """Test base class schema implementation."""
        tool = AutoTool("test_tool", "Test description")
        assert tool.get_schema() == {}

    def test_get_schema_override(self):
        """Test overridden schema implementation."""
        tool = TestTool("test_tool", "Test description")
        schema = tool.get_schema()

        assert schema["type"] == "object"
        assert "test_param" in schema["properties"]
        assert schema["properties"]["test_param"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_execute_base(self):
        """Test base class execute implementation."""
        tool = AutoTool("test_tool", "Test description")
        with pytest.raises(NotImplementedError):
            await tool.execute(param="value")

    @pytest.mark.asyncio
    async def test_execute_override(self):
        """Test overridden execute implementation."""
        tool = TestTool("test_tool", "Test description")
        result = await tool.execute(test_param="value")

        assert result["status"] == "success"
        assert result["params"]["test_param"] == "value"

    def test_name_property(self):
        """Test name property getter."""
        tool = TestTool("test_tool", "Test description")
        assert tool.name == "test_tool"

    def test_description_property(self):
        """Test description property getter."""
        tool = TestTool("test_tool", "Test description")
        assert tool.description == "Test description"

    def test_registration_failure(self, mock_registry):
        """Test handling registration failure."""
        mock_registry.register_tool.return_value = False
        tool = TestTool("test_tool", "Test description", registry=mock_registry)

        mock_registry.register_tool.assert_called_once_with(tool)
        assert tool.name == "test_tool"
        assert tool.description == "Test description"

"""
Tests for Jupiter Shield Tool.

Tests the JupiterShieldTool which checks token security
using Jupiter Shield API.
"""

import pytest
from unittest.mock import patch, AsyncMock

from solana_agent.tools.jupiter_shield import JupiterShieldTool, JupiterShieldPlugin


@pytest.fixture
def shield_tool():
    """Create a configured JupiterShieldTool."""
    tool = JupiterShieldTool()
    tool.configure(
        {
            "tools": {
                "jupiter_shield": {
                    "jupiter_api_key": "test-api-key",
                }
            }
        }
    )
    return tool


@pytest.fixture
def shield_tool_no_key():
    """Create a JupiterShieldTool without API key."""
    tool = JupiterShieldTool()
    tool.configure({"tools": {"jupiter_shield": {}}})
    return tool


class TestJupiterShieldToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, shield_tool):
        """Should have correct tool name."""
        assert shield_tool.name == "jupiter_shield"

    def test_schema_has_required_properties(self, shield_tool):
        """Should include mints in required properties."""
        schema = shield_tool.get_schema()
        assert "mints" in schema["properties"]
        assert "mints" in schema["required"]

    def test_schema_mints_is_array(self, shield_tool):
        """Should have mints as array type."""
        schema = shield_tool.get_schema()
        assert schema["properties"]["mints"]["type"] == "array"
        assert schema["properties"]["mints"]["items"]["type"] == "string"


class TestJupiterShieldToolConfigure:
    """Test configuration method."""

    def test_configure_stores_api_key(self, shield_tool):
        """Should store Jupiter API key from config."""
        assert shield_tool._jupiter_api_key == "test-api-key"

    def test_configure_without_api_key(self, shield_tool_no_key):
        """Should work without API key."""
        assert shield_tool_no_key._jupiter_api_key is None


class TestJupiterShieldToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_empty_mints_error(self, shield_tool):
        """Should return error for empty mints list."""
        result = await shield_tool.execute(mints=[])

        assert result["status"] == "error"
        assert "no mints" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_get_shield_success(self, shield_tool):
        """Should return shield data on success."""
        mock_shield = {
            "warnings": {
                "TokenMint123": [
                    {
                        "type": "low_liquidity",
                        "message": "Token has low liquidity",
                        "severity": "warning",
                    }
                ],
                "TokenMint456": [],
            }
        }

        with patch("solana_agent.tools.jupiter_shield.JupiterUltra") as MockUltra:
            mock_instance = AsyncMock()
            mock_instance.get_shield = AsyncMock(return_value=mock_shield)
            MockUltra.return_value = mock_instance

            result = await shield_tool.execute(mints=["TokenMint123", "TokenMint456"])

            assert result["status"] == "success"
            assert "shield" in result
            assert result["shield"]["TokenMint123"]["has_warnings"] is True
            assert result["shield"]["TokenMint456"]["has_warnings"] is False

    @pytest.mark.asyncio
    async def test_execute_processes_warnings_correctly(self, shield_tool):
        """Should correctly process warning details."""
        mock_shield = {
            "warnings": {
                "TokenMint123": [
                    {
                        "type": "rug_pull_risk",
                        "message": "High risk of rug pull",
                        "severity": "critical",
                    },
                    {
                        "type": "low_holders",
                        "message": "Few token holders",
                        "severity": "warning",
                    },
                ],
            }
        }

        with patch("solana_agent.tools.jupiter_shield.JupiterUltra") as MockUltra:
            mock_instance = AsyncMock()
            mock_instance.get_shield = AsyncMock(return_value=mock_shield)
            MockUltra.return_value = mock_instance

            result = await shield_tool.execute(mints=["TokenMint123"])

            assert result["shield"]["TokenMint123"]["warning_count"] == 2
            assert len(result["shield"]["TokenMint123"]["warnings"]) == 2

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, shield_tool):
        """Should return error on exception."""
        with patch("solana_agent.tools.jupiter_shield.JupiterUltra") as MockUltra:
            mock_instance = AsyncMock()
            mock_instance.get_shield = AsyncMock(side_effect=Exception("API Error"))
            MockUltra.return_value = mock_instance

            result = await shield_tool.execute(mints=["TokenMint123"])

            assert result["status"] == "error"
            assert "API Error" in result["message"]


class TestJupiterShieldPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = JupiterShieldPlugin()
        assert plugin.name == "jupiter_shield"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = JupiterShieldPlugin()
        assert (
            "security" in plugin.description.lower()
            or "shield" in plugin.description.lower()
        )

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = JupiterShieldPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        from unittest.mock import MagicMock

        plugin = JupiterShieldPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

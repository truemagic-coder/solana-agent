"""
Tests for Jupiter Holdings Tool.

Tests the JupiterHoldingsTool which retrieves token holdings
using Jupiter Ultra API.
"""

import pytest
from unittest.mock import patch, AsyncMock

from solana_agent.tools.jupiter_holdings import (
    JupiterHoldingsTool,
    JupiterHoldingsPlugin,
)


@pytest.fixture
def holdings_tool():
    """Create a configured JupiterHoldingsTool."""
    tool = JupiterHoldingsTool()
    tool.configure(
        {
            "tools": {
                "jupiter_holdings": {
                    "jupiter_api_key": "test-api-key",
                }
            }
        }
    )
    return tool


@pytest.fixture
def holdings_tool_no_key():
    """Create a JupiterHoldingsTool without API key (lite mode)."""
    tool = JupiterHoldingsTool()
    tool.configure({"tools": {"jupiter_holdings": {}}})
    return tool


class TestJupiterHoldingsToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, holdings_tool):
        """Should have correct tool name."""
        assert holdings_tool.name == "jupiter_holdings"

    def test_schema_has_required_properties(self, holdings_tool):
        """Should include wallet_address in required properties."""
        schema = holdings_tool.get_schema()
        assert "wallet_address" in schema["properties"]
        assert "wallet_address" in schema["required"]

    def test_schema_has_native_only_option(self, holdings_tool):
        """Should include native_only option."""
        schema = holdings_tool.get_schema()
        assert "native_only" in schema["properties"]
        assert schema["properties"]["native_only"]["type"] == "boolean"


class TestJupiterHoldingsToolConfigure:
    """Test configuration method."""

    def test_configure_stores_api_key(self, holdings_tool):
        """Should store Jupiter API key from config."""
        assert holdings_tool._jupiter_api_key == "test-api-key"

    def test_configure_without_api_key(self, holdings_tool_no_key):
        """Should work without API key (lite mode)."""
        assert holdings_tool_no_key._jupiter_api_key is None


class TestJupiterHoldingsToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_get_holdings_success(self, holdings_tool):
        """Should return holdings on success."""
        mock_holdings = {
            "nativeBalance": {"amount": "1000000000", "uiAmount": 1.0},
            "tokens": [
                {
                    "mint": "TokenMint123",
                    "amount": "1000000",
                    "uiAmount": 1.0,
                }
            ],
        }

        with patch("solana_agent.tools.jupiter_holdings.JupiterUltra") as MockUltra:
            mock_instance = AsyncMock()
            mock_instance.get_holdings = AsyncMock(return_value=mock_holdings)
            MockUltra.return_value = mock_instance

            result = await holdings_tool.execute(wallet_address="Wallet123...abc")

            assert result["status"] == "success"
            assert "holdings" in result

    @pytest.mark.asyncio
    async def test_execute_get_native_holdings(self, holdings_tool):
        """Should return native holdings when native_only=True."""
        mock_native = {"amount": "1000000000", "uiAmount": 1.0}

        with patch("solana_agent.tools.jupiter_holdings.JupiterUltra") as MockUltra:
            mock_instance = AsyncMock()
            mock_instance.get_native_holdings = AsyncMock(return_value=mock_native)
            MockUltra.return_value = mock_instance

            result = await holdings_tool.execute(
                wallet_address="Wallet123...abc",
                native_only=True,
            )

            assert result["status"] == "success"
            mock_instance.get_native_holdings.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, holdings_tool):
        """Should return error on exception."""
        with patch("solana_agent.tools.jupiter_holdings.JupiterUltra") as MockUltra:
            mock_instance = AsyncMock()
            mock_instance.get_holdings = AsyncMock(side_effect=Exception("API Error"))
            MockUltra.return_value = mock_instance

            result = await holdings_tool.execute(wallet_address="Wallet123...abc")

            assert result["status"] == "error"
            assert "API Error" in result["message"]


class TestJupiterHoldingsPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = JupiterHoldingsPlugin()
        assert plugin.name == "jupiter_holdings"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = JupiterHoldingsPlugin()
        assert "holdings" in plugin.description.lower()

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = JupiterHoldingsPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        from unittest.mock import MagicMock

        plugin = JupiterHoldingsPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

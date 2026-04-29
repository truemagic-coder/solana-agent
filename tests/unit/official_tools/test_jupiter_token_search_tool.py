"""
Tests for Jupiter Token Search Tool.

Tests the JupiterTokenSearchTool which searches for tokens
using Jupiter Ultra API.
"""

import pytest
from unittest.mock import patch, AsyncMock

from solana_agent.tools.jupiter_token_search import (
    JupiterTokenSearchTool,
    JupiterTokenSearchPlugin,
)


@pytest.fixture
def search_tool():
    """Create a configured JupiterTokenSearchTool."""
    tool = JupiterTokenSearchTool()
    tool.configure(
        {
            "tools": {
                "jupiter_token_search": {
                    "jupiter_api_key": "test-api-key",
                }
            }
        }
    )
    return tool


@pytest.fixture
def search_tool_no_key():
    """Create a JupiterTokenSearchTool without API key."""
    tool = JupiterTokenSearchTool()
    tool.configure({"tools": {"jupiter_token_search": {}}})
    return tool


class TestJupiterTokenSearchToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, search_tool):
        """Should have correct tool name."""
        assert search_tool.name == "jupiter_token_search"

    def test_schema_has_required_properties(self, search_tool):
        """Should include query in required properties."""
        schema = search_tool.get_schema()
        assert "query" in schema["properties"]
        assert "query" in schema["required"]

    def test_schema_query_type(self, search_tool):
        """Should have query as string type."""
        schema = search_tool.get_schema()
        assert schema["properties"]["query"]["type"] == "string"


class TestJupiterTokenSearchToolConfigure:
    """Test configuration method."""

    def test_configure_stores_api_key(self, search_tool):
        """Should store Jupiter API key from config."""
        assert search_tool._jupiter_api_key == "test-api-key"

    def test_configure_without_api_key(self, search_tool_no_key):
        """Should work without API key."""
        assert search_tool_no_key._jupiter_api_key is None


class TestJupiterTokenSearchToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_empty_query_error(self, search_tool):
        """Should return error for empty query."""
        result = await search_tool.execute(query="")

        assert result["status"] == "error"
        assert "no query" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_search_success(self, search_tool):
        """Should return tokens on success."""
        mock_tokens = [
            {
                "id": "So11111111111111111111111111111111111111112",
                "name": "Wrapped SOL",
                "symbol": "SOL",
                "decimals": 9,
                "icon": "https://example.com/sol.png",
                "isVerified": True,
                "usdPrice": 100.0,
                "mcap": 50000000000,
                "fdv": 50000000000,
                "liquidity": 1000000000,
                "holderCount": 1000000,
                "organicScore": 95,
                "organicScoreLabel": "Excellent",
                "tags": ["native"],
                "cexes": ["binance", "coinbase"],
                "audit": {
                    "mintAuthorityDisabled": True,
                    "freezeAuthorityDisabled": True,
                    "topHoldersPercentage": 5.0,
                },
                "stats24h": {
                    "priceChange": 2.5,
                    "volumeChange": 10.0,
                    "buyVolume": 500000000,
                    "sellVolume": 400000000,
                    "numBuys": 50000,
                    "numSells": 40000,
                    "numTraders": 25000,
                },
            }
        ]

        with patch("solana_agent.tools.jupiter_token_search.JupiterUltra") as MockUltra:
            mock_instance = AsyncMock()
            mock_instance.search_tokens = AsyncMock(return_value=mock_tokens)
            MockUltra.return_value = mock_instance

            result = await search_tool.execute(query="SOL")

            assert result["status"] == "success"
            assert result["count"] == 1
            assert len(result["tokens"]) == 1

    @pytest.mark.asyncio
    async def test_execute_formats_token_correctly(self, search_tool):
        """Should format token data correctly."""
        mock_tokens = [
            {
                "id": "TokenMint123",
                "name": "Test Token",
                "symbol": "TEST",
                "decimals": 6,
                "isVerified": False,
                "usdPrice": 1.5,
                "mcap": 1000000,
            }
        ]

        with patch("solana_agent.tools.jupiter_token_search.JupiterUltra") as MockUltra:
            mock_instance = AsyncMock()
            mock_instance.search_tokens = AsyncMock(return_value=mock_tokens)
            MockUltra.return_value = mock_instance

            result = await search_tool.execute(query="TEST")

            token = result["tokens"][0]
            assert token["mint"] == "TokenMint123"
            assert token["name"] == "Test Token"
            assert token["symbol"] == "TEST"
            assert token["decimals"] == 6
            assert token["is_verified"] is False
            assert token["price_usd"] == 1.5

    @pytest.mark.asyncio
    async def test_execute_handles_missing_fields(self, search_tool):
        """Should handle tokens with missing optional fields."""
        mock_tokens = [
            {
                "id": "TokenMint123",
                "name": "Minimal Token",
                "symbol": "MIN",
            }
        ]

        with patch("solana_agent.tools.jupiter_token_search.JupiterUltra") as MockUltra:
            mock_instance = AsyncMock()
            mock_instance.search_tokens = AsyncMock(return_value=mock_tokens)
            MockUltra.return_value = mock_instance

            result = await search_tool.execute(query="MIN")

            assert result["status"] == "success"
            token = result["tokens"][0]
            assert token["mint"] == "TokenMint123"
            assert token["tags"] == []
            assert token["cexes"] == []

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, search_tool):
        """Should return error on exception."""
        with patch("solana_agent.tools.jupiter_token_search.JupiterUltra") as MockUltra:
            mock_instance = AsyncMock()
            mock_instance.search_tokens = AsyncMock(side_effect=Exception("API Error"))
            MockUltra.return_value = mock_instance

            result = await search_tool.execute(query="TEST")

            assert result["status"] == "error"
            assert "API Error" in result["message"]


class TestJupiterTokenSearchPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = JupiterTokenSearchPlugin()
        assert plugin.name == "jupiter_token_search"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = JupiterTokenSearchPlugin()
        assert (
            "token" in plugin.description.lower()
            or "search" in plugin.description.lower()
        )

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = JupiterTokenSearchPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        from unittest.mock import MagicMock

        plugin = JupiterTokenSearchPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

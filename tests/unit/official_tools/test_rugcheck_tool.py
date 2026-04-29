"""
Tests for Rugcheck Tool.

Tests the RugCheckTool which checks token risk and liquidity
using rugcheck.xyz API.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.rugcheck import RugCheckTool, RugCheckPlugin, summarize_rugcheck


@pytest.fixture
def rugcheck_tool():
    """Create a configured RugCheckTool."""
    tool = RugCheckTool()
    tool.configure({})  # No config needed
    return tool


class TestRugCheckToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, rugcheck_tool):
        """Should have correct tool name."""
        assert rugcheck_tool.name == "rugcheck"

    def test_schema_has_required_properties(self, rugcheck_tool):
        """Should include mint in required properties."""
        schema = rugcheck_tool.get_schema()
        assert "mint" in schema["properties"]
        assert "mint" in schema["required"]

    def test_schema_property_types(self, rugcheck_tool):
        """Should have correct property types."""
        schema = rugcheck_tool.get_schema()
        assert schema["properties"]["mint"]["type"] == "string"


class TestRugCheckToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_success(self, rugcheck_tool):
        """Should return summary on success."""
        mock_data = {
            "mint": "TokenMint123...abc",
            "tokenMeta": {"name": "Test Token", "symbol": "TEST"},
            "fileMeta": {},
            "score": 85,
            "score_normalised": 0.85,
            "rugged": False,
            "totalHolders": 1000,
            "totalMarketLiquidity": 100000,
            "price": 1.5,
            "verification": {"jup_verified": True},
            "creator": "Creator123...abc",
            "topHolders": [
                {"address": "Holder1...abc", "pct": 5.5, "insider": False},
                {"address": "Holder2...abc", "pct": 3.2, "insider": True},
            ],
            "risks": [],
            "markets": [{"marketType": "raydium", "pubkey": "Market123...abc"}],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_data

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await rugcheck_tool.execute(mint="TokenMint123...abc")

            assert result["status"] == "success"
            assert "result" in result

    @pytest.mark.asyncio
    async def test_execute_api_error(self, rugcheck_tool):
        """Should return error on API failure."""
        import respx

        with respx.mock:
            respx.get(
                "https://api.rugcheck.xyz/v1/tokens/InvalidMint123/report"
            ).respond(404, text="Token not found")

            result = await rugcheck_tool.execute(mint="InvalidMint123")

            assert result["status"] == "error"
            assert "404" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, rugcheck_tool):
        """Should return error on network exception."""
        import respx
        import httpx

        with respx.mock:
            respx.get("https://api.rugcheck.xyz/v1/tokens/TokenMint123/report").mock(
                side_effect=httpx.ConnectError("Network error")
            )

            result = await rugcheck_tool.execute(mint="TokenMint123")

            assert result["status"] == "error"
            assert (
                "Network error" in result["message"]
                or "error" in result["message"].lower()
            )


class TestSummarizeRugcheck:
    """Test the summarize_rugcheck function."""

    def test_summarize_basic_info(self):
        """Should include basic token info in summary."""
        data = {
            "mint": "TokenMint123...abc",
            "tokenMeta": {"name": "Test Token", "symbol": "TEST"},
            "fileMeta": {},
            "score": 85,
            "score_normalised": 0.85,
            "rugged": False,
            "totalHolders": 1000,
            "totalMarketLiquidity": 100000,
            "price": 1.5,
            "verification": {"jup_verified": True},
            "creator": "Creator123...abc",
            "topHolders": [],
            "risks": [],
            "markets": [],
        }

        summary = summarize_rugcheck(data)

        assert "Test Token" in summary
        assert "TEST" in summary
        assert "TokenMint123...abc" in summary
        assert "85" in summary

    def test_summarize_with_risks(self):
        """Should include risks in summary."""
        data = {
            "mint": "TokenMint123",
            "tokenMeta": {"name": "Test", "symbol": "T"},
            "fileMeta": {},
            "score": 50,
            "score_normalised": 0.5,
            "rugged": False,
            "totalHolders": 100,
            "totalMarketLiquidity": 10000,
            "price": 0.5,
            "verification": {},
            "creator": "Creator123",
            "topHolders": [],
            "risks": ["Low liquidity", "Few holders"],
            "markets": [],
        }

        summary = summarize_rugcheck(data)

        assert "Low liquidity" in summary
        assert "Few holders" in summary

    def test_summarize_with_holders(self):
        """Should include top holders in summary."""
        data = {
            "mint": "TokenMint123",
            "tokenMeta": {"name": "Test", "symbol": "T"},
            "fileMeta": {},
            "score": 75,
            "score_normalised": 0.75,
            "rugged": False,
            "totalHolders": 500,
            "totalMarketLiquidity": 50000,
            "price": 1.0,
            "verification": {},
            "creator": "Creator123",
            "topHolders": [
                {"address": "Holder1...abc", "pct": 10.5, "insider": True},
                {"address": "Holder2...abc", "pct": 5.2, "insider": False},
            ],
            "risks": [],
            "markets": [],
        }

        summary = summarize_rugcheck(data)

        assert "Holder1...abc" in summary
        assert "10.50%" in summary
        assert "Insider: True" in summary

    def test_summarize_no_risks(self):
        """Should indicate no risks detected."""
        data = {
            "mint": "TokenMint123",
            "tokenMeta": {"name": "Safe Token", "symbol": "SAFE"},
            "fileMeta": {},
            "score": 95,
            "score_normalised": 0.95,
            "rugged": False,
            "totalHolders": 10000,
            "totalMarketLiquidity": 1000000,
            "price": 5.0,
            "verification": {"jup_verified": True},
            "creator": "Creator123",
            "topHolders": [],
            "risks": [],
            "markets": [],
        }

        summary = summarize_rugcheck(data)

        assert "Risks: None detected" in summary


class TestRugCheckPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = RugCheckPlugin()
        assert plugin.name == "rugcheck"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = RugCheckPlugin()
        assert (
            "risk" in plugin.description.lower()
            or "rugcheck" in plugin.description.lower()
        )

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = RugCheckPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = RugCheckPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

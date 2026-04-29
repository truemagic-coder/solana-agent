"""
Tests for Solana Ultra Quote Tool.

Tests the SolanaUltraQuoteTool which provides swap previews using Jupiter Ultra API
with a Solana keypair. Shows slippage and price impact without executing.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.solana_ultra_quote import (
    SolanaUltraQuoteTool,
    SolanaUltraQuotePlugin,
    get_plugin,
)


@pytest.fixture
def quote_tool():
    """Create a configured SolanaUltraQuoteTool."""
    tool = SolanaUltraQuoteTool()
    tool.configure(
        {
            "tools": {
                "solana_ultra_quote": {
                    "private_key": "5jGR...base58privatekey",
                    "jupiter_api_key": "test-jupiter-key",
                    "referral_account": "RefAcct123",
                    "referral_fee": 50,
                }
            }
        }
    )
    return tool


@pytest.fixture
def quote_tool_with_payer():
    """Create a SolanaUltraQuoteTool with payer."""
    tool = SolanaUltraQuoteTool()
    tool.configure(
        {
            "tools": {
                "solana_ultra_quote": {
                    "private_key": "5jGR...base58privatekey",
                    "jupiter_api_key": "test-jupiter-key",
                    "payer_private_key": "PayerPrivateKey...base58",
                }
            }
        }
    )
    return tool


@pytest.fixture
def quote_tool_no_key():
    """Create an unconfigured SolanaUltraQuoteTool."""
    tool = SolanaUltraQuoteTool()
    tool.configure({"tools": {"solana_ultra_quote": {}}})
    return tool


class TestSolanaUltraQuoteToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, quote_tool):
        """Should have correct tool name."""
        assert quote_tool.name == "solana_ultra_quote"

    def test_schema_has_required_properties(self, quote_tool):
        """Should include all required properties."""
        schema = quote_tool.get_schema()
        assert "input_mint" in schema["properties"]
        assert "output_mint" in schema["properties"]
        assert "amount" in schema["properties"]
        assert set(schema["required"]) == {"input_mint", "output_mint", "amount"}

    def test_schema_property_types(self, quote_tool):
        """Should have correct property types."""
        schema = quote_tool.get_schema()
        assert schema["properties"]["input_mint"]["type"] == "string"
        assert schema["properties"]["output_mint"]["type"] == "string"
        assert schema["properties"]["amount"]["type"] == "integer"


class TestSolanaUltraQuoteToolConfigure:
    """Test configuration method."""

    def test_configure_stores_private_key(self, quote_tool):
        """Should store private key from config."""
        assert quote_tool._private_key == "5jGR...base58privatekey"

    def test_configure_stores_jupiter_config(self, quote_tool):
        """Should store Jupiter configuration."""
        assert quote_tool._jupiter_api_key == "test-jupiter-key"
        assert quote_tool._referral_account == "RefAcct123"
        assert quote_tool._referral_fee == 50

    def test_configure_stores_payer_key(self, quote_tool_with_payer):
        """Should store payer private key."""
        assert quote_tool_with_payer._payer_private_key == "PayerPrivateKey...base58"


class TestSolanaUltraQuoteToolExecute:
    """Test execute method for quote preview."""

    @pytest.mark.asyncio
    async def test_execute_missing_private_key_error(self, quote_tool_no_key):
        """Should return error when private key is missing."""
        result = await quote_tool_no_key.execute(
            input_mint="So11111111111111111111111111111111111111112",
            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            amount=1000000000,
        )

        assert result["status"] == "error"
        assert (
            "private" in result["message"].lower() or "key" in result["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_execute_success_returns_quote(self, quote_tool):
        """Should return quote with slippage and price impact."""
        mock_order = MagicMock()
        mock_order.input_mint = "So11111111111111111111111111111111111111112"
        mock_order.output_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        mock_order.in_amount = "1000000000"
        mock_order.out_amount = "50000000"
        mock_order.slippage_bps = 100
        mock_order.price_impact = -0.5
        mock_order.in_usd_value = 50.0
        mock_order.out_usd_value = 49.75
        mock_order.swap_type = "ExactIn"
        mock_order.gasless = False

        with (
            patch("solana_agent.tools.solana_ultra_quote.Keypair") as MockKeypair,
            patch("solana_agent.tools.solana_ultra_quote.JupiterUltra") as MockUltra,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "TakerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_ultra_instance

            result = await quote_tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "success"
            assert result["input_mint"] == "So11111111111111111111111111111111111111112"
            assert (
                result["output_mint"] == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            )
            assert result["in_amount"] == "1000000000"
            assert result["out_amount"] == "50000000"
            assert result["slippage_bps"] == 100
            assert result["price_impact_pct"] == "-0.50%"
            assert result["in_usd_value"] == "$50.00"
            assert result["out_usd_value"] == "$49.75"
            assert result["swap_type"] == "ExactIn"
            assert result["gasless"] is False
            assert "preview only" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_with_referral(self, quote_tool):
        """Should include referral info in get_order call."""
        mock_order = MagicMock()
        mock_order.input_mint = "So11111111111111111111111111111111111111112"
        mock_order.output_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        mock_order.in_amount = "1000000000"
        mock_order.out_amount = "50000000"
        mock_order.slippage_bps = 100
        mock_order.price_impact = -0.3
        mock_order.in_usd_value = 50.0
        mock_order.out_usd_value = 49.85
        mock_order.swap_type = "ExactIn"
        mock_order.gasless = False

        with (
            patch("solana_agent.tools.solana_ultra_quote.Keypair") as MockKeypair,
            patch("solana_agent.tools.solana_ultra_quote.JupiterUltra") as MockUltra,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "TakerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_ultra_instance

            result = await quote_tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            # Verify get_order was called with referral params
            mock_ultra_instance.get_order.assert_called_once()
            call_args = mock_ultra_instance.get_order.call_args
            assert call_args.kwargs["referral_account"] == "RefAcct123"
            assert call_args.kwargs["referral_fee"] == 50

            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_with_payer(self, quote_tool_with_payer):
        """Should include payer in get_order call."""
        mock_order = MagicMock()
        mock_order.input_mint = "So11111111111111111111111111111111111111112"
        mock_order.output_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        mock_order.in_amount = "1000000000"
        mock_order.out_amount = "50000000"
        mock_order.slippage_bps = 100
        mock_order.price_impact = -0.5
        mock_order.in_usd_value = 50.0
        mock_order.out_usd_value = 49.75
        mock_order.swap_type = "ExactIn"
        mock_order.gasless = True

        with (
            patch("solana_agent.tools.solana_ultra_quote.Keypair") as MockKeypair,
            patch("solana_agent.tools.solana_ultra_quote.JupiterUltra") as MockUltra,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "TakerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_payer_keypair = MagicMock()
            mock_payer_keypair.pubkey.return_value = "PayerPubkey123"

            def keypair_side_effect(key):
                if key == "PayerPrivateKey...base58":
                    return mock_payer_keypair
                return mock_keypair

            MockKeypair.from_base58_string.side_effect = keypair_side_effect

            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_ultra_instance

            result = await quote_tool_with_payer.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            # Verify get_order was called with payer
            call_args = mock_ultra_instance.get_order.call_args
            assert call_args.kwargs["payer"] == "PayerPubkey123"

            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_jupiter_api_error(self, quote_tool):
        """Should handle Jupiter API errors gracefully."""
        with (
            patch("solana_agent.tools.solana_ultra_quote.Keypair") as MockKeypair,
            patch("solana_agent.tools.solana_ultra_quote.JupiterUltra") as MockUltra,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "TakerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )
            MockUltra.return_value = mock_ultra_instance

            result = await quote_tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "API rate limit" in result["message"]


class TestSolanaUltraQuotePlugin:
    """Test plugin functionality."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = SolanaUltraQuotePlugin()
        assert plugin.name == "solana_ultra_quote"

    def test_plugin_description(self):
        """Should have proper description."""
        plugin = SolanaUltraQuotePlugin()
        assert plugin.description is not None
        assert len(plugin.description) > 0

    def test_plugin_initialize(self):
        """Should initialize with tool registry."""
        plugin = SolanaUltraQuotePlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)
        assert plugin.tool_registry == mock_registry
        assert plugin._tool is not None

    def test_plugin_configure(self):
        """Should pass config to tool."""
        plugin = SolanaUltraQuotePlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)

        config = {
            "tools": {
                "solana_ultra_quote": {
                    "private_key": "test-key",
                    "jupiter_api_key": "test-api-key",
                }
            }
        }
        plugin.configure(config)
        assert plugin._tool._private_key == "test-key"
        assert plugin._tool._jupiter_api_key == "test-api-key"

    def test_plugin_get_tools(self):
        """Should return configured tools."""
        plugin = SolanaUltraQuotePlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)
        tools = plugin.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "solana_ultra_quote"


def test_get_plugin():
    """Should return SolanaUltraQuotePlugin instance."""
    plugin = get_plugin()
    assert isinstance(plugin, SolanaUltraQuotePlugin)

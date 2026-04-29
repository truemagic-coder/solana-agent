"""
Tests for Privy Ultra Quote Tool.

Tests the PrivyUltraQuoteTool which provides swap previews using Jupiter Ultra API
via Privy delegated wallets. Shows slippage and price impact without executing.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.privy_ultra_quote import (
    PrivyUltraQuoteTool,
    PrivyUltraQuotePlugin,
    get_plugin,
)


@pytest.fixture
def quote_tool():
    """Create a configured PrivyUltraQuoteTool."""
    tool = PrivyUltraQuoteTool()
    tool.configure(
        {
            "tools": {
                "privy_ultra_quote": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
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
    """Create a PrivyUltraQuoteTool with payer."""
    tool = PrivyUltraQuoteTool()
    tool.configure(
        {
            "tools": {
                "privy_ultra_quote": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                    "jupiter_api_key": "test-jupiter-key",
                    "payer_private_key": "PayerPrivateKey...base58",
                }
            }
        }
    )
    return tool


@pytest.fixture
def quote_tool_incomplete():
    """Create an incomplete PrivyUltraQuoteTool."""
    tool = PrivyUltraQuoteTool()
    tool.configure(
        {
            "tools": {
                "privy_ultra_quote": {
                    "app_id": "test-app-id",
                    # Missing app_secret
                }
            }
        }
    )
    return tool


class TestPrivyUltraQuoteToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, quote_tool):
        """Should have correct tool name."""
        assert quote_tool.name == "privy_ultra_quote"

    def test_schema_has_required_properties(self, quote_tool):
        """Should include all required properties."""
        schema = quote_tool.get_schema()
        assert "wallet_id" in schema["properties"]
        assert "wallet_public_key" in schema["properties"]
        assert "input_mint" in schema["properties"]
        assert "output_mint" in schema["properties"]
        assert "amount" in schema["properties"]
        assert set(schema["required"]) == {
            "wallet_id",
            "wallet_public_key",
            "input_mint",
            "output_mint",
            "amount",
        }

    def test_schema_property_types(self, quote_tool):
        """Should have correct property types."""
        schema = quote_tool.get_schema()
        assert schema["properties"]["wallet_id"]["type"] == "string"
        assert schema["properties"]["wallet_public_key"]["type"] == "string"
        assert schema["properties"]["input_mint"]["type"] == "string"
        assert schema["properties"]["output_mint"]["type"] == "string"
        assert schema["properties"]["amount"]["type"] == "integer"


class TestPrivyUltraQuoteToolConfigure:
    """Test configuration method."""

    def test_configure_stores_jupiter_config(self, quote_tool):
        """Should store Jupiter configuration."""
        assert quote_tool.jupiter_api_key == "test-jupiter-key"
        assert quote_tool.referral_account == "RefAcct123"
        assert quote_tool.referral_fee == 50

    def test_configure_stores_payer_key(self, quote_tool_with_payer):
        """Should store payer private key."""
        assert quote_tool_with_payer.payer_private_key == "PayerPrivateKey...base58"


class TestPrivyUltraQuoteToolExecute:
    """Test execute method for quote preview."""

    @pytest.mark.asyncio
    async def test_execute_missing_wallet_params(self, quote_tool):
        """Should return error when wallet params are missing."""
        result = await quote_tool.execute(
            wallet_id="",
            wallet_public_key="",
            input_mint="So11111111111111111111111111111111111111112",
            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            amount=1000000000,
        )

        assert result["status"] == "error"
        assert "wallet" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_success_with_wallet_params(self, quote_tool):
        """Should work with wallet_id and wallet_public_key."""
        mock_order = MagicMock()
        mock_order.input_mint = "So11111111111111111111111111111111111111112"
        mock_order.output_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        mock_order.in_amount = "1000000000"
        mock_order.out_amount = "50000000"
        mock_order.slippage_bps = 100
        mock_order.price_impact = -0.005
        mock_order.in_usd_value = 50.0
        mock_order.out_usd_value = 49.75
        mock_order.swap_type = "ExactIn"
        mock_order.gasless = False

        with (
            patch("solana_agent.tools.privy_ultra_quote.Keypair") as MockKeypair,
            patch("solana_agent.tools.privy_ultra_quote.JupiterUltra") as MockUltra,
        ):
            mock_payer_keypair = MagicMock()
            mock_payer_keypair.pubkey.return_value = "PayerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_payer_keypair

            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_ultra_instance

            result = await quote_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="UserPubkey123",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "success"

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

        with patch("solana_agent.tools.privy_ultra_quote.JupiterUltra") as MockUltra:
            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_ultra_instance

            result = await quote_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="UserPubkey123",
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
            patch("solana_agent.tools.privy_ultra_quote.Keypair") as MockKeypair,
            patch("solana_agent.tools.privy_ultra_quote.JupiterUltra") as MockUltra,
        ):
            mock_payer_keypair = MagicMock()
            mock_payer_keypair.pubkey.return_value = "PayerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_payer_keypair

            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_ultra_instance

            result = await quote_tool_with_payer.execute(
                wallet_id="wallet-123",
                wallet_public_key="UserPubkey123",
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
        with patch("solana_agent.tools.privy_ultra_quote.JupiterUltra") as MockUltra:
            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )
            MockUltra.return_value = mock_ultra_instance

            result = await quote_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="UserPubkey123",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "API rate limit" in result["message"]


class TestPrivyUltraQuotePlugin:
    """Test plugin functionality."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = PrivyUltraQuotePlugin()
        assert plugin.name == "privy_ultra_quote"

    def test_plugin_description(self):
        """Should have proper description."""
        plugin = PrivyUltraQuotePlugin()
        assert plugin.description is not None
        assert len(plugin.description) > 0

    def test_plugin_initialize(self):
        """Should initialize with tool registry."""
        plugin = PrivyUltraQuotePlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)
        assert plugin.tool_registry == mock_registry
        assert plugin._tool is not None

    def test_plugin_configure(self):
        """Should pass config to tool."""
        plugin = PrivyUltraQuotePlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)

        config = {
            "tools": {
                "privy_ultra_quote": {
                    "jupiter_api_key": "test-api-key",
                }
            }
        }
        plugin.configure(config)
        assert plugin._tool.jupiter_api_key == "test-api-key"

    def test_plugin_get_tools(self):
        """Should return configured tools."""
        plugin = PrivyUltraQuotePlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)
        tools = plugin.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "privy_ultra_quote"


def test_get_plugin():
    """Should return PrivyUltraQuotePlugin instance."""
    plugin = get_plugin()
    assert isinstance(plugin, PrivyUltraQuotePlugin)

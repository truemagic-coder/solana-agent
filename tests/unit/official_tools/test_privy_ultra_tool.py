"""
Tests for Privy Ultra Tool.

Tests the PrivyUltraTool which swaps tokens using Jupiter Ultra API
via Privy delegated wallets. Transactions are sent via Helius RPC
instead of Jupiter's /execute endpoint for reliability.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.privy_ultra import (
    PrivyUltraTool,
    PrivyUltraPlugin,
    get_plugin,
)


@pytest.fixture
def ultra_tool():
    """Create a configured PrivyUltraTool."""
    tool = PrivyUltraTool()
    tool.configure(
        {
            "tools": {
                "privy_ultra": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                    "signing_key": "wallet-auth:test-signing-key",
                    "jupiter_api_key": "test-jupiter-key",
                    "referral_account": "RefAcct123",
                    "referral_fee": 50,
                    "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                }
            }
        }
    )
    return tool


@pytest.fixture
def ultra_tool_incomplete():
    """Create an incomplete PrivyUltraTool."""
    tool = PrivyUltraTool()
    tool.configure(
        {
            "tools": {
                "privy_ultra": {
                    "app_id": "test-app-id",
                    # Missing app_secret and signing_key
                }
            }
        }
    )
    return tool


class TestPrivyUltraToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, ultra_tool):
        """Should have correct tool name."""
        assert ultra_tool.name == "privy_ultra"

    def test_schema_has_required_properties(self, ultra_tool):
        """Should include all required properties."""
        schema = ultra_tool.get_schema()
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

    def test_schema_property_types(self, ultra_tool):
        """Should have correct property types."""
        schema = ultra_tool.get_schema()
        assert schema["properties"]["wallet_id"]["type"] == "string"
        assert schema["properties"]["wallet_public_key"]["type"] == "string"
        assert schema["properties"]["input_mint"]["type"] == "string"
        assert schema["properties"]["output_mint"]["type"] == "string"
        assert schema["properties"]["amount"]["type"] == "integer"


class TestPrivyUltraToolConfigure:
    """Test configuration method."""

    def test_configure_stores_privy_config(self, ultra_tool):
        """Should store Privy configuration."""
        assert ultra_tool.app_id == "test-app-id"
        assert ultra_tool.app_secret == "test-app-secret"
        assert ultra_tool.signing_key == "wallet-auth:test-signing-key"

    def test_configure_stores_jupiter_config(self, ultra_tool):
        """Should store Jupiter configuration."""
        assert ultra_tool.jupiter_api_key == "test-jupiter-key"
        assert ultra_tool.referral_account == "RefAcct123"
        assert ultra_tool.referral_fee == 50

    def test_configure_stores_rpc_url(self, ultra_tool):
        """Should store RPC URL for direct transaction sending."""
        assert ultra_tool._rpc_url == "https://mainnet.helius-rpc.com/?api-key=test-key"

    def test_configure_stores_internal_signing_key(self, ultra_tool):
        """Should store internal signing key for _sign_and_execute."""
        assert ultra_tool._signing_key == "wallet-auth:test-signing-key"


class TestPrivyUltraToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_missing_wallet_params(self, ultra_tool):
        """Should return error when wallet params are missing."""
        result = await ultra_tool.execute(
            wallet_id="",
            wallet_public_key="",
            input_mint="So11111111111111111111111111111111111111112",
            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            amount=1000000000,
        )

        assert result["status"] == "error"
        assert (
            "wallet_id" in result["message"].lower()
            or "wallet_public_key" in result["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_execute_missing_config_error(self, ultra_tool_incomplete):
        """Should return error when config is incomplete."""
        result = await ultra_tool_incomplete.execute(
            wallet_id="wallet-123",
            wallet_public_key="WalletPubkey123",
            input_mint="So11111111111111111111111111111111111111112",
            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            amount=1000000000,
        )

        assert result["status"] == "error"
        assert "config" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_no_transaction_error(self, ultra_tool):
        """Should return error when no transaction returned."""
        mock_order = MagicMock()
        mock_order.transaction = None

        with patch("solana_agent.tools.privy_ultra.JupiterUltra") as MockUltra:
            mock_instance = AsyncMock()
            mock_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_instance

            result = await ultra_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123...abc",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "transaction" in result["message"].lower()


class TestPrivyUltraPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = PrivyUltraPlugin()
        assert plugin.name == "privy_ultra"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = PrivyUltraPlugin()
        assert (
            "swap" in plugin.description.lower()
            or "ultra" in plugin.description.lower()
        )

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = PrivyUltraPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = PrivyUltraPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

    def test_plugin_configure(self):
        """Should configure the tool after initialization."""
        plugin = PrivyUltraPlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)

        config = {
            "tools": {
                "privy_ultra": {
                    "app_id": "test-app",
                    "app_secret": "test-secret",
                    "signing_key": "wallet-auth:test-key",
                }
            }
        }
        plugin.configure(config)

        assert plugin.config == config
        assert plugin._tool.app_id == "test-app"


class TestPrivyUltraToolExecuteSuccess:
    """Test successful execute paths using RPC-based transaction sending."""

    TEST_EC_KEY_SEC1 = "MHcCAQEEIH6phbwVBTxg+QYJMSqHXcLoiTpmO163WjA8Td/+DqQ3oAoGCCqGSM49AwEHoUQDQgAEoY29/uiiWfItIYBAmejKuM17a0GackAbFG4sNs1ObTUilKQ2V/7WkTRC0xk7IgLwCRUI1e/Yk5wQFCjlajvilw=="

    @pytest.fixture
    def configured_ultra_tool(self):
        """Create a fully configured PrivyUltraTool with RPC URL."""
        tool = PrivyUltraTool()
        tool.configure(
            {
                "tools": {
                    "privy_ultra": {
                        "app_id": "test-app-id",
                        "app_secret": "test-app-secret",
                        "signing_key": f"wallet-auth:{self.TEST_EC_KEY_SEC1}",
                        "jupiter_api_key": "test-jupiter-key",
                        "referral_account": "RefAcct123",
                        "referral_fee": 50,
                        "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                    }
                }
            }
        )
        return tool

    @pytest.mark.asyncio
    async def test_execute_successful_swap(self, configured_ultra_tool):
        """Should successfully execute a swap via RPC."""
        mock_order = MagicMock()
        mock_order.transaction = "mock-transaction-base64"
        mock_order.request_id = "req-123"
        mock_order.swap_type = "ExactIn"
        mock_order.gasless = False
        mock_order.price_impact = -0.5
        mock_order.in_usd_value = 50.0
        mock_order.out_usd_value = 49.75
        mock_order.in_amount = "1000000000"
        mock_order.out_amount = "50000000"
        mock_order.slippage_bps = 100

        # Mock _sign_and_execute to return success
        mock_exec_result = {
            "status": "success",
            "signature": "sig-456",
        }

        with (
            patch("solana_agent.tools.privy_ultra.JupiterUltra") as MockUltra,
            patch.object(
                configured_ultra_tool,
                "_sign_and_execute",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ),
        ):
            mock_instance = AsyncMock()
            mock_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_instance

            result = await configured_ultra_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "success"
            assert result["signature"] == "sig-456"
            assert result["swap_type"] == "ExactIn"
            assert result["gasless"] is False

    @pytest.mark.asyncio
    async def test_execute_sign_and_execute_failure(self, configured_ultra_tool):
        """Should return error when _sign_and_execute fails."""
        mock_order = MagicMock()
        mock_order.transaction = "mock-transaction-base64"
        mock_order.request_id = "req-123"
        mock_order.price_impact = -0.5
        mock_order.in_usd_value = 50.0
        mock_order.out_usd_value = 49.75

        # Mock _sign_and_execute to return error
        mock_exec_result = {
            "status": "error",
            "message": "Failed to send transaction",
        }

        with (
            patch("solana_agent.tools.privy_ultra.JupiterUltra") as MockUltra,
            patch.object(
                configured_ultra_tool,
                "_sign_and_execute",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ),
        ):
            mock_instance = AsyncMock()
            mock_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_instance

            result = await configured_ultra_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "Failed to send transaction" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_rpc_blockhash_failure(self, configured_ultra_tool):
        """Should return error when blockhash fetch fails."""
        mock_order = MagicMock()
        mock_order.transaction = "mock-transaction-base64"
        mock_order.request_id = "req-123"
        mock_order.price_impact = -0.5
        mock_order.in_usd_value = 50.0
        mock_order.out_usd_value = 49.75

        # Mock _sign_and_execute to return blockhash error
        mock_exec_result = {
            "status": "error",
            "message": "Failed to get blockhash: RPC error",
        }

        with (
            patch("solana_agent.tools.privy_ultra.JupiterUltra") as MockUltra,
            patch.object(
                configured_ultra_tool,
                "_sign_and_execute",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ),
        ):
            mock_instance = AsyncMock()
            mock_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_instance

            result = await configured_ultra_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "blockhash" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_privy_signing_failure(self, configured_ultra_tool):
        """Should return error when Privy signing fails."""
        mock_order = MagicMock()
        mock_order.transaction = "mock-transaction-base64"
        mock_order.request_id = "req-123"
        mock_order.price_impact = -0.5
        mock_order.in_usd_value = 50.0
        mock_order.out_usd_value = 49.75

        # Mock _sign_and_execute to return signing error
        mock_exec_result = {
            "status": "error",
            "message": "Failed to sign transaction via Privy.",
        }

        with (
            patch("solana_agent.tools.privy_ultra.JupiterUltra") as MockUltra,
            patch.object(
                configured_ultra_tool,
                "_sign_and_execute",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ),
        ):
            mock_instance = AsyncMock()
            mock_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_instance

            result = await configured_ultra_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "sign" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, configured_ultra_tool):
        """Should handle exceptions gracefully."""
        with (
            patch("solana_agent.tools.privy_ultra.JupiterUltra") as MockUltra,
        ):
            mock_instance = AsyncMock()
            mock_instance.get_order = AsyncMock(side_effect=Exception("Network error"))
            MockUltra.return_value = mock_instance

            result = await configured_ultra_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "Network error" in result["message"]


class TestGetPlugin:
    """Test get_plugin function."""

    def test_get_plugin_returns_instance(self):
        """Should return a PrivyUltraPlugin instance."""
        plugin = get_plugin()
        assert isinstance(plugin, PrivyUltraPlugin)
        assert plugin.name == "privy_ultra"

"""
Tests for Privy Transfer Tool.

Tests the PrivyTransferTool which transfers SOL and SPL tokens
using Privy delegated wallets.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.privy_transfer import PrivyTransferTool, PrivyTransferPlugin


@pytest.fixture
def transfer_tool():
    """Create a configured PrivyTransferTool."""
    tool = PrivyTransferTool()
    tool.configure(
        {
            "tools": {
                "privy_transfer": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                    "signing_key": "wallet-auth:test-signing-key",
                    "rpc_url": "https://api.mainnet-beta.solana.com",
                    "fee_payer": "FeePayer123...base58",
                }
            }
        }
    )
    return tool


@pytest.fixture
def transfer_tool_incomplete():
    """Create an incomplete PrivyTransferTool."""
    tool = PrivyTransferTool()
    tool.configure(
        {
            "tools": {
                "privy_transfer": {
                    "app_id": "test-app-id",
                    # Missing other required fields
                }
            }
        }
    )
    return tool


class TestPrivyTransferToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, transfer_tool):
        """Should have correct tool name."""
        assert transfer_tool.name == "privy_transfer"

    def test_schema_has_required_properties(self, transfer_tool):
        """Should include all required properties."""
        schema = transfer_tool.get_schema()
        assert "wallet_id" in schema["properties"]
        assert "wallet_public_key" in schema["properties"]
        assert "to_address" in schema["properties"]
        assert "amount" in schema["properties"]
        assert "mint" in schema["properties"]
        assert "memo" in schema["properties"]
        assert set(schema["required"]) == {
            "wallet_id",
            "wallet_public_key",
            "to_address",
            "amount",
            "mint",
            "memo",
        }

    def test_schema_property_types(self, transfer_tool):
        """Should have correct property types."""
        schema = transfer_tool.get_schema()
        assert schema["properties"]["wallet_id"]["type"] == "string"
        assert schema["properties"]["wallet_public_key"]["type"] == "string"
        assert schema["properties"]["to_address"]["type"] == "string"
        assert schema["properties"]["amount"]["type"] == "number"
        assert schema["properties"]["mint"]["type"] == "string"
        assert schema["properties"]["memo"]["type"] == "string"


class TestPrivyTransferToolConfigure:
    """Test configuration method."""

    def test_configure_stores_privy_config(self, transfer_tool):
        """Should store Privy configuration."""
        assert transfer_tool.app_id == "test-app-id"
        assert transfer_tool.app_secret == "test-app-secret"
        assert transfer_tool.signing_key == "wallet-auth:test-signing-key"

    def test_configure_stores_rpc_config(self, transfer_tool):
        """Should store RPC configuration."""
        assert transfer_tool.rpc_url == "https://api.mainnet-beta.solana.com"
        assert transfer_tool.fee_payer == "FeePayer123...base58"


class TestPrivyTransferToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_missing_wallet_params(self, transfer_tool):
        """Should return error when wallet params are missing."""
        result = await transfer_tool.execute(
            wallet_id="",
            wallet_public_key="",
            to_address="Recipient123...abc",
            amount=1.0,
            mint="So11111111111111111111111111111111111111112",
        )

        assert result["status"] == "error"
        assert (
            "wallet_id" in result["message"].lower()
            or "wallet_public_key" in result["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_execute_missing_config_error(self, transfer_tool_incomplete):
        """Should return error when config is incomplete."""
        result = await transfer_tool_incomplete.execute(
            wallet_id="wallet-123",
            wallet_public_key="WalletPubkey123",
            to_address="Recipient123...abc",
            amount=1.0,
            mint="So11111111111111111111111111111111111111112",
        )

        assert result["status"] == "error"
        assert "config" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_success(self, transfer_tool):
        """Should return success on successful transfer."""
        with (
            patch("solana_agent.tools.privy_transfer.SolanaWalletClient") as MockWallet,
            patch(
                "solana_agent.tools.privy_transfer.TokenTransferManager"
            ) as MockTransfer,
            patch(
                "solana_agent.tools.privy_transfer.privy_sign_and_send",
                new_callable=AsyncMock,
                return_value={"signature": "TxSig123...abc"},
            ),
        ):
            # Setup mocks
            mock_wallet_instance = MagicMock()
            MockWallet.return_value = mock_wallet_instance

            mock_tx = MagicMock()
            mock_tx.__bytes__ = MagicMock(return_value=b"transaction_bytes")
            MockTransfer.transfer = AsyncMock(return_value=mock_tx)

            result = await transfer_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123...abc",
                to_address="Recipient123...abc",
                amount=1.0,
                mint="So11111111111111111111111111111111111111112",
            )

            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, transfer_tool):
        """Should return error on exception."""
        with patch(
            "solana_agent.tools.privy_transfer.SolanaWalletClient",
            side_effect=Exception("Connection error"),
        ):
            result = await transfer_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123...abc",
                to_address="Recipient123...abc",
                amount=1.0,
                mint="So11111111111111111111111111111111111111112",
            )

            assert result["status"] == "error"
            assert "Connection error" in result["message"]


class TestPrivyTransferPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = PrivyTransferPlugin()
        assert plugin.name == "privy_transfer"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = PrivyTransferPlugin()
        assert "transfer" in plugin.description.lower()

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = PrivyTransferPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = PrivyTransferPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

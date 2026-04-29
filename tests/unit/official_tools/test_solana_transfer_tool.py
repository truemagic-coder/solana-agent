"""
Tests for Solana Transfer Tool.

Tests the SolanaTransferTool which transfers SOL and SPL tokens
using a Solana keypair.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.solana_transfer import SolanaTransferTool, SolanaTransferPlugin


@pytest.fixture
def transfer_tool():
    """Create a configured SolanaTransferTool."""
    tool = SolanaTransferTool()
    tool.configure(
        {
            "tools": {
                "solana_transfer": {
                    "rpc_url": "https://api.mainnet-beta.solana.com",
                    "private_key": "5jGR...base58privatekey",
                }
            }
        }
    )
    return tool


@pytest.fixture
def transfer_tool_helius():
    """Create a SolanaTransferTool with Helius RPC."""
    tool = SolanaTransferTool()
    tool.configure(
        {
            "tools": {
                "solana_transfer": {
                    "rpc_url": "https://rpc.helius.xyz/?api-key=test",
                    "private_key": "5jGR...base58privatekey",
                }
            }
        }
    )
    return tool


@pytest.fixture
def transfer_tool_no_config():
    """Create an unconfigured SolanaTransferTool."""
    tool = SolanaTransferTool()
    tool.configure({"tools": {"solana_transfer": {}}})
    return tool


class TestSolanaTransferToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, transfer_tool):
        """Should have correct tool name."""
        assert transfer_tool.name == "solana_transfer"

    def test_schema_has_required_properties(self, transfer_tool):
        """Should include all required properties."""
        schema = transfer_tool.get_schema()
        assert "to_address" in schema["properties"]
        assert "amount" in schema["properties"]
        assert "mint" in schema["properties"]
        assert set(schema["required"]) == {"to_address", "amount", "mint"}

    def test_schema_property_types(self, transfer_tool):
        """Should have correct property types."""
        schema = transfer_tool.get_schema()
        assert schema["properties"]["to_address"]["type"] == "string"
        assert schema["properties"]["amount"]["type"] == "number"
        assert schema["properties"]["mint"]["type"] == "string"


class TestSolanaTransferToolConfigure:
    """Test configuration method."""

    def test_configure_stores_rpc_url(self, transfer_tool):
        """Should store RPC URL from config."""
        assert transfer_tool._rpc_url == "https://api.mainnet-beta.solana.com"

    def test_configure_stores_private_key(self, transfer_tool):
        """Should store private key from config."""
        assert transfer_tool._private_key == "5jGR...base58privatekey"

    def test_configure_detects_helius(self, transfer_tool_helius):
        """Should detect Helius RPC URL."""
        assert "helius" in transfer_tool_helius._rpc_url.lower()


class TestSolanaTransferToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_missing_rpc_url_error(self, transfer_tool_no_config):
        """Should return error when RPC URL is missing."""
        result = await transfer_tool_no_config.execute(
            to_address="Recipient123...abc",
            amount=1.0,
            mint="So11111111111111111111111111111111111111112",
        )

        assert result["status"] == "error"
        assert "rpc" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_private_key_error(self):
        """Should return error when private key is missing."""
        tool = SolanaTransferTool()
        tool.configure(
            {
                "tools": {
                    "solana_transfer": {
                        "rpc_url": "https://api.mainnet-beta.solana.com",
                        # Missing private_key
                    }
                }
            }
        )

        result = await tool.execute(
            to_address="Recipient123...abc",
            amount=1.0,
            mint="So11111111111111111111111111111111111111112",
        )

        assert result["status"] == "error"
        assert (
            "private" in result["message"].lower() or "key" in result["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_execute_success(self, transfer_tool):
        """Should return success on successful transfer."""
        mock_signature = MagicMock()
        mock_signature.value = "TxSignature123...abc"

        with (
            patch("solana_agent.tools.solana_transfer.Keypair") as MockKeypair,
            patch(
                "solana_agent.tools.solana_transfer.SolanaWalletClient"
            ) as MockWallet,
            patch(
                "solana_agent.tools.solana_transfer.TokenTransferManager"
            ) as MockTransfer,
        ):
            # Setup mocks
            mock_keypair = MagicMock()
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_wallet_instance = MagicMock()
            mock_wallet_instance.client.send_transaction = AsyncMock(
                return_value=mock_signature
            )
            MockWallet.return_value = mock_wallet_instance

            mock_tx = MagicMock()
            MockTransfer.transfer = AsyncMock(return_value=mock_tx)

            result = await transfer_tool.execute(
                to_address="Recipient123...abc",
                amount=1.0,
                mint="So11111111111111111111111111111111111111112",
            )

            assert result["status"] == "success"
            assert result["result"] == "TxSignature123...abc"

    @pytest.mark.asyncio
    async def test_execute_transfer_exception(self, transfer_tool):
        """Should return error on transfer exception."""
        with (
            patch("solana_agent.tools.solana_transfer.Keypair") as MockKeypair,
            patch(
                "solana_agent.tools.solana_transfer.SolanaWalletClient"
            ) as MockWalletClient,
            patch(
                "solana_agent.tools.solana_transfer.TokenTransferManager"
            ) as MockTransferManager,
        ):
            mock_keypair = MagicMock()
            MockKeypair.from_base58_string.return_value = mock_keypair
            MockWalletClient.return_value = MagicMock()
            MockTransferManager.transfer = AsyncMock(
                side_effect=Exception("Transfer failed")
            )

            result = await transfer_tool.execute(
                to_address="Recipient123...abc",
                amount=1.0,
                mint="So11111111111111111111111111111111111111112",
            )

            assert result["status"] == "error"
            assert "Transfer failed" in result["message"]


class TestSolanaTransferPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = SolanaTransferPlugin()
        assert plugin.name == "solana_transfer"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = SolanaTransferPlugin()
        assert "transfer" in plugin.description.lower()

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = SolanaTransferPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = SolanaTransferPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

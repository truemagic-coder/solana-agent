"""
Tests for Solana Ultra Tool.

Tests the SolanaUltraTool which swaps tokens using Jupiter Ultra API
with a Solana keypair. Transactions are sent via Helius RPC instead
of Jupiter's /execute endpoint for reliability.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.solana_ultra import SolanaUltraTool, SolanaUltraPlugin


@pytest.fixture
def ultra_tool():
    """Create a configured SolanaUltraTool."""
    tool = SolanaUltraTool()
    tool.configure(
        {
            "tools": {
                "solana_ultra": {
                    "private_key": "5jGR...base58privatekey",
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
def ultra_tool_with_payer():
    """Create a SolanaUltraTool with gasless payer."""
    tool = SolanaUltraTool()
    tool.configure(
        {
            "tools": {
                "solana_ultra": {
                    "private_key": "5jGR...base58privatekey",
                    "jupiter_api_key": "test-jupiter-key",
                    "payer_private_key": "PayerPrivateKey...base58",
                    "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                }
            }
        }
    )
    return tool


@pytest.fixture
def ultra_tool_no_key():
    """Create an unconfigured SolanaUltraTool."""
    tool = SolanaUltraTool()
    tool.configure({"tools": {"solana_ultra": {}}})
    return tool


class TestSolanaUltraToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, ultra_tool):
        """Should have correct tool name."""
        assert ultra_tool.name == "solana_ultra"

    def test_schema_has_required_properties(self, ultra_tool):
        """Should include all required properties."""
        schema = ultra_tool.get_schema()
        assert "input_mint" in schema["properties"]
        assert "output_mint" in schema["properties"]
        assert "amount" in schema["properties"]
        assert set(schema["required"]) == {"input_mint", "output_mint", "amount"}

    def test_schema_property_types(self, ultra_tool):
        """Should have correct property types."""
        schema = ultra_tool.get_schema()
        assert schema["properties"]["input_mint"]["type"] == "string"
        assert schema["properties"]["output_mint"]["type"] == "string"
        assert schema["properties"]["amount"]["type"] == "integer"


class TestSolanaUltraToolConfigure:
    """Test configuration method."""

    def test_configure_stores_private_key(self, ultra_tool):
        """Should store private key from config."""
        assert ultra_tool._private_key == "5jGR...base58privatekey"

    def test_configure_stores_jupiter_config(self, ultra_tool):
        """Should store Jupiter configuration."""
        assert ultra_tool._jupiter_api_key == "test-jupiter-key"
        assert ultra_tool._referral_account == "RefAcct123"
        assert ultra_tool._referral_fee == 50

    def test_configure_stores_payer_key(self, ultra_tool_with_payer):
        """Should store payer private key for gasless."""
        assert ultra_tool_with_payer._payer_private_key == "PayerPrivateKey...base58"

    def test_configure_stores_rpc_url(self, ultra_tool):
        """Should store RPC URL for direct transaction sending."""
        assert ultra_tool._rpc_url == "https://mainnet.helius-rpc.com/?api-key=test-key"


class TestSolanaUltraToolExecute:
    """Test execute method using RPC-based transaction sending."""

    @pytest.mark.asyncio
    async def test_execute_missing_private_key_error(self, ultra_tool_no_key):
        """Should return error when private key is missing."""
        result = await ultra_tool_no_key.execute(
            input_mint="So11111111111111111111111111111111111111112",
            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            amount=1000000000,
        )

        assert result["status"] == "error"
        assert (
            "private" in result["message"].lower() or "key" in result["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_execute_no_transaction_error(self, ultra_tool):
        """Should return error when no transaction returned."""
        mock_order = MagicMock()
        mock_order.transaction = None

        with (
            patch("solana_agent.tools.solana_ultra.Keypair") as MockKeypair,
            patch("solana_agent.tools.solana_ultra.JupiterUltra") as MockUltra,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "TakerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_ultra_instance

            result = await ultra_tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "transaction" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_success(self, ultra_tool):
        """Should return success on successful swap via RPC."""
        mock_order = MagicMock()
        mock_order.transaction = "base64encodedtransaction"
        mock_order.request_id = "request-123"
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
            "signature": "TxSig123...abc",
        }

        with (
            patch("solana_agent.tools.solana_ultra.Keypair") as MockKeypair,
            patch("solana_agent.tools.solana_ultra.JupiterUltra") as MockUltra,
            patch.object(
                ultra_tool,
                "_sign_and_execute",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ),
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "TakerPubkey123"
            mock_keypair.sign_message = MagicMock(return_value=b"signature")
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_ultra_instance

            result = await ultra_tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "success"
            assert result["signature"] == "TxSig123...abc"
            assert result["swap_type"] == "ExactIn"
            assert result["gasless"] is False

    @pytest.mark.asyncio
    async def test_execute_sign_and_execute_failure(self, ultra_tool):
        """Should return error when _sign_and_execute fails."""
        mock_order = MagicMock()
        mock_order.transaction = "base64encodedtransaction"
        mock_order.request_id = "request-123"
        mock_order.price_impact = -0.5
        mock_order.in_usd_value = 50.0
        mock_order.out_usd_value = 49.75

        # Mock _sign_and_execute to return error
        mock_exec_result = {
            "status": "error",
            "message": "Failed to send transaction",
        }

        with (
            patch("solana_agent.tools.solana_ultra.Keypair") as MockKeypair,
            patch("solana_agent.tools.solana_ultra.JupiterUltra") as MockUltra,
            patch.object(
                ultra_tool,
                "_sign_and_execute",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ),
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "TakerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_ultra_instance

            result = await ultra_tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "Failed to send transaction" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_rpc_blockhash_failure(self, ultra_tool):
        """Should return error when blockhash fetch fails."""
        mock_order = MagicMock()
        mock_order.transaction = "base64encodedtransaction"
        mock_order.request_id = "request-123"
        mock_order.price_impact = -0.5
        mock_order.in_usd_value = 50.0
        mock_order.out_usd_value = 49.75

        # Mock _sign_and_execute to return blockhash error
        mock_exec_result = {
            "status": "error",
            "message": "Failed to get blockhash: RPC error",
        }

        with (
            patch("solana_agent.tools.solana_ultra.Keypair") as MockKeypair,
            patch("solana_agent.tools.solana_ultra.JupiterUltra") as MockUltra,
            patch.object(
                ultra_tool,
                "_sign_and_execute",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ),
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "TakerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_ultra_instance = AsyncMock()
            mock_ultra_instance.get_order = AsyncMock(return_value=mock_order)
            MockUltra.return_value = mock_ultra_instance

            result = await ultra_tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "blockhash" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, ultra_tool):
        """Should return error on exception."""
        with patch("solana_agent.tools.solana_ultra.Keypair") as MockKeypair:
            MockKeypair.from_base58_string.side_effect = Exception("Invalid key")

            result = await ultra_tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "Invalid key" in result["message"]


class TestSolanaUltraPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = SolanaUltraPlugin()
        assert plugin.name == "solana_ultra"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = SolanaUltraPlugin()
        assert (
            "swap" in plugin.description.lower()
            or "ultra" in plugin.description.lower()
        )

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = SolanaUltraPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = SolanaUltraPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

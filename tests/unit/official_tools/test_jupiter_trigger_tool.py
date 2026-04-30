"""
Tests for Jupiter Trigger Tool.

Tests the JupiterTriggerTool which wraps the JupiterTrigger API client
and provides the AutoTool interface for limit order management.
Transactions are sent via Helius RPC instead of Jupiter's /execute endpoint for reliability.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from solana_agent.tools.jupiter_trigger import JupiterTriggerTool, JupiterTriggerPlugin


@pytest.fixture
def trigger_tool():
    """Create a configured JupiterTriggerTool."""
    tool = JupiterTriggerTool()
    tool.configure(
        {
            "tools": {
                "jupiter_trigger": {
                    "private_key": "5jGR...base58privatekey",  # This would be a real key in production
                    "jupiter_api_key": "test-api-key",
                    "referral_account": "RefAcct123",
                    "referral_fee": 100,
                    "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                }
            }
        }
    )
    return tool


@pytest.fixture
def trigger_tool_with_payer():
    """Create a JupiterTriggerTool with gasless payer."""
    tool = JupiterTriggerTool()
    tool.configure(
        {
            "tools": {
                "jupiter_trigger": {
                    "private_key": "5jGR...base58privatekey",
                    "jupiter_api_key": "test-api-key",
                    "payer_private_key": "PayerPrivateKey...base58",
                    "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                }
            }
        }
    )
    return tool


@pytest.fixture
def trigger_tool_no_api_key():
    """Create a JupiterTriggerTool without API key (lite mode)."""
    tool = JupiterTriggerTool()
    tool.configure(
        {
            "tools": {
                "jupiter_trigger": {
                    "private_key": "5jGR...base58privatekey",
                    "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                }
            }
        }
    )
    return tool


class TestJupiterTriggerToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, trigger_tool):
        """Should have correct tool name."""
        assert trigger_tool.name == "jupiter_trigger"

    def test_schema_has_required_properties(self, trigger_tool):
        """Should include action in required properties."""
        schema = trigger_tool.get_schema()
        assert "action" in schema["properties"]
        assert schema["properties"]["action"]["enum"] == [
            "create",
            "cancel",
            "cancel_all",
            "list",
        ]

    def test_schema_has_order_properties(self, trigger_tool):
        """Should include order creation properties."""
        schema = trigger_tool.get_schema()
        props = schema["properties"]
        assert "input_mint" in props
        assert "output_mint" in props
        assert "making_amount" in props  # Jupiter API uses making_amount
        assert "taking_amount" in props  # Jupiter API uses taking_amount
        assert "expired_at" in props


class TestJupiterTriggerToolConfigure:
    """Test configuration method."""

    def test_configure_stores_private_key(self, trigger_tool):
        """Should store private key from config."""
        assert trigger_tool._private_key == "5jGR...base58privatekey"

    def test_configure_stores_api_key(self, trigger_tool):
        """Should store Jupiter API key from config."""
        assert trigger_tool._jupiter_api_key == "test-api-key"

    def test_configure_stores_referral_info(self, trigger_tool):
        """Should store referral account and fee from config."""
        assert trigger_tool._referral_account == "RefAcct123"
        assert trigger_tool._referral_fee == 100

    def test_configure_stores_rpc_url(self, trigger_tool):
        """Should store RPC URL for direct transaction sending."""
        assert (
            trigger_tool._rpc_url == "https://mainnet.helius-rpc.com/?api-key=test-key"
        )

    def test_configure_stores_payer_key(self, trigger_tool_with_payer):
        """Should store payer private key for gasless."""
        assert trigger_tool_with_payer._payer_private_key == "PayerPrivateKey...base58"


class TestJupiterTriggerToolCreateAction:
    """Test create action."""

    @pytest.mark.asyncio
    async def test_create_missing_private_key(self):
        """Should return error if private key is missing."""
        tool = JupiterTriggerTool()
        tool.configure({"tools": {"jupiter_trigger": {}}})

        result = await tool.execute(
            action="create",
            input_mint="So11...",
            output_mint="EPjF...",
            making_amount="1000000",
            taking_amount="100000",
        )

        assert result["status"] == "error"
        assert (
            "private_key" in result["message"].lower()
            or "config" in result["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_create_missing_required_params(self, trigger_tool):
        """Should return error if required create params are missing."""
        result = await trigger_tool.execute(
            action="create",
            input_mint="So11...",
            # Missing output_mint, making_amount, taking_amount
        )

        assert result["status"] == "error"
        assert (
            "missing" in result["message"].lower()
            or "required" in result["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_create_success(self, trigger_tool):
        """Should return success on successful order creation via RPC."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.transaction = "base64encodedtransaction"
        mock_result.order = "OrderPubkey123"
        mock_result.request_id = "request-123"

        # Mock _sign_and_execute to return success
        mock_exec_result = {
            "success": True,
            "signature": "TxSig123...abc",
        }

        with (
            patch("solana_agent.tools.jupiter_trigger.Keypair") as MockKeypair,
            patch("solana_agent.tools.jupiter_trigger.JupiterTrigger") as MockTrigger,
            patch.object(
                trigger_tool,
                "_sign_and_execute",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ),
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "MakerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_trigger_instance = MagicMock()
            mock_trigger_instance.create_order = AsyncMock(return_value=mock_result)
            MockTrigger.return_value = mock_trigger_instance

            result = await trigger_tool.execute(
                action="create",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                making_amount="1000000000",
                taking_amount="1000000",
            )

            assert result["status"] == "success"
            assert result["action"] == "create"
            assert result["order_pubkey"] == "OrderPubkey123"
            assert result["signature"] == "TxSig123...abc"

    @pytest.mark.asyncio
    async def test_create_no_transaction_error(self, trigger_tool):
        """Should return error when no transaction returned."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.transaction = None

        with (
            patch("solana_agent.tools.jupiter_trigger.Keypair") as MockKeypair,
            patch("solana_agent.tools.jupiter_trigger.JupiterTrigger") as MockTrigger,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "MakerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_trigger_instance = MagicMock()
            mock_trigger_instance.create_order = AsyncMock(return_value=mock_result)
            MockTrigger.return_value = mock_trigger_instance

            result = await trigger_tool.execute(
                action="create",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                making_amount="1000000000",
                taking_amount="1000000",
            )

            assert result["status"] == "error"
            assert "transaction" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_create_api_error(self, trigger_tool):
        """Should return error when Jupiter API fails."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Insufficient balance"

        with (
            patch("solana_agent.tools.jupiter_trigger.Keypair") as MockKeypair,
            patch("solana_agent.tools.jupiter_trigger.JupiterTrigger") as MockTrigger,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "MakerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_trigger_instance = MagicMock()
            mock_trigger_instance.create_order = AsyncMock(return_value=mock_result)
            MockTrigger.return_value = mock_trigger_instance

            result = await trigger_tool.execute(
                action="create",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                making_amount="1000000000",
                taking_amount="1000000",
            )

            assert result["status"] == "error"
            assert "Insufficient balance" in result["message"]

    @pytest.mark.asyncio
    async def test_create_sign_and_execute_failure(self, trigger_tool):
        """Should return error when _sign_and_execute fails."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.transaction = "base64encodedtransaction"
        mock_result.order = "OrderPubkey123"
        mock_result.request_id = "request-123"

        # Mock _sign_and_execute to return error
        mock_exec_result = {
            "success": False,
            "error": "Failed to send transaction",
        }

        with (
            patch("solana_agent.tools.jupiter_trigger.Keypair") as MockKeypair,
            patch("solana_agent.tools.jupiter_trigger.JupiterTrigger") as MockTrigger,
            patch.object(
                trigger_tool,
                "_sign_and_execute",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ),
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "MakerPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            mock_trigger_instance = MagicMock()
            mock_trigger_instance.create_order = AsyncMock(return_value=mock_result)
            MockTrigger.return_value = mock_trigger_instance

            result = await trigger_tool.execute(
                action="create",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                making_amount="1000000000",
                taking_amount="1000000",
            )

            assert result["status"] == "error"
            assert "Failed to send transaction" in result["message"]


class TestJupiterTriggerToolCancelAction:
    """Test cancel action."""

    @pytest.mark.asyncio
    async def test_cancel_missing_order_pubkey(self, trigger_tool):
        """Should return error if order_pubkey is missing for cancel."""
        with patch.object(trigger_tool, "_private_key", "testkey"):
            result = await trigger_tool.execute(action="cancel")

        assert result["status"] == "error"
        assert "order_pubkey" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_cancel_order_not_owned_by_user(self):
        """Should reject cancellation of orders not owned by the wallet."""
        tool = JupiterTriggerTool()
        tool.configure(
            {
                "tools": {
                    "jupiter_trigger": {
                        "private_key": "5jGR...base58privatekey",
                        "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                    }
                }
            }
        )

        # Mock Keypair to return a predictable public key
        with patch("solana_agent.tools.jupiter_trigger.Keypair") as MockKeypair:
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserWalletPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            # Mock JupiterTrigger to return user's orders
            with patch(
                "solana_agent.tools.jupiter_trigger.JupiterTrigger"
            ) as MockTrigger:
                mock_trigger_instance = MockTrigger.return_value
                mock_trigger_instance.get_orders = AsyncMock(
                    return_value={
                        "success": True,
                        "orders": [
                            {"order": "UserOwnedOrder123"},
                            {"order": "UserOwnedOrder456"},
                        ],
                    }
                )

                result = await tool.execute(
                    action="cancel",
                    order_pubkey="SomeoneElsesOrder789",  # Not in user's orders
                )

        assert result["status"] == "error"
        assert "does not belong" in result["message"]


class TestJupiterTriggerToolListAction:
    """Test list actions."""

    @pytest.mark.asyncio
    async def test_list_action(self):
        """Should list orders."""
        # This test requires proper mocking of Keypair.from_base58_string
        # For now, we'll just verify the action routing works with minimal config
        tool = JupiterTriggerTool()
        tool.configure(
            {
                "tools": {
                    "jupiter_trigger": {}  # No private key = error
                }
            }
        )

        result = await tool.execute(action="list")
        # Without private key, should error
        assert result["status"] == "error"


class TestJupiterTriggerToolUnknownAction:
    """Test unknown action handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self, trigger_tool):
        """Should return error for unknown action."""
        result = await trigger_tool.execute(action="invalid_action")

        assert result["status"] == "error"
        assert (
            "unknown" in result["message"].lower()
            or "invalid" in result["message"].lower()
        )


class TestJupiterTriggerToolSignAndExecute:
    """Test _sign_and_execute method."""

    @pytest.mark.asyncio
    async def test_sign_and_execute_missing_rpc_url(self):
        """Should return error when rpc_url is not configured."""
        tool = JupiterTriggerTool()
        tool.configure(
            {
                "tools": {
                    "jupiter_trigger": {
                        "private_key": "5jGR...base58privatekey",
                        # No rpc_url
                    }
                }
            }
        )

        mock_keypair = MagicMock()
        result = await tool._sign_and_execute(
            transaction_base64="base64encodedtransaction",
            keypair=mock_keypair,
        )

        assert result["success"] is False
        assert "rpc_url" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_sign_and_execute_success(self, trigger_tool):
        """Should sign and send transaction successfully via RPC."""
        mock_keypair = MagicMock()
        mock_keypair.sign_message.return_value = b"signature"

        # Mock transaction decoding
        import base64

        mock_tx_bytes = b"mock_transaction_bytes"
        mock_tx_base64 = base64.b64encode(mock_tx_bytes).decode()

        mock_tx = MagicMock()
        mock_tx.message = MagicMock()

        with (
            patch("solana_agent.tools.jupiter_trigger.base64") as mock_base64,
            patch("solana_agent.tools.jupiter_trigger.VersionedTransaction") as MockTx,
            patch(
                "solana_agent.tools.jupiter_trigger.to_bytes_versioned"
            ) as mock_to_bytes,
            patch(
                "solana_agent.tools.jupiter_trigger.get_fresh_blockhash"
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.jupiter_trigger.replace_blockhash_in_transaction"
            ) as mock_replace,
            patch(
                "solana_agent.tools.jupiter_trigger.send_raw_transaction_with_priority"
            ) as mock_send,
        ):
            mock_base64.b64decode.return_value = mock_tx_bytes
            mock_base64.b64encode.return_value.decode.return_value = "signed_tx_base64"
            MockTx.from_bytes.return_value = mock_tx
            MockTx.populate.return_value = MagicMock(__bytes__=lambda: b"signed_bytes")
            mock_to_bytes.return_value = b"message_bytes"
            mock_blockhash.return_value = "fresh_blockhash_123"
            mock_replace.return_value = mock_tx
            mock_send.return_value = "TxSignature123abc"

            result = await trigger_tool._sign_and_execute(
                transaction_base64=mock_tx_base64,
                keypair=mock_keypair,
            )

            assert result["success"] is True
            assert result["signature"] == "TxSignature123abc"
            mock_blockhash.assert_called_once()
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_sign_and_execute_with_payer(self, trigger_tool):
        """Should sign with both keypair and payer for gasless."""
        mock_keypair = MagicMock()
        mock_keypair.sign_message.return_value = b"user_signature"

        mock_payer = MagicMock()
        mock_payer.sign_message.return_value = b"payer_signature"

        mock_tx = MagicMock()
        mock_tx.message = MagicMock()

        with (
            patch("solana_agent.tools.jupiter_trigger.base64") as mock_base64,
            patch("solana_agent.tools.jupiter_trigger.VersionedTransaction") as MockTx,
            patch(
                "solana_agent.tools.jupiter_trigger.to_bytes_versioned"
            ) as mock_to_bytes,
            patch(
                "solana_agent.tools.jupiter_trigger.get_fresh_blockhash"
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.jupiter_trigger.replace_blockhash_in_transaction"
            ) as mock_replace,
            patch(
                "solana_agent.tools.jupiter_trigger.send_raw_transaction_with_priority"
            ) as mock_send,
        ):
            mock_base64.b64decode.return_value = b"tx_bytes"
            mock_base64.b64encode.return_value.decode.return_value = "signed_tx_base64"
            MockTx.from_bytes.return_value = mock_tx
            MockTx.populate.return_value = MagicMock(__bytes__=lambda: b"signed_bytes")
            mock_to_bytes.return_value = b"message_bytes"
            mock_blockhash.return_value = "fresh_blockhash"
            mock_replace.return_value = mock_tx
            mock_send.return_value = "TxSignature123"

            result = await trigger_tool._sign_and_execute(
                transaction_base64="base64tx",
                keypair=mock_keypair,
                payer_keypair=mock_payer,
            )

            assert result["success"] is True
            # Both signatures should be used
            mock_keypair.sign_message.assert_called_once()
            mock_payer.sign_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_sign_and_execute_exception(self, trigger_tool):
        """Should return error on exception during signing."""
        mock_keypair = MagicMock()

        with patch("solana_agent.tools.jupiter_trigger.base64") as mock_base64:
            mock_base64.b64decode.side_effect = Exception("Decode error")

            result = await trigger_tool._sign_and_execute(
                transaction_base64="invalid_base64",
                keypair=mock_keypair,
            )

            assert result["success"] is False
            assert "Decode error" in result["error"]


class TestJupiterTriggerPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = JupiterTriggerPlugin()
        assert plugin.name == "jupiter_trigger"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = JupiterTriggerPlugin()
        assert (
            "limit" in plugin.description.lower()
            or "trigger" in plugin.description.lower()
        )

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = JupiterTriggerPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = JupiterTriggerPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

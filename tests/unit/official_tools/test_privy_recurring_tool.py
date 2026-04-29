"""
Tests for Privy Recurring Tool.

Tests the PrivyRecurringTool which uses Jupiter Recurring API with
Privy delegated wallets for transaction signing.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.privy_recurring import (
    PrivyRecurringTool,
    PrivyRecurringPlugin,
    _privy_sign_transaction,
    get_plugin,
)


# Test EC private key in SEC1 format (base64-encoded DER)
# This is a valid test P-256 key generated for testing
TEST_EC_KEY_SEC1 = "MHcCAQEEIH6phbwVBTxg+QYJMSqHXcLoiTpmO163WjA8Td/+DqQ3oAoGCCqGSM49AwEHoUQDQgAEoY29/uiiWfItIYBAmejKuM17a0GackAbFG4sNs1ObTUilKQ2V/7WkTRC0xk7IgLwCRUI1e/Yk5wQFCjlajvilw=="


@pytest.fixture
def privy_recurring_tool():
    """Create a configured PrivyRecurringTool."""
    tool = PrivyRecurringTool()
    tool.configure(
        {
            "tools": {
                "privy_recurring": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                    "signing_key": f"wallet-auth:{TEST_EC_KEY_SEC1}",
                    "jupiter_api_key": "test-api-key",
                }
            }
        }
    )
    return tool


@pytest.fixture
def privy_recurring_tool_with_payer():
    """Create a PrivyRecurringTool configured with a payer key."""
    tool = PrivyRecurringTool()
    # Using a test Solana keypair (base58)
    test_payer_key = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviUkauRiTMD8DrESdrNjN8zd9mTmVhRvBJeg5vhyvgrAhG"
    tool.configure(
        {
            "tools": {
                "privy_recurring": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                    "signing_key": f"wallet-auth:{TEST_EC_KEY_SEC1}",
                    "jupiter_api_key": "test-api-key",
                    "payer_private_key": test_payer_key,
                }
            }
        }
    )
    return tool


class TestPrivySignTransaction:
    """Test _privy_sign_transaction function."""

    @pytest.mark.asyncio
    async def test_successful_sign_transaction(self):
        """Should successfully sign transaction."""
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = MagicMock()
        mock_rpc_result.data.signed_transaction = "signed-tx-base64"

        mock_privy_client = AsyncMock()
        mock_privy_client.app_id = "test-app-id"
        mock_privy_client.wallets.rpc = AsyncMock(return_value=mock_rpc_result)

        with (
            patch(
                "solana_agent.tools.privy_recurring.get_authorization_signature",
                return_value="mock-signature",
            ),
            patch(
                "solana_agent.tools.privy_recurring._convert_key_to_pkcs8_pem",
                return_value="mock-pem-key",
            ),
        ):
            result = await _privy_sign_transaction(
                privy_client=mock_privy_client,
                wallet_id="wallet-123",
                encoded_tx="encoded-tx-base64",
                signing_key="wallet-auth:test-key",
            )

            assert result == "signed-tx-base64"

    @pytest.mark.asyncio
    async def test_returns_none_on_api_error(self):
        """Should return None on API error."""
        mock_privy_client = AsyncMock()
        mock_privy_client.app_id = "test-app-id"
        mock_privy_client.wallets.rpc = AsyncMock(side_effect=Exception("API Error"))

        with patch(
            "solana_agent.tools.privy_recurring.get_authorization_signature",
            return_value="mock-signature",
        ):
            result = await _privy_sign_transaction(
                privy_client=mock_privy_client,
                wallet_id="wallet-123",
                encoded_tx="encoded-tx-base64",
                signing_key="wallet-auth:test-key",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_signed_transaction(self):
        """Should return None when response lacks signedTransaction."""
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = None

        mock_privy_client = AsyncMock()
        mock_privy_client.app_id = "test-app-id"
        mock_privy_client.wallets.rpc = AsyncMock(return_value=mock_rpc_result)

        with patch(
            "solana_agent.tools.privy_recurring.get_authorization_signature",
            return_value="mock-signature",
        ):
            result = await _privy_sign_transaction(
                privy_client=mock_privy_client,
                wallet_id="wallet-123",
                encoded_tx="encoded-tx-base64",
                signing_key="wallet-auth:test-key",
            )

            assert result is None


class TestPrivyRecurringToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, privy_recurring_tool):
        """Should have correct tool name."""
        assert privy_recurring_tool.name == "privy_recurring"

    def test_schema_has_wallet_params(self, privy_recurring_tool):
        """Should require wallet_id and wallet_public_key."""
        schema = privy_recurring_tool.get_schema()
        assert "wallet_id" in schema["properties"]
        assert "wallet_public_key" in schema["properties"]
        assert "wallet_id" in schema["required"]
        assert "wallet_public_key" in schema["required"]

    def test_schema_has_actions(self, privy_recurring_tool):
        """Should include action in required properties."""
        schema = privy_recurring_tool.get_schema()
        assert "action" in schema["properties"]
        assert schema["properties"]["action"]["enum"] == ["create", "cancel", "list"]

    def test_schema_has_dca_properties(self, privy_recurring_tool):
        """Should include DCA order creation properties."""
        schema = privy_recurring_tool.get_schema()
        props = schema["properties"]
        assert "input_mint" in props
        assert "output_mint" in props
        assert "in_amount" in props
        assert "order_count" in props
        assert "frequency" in props
        assert "min_out_amount" in props
        assert "max_out_amount" in props

    def test_schema_has_order_pubkey(self, privy_recurring_tool):
        """Should include order_pubkey for cancel action."""
        schema = privy_recurring_tool.get_schema()
        assert "order_pubkey" in schema["properties"]

    def test_schema_has_start_at(self, privy_recurring_tool):
        """Should include start_at for scheduling DCA start."""
        schema = privy_recurring_tool.get_schema()
        assert "start_at" in schema["properties"]


class TestPrivyRecurringToolConfigure:
    """Test configuration method."""

    def test_configure_stores_privy_config(self, privy_recurring_tool):
        """Should store Privy credentials from config."""
        assert privy_recurring_tool._app_id == "test-app-id"
        assert privy_recurring_tool._app_secret == "test-app-secret"
        assert privy_recurring_tool._signing_key == f"wallet-auth:{TEST_EC_KEY_SEC1}"

    def test_configure_stores_api_key(self, privy_recurring_tool):
        """Should store Jupiter API key from config."""
        assert privy_recurring_tool._jupiter_api_key == "test-api-key"

    def test_configure_stores_payer_key(self, privy_recurring_tool_with_payer):
        """Should store payer private key from config."""
        assert privy_recurring_tool_with_payer._payer_private_key is not None

    def test_configure_with_empty_config(self):
        """Should handle empty config gracefully."""
        tool = PrivyRecurringTool()
        tool.configure({"tools": {}})
        assert tool._app_id is None
        assert tool._app_secret is None


class TestPrivyRecurringToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_missing_privy_config(self):
        """Should return error if Privy config is missing."""
        tool = PrivyRecurringTool()
        tool.configure({"tools": {"privy_recurring": {}}})

        result = await tool.execute(
            wallet_id="wallet-123",
            wallet_public_key="PublicKey123",
            action="list",
        )

        assert result["status"] == "error"
        assert "config" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_wallet_params(self, privy_recurring_tool):
        """Should return error if wallet params are missing."""
        result = await privy_recurring_tool.execute(
            wallet_id="",
            wallet_public_key="",
            action="list",
        )

        assert result["status"] == "error"
        assert "wallet" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_list_success(self, privy_recurring_tool):
        """Should list active DCA orders for user's wallet."""
        with patch(
            "solana_agent.tools.privy_recurring.JupiterRecurring"
        ) as MockRecurring:
            mock_instance = MockRecurring.return_value
            mock_instance.get_orders = AsyncMock(
                return_value={"success": True, "orders": []}
            )

            result = await privy_recurring_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="PublicKey123",
                action="list",
            )

        assert result["status"] == "success"
        assert result["action"] == "list"


class TestPrivyRecurringToolCreateAction:
    """Test create action."""

    @pytest.mark.asyncio
    async def test_create_missing_required_params(self, privy_recurring_tool):
        """Should return error if required create params are missing."""
        result = await privy_recurring_tool.execute(
            wallet_id="wallet-123",
            wallet_public_key="PublicKey123",
            action="create",
            input_mint="So11...",
            # Missing output_mint, in_amount, order_count, frequency
        )

        assert result["status"] == "error"
        assert (
            "missing" in result["message"].lower()
            or "required" in result["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_create_order_success(self, privy_recurring_tool):
        """Should successfully create a DCA order."""
        mock_create_result = MagicMock()
        mock_create_result.success = True
        mock_create_result.transaction = "mock-tx-base64"
        mock_create_result.request_id = "req-123"
        mock_create_result.order = "order-pubkey-123"

        with (
            patch(
                "solana_agent.tools.privy_recurring.JupiterRecurring"
            ) as MockRecurring,
            patch(
                "solana_agent.tools.privy_recurring._privy_sign_transaction",
                new_callable=AsyncMock,
                return_value="signed-tx-base64",
            ),
        ):
            mock_instance = MockRecurring.return_value
            mock_instance.create_order = AsyncMock(return_value=mock_create_result)

            mock_exec_result = MagicMock()
            mock_exec_result.success = True
            mock_exec_result.signature = "tx-sig-456"
            mock_instance.execute = AsyncMock(return_value=mock_exec_result)

            result = await privy_recurring_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="PublicKey123",
                action="create",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                in_amount="1000000000",
                order_count=10,
                frequency="3600",
            )

        assert result["status"] == "success"
        assert result["action"] == "create"
        assert result["order_pubkey"] == "order-pubkey-123"

    @pytest.mark.asyncio
    async def test_create_order_no_transaction(self, privy_recurring_tool):
        """Should return error when Jupiter returns no transaction."""
        mock_create_result = MagicMock()
        mock_create_result.success = True
        mock_create_result.transaction = None
        mock_create_result.request_id = "req-123"

        with patch(
            "solana_agent.tools.privy_recurring.JupiterRecurring"
        ) as MockRecurring:
            mock_instance = MockRecurring.return_value
            mock_instance.create_order = AsyncMock(return_value=mock_create_result)

            result = await privy_recurring_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="PublicKey123",
                action="create",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                in_amount="1000000000",
                order_count=10,
                frequency="3600",
            )

        assert result["status"] == "error"
        assert "transaction" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_create_order_jupiter_error(self, privy_recurring_tool):
        """Should return error when Jupiter API fails."""
        mock_create_result = MagicMock()
        mock_create_result.success = False
        mock_create_result.error = "Insufficient balance"

        with patch(
            "solana_agent.tools.privy_recurring.JupiterRecurring"
        ) as MockRecurring:
            mock_instance = MockRecurring.return_value
            mock_instance.create_order = AsyncMock(return_value=mock_create_result)

            result = await privy_recurring_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="PublicKey123",
                action="create",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                in_amount="1000000000",
                order_count=10,
                frequency="3600",
            )

        assert result["status"] == "error"
        assert "Insufficient balance" in result["message"]

    @pytest.mark.asyncio
    async def test_create_order_signing_failure(self, privy_recurring_tool):
        """Should return error when signing fails."""
        mock_create_result = MagicMock()
        mock_create_result.success = True
        mock_create_result.transaction = "mock-tx-base64"
        mock_create_result.request_id = "req-123"
        mock_create_result.order = "order-pubkey-123"

        with (
            patch(
                "solana_agent.tools.privy_recurring.JupiterRecurring"
            ) as MockRecurring,
            patch(
                "solana_agent.tools.privy_recurring._privy_sign_transaction",
                new_callable=AsyncMock,
                return_value=None,  # Signing failed
            ),
        ):
            mock_instance = MockRecurring.return_value
            mock_instance.create_order = AsyncMock(return_value=mock_create_result)

            result = await privy_recurring_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="PublicKey123",
                action="create",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                in_amount="1000000000",
                order_count=10,
                frequency="3600",
            )

        assert result["status"] == "error"
        assert "sign" in result["message"].lower()


class TestPrivyRecurringToolCancelAction:
    """Test cancel action."""

    @pytest.mark.asyncio
    async def test_cancel_missing_order_pubkey(self, privy_recurring_tool):
        """Should return error if order_pubkey is missing for cancel."""
        result = await privy_recurring_tool.execute(
            wallet_id="wallet-123",
            wallet_public_key="PublicKey123",
            action="cancel",
        )

        assert result["status"] == "error"
        assert "order_pubkey" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_cancel_order_not_owned_by_user(self, privy_recurring_tool):
        """Should reject cancellation of orders not owned by the user."""
        with patch(
            "solana_agent.tools.privy_recurring.JupiterRecurring"
        ) as MockRecurring:
            mock_recurring_instance = MockRecurring.return_value
            # User's orders - does NOT include the order they're trying to cancel
            mock_recurring_instance.get_orders = AsyncMock(
                return_value={
                    "success": True,
                    "orders": [
                        {"order": "UserOwnedDCA123"},
                        {"order": "UserOwnedDCA456"},
                    ],
                }
            )

            result = await privy_recurring_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="UserPublicKey123",
                action="cancel",
                order_pubkey="SomeoneElsesDCA789",  # Not in user's orders
            )

        assert result["status"] == "error"
        assert "does not belong" in result["message"]

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, privy_recurring_tool):
        """Should successfully cancel a DCA order."""
        mock_cancel_result = MagicMock()
        mock_cancel_result.success = True
        mock_cancel_result.transaction = "cancel-tx-base64"
        mock_cancel_result.request_id = "req-cancel-123"

        with (
            patch(
                "solana_agent.tools.privy_recurring.JupiterRecurring"
            ) as MockRecurring,
            patch(
                "solana_agent.tools.privy_recurring._privy_sign_transaction",
                new_callable=AsyncMock,
                return_value="signed-cancel-tx-base64",
            ),
        ):
            mock_instance = MockRecurring.return_value
            mock_instance.get_orders = AsyncMock(
                return_value={
                    "success": True,
                    "orders": [{"order": "UserOwnedDCA123"}],
                }
            )
            mock_instance.cancel_order = AsyncMock(return_value=mock_cancel_result)

            mock_exec_result = MagicMock()
            mock_exec_result.success = True
            mock_exec_result.signature = "cancel-sig-789"
            mock_instance.execute = AsyncMock(return_value=mock_exec_result)

            result = await privy_recurring_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="UserPublicKey123",
                action="cancel",
                order_pubkey="UserOwnedDCA123",
            )

        assert result["status"] == "success"
        assert result["action"] == "cancel"
        assert result["order_pubkey"] == "UserOwnedDCA123"


class TestPrivyRecurringToolListAction:
    """Test list action."""

    @pytest.mark.asyncio
    async def test_list_orders_success(self, privy_recurring_tool):
        """Should successfully list DCA orders."""
        with patch(
            "solana_agent.tools.privy_recurring.JupiterRecurring"
        ) as MockRecurring:
            mock_instance = MockRecurring.return_value
            mock_instance.get_orders = AsyncMock(
                return_value={
                    "success": True,
                    "orders": [
                        {
                            "order": "order-123",
                            "inputMint": "So11...",
                            "outputMint": "EPj...",
                            "depositAmount": "1000000000",
                            "orderCount": 10,
                            "executedCount": 3,
                            "frequency": "3600",
                            "nextExecution": "2024-01-02T00:00:00Z",
                            "createdAt": "2024-01-01T00:00:00Z",
                        }
                    ],
                }
            )

            result = await privy_recurring_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="UserPublicKey123",
                action="list",
            )

        assert result["status"] == "success"
        assert result["action"] == "list"
        assert result["order_count"] == 1
        assert len(result["orders"]) == 1
        assert result["orders"][0]["order_pubkey"] == "order-123"
        assert result["orders"][0]["executed_count"] == 3

    @pytest.mark.asyncio
    async def test_list_orders_empty(self, privy_recurring_tool):
        """Should handle empty order list."""
        with patch(
            "solana_agent.tools.privy_recurring.JupiterRecurring"
        ) as MockRecurring:
            mock_instance = MockRecurring.return_value
            mock_instance.get_orders = AsyncMock(
                return_value={"success": True, "orders": []}
            )

            result = await privy_recurring_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="UserPublicKey123",
                action="list",
            )

        assert result["status"] == "success"
        assert result["order_count"] == 0
        assert "no active" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_list_orders_api_failure(self, privy_recurring_tool):
        """Should handle API failure when listing orders."""
        with patch(
            "solana_agent.tools.privy_recurring.JupiterRecurring"
        ) as MockRecurring:
            mock_instance = MockRecurring.return_value
            mock_instance.get_orders = AsyncMock(
                return_value={"success": False, "error": "API unavailable"}
            )

            result = await privy_recurring_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="UserPublicKey123",
                action="list",
            )

        assert result["status"] == "error"
        assert "API unavailable" in result["message"]


class TestPrivyRecurringToolUnknownAction:
    """Test unknown action handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self, privy_recurring_tool):
        """Should return error for unknown action."""
        result = await privy_recurring_tool.execute(
            wallet_id="wallet-123",
            wallet_public_key="PublicKey123",
            action="invalid_action",
        )

        assert result["status"] == "error"
        assert (
            "unknown" in result["message"].lower()
            or "invalid" in result["message"].lower()
        )


class TestPrivyRecurringPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = PrivyRecurringPlugin()
        assert plugin.name == "privy_recurring"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = PrivyRecurringPlugin()
        assert (
            "recurring" in plugin.description.lower()
            or "dca" in plugin.description.lower()
        )

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = PrivyRecurringPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = PrivyRecurringPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

    def test_plugin_configure(self):
        """Should configure the tool after initialization."""
        plugin = PrivyRecurringPlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)

        config = {
            "tools": {
                "privy_recurring": {
                    "app_id": "test-app",
                    "app_secret": "test-secret",
                    "signing_key": f"wallet-auth:{TEST_EC_KEY_SEC1}",
                }
            }
        }
        plugin.configure(config)

        assert plugin.config == config
        assert plugin._tool._app_id == "test-app"


class TestGetPlugin:
    """Test get_plugin function."""

    def test_get_plugin_returns_instance(self):
        """Should return a PrivyRecurringPlugin instance."""
        plugin = get_plugin()
        assert isinstance(plugin, PrivyRecurringPlugin)
        assert plugin.name == "privy_recurring"

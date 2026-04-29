"""
Tests for Jupiter Recurring Tool.

Tests the JupiterRecurringTool which wraps the JupiterRecurring API client
and provides the AutoTool interface for DCA order management.
"""

import pytest
from unittest.mock import patch

from solana_agent.tools.jupiter_recurring import JupiterRecurringTool


@pytest.fixture
def recurring_tool():
    """Create a configured JupiterRecurringTool."""
    tool = JupiterRecurringTool()
    tool.configure(
        {
            "tools": {
                "jupiter_recurring": {
                    "private_key": "5jGR...base58privatekey",
                    "jupiter_api_key": "test-api-key",
                }
            }
        }
    )
    return tool


class TestJupiterRecurringToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, recurring_tool):
        """Should have correct tool name."""
        assert recurring_tool.name == "jupiter_recurring"

    def test_schema_has_required_properties(self, recurring_tool):
        """Should include action in required properties."""
        schema = recurring_tool.get_schema()
        assert "action" in schema["properties"]
        assert schema["properties"]["action"]["enum"] == ["create", "cancel", "list"]

    def test_schema_has_dca_properties(self, recurring_tool):
        """Should include DCA order creation properties."""
        schema = recurring_tool.get_schema()
        props = schema["properties"]
        assert "input_mint" in props
        assert "output_mint" in props
        assert "in_amount" in props
        assert "order_count" in props
        assert "frequency" in props
        assert "min_out_amount" in props
        assert "max_out_amount" in props


class TestJupiterRecurringToolConfigure:
    """Test configuration method."""

    def test_configure_stores_private_key(self, recurring_tool):
        """Should store private key from config."""
        assert recurring_tool._private_key == "5jGR...base58privatekey"

    def test_configure_stores_api_key(self, recurring_tool):
        """Should store Jupiter API key from config."""
        assert recurring_tool._jupiter_api_key == "test-api-key"


class TestJupiterRecurringToolCreateAction:
    """Test create action."""

    @pytest.mark.asyncio
    async def test_create_missing_private_key(self):
        """Should return error if private key is missing."""
        tool = JupiterRecurringTool()
        tool.configure({"tools": {"jupiter_recurring": {}}})

        result = await tool.execute(
            action="create",
            input_mint="So11...",
            output_mint="EPjF...",
            in_amount="1000000000",
            order_count=10,
            frequency="3600",
        )

        assert result["status"] == "error"
        assert (
            "private_key" in result["message"].lower()
            or "config" in result["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_create_missing_required_params(self, recurring_tool):
        """Should return error if required create params are missing."""
        result = await recurring_tool.execute(
            action="create",
            input_mint="So11...",
            # Missing output_mint, in_amount, order_count, frequency
        )

        assert result["status"] == "error"
        assert (
            "missing" in result["message"].lower()
            or "required" in result["message"].lower()
        )


class TestJupiterRecurringToolCancelAction:
    """Test cancel action."""

    @pytest.mark.asyncio
    async def test_cancel_missing_order_pubkey(self, recurring_tool):
        """Should return error if order_pubkey is missing for cancel."""
        with patch.object(recurring_tool, "_private_key", "testkey"):
            result = await recurring_tool.execute(action="cancel")

        assert result["status"] == "error"
        assert "order_pubkey" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_cancel_order_not_owned_by_user(self):
        """Should reject cancellation of orders not owned by the wallet."""
        from unittest.mock import MagicMock, AsyncMock

        tool = JupiterRecurringTool()
        tool.configure(
            {
                "tools": {
                    "jupiter_recurring": {
                        "private_key": "5jGR...base58privatekey",
                    }
                }
            }
        )

        # Mock Keypair to return a predictable public key
        with patch("solana_agent.tools.jupiter_recurring.Keypair") as MockKeypair:
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserWalletPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            # Mock JupiterRecurring to return user's orders
            with patch(
                "solana_agent.tools.jupiter_recurring.JupiterRecurring"
            ) as MockRecurring:
                mock_recurring_instance = MockRecurring.return_value
                mock_recurring_instance.get_orders = AsyncMock(
                    return_value={
                        "success": True,
                        "orders": [
                            {"order": "UserOwnedDCA123"},
                            {"order": "UserOwnedDCA456"},
                        ],
                    }
                )

                result = await tool.execute(
                    action="cancel",
                    order_pubkey="SomeoneElsesDCA789",  # Not in user's orders
                )

        assert result["status"] == "error"
        assert "does not belong" in result["message"]


class TestJupiterRecurringToolListAction:
    """Test list action."""

    @pytest.mark.asyncio
    async def test_list_action(self):
        """Should list active DCA orders."""
        # This test requires proper mocking of Keypair.from_base58_string
        # For now, we verify the action routing works
        tool = JupiterRecurringTool()
        tool.configure(
            {
                "tools": {
                    "jupiter_recurring": {}  # No private key = error
                }
            }
        )

        result = await tool.execute(action="list")
        # Without private key, should error
        assert result["status"] == "error"


class TestJupiterRecurringToolUnknownAction:
    """Test unknown action handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self, recurring_tool):
        """Should return error for unknown action."""
        result = await recurring_tool.execute(action="invalid_action")

        assert result["status"] == "error"
        assert (
            "unknown" in result["message"].lower()
            or "invalid" in result["message"].lower()
        )

"""
Tests for Privy Wallet Address Tool.

Tests the PrivyWalletAddressCheckerTool which gets wallet addresses
for Privy delegated wallets.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.privy_wallet_address import (
    PrivyWalletAddressCheckerTool,
    PrivyWalletAddressCheckerPlugin,
    get_privy_embedded_wallet_address,
)


@pytest.fixture
def wallet_tool():
    """Create a configured PrivyWalletAddressCheckerTool."""
    tool = PrivyWalletAddressCheckerTool()
    tool.configure(
        {
            "tools": {
                "privy_balance": {  # Note: config key is privy_balance
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                }
            }
        }
    )
    return tool


@pytest.fixture
def wallet_tool_incomplete():
    """Create an incomplete PrivyWalletAddressCheckerTool."""
    tool = PrivyWalletAddressCheckerTool()
    tool.configure(
        {
            "tools": {
                "privy_balance": {
                    "app_id": "test-app-id",
                    # Missing app_secret
                }
            }
        }
    )
    return tool


class TestPrivyWalletAddressToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, wallet_tool):
        """Should have correct tool name."""
        assert wallet_tool.name == "privy_wallet_address"

    def test_schema_has_required_properties(self, wallet_tool):
        """Should include user_id in required properties."""
        schema = wallet_tool.get_schema()
        assert "user_id" in schema["properties"]
        assert "user_id" in schema["required"]

    def test_schema_property_types(self, wallet_tool):
        """Should have correct property types."""
        schema = wallet_tool.get_schema()
        assert schema["properties"]["user_id"]["type"] == "string"


class TestPrivyWalletAddressToolConfigure:
    """Test configuration method."""

    def test_configure_stores_privy_config(self, wallet_tool):
        """Should store Privy configuration."""
        assert wallet_tool.app_id == "test-app-id"
        assert wallet_tool.app_secret == "test-app-secret"


class TestPrivyWalletAddressToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_missing_config_error(self, wallet_tool_incomplete):
        """Should return error when config is incomplete."""
        result = await wallet_tool_incomplete.execute(
            user_id="did:privy:user123",
        )

        assert result["status"] == "error"
        assert "config" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_no_wallet_found_error(self, wallet_tool):
        """Should return error when no delegated wallet found."""
        with patch(
            "solana_agent.tools.privy_wallet_address.get_privy_embedded_wallet_address",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await wallet_tool.execute(
                user_id="did:privy:user123",
            )

            assert result["status"] == "error"
            assert "wallet" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_success(self, wallet_tool):
        """Should return wallet address on success."""
        with patch(
            "solana_agent.tools.privy_wallet_address.get_privy_embedded_wallet_address",
            new_callable=AsyncMock,
            return_value="WalletPubkey123...abc",
        ):
            result = await wallet_tool.execute(
                user_id="did:privy:user123",
            )

            assert result["status"] == "success"
            assert result["result"] == "WalletPubkey123...abc"


class TestGetPrivyEmbeddedWalletAddress:
    """Test the helper function."""

    @pytest.mark.asyncio
    async def test_returns_public_key_for_delegated_wallet(self):
        """Should return public key when delegated wallet exists."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "linked_accounts": [
                {
                    "connector_type": "embedded",
                    "delegated": True,
                    "public_key": "WalletPubkey123...abc",
                },
            ]
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await get_privy_embedded_wallet_address(
                "user123", "app-id", "app-secret"
            )

            assert result == "WalletPubkey123...abc"

    @pytest.mark.asyncio
    async def test_returns_none_for_non_delegated_wallet(self):
        """Should return None when wallet is not delegated."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "linked_accounts": [
                {
                    "connector_type": "embedded",
                    "delegated": False,
                    "public_key": "WalletPubkey123...abc",
                },
            ]
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await get_privy_embedded_wallet_address(
                "user123", "app-id", "app-secret"
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_no_linked_accounts(self):
        """Should return None when no linked accounts."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"linked_accounts": []}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await get_privy_embedded_wallet_address(
                "user123", "app-id", "app-secret"
            )

            assert result is None


class TestPrivyWalletAddressPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = PrivyWalletAddressCheckerPlugin()
        assert plugin.name == "privy_wallet_address"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = PrivyWalletAddressCheckerPlugin()
        assert "wallet" in plugin.description.lower()

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = PrivyWalletAddressCheckerPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = PrivyWalletAddressCheckerPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

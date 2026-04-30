"""
Tests for Privy Get User By Telegram Tool.

Tests the PrivyGetUserByTelegramTool which looks up existing Privy users
by their Telegram user ID.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.privy_get_user_by_telegram import (
    PrivyGetUserByTelegramTool,
    PrivyGetUserByTelegramPlugin,
    get_privy_user_by_telegram,
    extract_wallet_info,
)


@pytest.fixture
def get_user_tool():
    """Create a configured PrivyGetUserByTelegramTool."""
    tool = PrivyGetUserByTelegramTool()
    tool.configure(
        {
            "tools": {
                "privy_get_user_by_telegram": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                }
            }
        }
    )
    return tool


@pytest.fixture
def get_user_tool_incomplete():
    """Create an incomplete PrivyGetUserByTelegramTool."""
    tool = PrivyGetUserByTelegramTool()
    tool.configure(
        {
            "tools": {
                "privy_get_user_by_telegram": {
                    "app_id": "test-app-id",
                    # Missing app_secret
                }
            }
        }
    )
    return tool


class TestPrivyGetUserByTelegramToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, get_user_tool):
        """Should have correct tool name."""
        assert get_user_tool.name == "privy_get_user_by_telegram"

    def test_schema_has_required_properties(self, get_user_tool):
        """Should include telegram_user_id in required properties."""
        schema = get_user_tool.get_schema()
        assert "telegram_user_id" in schema["properties"]
        assert "telegram_user_id" in schema["required"]

    def test_schema_property_types(self, get_user_tool):
        """Should have correct property types."""
        schema = get_user_tool.get_schema()
        assert schema["properties"]["telegram_user_id"]["type"] == "string"

    def test_schema_additional_properties_false(self, get_user_tool):
        """Should have additionalProperties set to false for OpenAI compliance."""
        schema = get_user_tool.get_schema()
        assert schema["additionalProperties"] is False


class TestPrivyGetUserByTelegramToolConfigure:
    """Test configuration method."""

    def test_configure_stores_privy_config(self, get_user_tool):
        """Should store Privy configuration."""
        assert get_user_tool.app_id == "test-app-id"
        assert get_user_tool.app_secret == "test-app-secret"


class TestPrivyGetUserByTelegramToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_missing_config_error(self, get_user_tool_incomplete):
        """Should return error when config is incomplete."""
        result = await get_user_tool_incomplete.execute(
            telegram_user_id="123456789",
        )

        assert result["status"] == "error"
        assert "config" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_user_not_found(self, get_user_tool):
        """Should return not_found when user doesn't exist."""
        with patch(
            "solana_agent.tools.privy_get_user_by_telegram.get_privy_user_by_telegram",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await get_user_tool.execute(
                telegram_user_id="123456789",
            )

            assert result["status"] == "not_found"
            assert "123456789" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_success_with_wallet(self, get_user_tool):
        """Should return user data with wallets on success."""
        mock_user_data = {
            "id": "did:privy:user123",
            "created_at": 1731974895,
            "linked_accounts": [
                {
                    "type": "telegram",
                    "telegram_user_id": "123456789",
                },
                {
                    "type": "solana_embedded_wallet",
                    "id": "wallet-123",
                    "address": "SoLaNaWaLLeTaDdReSs...",
                    "chain_type": "solana",
                    "delegated": True,
                },
            ],
        }

        with patch(
            "solana_agent.tools.privy_get_user_by_telegram.get_privy_user_by_telegram",
            new_callable=AsyncMock,
            return_value=mock_user_data,
        ):
            result = await get_user_tool.execute(
                telegram_user_id="123456789",
            )

            assert result["status"] == "success"
            assert result["result"]["user_id"] == "did:privy:user123"
            assert result["result"]["has_wallet"] is True
            assert len(result["result"]["wallets"]) == 1
            assert result["result"]["wallets"][0]["address"] == "SoLaNaWaLLeTaDdReSs..."

    @pytest.mark.asyncio
    async def test_execute_success_without_wallet(self, get_user_tool):
        """Should return user data without wallets when none exist."""
        mock_user_data = {
            "id": "did:privy:user456",
            "created_at": 1731974895,
            "linked_accounts": [
                {
                    "type": "telegram",
                    "telegram_user_id": "123456789",
                },
            ],
        }

        with patch(
            "solana_agent.tools.privy_get_user_by_telegram.get_privy_user_by_telegram",
            new_callable=AsyncMock,
            return_value=mock_user_data,
        ):
            result = await get_user_tool.execute(
                telegram_user_id="123456789",
            )

            assert result["status"] == "success"
            assert result["result"]["user_id"] == "did:privy:user456"
            assert result["result"]["has_wallet"] is False
            assert len(result["result"]["wallets"]) == 0

    @pytest.mark.asyncio
    async def test_execute_http_error(self, get_user_tool):
        """Should return error on HTTP failure."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch(
            "solana_agent.tools.privy_get_user_by_telegram.get_privy_user_by_telegram",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Server Error", request=MagicMock(), response=mock_response
            ),
        ):
            result = await get_user_tool.execute(
                telegram_user_id="123456789",
            )

            assert result["status"] == "error"
            assert "500" in result["message"]


class TestGetPrivyUserByTelegram:
    """Test the helper function."""

    @pytest.mark.asyncio
    async def test_returns_user_data_when_found(self):
        """Should return user data when user exists."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "did:privy:test123",
            "created_at": 1731974895,
            "linked_accounts": [],
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await get_privy_user_by_telegram(
                "123456789", "app-id", "app-secret"
            )

            assert result["id"] == "did:privy:test123"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        """Should return None when user doesn't exist (404)."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await get_privy_user_by_telegram(
                "123456789", "app-id", "app-secret"
            )

            assert result is None


class TestExtractWalletInfo:
    """Test the wallet extraction helper function."""

    def test_extracts_embedded_wallet_by_type(self):
        """Should extract wallets with embedded_wallet in type."""
        user_data = {
            "linked_accounts": [
                {
                    "type": "solana_embedded_wallet",
                    "id": "wallet-123",
                    "address": "WalletAddress...",
                    "chain_type": "solana",
                    "delegated": True,
                },
            ]
        }

        wallets = extract_wallet_info(user_data)

        assert len(wallets) == 1
        assert wallets[0]["wallet_id"] == "wallet-123"
        assert wallets[0]["address"] == "WalletAddress..."
        assert wallets[0]["delegated"] is True

    def test_extracts_embedded_wallet_by_connector_type(self):
        """Should extract wallets with connector_type == embedded."""
        user_data = {
            "linked_accounts": [
                {
                    "type": "wallet",
                    "connector_type": "embedded",
                    "id": "wallet-456",
                    "public_key": "PublicKey...",
                    "delegated": False,
                },
            ]
        }

        wallets = extract_wallet_info(user_data)

        assert len(wallets) == 1
        assert wallets[0]["wallet_id"] == "wallet-456"
        assert wallets[0]["address"] == "PublicKey..."
        assert wallets[0]["delegated"] is False

    def test_ignores_non_embedded_wallets(self):
        """Should ignore wallets that aren't embedded."""
        user_data = {
            "linked_accounts": [
                {
                    "type": "telegram",
                    "telegram_user_id": "123456789",
                },
                {
                    "type": "external_wallet",
                    "address": "ExternalWallet...",
                },
            ]
        }

        wallets = extract_wallet_info(user_data)

        assert len(wallets) == 0

    def test_handles_empty_linked_accounts(self):
        """Should handle empty linked_accounts."""
        user_data = {"linked_accounts": []}

        wallets = extract_wallet_info(user_data)

        assert len(wallets) == 0

    def test_handles_missing_linked_accounts(self):
        """Should handle missing linked_accounts key."""
        user_data = {}

        wallets = extract_wallet_info(user_data)

        assert len(wallets) == 0


class TestPrivyGetUserByTelegramPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = PrivyGetUserByTelegramPlugin()
        assert plugin.name == "privy_get_user_by_telegram"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = PrivyGetUserByTelegramPlugin()
        assert "telegram" in plugin.description.lower()

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = PrivyGetUserByTelegramPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = PrivyGetUserByTelegramPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

    def test_plugin_configure(self):
        """Should configure tool with config."""
        plugin = PrivyGetUserByTelegramPlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)

        config = {
            "tools": {
                "privy_get_user_by_telegram": {
                    "app_id": "test-id",
                    "app_secret": "test-secret",
                }
            }
        }
        plugin.configure(config)

        assert plugin._tool.app_id == "test-id"
        assert plugin._tool.app_secret == "test-secret"

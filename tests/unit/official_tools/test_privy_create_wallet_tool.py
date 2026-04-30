"""
Tests for Privy Create Wallet Tool.

Tests the PrivyCreateWalletTool which creates new Solana wallets
for Privy users.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.privy_create_wallet import (
    PrivyCreateWalletTool,
    PrivyCreateWalletPlugin,
    create_privy_wallet,
)


@pytest.fixture
def create_wallet_tool():
    """Create a configured PrivyCreateWalletTool."""
    tool = PrivyCreateWalletTool()
    tool.configure(
        {
            "tools": {
                "privy_create_wallet": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                }
            }
        }
    )
    return tool


@pytest.fixture
def create_wallet_tool_incomplete():
    """Create an incomplete PrivyCreateWalletTool."""
    tool = PrivyCreateWalletTool()
    tool.configure(
        {
            "tools": {
                "privy_create_wallet": {
                    "app_id": "test-app-id",
                    # Missing app_secret
                }
            }
        }
    )
    return tool


class TestPrivyCreateWalletToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, create_wallet_tool):
        """Should have correct tool name."""
        assert create_wallet_tool.name == "privy_create_wallet"

    def test_schema_has_required_properties(self, create_wallet_tool):
        """Should include required properties."""
        schema = create_wallet_tool.get_schema()
        assert "user_id" in schema["properties"]
        assert "chain_type" in schema["properties"]
        assert "user_id" in schema["required"]
        assert "chain_type" in schema["required"]

    def test_schema_property_types(self, create_wallet_tool):
        """Should have correct property types."""
        schema = create_wallet_tool.get_schema()
        assert schema["properties"]["user_id"]["type"] == "string"
        assert schema["properties"]["chain_type"]["type"] == "string"

    def test_schema_chain_type_enum(self, create_wallet_tool):
        """Should have chain_type enum values."""
        schema = create_wallet_tool.get_schema()
        assert "solana" in schema["properties"]["chain_type"]["enum"]
        assert "ethereum" in schema["properties"]["chain_type"]["enum"]

    def test_schema_additional_properties_false(self, create_wallet_tool):
        """Should have additionalProperties set to false for OpenAI compliance."""
        schema = create_wallet_tool.get_schema()
        assert schema["additionalProperties"] is False


class TestPrivyCreateWalletToolConfigure:
    """Test configuration method."""

    def test_configure_stores_privy_config(self, create_wallet_tool):
        """Should store Privy configuration."""
        assert create_wallet_tool.app_id == "test-app-id"
        assert create_wallet_tool.app_secret == "test-app-secret"


class TestPrivyCreateWalletToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_missing_config_error(self, create_wallet_tool_incomplete):
        """Should return error when config is incomplete."""
        result = await create_wallet_tool_incomplete.execute(
            user_id="did:privy:user123",
            chain_type="solana",
        )

        assert result["status"] == "error"
        assert "config" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_success(self, create_wallet_tool):
        """Should return wallet data on success."""
        mock_wallet_data = {
            "id": "wallet-id-123",
            "address": "SoLaNaWaLLeTaDdReSs123...",
            "chain_type": "solana",
            "created_at": 1741834854578,
            "owner_id": "rkiz0ivz254drv1xw982v3jq",
            "additional_signers": [],
        }

        with patch(
            "solana_agent.tools.privy_create_wallet.create_privy_wallet",
            new_callable=AsyncMock,
            return_value=mock_wallet_data,
        ):
            result = await create_wallet_tool.execute(
                user_id="did:privy:user123",
                chain_type="solana",
            )

            assert result["status"] == "success"
            assert result["result"]["wallet_id"] == "wallet-id-123"
            assert result["result"]["address"] == "SoLaNaWaLLeTaDdReSs123..."
            assert result["result"]["chain_type"] == "solana"
            assert result["result"]["additional_signers"] == []

    @pytest.mark.asyncio
    async def test_execute_http_error(self, create_wallet_tool):
        """Should return error on HTTP failure."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch(
            "solana_agent.tools.privy_create_wallet.create_privy_wallet",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Bad Request", request=MagicMock(), response=mock_response
            ),
        ):
            result = await create_wallet_tool.execute(
                user_id="did:privy:user123",
                chain_type="solana",
            )

            assert result["status"] == "error"
            assert "400" in result["message"]


class TestCreatePrivyWallet:
    """Test the helper function."""

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_creates_wallet_without_signer(self):
        """Should create wallet."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "wallet-456",
            "address": "WalletAddress...",
            "chain_type": "solana",
            "additional_signers": [],
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await create_privy_wallet(
                user_id="did:privy:user123",
                app_id="app-id",
                app_secret="app-secret",
                chain_type="solana",
            )

            assert result["id"] == "wallet-456"
            assert result["additional_signers"] == []


class TestPrivyCreateWalletPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = PrivyCreateWalletPlugin()
        assert plugin.name == "privy_create_wallet"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = PrivyCreateWalletPlugin()
        assert "wallet" in plugin.description.lower()

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = PrivyCreateWalletPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = PrivyCreateWalletPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

    def test_plugin_configure(self):
        """Should configure tool with config."""
        plugin = PrivyCreateWalletPlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)

        config = {
            "tools": {
                "privy_create_wallet": {
                    "app_id": "test-id",
                    "app_secret": "test-secret",
                }
            }
        }
        plugin.configure(config)

        assert plugin._tool.app_id == "test-id"
        assert plugin._tool.app_secret == "test-secret"

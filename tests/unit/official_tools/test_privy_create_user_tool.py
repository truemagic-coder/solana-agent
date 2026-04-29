"""
Tests for Privy Create User Tool.

Tests the PrivyCreateUserTool which creates new Privy users
with linked Telegram accounts.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.privy_create_user import (
    PrivyCreateUserTool,
    PrivyCreateUserPlugin,
    create_privy_user_with_telegram,
)


@pytest.fixture
def create_user_tool():
    """Create a configured PrivyCreateUserTool."""
    tool = PrivyCreateUserTool()
    tool.configure(
        {
            "tools": {
                "privy_create_user": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                }
            }
        }
    )
    return tool


@pytest.fixture
def create_user_tool_incomplete():
    """Create an incomplete PrivyCreateUserTool."""
    tool = PrivyCreateUserTool()
    tool.configure(
        {
            "tools": {
                "privy_create_user": {
                    "app_id": "test-app-id",
                    # Missing app_secret
                }
            }
        }
    )
    return tool


class TestPrivyCreateUserToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, create_user_tool):
        """Should have correct tool name."""
        assert create_user_tool.name == "privy_create_user"

    def test_schema_has_required_properties(self, create_user_tool):
        """Should include telegram_user_id in required properties."""
        schema = create_user_tool.get_schema()
        assert "telegram_user_id" in schema["properties"]
        assert "telegram_user_id" in schema["required"]

    def test_schema_property_types(self, create_user_tool):
        """Should have correct property types."""
        schema = create_user_tool.get_schema()
        assert schema["properties"]["telegram_user_id"]["type"] == "string"

    def test_schema_additional_properties_false(self, create_user_tool):
        """Should have additionalProperties set to false for OpenAI compliance."""
        schema = create_user_tool.get_schema()
        assert schema["additionalProperties"] is False


class TestPrivyCreateUserToolConfigure:
    """Test configuration method."""

    def test_configure_stores_privy_config(self, create_user_tool):
        """Should store Privy configuration."""
        assert create_user_tool.app_id == "test-app-id"
        assert create_user_tool.app_secret == "test-app-secret"


class TestPrivyCreateUserToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_missing_config_error(self, create_user_tool_incomplete):
        """Should return error when config is incomplete."""
        result = await create_user_tool_incomplete.execute(
            telegram_user_id="123456789",
        )

        assert result["status"] == "error"
        assert "config" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_success(self, create_user_tool):
        """Should return user data on success."""
        mock_user_data = {
            "id": "did:privy:cm3np4u9j001rc8b73seqmqqk",
            "created_at": 1731974895,
            "linked_accounts": [
                {
                    "type": "telegram",
                    "telegram_user_id": "123456789",
                }
            ],
        }

        with patch(
            "solana_agent.tools.privy_create_user.create_privy_user_with_telegram",
            new_callable=AsyncMock,
            return_value=mock_user_data,
        ):
            result = await create_user_tool.execute(
                telegram_user_id="123456789",
            )

            assert result["status"] == "success"
            assert result["result"]["user_id"] == "did:privy:cm3np4u9j001rc8b73seqmqqk"
            assert result["result"]["created_at"] == 1731974895
            assert len(result["result"]["linked_accounts"]) == 1

    @pytest.mark.asyncio
    async def test_execute_http_error(self, create_user_tool):
        """Should return error on HTTP failure."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch(
            "solana_agent.tools.privy_create_user.create_privy_user_with_telegram",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Bad Request", request=MagicMock(), response=mock_response
            ),
        ):
            result = await create_user_tool.execute(
                telegram_user_id="123456789",
            )

            assert result["status"] == "error"
            assert "400" in result["message"]


class TestCreatePrivyUserWithTelegram:
    """Test the helper function."""

    @pytest.mark.asyncio
    async def test_creates_user_successfully(self):
        """Should create user and return data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "did:privy:test123",
            "created_at": 1731974895,
            "linked_accounts": [{"type": "telegram", "telegram_user_id": "123456789"}],
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await create_privy_user_with_telegram(
                "123456789", "app-id", "app-secret"
            )

            assert result["id"] == "did:privy:test123"
            assert result["linked_accounts"][0]["telegram_user_id"] == "123456789"

    @pytest.mark.asyncio
    async def test_sends_correct_request_body(self):
        """Should send correct request body with telegram linked account."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "test"}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            await create_privy_user_with_telegram("123456789", "app-id", "app-secret")

            # Verify the call was made with correct body
            call_args = mock_client_instance.post.call_args
            assert call_args[1]["json"]["linked_accounts"][0]["type"] == "telegram"
            assert (
                call_args[1]["json"]["linked_accounts"][0]["telegram_user_id"]
                == "123456789"
            )


class TestPrivyCreateUserPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = PrivyCreateUserPlugin()
        assert plugin.name == "privy_create_user"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = PrivyCreateUserPlugin()
        assert "telegram" in plugin.description.lower()

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = PrivyCreateUserPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = PrivyCreateUserPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

    def test_plugin_configure(self):
        """Should configure tool with config."""
        plugin = PrivyCreateUserPlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)

        config = {
            "tools": {
                "privy_create_user": {
                    "app_id": "test-id",
                    "app_secret": "test-secret",
                }
            }
        }
        plugin.configure(config)

        assert plugin._tool.app_id == "test-id"
        assert plugin._tool.app_secret == "test-secret"

"""Tests for Privy DFlow swap tool."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from solana_agent.tools.privy_dflow_swap import (
    PrivyDFlowSwapTool,
    PrivyDFlowSwapPlugin,
    get_plugin,
)


class TestPrivyDFlowSwapToolInit:
    """Tests for PrivyDFlowSwapTool initialization."""

    def test_tool_name(self):
        """Should have correct tool name."""
        tool = PrivyDFlowSwapTool()
        assert tool.name == "privy_dflow_swap"

    def test_tool_description(self):
        """Should have meaningful description."""
        tool = PrivyDFlowSwapTool()
        assert "DFlow" in tool.description
        assert "swap" in tool.description.lower()


class TestPrivyDFlowSwapToolSchema:
    """Tests for PrivyDFlowSwapTool schema."""

    def test_schema_has_required_fields(self):
        """Should have all required fields in schema."""
        tool = PrivyDFlowSwapTool()
        schema = tool.get_schema()

        assert "properties" in schema
        props = schema["properties"]

        assert "wallet_id" in props
        assert "wallet_public_key" in props
        assert "input_mint" in props
        assert "output_mint" in props
        assert "amount" in props

    def test_schema_required_fields(self):
        """Should mark correct fields as required."""
        tool = PrivyDFlowSwapTool()
        schema = tool.get_schema()

        required = schema.get("required", [])
        assert "wallet_id" in required
        assert "wallet_public_key" in required
        assert "input_mint" in required
        assert "output_mint" in required
        assert "amount" in required


class TestPrivyDFlowSwapToolConfigure:
    """Tests for PrivyDFlowSwapTool configuration."""

    def test_configure_sets_credentials(self):
        """Should set credentials from config."""
        tool = PrivyDFlowSwapTool()
        config = {
            "tools": {
                "privy_dflow_swap": {
                    "app_id": "test_app_id",
                    "app_secret": "test_app_secret",
                    "signing_key": "test_signing_key",
                    "payer_private_key": "payer_key",
                    "rpc_url": "https://custom-rpc.com",
                }
            }
        }

        tool.configure(config)

        assert tool._app_id == "test_app_id"
        assert tool._app_secret == "test_app_secret"
        assert tool._signing_key == "test_signing_key"
        assert tool._payer_private_key == "payer_key"
        assert tool._rpc_url == "https://custom-rpc.com"


class TestPrivyDFlowSwapToolExecute:
    """Tests for PrivyDFlowSwapTool execute method."""

    @pytest.fixture
    def configured_tool(self):
        """Create a configured tool for testing."""
        tool = PrivyDFlowSwapTool()
        tool.configure(
            {
                "tools": {
                    "privy_dflow_swap": {
                        "app_id": "test_app_id",
                        "app_secret": "test_app_secret",
                        "signing_key": "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg"
                        + "A" * 43
                        + "=",
                        "rpc_url": "https://api.mainnet-beta.solana.com",
                    }
                }
            }
        )
        return tool

    @pytest.mark.asyncio
    async def test_execute_missing_config(self):
        """Should return error when config is missing."""
        tool = PrivyDFlowSwapTool()

        result = await tool.execute(
            wallet_id="wallet-123",
            wallet_public_key="WalletPubkey123",
            input_mint="So11111111111111111111111111111111111111112",
            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            amount="1000000000",
        )

        assert result["status"] == "error"
        assert "config" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_wallet_params(self, configured_tool):
        """Should return error when wallet params are missing."""
        result = await configured_tool.execute(
            wallet_id="",
            wallet_public_key="",
            input_mint="So11111111111111111111111111111111111111112",
            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            amount="1000000000",
        )

        assert result["status"] == "error"
        assert "wallet" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_dflow_api_error(self, configured_tool):
        """Should return error when DFlow API fails."""
        with patch("solana_agent.tools.privy_dflow_swap.AsyncPrivyAPI") as MockPrivy:
            mock_privy_instance = AsyncMock()
            mock_privy_instance.close = AsyncMock()
            MockPrivy.return_value = mock_privy_instance

            with patch("solana_agent.tools.privy_dflow_swap.DFlowSwap") as MockDFlow:
                mock_dflow_instance = MagicMock()
                mock_dflow_instance.get_order = AsyncMock(
                    return_value=MagicMock(
                        success=False,
                        error="Insufficient liquidity",
                    )
                )
                MockDFlow.return_value = mock_dflow_instance

                result = await configured_tool.execute(
                    wallet_id="wallet-123",
                    wallet_public_key="WalletAddress123",
                    input_mint="So11111111111111111111111111111111111111112",
                    output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    amount="1000000000",
                )

                assert result["status"] == "error"
                assert "liquidity" in result["message"].lower()


class TestPrivyDFlowSwapPlugin:
    """Tests for PrivyDFlowSwapPlugin."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = PrivyDFlowSwapPlugin()
        assert plugin.name == "privy_dflow_swap"

    def test_plugin_description(self):
        """Should have meaningful description."""
        plugin = PrivyDFlowSwapPlugin()
        assert "DFlow" in plugin.description

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = PrivyDFlowSwapPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should create tool on initialize."""
        plugin = PrivyDFlowSwapPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        tools = plugin.get_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], PrivyDFlowSwapTool)

    def test_plugin_configure(self):
        """Should configure the tool."""
        plugin = PrivyDFlowSwapPlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)

        config = {
            "tools": {
                "privy_dflow_swap": {
                    "app_id": "test_app_id",
                    "app_secret": "test_app_secret",
                    "signing_key": "test_signing_key",
                }
            }
        }
        plugin.configure(config)

        tool = plugin.get_tools()[0]
        assert tool._app_id == "test_app_id"


class TestGetPlugin:
    """Tests for get_plugin function."""

    def test_get_plugin_returns_instance(self):
        """Should return PrivyDFlowSwapPlugin instance."""
        plugin = get_plugin()
        assert isinstance(plugin, PrivyDFlowSwapPlugin)

"""Tests for Solana DFlow swap tool."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from solana_agent.tools.solana_dflow_swap import (
    SolanaDFlowSwapTool,
    SolanaDFlowSwapPlugin,
    get_plugin,
)


class TestSolanaDFlowSwapToolInit:
    """Tests for SolanaDFlowSwapTool initialization."""

    def test_tool_name(self):
        """Should have correct tool name."""
        tool = SolanaDFlowSwapTool()
        assert tool.name == "solana_dflow_swap"

    def test_tool_description(self):
        """Should have meaningful description."""
        tool = SolanaDFlowSwapTool()
        assert "DFlow" in tool.description
        assert "swap" in tool.description.lower()


class TestSolanaDFlowSwapToolSchema:
    """Tests for SolanaDFlowSwapTool schema."""

    def test_schema_has_required_fields(self):
        """Should have all required fields in schema."""
        tool = SolanaDFlowSwapTool()
        schema = tool.get_schema()

        assert "properties" in schema
        props = schema["properties"]

        assert "input_mint" in props
        assert "output_mint" in props
        assert "amount" in props

    def test_schema_required_fields(self):
        """Should mark correct fields as required."""
        tool = SolanaDFlowSwapTool()
        schema = tool.get_schema()

        required = schema.get("required", [])
        assert "input_mint" in required
        assert "output_mint" in required
        assert "amount" in required

    def test_schema_has_slippage_bps(self):
        """Should have optional slippage_bps field."""
        tool = SolanaDFlowSwapTool()
        schema = tool.get_schema()

        assert "slippage_bps" in schema["properties"]
        assert "slippage_bps" not in schema.get("required", [])


class TestSolanaDFlowSwapToolConfigure:
    """Tests for SolanaDFlowSwapTool configuration."""

    def test_configure_sets_credentials(self):
        """Should set credentials from config."""
        tool = SolanaDFlowSwapTool()
        config = {
            "tools": {
                "solana_dflow_swap": {
                    "private_key": "test_private_key",
                    "payer_private_key": "payer_key",
                    "rpc_url": "https://custom-rpc.com",
                }
            }
        }

        tool.configure(config)

        assert tool._private_key == "test_private_key"
        assert tool._payer_private_key == "payer_key"
        assert tool._rpc_url == "https://custom-rpc.com"

    def test_configure_uses_default_rpc_url(self):
        """Should use default RPC URL when not provided."""
        tool = SolanaDFlowSwapTool()
        config = {
            "tools": {
                "solana_dflow_swap": {
                    "private_key": "test_private_key",
                }
            }
        }

        tool.configure(config)

        assert tool._rpc_url == "https://api.mainnet-beta.solana.com"


class TestSolanaDFlowSwapToolExecute:
    """Tests for SolanaDFlowSwapTool execute method."""

    @pytest.mark.asyncio
    async def test_execute_missing_config(self):
        """Should return error when private key is missing."""
        tool = SolanaDFlowSwapTool()

        result = await tool.execute(
            input_mint="So11111111111111111111111111111111111111112",
            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            amount=1000000000,
        )

        assert result["status"] == "error"
        assert "private key" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_dflow_api_error(self):
        """Should return error when DFlow API fails."""
        tool = SolanaDFlowSwapTool()
        # Use a valid base58 keypair for testing
        tool._private_key = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviUkauRiTMD8DrESdrNjN8zd9mTmVhRvBJeg5vhyvgrAhG"
        tool._rpc_url = "https://api.mainnet-beta.solana.com"

        with patch("solana_agent.tools.solana_dflow_swap.DFlowSwap") as MockDFlow:
            mock_dflow_instance = MagicMock()
            mock_dflow_instance.get_order = AsyncMock(
                return_value=MagicMock(
                    success=False,
                    error="Insufficient liquidity",
                )
            )
            MockDFlow.return_value = mock_dflow_instance

            result = await tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "liquidity" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_no_transaction_returned(self):
        """Should return error when no transaction is returned."""
        tool = SolanaDFlowSwapTool()
        tool._private_key = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviUkauRiTMD8DrESdrNjN8zd9mTmVhRvBJeg5vhyvgrAhG"
        tool._rpc_url = "https://api.mainnet-beta.solana.com"

        with patch("solana_agent.tools.solana_dflow_swap.DFlowSwap") as MockDFlow:
            mock_dflow_instance = MagicMock()
            mock_dflow_instance.get_order = AsyncMock(
                return_value=MagicMock(
                    success=True,
                    transaction=None,
                    error=None,
                )
            )
            MockDFlow.return_value = mock_dflow_instance

            result = await tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "no transaction" in result["message"].lower()


class TestSolanaDFlowSwapPlugin:
    """Tests for SolanaDFlowSwapPlugin."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = SolanaDFlowSwapPlugin()
        assert plugin.name == "solana_dflow_swap"

    def test_plugin_description(self):
        """Should have meaningful description."""
        plugin = SolanaDFlowSwapPlugin()
        assert "DFlow" in plugin.description

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = SolanaDFlowSwapPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should create tool on initialize."""
        plugin = SolanaDFlowSwapPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        tools = plugin.get_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], SolanaDFlowSwapTool)

    def test_plugin_configure(self):
        """Should configure the tool."""
        plugin = SolanaDFlowSwapPlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)

        config = {
            "tools": {
                "solana_dflow_swap": {
                    "private_key": "test_private_key",
                    "payer_private_key": "payer_key",
                }
            }
        }
        plugin.configure(config)

        tool = plugin.get_tools()[0]
        assert tool._private_key == "test_private_key"
        assert tool._payer_private_key == "payer_key"


class TestSignDFlowTransaction:
    """Tests for _sign_dflow_transaction function."""

    def test_sign_transaction_without_payer(self):
        """Should sign transaction with user keypair only."""
        from solana_agent.tools.solana_dflow_swap import _sign_dflow_transaction

        # Create a mock transaction that looks valid
        with (
            patch(
                "solana_agent.tools.solana_dflow_swap.VersionedTransaction"
            ) as MockVTx,
            patch(
                "solana_agent.tools.solana_dflow_swap.to_bytes_versioned"
            ) as mock_to_bytes,
        ):
            mock_tx_instance = MagicMock()
            mock_tx_instance.message = MagicMock()
            MockVTx.from_bytes.return_value = mock_tx_instance

            mock_to_bytes.return_value = b"message_bytes"

            mock_sign_func = MagicMock(return_value=b"user_signature")

            # Create a mock populated transaction
            mock_signed_tx = MagicMock()
            mock_signed_tx.__bytes__ = MagicMock(return_value=b"signed_tx_bytes")
            MockVTx.populate.return_value = mock_signed_tx

            _sign_dflow_transaction(
                transaction_base64="dHJhbnNhY3Rpb24=",  # "transaction" in base64
                sign_message_func=mock_sign_func,
            )

            # Verify sign function was called
            mock_sign_func.assert_called_once_with(b"message_bytes")
            # Verify populate was called with just user signature
            MockVTx.populate.assert_called_once()

    def test_sign_transaction_with_payer(self):
        """Should sign transaction with payer and user keypairs."""
        from solana_agent.tools.solana_dflow_swap import _sign_dflow_transaction

        with (
            patch(
                "solana_agent.tools.solana_dflow_swap.VersionedTransaction"
            ) as MockVTx,
            patch(
                "solana_agent.tools.solana_dflow_swap.to_bytes_versioned"
            ) as mock_to_bytes,
        ):
            mock_tx_instance = MagicMock()
            mock_tx_instance.message = MagicMock()
            MockVTx.from_bytes.return_value = mock_tx_instance

            mock_to_bytes.return_value = b"message_bytes"

            mock_user_sign = MagicMock(return_value=b"user_signature")
            mock_payer_sign = MagicMock(return_value=b"payer_signature")

            mock_signed_tx = MagicMock()
            mock_signed_tx.__bytes__ = MagicMock(return_value=b"signed_tx_bytes")
            MockVTx.populate.return_value = mock_signed_tx

            _sign_dflow_transaction(
                transaction_base64="dHJhbnNhY3Rpb24=",
                sign_message_func=mock_user_sign,
                payer_sign_func=mock_payer_sign,
            )

            # Verify both sign functions were called
            mock_user_sign.assert_called_once_with(b"message_bytes")
            mock_payer_sign.assert_called_once_with(b"message_bytes")


class TestSolanaDFlowSwapToolExecuteAdvanced:
    """Advanced tests for SolanaDFlowSwapTool execute method."""

    @pytest.mark.asyncio
    async def test_execute_success_flow(self):
        """Should complete a successful swap flow."""
        tool = SolanaDFlowSwapTool()
        tool._private_key = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviUkauRiTMD8DrESdrNjN8zd9mTmVhRvBJeg5vhyvgrAhG"
        tool._rpc_url = "https://api.mainnet-beta.solana.com"

        with (
            patch("solana_agent.tools.solana_dflow_swap.DFlowSwap") as MockDFlow,
            patch(
                "solana_agent.tools.solana_dflow_swap._sign_dflow_transaction"
            ) as mock_sign,
            patch.object(
                tool, "_send_transaction", new_callable=AsyncMock
            ) as mock_send,
        ):
            # Mock DFlow order response
            mock_dflow_instance = MagicMock()
            mock_order_result = MagicMock(
                success=True,
                transaction="dHJhbnNhY3Rpb24=",
                in_amount="1000000000",
                out_amount="50000000",
                min_out_amount="49500000",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                price_impact_pct="0.1",
                platform_fee="100",
                execution_mode="direct",
                error=None,
            )
            mock_dflow_instance.get_order = AsyncMock(return_value=mock_order_result)
            MockDFlow.return_value = mock_dflow_instance

            # Mock signing
            mock_sign.return_value = "c2lnbmVkX3R4"

            # Mock sending
            mock_send.return_value = "5abc123def456"

            result = await tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "success"
            assert result["signature"] == "5abc123def456"
            assert result["input_amount"] == "1000000000"
            assert result["output_amount"] == "50000000"

    @pytest.mark.asyncio
    async def test_execute_with_payer_keypair(self):
        """Should use payer keypair for gasless transactions."""
        tool = SolanaDFlowSwapTool()
        tool._private_key = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviUkauRiTMD8DrESdrNjN8zd9mTmVhRvBJeg5vhyvgrAhG"
        tool._payer_private_key = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviUkauRiTMD8DrESdrNjN8zd9mTmVhRvBJeg5vhyvgrAhG"
        tool._rpc_url = "https://api.mainnet-beta.solana.com"

        with (
            patch("solana_agent.tools.solana_dflow_swap.DFlowSwap") as MockDFlow,
            patch(
                "solana_agent.tools.solana_dflow_swap._sign_dflow_transaction"
            ) as mock_sign,
            patch.object(
                tool, "_send_transaction", new_callable=AsyncMock
            ) as mock_send,
        ):
            mock_dflow_instance = MagicMock()
            mock_order_result = MagicMock(
                success=True,
                transaction="dHJhbnNhY3Rpb24=",
                in_amount="1000000000",
                out_amount="50000000",
                min_out_amount="49500000",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                price_impact_pct="0.1",
                platform_fee="100",
                execution_mode="direct",
                error=None,
            )
            mock_dflow_instance.get_order = AsyncMock(return_value=mock_order_result)
            MockDFlow.return_value = mock_dflow_instance

            mock_sign.return_value = "c2lnbmVkX3R4"
            mock_send.return_value = "5abc123def456"

            await tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            # Should have sponsor set
            call_kwargs = mock_dflow_instance.get_order.call_args.kwargs
            assert call_kwargs.get("sponsor") is not None

    @pytest.mark.asyncio
    async def test_execute_send_transaction_fails(self):
        """Should return error when send transaction fails."""
        tool = SolanaDFlowSwapTool()
        tool._private_key = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviUkauRiTMD8DrESdrNjN8zd9mTmVhRvBJeg5vhyvgrAhG"
        tool._rpc_url = "https://api.mainnet-beta.solana.com"

        with (
            patch("solana_agent.tools.solana_dflow_swap.DFlowSwap") as MockDFlow,
            patch(
                "solana_agent.tools.solana_dflow_swap._sign_dflow_transaction"
            ) as mock_sign,
            patch.object(
                tool, "_send_transaction", new_callable=AsyncMock
            ) as mock_send,
        ):
            mock_dflow_instance = MagicMock()
            mock_order_result = MagicMock(
                success=True,
                transaction="dHJhbnNhY3Rpb24=",
                error=None,
            )
            mock_dflow_instance.get_order = AsyncMock(return_value=mock_order_result)
            MockDFlow.return_value = mock_dflow_instance

            mock_sign.return_value = "c2lnbmVkX3R4"
            mock_send.return_value = None  # Simulate send failure

            result = await tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "failed to send" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_with_slippage_bps(self):
        """Should pass slippage_bps when provided."""
        tool = SolanaDFlowSwapTool()
        tool._private_key = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviUkauRiTMD8DrESdrNjN8zd9mTmVhRvBJeg5vhyvgrAhG"
        tool._rpc_url = "https://api.mainnet-beta.solana.com"

        with (
            patch("solana_agent.tools.solana_dflow_swap.DFlowSwap") as MockDFlow,
            patch(
                "solana_agent.tools.solana_dflow_swap._sign_dflow_transaction"
            ) as mock_sign,
            patch.object(
                tool, "_send_transaction", new_callable=AsyncMock
            ) as mock_send,
        ):
            mock_dflow_instance = MagicMock()
            mock_order_result = MagicMock(
                success=True,
                transaction="dHJhbnNhY3Rpb24=",
                in_amount="1000000000",
                out_amount="50000000",
                min_out_amount="49500000",
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                price_impact_pct="0.1",
                platform_fee="100",
                execution_mode="direct",
                error=None,
            )
            mock_dflow_instance.get_order = AsyncMock(return_value=mock_order_result)
            MockDFlow.return_value = mock_dflow_instance

            mock_sign.return_value = "c2lnbmVkX3R4"
            mock_send.return_value = "5abc123def456"

            await tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
                slippage_bps=100,
            )

            # Verify slippage was passed
            call_kwargs = mock_dflow_instance.get_order.call_args.kwargs
            assert call_kwargs.get("slippage_bps") == 100

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self):
        """Should handle exceptions gracefully."""
        tool = SolanaDFlowSwapTool()
        tool._private_key = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviUkauRiTMD8DrESdrNjN8zd9mTmVhRvBJeg5vhyvgrAhG"
        tool._rpc_url = "https://api.mainnet-beta.solana.com"

        with patch("solana_agent.tools.solana_dflow_swap.DFlowSwap") as MockDFlow:
            mock_dflow_instance = MagicMock()
            mock_dflow_instance.get_order = AsyncMock(
                side_effect=Exception("Network error")
            )
            MockDFlow.return_value = mock_dflow_instance

            result = await tool.execute(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
            )

            assert result["status"] == "error"
            assert "network error" in result["message"].lower()


class TestSendTransaction:
    """Tests for _send_transaction method."""

    @pytest.mark.asyncio
    async def test_send_transaction_success(self):
        """Should return signature on successful send."""
        tool = SolanaDFlowSwapTool()
        tool._rpc_url = "https://api.mainnet-beta.solana.com"

        with patch("solana_agent.tools.solana_dflow_swap.AsyncClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.value = "5abc123def456signature"
            mock_client_instance.send_raw_transaction = AsyncMock(
                return_value=mock_result
            )
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client_instance

            result = await tool._send_transaction("dHJhbnNhY3Rpb24=")

            assert result == "5abc123def456signature"

    @pytest.mark.asyncio
    async def test_send_transaction_no_value(self):
        """Should return None when result has no value."""
        tool = SolanaDFlowSwapTool()
        tool._rpc_url = "https://api.mainnet-beta.solana.com"

        with patch("solana_agent.tools.solana_dflow_swap.AsyncClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.value = None
            mock_client_instance.send_raw_transaction = AsyncMock(
                return_value=mock_result
            )
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client_instance

            result = await tool._send_transaction("dHJhbnNhY3Rpb24=")

            assert result is None

    @pytest.mark.asyncio
    async def test_send_transaction_exception(self):
        """Should return None on exception."""
        tool = SolanaDFlowSwapTool()
        tool._rpc_url = "https://api.mainnet-beta.solana.com"

        with patch("solana_agent.tools.solana_dflow_swap.AsyncClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.send_raw_transaction = AsyncMock(
                side_effect=Exception("RPC error")
            )
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client_instance

            result = await tool._send_transaction("dHJhbnNhY3Rpb24=")

            assert result is None


class TestGetPlugin:
    """Tests for get_plugin function."""

    def test_get_plugin_returns_instance(self):
        """Should return SolanaDFlowSwapPlugin instance."""
        plugin = get_plugin()
        assert isinstance(plugin, SolanaDFlowSwapPlugin)

"""Tests for DFlow Swap API utilities."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from solana_agent.tools.utils.dflow import (
    DFlowSwap,
    DFlowOrderResponse,
    DFlowOrderStatusResponse,
    DFLOW_API_URL,
)


class TestDFlowSwapInit:
    """Tests for DFlowSwap initialization."""

    def test_default_base_url(self):
        """Should use default DFlow API URL."""
        client = DFlowSwap()
        assert client.base_url == DFLOW_API_URL

    def test_custom_base_url(self):
        """Should allow custom base URL."""
        custom_url = "https://custom.dflow.net"
        client = DFlowSwap(base_url=custom_url)
        assert client.base_url == custom_url


class TestDFlowGetOrder:
    """Tests for DFlowSwap.get_order method."""

    @pytest.fixture
    def dflow_client(self):
        return DFlowSwap()

    @pytest.mark.asyncio
    async def test_get_order_success(self, dflow_client):
        """Should return order with transaction on success."""
        mock_response = {
            "transaction": "base64encodedtx==",
            "inAmount": "1000000000",
            "outAmount": "50000000",
            "minOutAmount": "49500000",
            "inputMint": "So11111111111111111111111111111111111111112",
            "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "slippageBps": 50,
            "executionMode": "sync",
            "priceImpactPct": "0.01",
            "contextSlot": 12345678,
            "lastValidBlockHeight": 12345700,
            "computeUnitLimit": 200000,
            "prioritizationFeeLamports": 5000,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
                user_public_key="WalletPubkey123",
            )

            assert result.success is True
            assert result.transaction == "base64encodedtx=="
            assert result.in_amount == "1000000000"
            assert result.out_amount == "50000000"
            assert result.execution_mode == "sync"

    @pytest.mark.asyncio
    async def test_get_order_with_platform_fee(self, dflow_client):
        """Should include platform fee parameters."""
        mock_response = {
            "transaction": "base64encodedtx==",
            "inAmount": "1000000000",
            "outAmount": "49750000",
            "minOutAmount": "49252500",
            "inputMint": "So11111111111111111111111111111111111111112",
            "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "slippageBps": 50,
            "executionMode": "sync",
            "priceImpactPct": "0.01",
            "platformFee": {
                "amount": "250000",
                "feeBps": 50,
            },
            "contextSlot": 12345678,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
                user_public_key="WalletPubkey123",
                platform_fee_bps=50,
                fee_account="FeeAccountPubkey123",
            )

            # Verify fee params were included in request
            call_args = mock_instance.get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert params.get("platformFeeBps") == 50
            assert params.get("feeAccount") == "FeeAccountPubkey123"

            assert result.success is True
            assert result.platform_fee is not None
            assert result.platform_fee.get("feeBps") == 50

    @pytest.mark.asyncio
    async def test_get_order_with_sponsor(self, dflow_client):
        """Should include sponsor for gasless swaps."""
        mock_response = {
            "transaction": "base64encodedtx==",
            "inAmount": "1000000000",
            "outAmount": "50000000",
            "minOutAmount": "49500000",
            "inputMint": "So11111111111111111111111111111111111111112",
            "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "slippageBps": 50,
            "executionMode": "sync",
            "contextSlot": 12345678,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
                user_public_key="WalletPubkey123",
                sponsor="SponsorPubkey123",
            )

            # Verify sponsor param was included
            call_args = mock_instance.get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert params.get("sponsor") == "SponsorPubkey123"

            assert result.success is True

    @pytest.mark.asyncio
    async def test_get_order_api_error(self, dflow_client):
        """Should return error on API failure."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                return_value=MagicMock(
                    status_code=400,
                    text="Bad Request",
                    json=lambda: {"error": "Invalid mint address"},
                )
            )
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order(
                input_mint="invalid",
                output_mint="invalid",
                amount=0,
                user_public_key="WalletPubkey123",
            )

            assert result.success is False
            assert "Invalid mint address" in result.error


class TestDFlowGetOrderStatus:
    """Tests for DFlowSwap.get_order_status method."""

    @pytest.fixture
    def dflow_client(self):
        return DFlowSwap()

    @pytest.mark.asyncio
    async def test_get_order_status_success(self, dflow_client):
        """Should return order status."""
        mock_response = {
            "status": "closed",
            "inAmount": "1000000000",
            "outAmount": "50000000",
            "fills": [
                {
                    "signature": "fillsig123",
                    "inAmount": "1000000000",
                    "outAmount": "50000000",
                    "inputMint": "So11111111111111111111111111111111111111112",
                    "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                }
            ],
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order_status(
                signature="txsig123",
            )

            assert result.success is True
            assert result.status == "closed"
            assert result.in_amount == "1000000000"
            assert result.out_amount == "50000000"
            assert len(result.fills) == 1

    @pytest.mark.asyncio
    async def test_get_order_status_pending(self, dflow_client):
        """Should return pending status."""
        mock_response = {
            "status": "pending",
            "inAmount": "0",
            "outAmount": "0",
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order_status(
                signature="txsig123",
            )

            assert result.success is True
            assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_get_order_status_not_found(self, dflow_client):
        """Should return error when order not found."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                return_value=MagicMock(status_code=404, text="Not Found")
            )
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order_status(
                signature="nonexistent",
            )

            assert result.success is False
            assert "not found" in result.error.lower()


class TestDFlowOrderResponse:
    """Tests for DFlowOrderResponse dataclass."""

    def test_order_response_success(self):
        """Should create successful response."""
        response = DFlowOrderResponse(
            success=True,
            transaction="base64tx==",
            in_amount="1000000000",
            out_amount="50000000",
            input_mint="So11111111111111111111111111111111111111112",
            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        )
        assert response.success is True
        assert response.transaction == "base64tx=="

    def test_order_response_error(self):
        """Should create error response."""
        response = DFlowOrderResponse(
            success=False,
            error="Insufficient liquidity",
        )
        assert response.success is False
        assert response.error == "Insufficient liquidity"


class TestDFlowOrderStatusResponse:
    """Tests for DFlowOrderStatusResponse dataclass."""

    def test_status_response_success(self):
        """Should create successful status response."""
        response = DFlowOrderStatusResponse(
            success=True,
            status="closed",
            in_amount="1000000000",
            out_amount="50000000",
        )
        assert response.success is True
        assert response.status == "closed"

    def test_status_response_with_fills(self):
        """Should include fills when present."""
        fills = [
            {"signature": "sig123", "inAmount": "1000000000", "outAmount": "50000000"}
        ]
        response = DFlowOrderStatusResponse(
            success=True,
            status="closed",
            in_amount="1000000000",
            out_amount="50000000",
            fills=fills,
        )
        assert response.fills == fills


class TestDFlowGetOrderTimeout:
    """Tests for timeout handling in DFlowSwap.get_order."""

    @pytest.fixture
    def dflow_client(self):
        return DFlowSwap()

    @pytest.mark.asyncio
    async def test_get_order_timeout(self, dflow_client):
        """Should return error on timeout."""
        import httpx

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
                user_public_key="WalletPubkey123",
            )

            assert result.success is False
            assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_order_generic_exception(self, dflow_client):
        """Should return error on generic exception."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=Exception("Network error"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
                user_public_key="WalletPubkey123",
            )

            assert result.success is False
            assert "Network error" in result.error


class TestDFlowGetOrderParams:
    """Tests for parameter handling in DFlowSwap.get_order."""

    @pytest.fixture
    def dflow_client(self):
        return DFlowSwap()

    @pytest.mark.asyncio
    async def test_get_order_with_all_optional_params(self, dflow_client):
        """Should include all optional parameters when provided."""
        mock_response = {
            "transaction": "base64encodedtx==",
            "inAmount": "1000000000",
            "outAmount": "50000000",
            "inputMint": "So11111111111111111111111111111111111111112",
            "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "slippageBps": 100,
            "executionMode": "sync",
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
                user_public_key="WalletPubkey123",
                slippage_bps=100,
                platform_fee_bps=50,
                platform_fee_mode="outputMint",
                fee_account="FeeAccount123",
                referral_account="RefAccount123",
                destination_wallet="DestWallet123",
                prioritization_fee_lamports=10000,
                only_direct_routes=True,
                max_route_length=3,
                wrap_and_unwrap_sol=True,
                dynamic_compute_unit_limit=True,
            )

            call_args = mock_instance.get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")

            assert params.get("slippageBps") == 100
            assert params.get("platformFeeBps") == 50
            assert params.get("platformFeeMode") == "outputMint"
            assert params.get("feeAccount") == "FeeAccount123"
            assert params.get("referralAccount") == "RefAccount123"
            assert params.get("destinationWallet") == "DestWallet123"
            assert params.get("prioritizationFeeLamports") == 10000
            assert params.get("onlyDirectRoutes") == "true"
            assert params.get("maxRouteLength") == 3
            assert result.success is True

    @pytest.mark.asyncio
    async def test_get_order_error_json_parse_failure(self, dflow_client):
        """Should handle error response with non-JSON body."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=500, text="Internal Server Error")
            mock_response_obj.json = MagicMock(side_effect=Exception("Not JSON"))
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
                user_public_key="WalletPubkey123",
            )

            assert result.success is False
            assert "500" in result.error


class TestDFlowGetOrderStatusTimeout:
    """Tests for timeout handling in DFlowSwap.get_order_status."""

    @pytest.fixture
    def dflow_client(self):
        return DFlowSwap()

    @pytest.mark.asyncio
    async def test_get_order_status_timeout(self, dflow_client):
        """Should return error on timeout."""
        import httpx

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order_status(signature="txsig123")

            assert result.success is False
            assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_order_status_with_block_height(self, dflow_client):
        """Should include last_valid_block_height in params."""
        mock_response = {
            "status": "closed",
            "inAmount": "1000000000",
            "outAmount": "50000000",
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order_status(
                signature="txsig123",
                last_valid_block_height=12345678,
            )

            call_args = mock_instance.get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert params.get("lastValidBlockHeight") == 12345678
            assert result.success is True

    @pytest.mark.asyncio
    async def test_get_order_status_generic_exception(self, dflow_client):
        """Should return error on generic exception."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await dflow_client.get_order_status(signature="txsig123")

            assert result.success is False
            assert "Connection refused" in result.error

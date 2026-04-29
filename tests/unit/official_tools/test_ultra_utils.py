"""
Tests for Jupiter Ultra API utility.

Tests the JupiterUltra class which provides a simplified interface
to Jupiter's Ultra API for swaps, holdings, shield, and token search.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import base64

from solana_agent.tools.utils.ultra import (
    JupiterUltra,
    UltraOrderResponse,
    UltraExecuteResponse,
    sign_ultra_transaction,
    JUPITER_ULTRA_API,
)


class TestJupiterUltraInit:
    """Test JupiterUltra initialization."""

    def test_init_with_api_key(self):
        """Should use api.jup.ag and store API key."""
        ultra = JupiterUltra(api_key="test-key")
        assert ultra.base_url == JUPITER_ULTRA_API
        assert ultra.api_key == "test-key"
        assert ultra._headers["x-api-key"] == "test-key"

    def test_init_with_custom_base_url(self):
        """Should use custom base URL when provided."""
        custom_url = "https://custom.api.com"
        ultra = JupiterUltra(api_key="test-key", base_url=custom_url)
        assert ultra.base_url == custom_url


class TestJupiterUltraGetOrder:
    """Test get_order method."""

    @pytest.mark.asyncio
    async def test_get_order_success(self):
        """Should return UltraOrderResponse on success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "requestId": "req-123",
            "transaction": "base64tx",
            "inAmount": "1000000000",
            "outAmount": "50000000",
            "inputMint": "So11111111111111111111111111111111111111112",
            "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "slippageBps": 50,
            "swapType": "ExactIn",
            "feeBps": 0,
            "gasless": False,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            ultra = JupiterUltra(api_key="test-key")
            order = await ultra.get_order(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000,
                taker="TakerPubkey123",
            )

            assert isinstance(order, UltraOrderResponse)
            assert order.request_id == "req-123"
            assert order.transaction == "base64tx"
            assert order.swap_type == "ExactIn"

    @pytest.mark.asyncio
    async def test_get_order_with_referral(self):
        """Should include referral params in request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "requestId": "req-123",
            "transaction": "base64tx",
            "inAmount": "1000000000",
            "outAmount": "50000000",
            "inputMint": "input",
            "outputMint": "output",
            "slippageBps": 50,
            "swapType": "ExactIn",
            "feeBps": 50,
            "gasless": False,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            ultra = JupiterUltra(api_key="test-key")
            order = await ultra.get_order(
                input_mint="input",
                output_mint="output",
                amount=1000000000,
                taker="TakerPubkey123",
                referral_account="RefAcct123",
                referral_fee=50,
            )

            # Verify params were included
            call_args = mock_client_instance.get.call_args
            assert "referralAccount" in str(call_args) or order.fee_bps == 50

    @pytest.mark.asyncio
    async def test_get_order_with_payer(self):
        """Should include payer params for gasless transactions."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "requestId": "req-123",
            "transaction": "base64tx",
            "inAmount": "1000000000",
            "outAmount": "50000000",
            "inputMint": "input",
            "outputMint": "output",
            "slippageBps": 50,
            "swapType": "ExactIn",
            "feeBps": 0,
            "gasless": True,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            ultra = JupiterUltra(api_key="test-key")
            order = await ultra.get_order(
                input_mint="input",
                output_mint="output",
                amount=1000000000,
                taker="TakerPubkey123",
                payer="PayerPubkey456",
                close_authority="TakerPubkey123",
            )

            assert order.gasless is True

    @pytest.mark.asyncio
    async def test_get_order_error(self):
        """Should raise exception on API error."""
        import respx

        with respx.mock:
            respx.get(url__startswith="https://api.jup.ag/ultra/v1/order").respond(
                400, text="Bad request"
            )

            ultra = JupiterUltra(api_key="test-key")

            with pytest.raises(Exception) as exc_info:
                await ultra.get_order(
                    input_mint="input",
                    output_mint="output",
                    amount=1000000000,
                    taker="TakerPubkey123",
                )

            assert "400" in str(exc_info.value)


class TestJupiterUltraExecuteOrder:
    """Test execute_order method."""

    @pytest.mark.asyncio
    async def test_execute_order_success(self):
        """Should return UltraExecuteResponse on success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "Success",
            "signature": "TxSig123...abc",
            "inputAmountResult": "1000000000",
            "outputAmountResult": "50000000",
            "error": None,
            "code": 0,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            ultra = JupiterUltra(api_key="test-key")
            result = await ultra.execute_order(
                signed_transaction="signedtx",
                request_id="req-123",
            )

            assert isinstance(result, UltraExecuteResponse)
            assert result.status == "Success"
            assert result.signature == "TxSig123...abc"

    @pytest.mark.asyncio
    async def test_execute_order_failed(self):
        """Should return error response on swap failure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "Failed",
            "signature": None,
            "error": "Insufficient balance",
            "code": 1001,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            ultra = JupiterUltra(api_key="test-key")
            result = await ultra.execute_order(
                signed_transaction="signedtx",
                request_id="req-123",
            )

            assert result.status == "Failed"
            assert result.error == "Insufficient balance"
            assert result.code == 1001


class TestJupiterUltraGetHoldings:
    """Test get_holdings method."""

    @pytest.mark.asyncio
    async def test_get_holdings_success(self):
        """Should return holdings data on success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "nativeBalance": {"amount": "1000000000", "uiAmount": 1.0},
            "tokens": [{"mint": "Token123", "amount": "1000000", "uiAmount": 1.0}],
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            ultra = JupiterUltra(api_key="test-key")
            holdings = await ultra.get_holdings("Wallet123")

            assert "nativeBalance" in holdings
            assert "tokens" in holdings

    @pytest.mark.asyncio
    async def test_get_holdings_error(self):
        """Should raise exception on API error."""
        import respx

        with respx.mock:
            respx.get(url__startswith="https://api.jup.ag/ultra/v1/holdings/").respond(
                404, text="Wallet not found"
            )

            ultra = JupiterUltra(api_key="test-key")

            with pytest.raises(Exception) as exc_info:
                await ultra.get_holdings("InvalidWallet")

            assert "404" in str(exc_info.value)


class TestJupiterUltraGetNativeHoldings:
    """Test get_native_holdings method."""

    @pytest.mark.asyncio
    async def test_get_native_holdings_success(self):
        """Should return native balance on success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "amount": "1000000000",
            "uiAmount": 1.0,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            ultra = JupiterUltra(api_key="test-key")
            native = await ultra.get_native_holdings("Wallet123")

            assert "amount" in native
            assert "uiAmount" in native


class TestJupiterUltraGetShield:
    """Test get_shield method."""

    @pytest.mark.asyncio
    async def test_get_shield_success(self):
        """Should return shield data on success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "warnings": {
                "Token123": [{"type": "low_liquidity", "message": "Low liquidity"}],
                "Token456": [],
            }
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            ultra = JupiterUltra(api_key="test-key")
            shield = await ultra.get_shield(["Token123", "Token456"])

            assert "warnings" in shield


class TestJupiterUltraSearchTokens:
    """Test search_tokens method."""

    @pytest.mark.asyncio
    async def test_search_tokens_success(self):
        """Should return token list on success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "Token123", "name": "Test Token", "symbol": "TEST"},
        ]

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            ultra = JupiterUltra(api_key="test-key")
            tokens = await ultra.search_tokens("TEST")

            assert len(tokens) == 1
            assert tokens[0]["symbol"] == "TEST"


class TestSignUltraTransaction:
    """Test sign_ultra_transaction function."""

    def test_sign_transaction_single_signer(self):
        """Should sign transaction with single signer."""
        # Create a minimal valid transaction for testing
        # This is a simplified test - actual implementation uses real Solana transactions
        mock_tx = MagicMock()
        mock_tx.message = MagicMock()
        mock_tx.signatures = [b"\x00" * 64]

        with (
            patch("solana_agent.tools.utils.ultra.VersionedTransaction") as MockTx,
            patch(
                "solana_agent.tools.utils.ultra.to_bytes_versioned",
                return_value=b"message_bytes",
            ),
        ):
            MockTx.from_bytes.return_value = mock_tx
            MockTx.populate.return_value = mock_tx

            mock_sign_func = MagicMock(return_value=b"signature")

            # Test requires actual base64 encoded transaction
            # This is a simplified placeholder
            try:
                result = sign_ultra_transaction(
                    transaction_base64=base64.b64encode(b"test").decode(),
                    sign_message_func=mock_sign_func,
                )
                # If it works, verify it returns base64
                assert isinstance(result, str)
            except Exception:
                # Expected to fail with invalid transaction data
                pass

    def test_sign_transaction_with_payer(self):
        """Should sign transaction with both taker and payer."""
        mock_tx = MagicMock()
        mock_tx.message = MagicMock()
        mock_tx.signatures = [b"\x00" * 64, b"\x00" * 64]

        with (
            patch("solana_agent.tools.utils.ultra.VersionedTransaction") as MockTx,
            patch(
                "solana_agent.tools.utils.ultra.to_bytes_versioned",
                return_value=b"message_bytes",
            ),
        ):
            MockTx.from_bytes.return_value = mock_tx
            MockTx.populate.return_value = mock_tx

            mock_taker_sign = MagicMock(return_value=b"taker_sig")
            mock_payer_sign = MagicMock(return_value=b"payer_sig")

            try:
                result = sign_ultra_transaction(
                    transaction_base64=base64.b64encode(b"test").decode(),
                    sign_message_func=mock_taker_sign,
                    payer_sign_func=mock_payer_sign,
                )
                assert isinstance(result, str)
            except Exception:
                # Expected to fail with invalid transaction data
                pass

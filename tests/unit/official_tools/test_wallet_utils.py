"""
Tests for Solana Wallet utility.

Tests the SolanaWalletClient which provides wallet operations
for Solana transactions.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestSolanaWalletClientInit:
    """Test SolanaWalletClient initialization."""

    def test_init_with_keypair(self):
        """Should initialize with keypair."""
        from solana_agent.tools.utils.wallet import SolanaWalletClient

        mock_keypair = MagicMock()
        mock_keypair.pubkey.return_value = "TestPubkey123"

        with patch("solana_agent.tools.utils.wallet.AsyncClient"):
            wallet = SolanaWalletClient(
                rpc_url="https://api.mainnet-beta.solana.com",
                keypair=mock_keypair,
            )

            assert wallet.keypair == mock_keypair
            assert wallet.pubkey == "TestPubkey123"

    def test_init_with_pubkey_string(self):
        """Should initialize with public key string."""
        from solana_agent.tools.utils.wallet import SolanaWalletClient

        with (
            patch("solana_agent.tools.utils.wallet.AsyncClient"),
            patch("solana_agent.tools.utils.wallet.Pubkey") as MockPubkey,
        ):
            MockPubkey.from_string.return_value = "ParsedPubkey123"

            wallet = SolanaWalletClient(
                rpc_url="https://api.mainnet-beta.solana.com",
                pubkey="TestPubkeyString",
            )

            assert wallet.pubkey == "ParsedPubkey123"

    def test_init_stores_rpc_url(self):
        """Should store RPC URL."""
        from solana_agent.tools.utils.wallet import SolanaWalletClient

        with patch("solana_agent.tools.utils.wallet.AsyncClient"):
            wallet = SolanaWalletClient(
                rpc_url="https://api.mainnet-beta.solana.com",
            )

            assert wallet.rpc_url == "https://api.mainnet-beta.solana.com"

    def test_init_fee_payer_defaults_to_none(self):
        """Should always set fee_payer attribute (default None)."""
        from solana_agent.tools.utils.wallet import SolanaWalletClient

        with patch("solana_agent.tools.utils.wallet.AsyncClient"):
            wallet = SolanaWalletClient(
                rpc_url="https://api.mainnet-beta.solana.com",
            )

            assert hasattr(wallet, "fee_payer")
            assert wallet.fee_payer is None


class TestSolanaWalletClientSignMessage:
    """Test sign_message method."""

    def test_sign_message(self):
        """Should sign message with keypair."""
        from solana_agent.tools.utils.wallet import SolanaWalletClient

        mock_keypair = MagicMock()
        mock_keypair.pubkey.return_value = "TestPubkey123"
        mock_keypair.secret.return_value = b"\x00" * 32

        with (
            patch("solana_agent.tools.utils.wallet.AsyncClient"),
            patch("nacl.signing.SigningKey") as MockSigningKey,
        ):
            mock_signing_key = MagicMock()
            mock_signed = MagicMock()
            mock_signed.signature = b"\x01" * 64
            mock_signing_key.sign.return_value = mock_signed
            MockSigningKey.return_value = mock_signing_key

            wallet = SolanaWalletClient(
                rpc_url="https://api.mainnet-beta.solana.com",
                keypair=mock_keypair,
            )

            signature = wallet.sign_message(b"test message")

            # Should return a Signature object
            assert signature is not None


class TestSolanaWalletClientGetPriorityFeeEstimate:
    """Test get_priority_fee_estimate_helius method."""

    @pytest.mark.asyncio
    async def test_get_priority_fee_estimate_success(self):
        """Should return priority fee estimate."""
        from solana_agent.tools.utils.wallet import SolanaWalletClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"priorityFeeEstimate": 1000}}

        with (
            patch("solana_agent.tools.utils.wallet.AsyncClient"),
            patch("httpx.AsyncClient") as MockHttpx,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            MockHttpx.return_value = mock_client

            wallet = SolanaWalletClient(
                rpc_url="https://rpc.helius.xyz/?api-key=test",
            )

            fee = await wallet.get_priority_fee_estimate_helius("base64tx")

            assert fee == 1000

    @pytest.mark.asyncio
    async def test_get_priority_fee_estimate_error(self):
        """Should raise exception on API error."""
        from solana_agent.tools.utils.wallet import SolanaWalletClient
        import respx

        with respx.mock:
            respx.post("https://rpc.helius.xyz/?api-key=test").respond(
                200, json={"error": {"message": "Invalid transaction"}}
            )

            with patch("solana_agent.tools.utils.wallet.AsyncClient"):
                wallet = SolanaWalletClient(
                    rpc_url="https://rpc.helius.xyz/?api-key=test",
                )

                with pytest.raises(RuntimeError) as exc_info:
                    await wallet.get_priority_fee_estimate_helius("invalidtx")

                assert "failed" in str(exc_info.value).lower()


class TestSolanaTransaction:
    """Test SolanaTransaction class."""

    def test_init_with_instructions(self):
        """Should initialize with instructions."""
        from solana_agent.tools.utils.wallet import SolanaTransaction

        mock_instruction = MagicMock()
        tx = SolanaTransaction(instructions=[mock_instruction])

        assert len(tx.instructions) == 1
        assert tx.instructions[0] == mock_instruction

    def test_init_with_signers(self):
        """Should initialize with accounts to sign."""
        from solana_agent.tools.utils.wallet import SolanaTransaction

        mock_instruction = MagicMock()
        mock_signer = MagicMock()
        tx = SolanaTransaction(
            instructions=[mock_instruction],
            accounts_to_sign=[mock_signer],
        )

        assert tx.accounts_to_sign == [mock_signer]

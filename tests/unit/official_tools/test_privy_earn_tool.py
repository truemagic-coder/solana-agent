"""
Tests for Privy Earn Tool.
"""

import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from solders.hash import Hash

from solana_agent.tools.privy_earn import (
    PrivyEarnTool,
    PrivyEarnPlugin,
    SOL_MINT,
    USDC_MINT,
    _build_instruction,
    _normalize_asset,
    get_plugin,
)


@pytest.fixture
def privy_earn_tool():
    tool = PrivyEarnTool()
    tool.configure(
        {
            "tools": {
                "privy_earn": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                    "signing_key": "wallet-auth:test-signing-key",
                    "jupiter_api_key": "test-api-key",
                    "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                }
            }
        }
    )
    return tool


@pytest.fixture
def privy_earn_tool_incomplete():
    tool = PrivyEarnTool()
    tool.configure({"tools": {"privy_earn": {"app_id": "test-app-id"}}})
    return tool


class TestPrivyEarnToolSchema:
    def test_tool_name(self, privy_earn_tool):
        assert privy_earn_tool.name == "privy_earn"

    def test_schema_required_fields(self, privy_earn_tool):
        schema = privy_earn_tool.get_schema()
        assert "wallet_id" in schema["required"]
        assert "wallet_public_key" in schema["required"]
        assert "action" in schema["required"]


class TestPrivyEarnToolExecute:
    @pytest.mark.asyncio
    async def test_execute_missing_wallet_params(self, privy_earn_tool):
        result = await privy_earn_tool.execute(
            wallet_id="",
            wallet_public_key="",
            action="deposit",
            asset=SOL_MINT,
            amount="1",
        )

        assert result["status"] == "error"
        assert "wallet_id" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_config(self, privy_earn_tool_incomplete):
        result = await privy_earn_tool_incomplete.execute(
            wallet_id="wallet-123",
            wallet_public_key="WalletPubkey123",
            action="deposit",
            asset=SOL_MINT,
            amount="1",
        )

        assert result["status"] == "error"
        assert "config" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_api_key(self, privy_earn_tool):
        privy_earn_tool._jupiter_api_key = None

        result = await privy_earn_tool.execute(
            wallet_id="wallet-123",
            wallet_public_key="WalletPubkey123",
            action="tokens",
        )

        assert result["status"] == "error"
        assert "api key" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_rpc_url(self, privy_earn_tool):
        privy_earn_tool._rpc_url = None

        with patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy:
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "error"
        assert "rpc" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_asset(self, privy_earn_tool):
        with patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy:
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="deposit",
                asset="BadMint",
                amount="1",
            )

        assert result["status"] == "error"
        assert "sol" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_amount(self, privy_earn_tool):
        with patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy:
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="deposit",
                asset=SOL_MINT,
            )

        assert result["status"] == "error"
        assert "amount" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_shares(self, privy_earn_tool):
        with patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy:
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="mint",
                asset=SOL_MINT,
            )

        assert result["status"] == "error"
        assert "shares" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_instruction_error(self, privy_earn_tool):
        with (
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
        ):
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            mock_earn = MagicMock()
            mock_earn.get_deposit_instructions = AsyncMock(
                return_value=MagicMock(success=False, error="nope", instruction=None)
            )
            MockEarn.return_value = mock_earn

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "error"
        assert "nope" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_build_instruction_error(self, privy_earn_tool):
        bad_instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": "not-base64",
        }
        with (
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
        ):
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            mock_earn = MagicMock()
            mock_earn.get_deposit_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=bad_instruction
                )
            )
            MockEarn.return_value = mock_earn

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "error"
        assert "invalid instruction data" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_blockhash_error(self, privy_earn_tool):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }
        with (
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch(
                "solana_agent.tools.privy_earn.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
        ):
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            mock_earn = MagicMock()
            mock_earn.get_deposit_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=instruction
                )
            )
            MockEarn.return_value = mock_earn
            mock_blockhash.return_value = {"error": "no blockhash"}

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "error"
        assert "blockhash" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_privy_sign_error(self, privy_earn_tool):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }
        with (
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch(
                "solana_agent.tools.privy_earn.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.privy_earn._privy_sign_transaction",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            mock_earn = MagicMock()
            mock_earn.get_deposit_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=instruction
                )
            )
            MockEarn.return_value = mock_earn
            mock_blockhash.return_value = {"blockhash": Hash.default()}

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key=SOL_MINT,
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "error"
        assert "sign transaction" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_send_error(self, privy_earn_tool):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }
        signed_tx = base64.b64encode(b"signed").decode("utf-8")
        with (
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch(
                "solana_agent.tools.privy_earn.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.privy_earn._privy_sign_transaction",
                new_callable=AsyncMock,
                return_value=signed_tx,
            ),
            patch(
                "solana_agent.tools.privy_earn.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            mock_earn = MagicMock()
            mock_earn.get_deposit_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=instruction
                )
            )
            MockEarn.return_value = mock_earn
            mock_blockhash.return_value = {"blockhash": Hash.default()}
            mock_send.return_value = {"success": False, "error": "send failed"}

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key=SOL_MINT,
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "error"
        assert "send failed" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_success_deposit(self, privy_earn_tool):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }
        signed_tx = base64.b64encode(b"signed").decode("utf-8")
        with (
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch(
                "solana_agent.tools.privy_earn.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.privy_earn._privy_sign_transaction",
                new_callable=AsyncMock,
                return_value=signed_tx,
            ),
            patch(
                "solana_agent.tools.privy_earn.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            mock_earn = MagicMock()
            mock_earn.get_deposit_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=instruction
                )
            )
            MockEarn.return_value = mock_earn
            mock_blockhash.return_value = {"blockhash": Hash.default()}
            mock_send.return_value = {"success": True, "signature": "sig"}

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key=SOL_MINT,
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "success"
        assert result["signature"] == "sig"

    @pytest.mark.asyncio
    async def test_tokens_action_filters(self, privy_earn_tool):
        mock_tokens = [
            {"asset": {"address": SOL_MINT}},
            {"asset": {"address": USDC_MINT}},
            {"asset": {"address": "OtherMint"}},
        ]

        with (
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
        ):
            mock_earn = MagicMock()
            mock_earn.get_tokens = AsyncMock(
                return_value={"success": True, "tokens": mock_tokens}
            )
            MockEarn.return_value = mock_earn

            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="tokens",
            )

        assert result["status"] == "success"
        assert len(result["tokens"]) == 2

    @pytest.mark.asyncio
    async def test_tokens_error(self, privy_earn_tool):
        with (
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
        ):
            mock_earn = MagicMock()
            mock_earn.get_tokens = AsyncMock(
                return_value={"success": False, "error": "token error"}
            )
            MockEarn.return_value = mock_earn

            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="tokens",
            )

        assert result["status"] == "error"
        assert "token error" in result["message"]

    @pytest.mark.asyncio
    async def test_positions_error(self, privy_earn_tool):
        with (
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
        ):
            mock_earn = MagicMock()
            mock_earn.get_positions = AsyncMock(
                return_value={"success": False, "error": "positions error"}
            )
            MockEarn.return_value = mock_earn

            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="positions",
                users="UserA",
            )

        assert result["status"] == "error"
        assert "positions error" in result["message"]

    @pytest.mark.asyncio
    async def test_positions_defaults_to_wallet(self, privy_earn_tool):
        with (
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
        ):
            mock_earn = MagicMock()
            mock_earn.get_positions = AsyncMock(
                return_value={"success": True, "positions": []}
            )
            MockEarn.return_value = mock_earn

            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="positions",
            )

        assert result["status"] == "success"
        mock_earn.get_positions.assert_awaited_once_with(["WalletPubkey123"])

    @pytest.mark.asyncio
    async def test_earnings_requires_positions(self, privy_earn_tool):
        with patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy:
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="earnings",
            )

        assert result["status"] == "error"
        assert "positions" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_earnings_empty_positions_list(self, privy_earn_tool):
        with patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy:
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="earnings",
                positions=",",
            )

        assert result["status"] == "error"
        assert "positions" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_earnings_error(self, privy_earn_tool):
        with (
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
        ):
            mock_earn = MagicMock()
            mock_earn.get_earnings = AsyncMock(
                return_value={"success": False, "error": "earnings error"}
            )
            MockEarn.return_value = mock_earn

            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="earnings",
                user="UserA",
                positions="Pos1",
            )

        assert result["status"] == "error"
        assert "earnings error" in result["message"]

    @pytest.mark.asyncio
    async def test_earnings_default_user(self, privy_earn_tool):
        with (
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
        ):
            mock_earn = MagicMock()
            mock_earn.get_earnings = AsyncMock(
                return_value={"success": True, "earnings": []}
            )
            MockEarn.return_value = mock_earn

            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="earnings",
                positions="Pos1",
            )

        assert result["status"] == "success"
        mock_earn.get_earnings.assert_awaited_once_with("WalletPubkey123", ["Pos1"])

    @pytest.mark.asyncio
    async def test_unknown_action(self, privy_earn_tool):
        with patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy:
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="unknown",
            )

        assert result["status"] == "error"
        assert "unknown" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_withdraw_success(self, privy_earn_tool):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }
        signed_tx = base64.b64encode(b"signed").decode("utf-8")
        with (
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch(
                "solana_agent.tools.privy_earn.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.privy_earn._privy_sign_transaction",
                new_callable=AsyncMock,
                return_value=signed_tx,
            ),
            patch(
                "solana_agent.tools.privy_earn.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            mock_earn = MagicMock()
            mock_earn.get_withdraw_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=instruction
                )
            )
            MockEarn.return_value = mock_earn
            mock_blockhash.return_value = {"blockhash": Hash.default()}
            mock_send.return_value = {"success": True, "signature": "sig"}

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key=SOL_MINT,
                action="withdraw",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_redeem_success(self, privy_earn_tool):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }
        signed_tx = base64.b64encode(b"signed").decode("utf-8")
        with (
            patch("solana_agent.tools.privy_earn.AsyncPrivyAPI") as MockPrivy,
            patch("solana_agent.tools.privy_earn.JupiterEarn") as MockEarn,
            patch(
                "solana_agent.tools.privy_earn.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.privy_earn._privy_sign_transaction",
                new_callable=AsyncMock,
                return_value=signed_tx,
            ),
            patch(
                "solana_agent.tools.privy_earn.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            MockPrivy.return_value = mock_client

            mock_earn = MagicMock()
            mock_earn.get_redeem_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=instruction
                )
            )
            MockEarn.return_value = mock_earn
            mock_blockhash.return_value = {"blockhash": Hash.default()}
            mock_send.return_value = {"success": True, "signature": "sig"}

            result = await privy_earn_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key=SOL_MINT,
                action="redeem",
                asset=SOL_MINT,
                shares="1",
            )

        assert result["status"] == "success"


class TestPrivyEarnPlugin:
    def test_plugin_name(self):
        plugin = PrivyEarnPlugin()
        assert plugin.name == "privy_earn"

    def test_plugin_description(self):
        plugin = PrivyEarnPlugin()
        assert "jupiter" in plugin.description.lower()

    def test_get_plugin(self):
        plugin = get_plugin()
        assert plugin.name == "privy_earn"


class TestPrivyEarnHelpers:
    def test_normalize_asset(self):
        assert _normalize_asset("sol") == SOL_MINT
        assert _normalize_asset("USDC") == USDC_MINT
        assert _normalize_asset("Other") == "Other"
        assert _normalize_asset("") is None

    def test_build_instruction(self):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [
                {
                    "pubkey": USDC_MINT,
                    "isSigner": True,
                    "isWritable": False,
                }
            ],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }

        ix = _build_instruction(instruction)

        assert len(ix.accounts) == 1

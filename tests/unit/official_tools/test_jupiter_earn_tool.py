"""
Tests for Jupiter Earn Tool.
"""

import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from solders.keypair import Keypair
from solders.hash import Hash

from solana_agent.tools.jupiter_earn import (
    JupiterEarnTool,
    JupiterEarnPlugin,
    SOL_MINT,
    USDC_MINT,
    _build_instruction,
    _normalize_asset,
    get_plugin,
)
from solana_agent.tools.utils.earn import JupiterEarn


@pytest.fixture
def earn_tool():
    tool = JupiterEarnTool()
    tool.configure(
        {
            "tools": {
                "jupiter_earn": {
                    "private_key": "5jGR...base58privatekey",
                    "jupiter_api_key": "test-api-key",
                    "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                }
            }
        }
    )
    return tool


class TestJupiterEarnToolSchema:
    def test_tool_name(self, earn_tool):
        assert earn_tool.name == "jupiter_earn"

    def test_schema_has_actions(self, earn_tool):
        schema = earn_tool.get_schema()
        assert "action" in schema["properties"]
        assert "deposit" in schema["properties"]["action"]["enum"]
        assert "earnings" in schema["properties"]["action"]["enum"]

    def test_schema_required_fields(self, earn_tool):
        schema = earn_tool.get_schema()
        assert "asset" in schema["required"]
        assert "amount" in schema["required"]
        assert "shares" in schema["required"]


class TestJupiterEarnToolConfigure:
    def test_configure_stores_config(self, earn_tool):
        assert earn_tool._private_key == "5jGR...base58privatekey"
        assert earn_tool._jupiter_api_key == "test-api-key"
        assert earn_tool._rpc_url == "https://mainnet.helius-rpc.com/?api-key=test-key"


class TestJupiterEarnToolExecute:
    @pytest.mark.asyncio
    async def test_execute_missing_private_key(self):
        tool = JupiterEarnTool()
        tool.configure({"tools": {"jupiter_earn": {"jupiter_api_key": "key"}}})

        result = await tool.execute(action="deposit", asset=SOL_MINT, amount="1")

        assert result["status"] == "error"
        assert "private" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_api_key(self):
        tool = JupiterEarnTool()
        tool.configure({"tools": {"jupiter_earn": {"private_key": "key"}}})

        result = await tool.execute(action="tokens")

        assert result["status"] == "error"
        assert "api key" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_rpc_url(self, earn_tool):
        earn_tool._rpc_url = None

        result = await earn_tool.execute(action="deposit", asset=SOL_MINT, amount="1")

        assert result["status"] == "error"
        assert "rpc" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_asset(self, earn_tool):
        result = await earn_tool.execute(
            action="deposit",
            asset="BadMint",
            amount="1",
        )

        assert result["status"] == "error"
        assert "sol" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_amount(self, earn_tool):
        with patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair:
            MockKeypair.from_base58_string.return_value = Keypair()

            result = await earn_tool.execute(action="deposit", asset=SOL_MINT)

        assert result["status"] == "error"
        assert "amount" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_shares(self, earn_tool):
        with patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair:
            MockKeypair.from_base58_string.return_value = Keypair()

            result = await earn_tool.execute(action="mint", asset=SOL_MINT)

        assert result["status"] == "error"
        assert "shares" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_instruction_error(self, earn_tool):
        with (
            patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn,
            patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair,
        ):
            MockKeypair.from_base58_string.return_value = Keypair()

            mock_earn = MagicMock()
            mock_earn.get_deposit_instructions = AsyncMock(
                return_value=MagicMock(success=False, error="nope", instruction=None)
            )
            MockEarn.return_value = mock_earn

            result = await earn_tool.execute(
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "error"
        assert "nope" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_build_instruction_error(self, earn_tool):
        bad_instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": "not-base64",
        }
        with (
            patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn,
            patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair,
        ):
            MockKeypair.from_base58_string.return_value = Keypair()

            mock_earn = MagicMock()
            mock_earn.get_deposit_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=bad_instruction
                )
            )
            MockEarn.return_value = mock_earn

            result = await earn_tool.execute(
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "error"
        assert "invalid instruction data" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_blockhash_error(self, earn_tool):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }
        with (
            patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn,
            patch(
                "solana_agent.tools.jupiter_earn.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair,
        ):
            MockKeypair.from_base58_string.return_value = Keypair()

            mock_earn = MagicMock()
            mock_earn.get_deposit_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=instruction
                )
            )
            MockEarn.return_value = mock_earn
            mock_blockhash.return_value = {"error": "no blockhash"}

            result = await earn_tool.execute(
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "error"
        assert "blockhash" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_send_error(self, earn_tool):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }
        with (
            patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn,
            patch(
                "solana_agent.tools.jupiter_earn.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.jupiter_earn.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
            patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair,
        ):
            MockKeypair.from_base58_string.return_value = Keypair()

            mock_earn = MagicMock()
            mock_earn.get_deposit_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=instruction
                )
            )
            MockEarn.return_value = mock_earn
            mock_blockhash.return_value = {"blockhash": Hash.default()}
            mock_send.return_value = {"success": False, "error": "send failed"}

            result = await earn_tool.execute(
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "error"
        assert "send failed" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_success_deposit(self, earn_tool):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }
        with (
            patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn,
            patch(
                "solana_agent.tools.jupiter_earn.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.jupiter_earn.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
            patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair,
        ):
            MockKeypair.from_base58_string.return_value = Keypair()

            mock_earn = MagicMock()
            mock_earn.get_deposit_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=instruction
                )
            )
            MockEarn.return_value = mock_earn
            mock_blockhash.return_value = {"blockhash": Hash.default()}
            mock_send.return_value = {"success": True, "signature": "sig"}

            result = await earn_tool.execute(
                action="deposit",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "success"
        assert result["signature"] == "sig"

    @pytest.mark.asyncio
    async def test_positions_defaults_to_configured_wallet(self, earn_tool):
        with (
            patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair,
            patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn,
        ):
            keypair = Keypair()
            MockKeypair.from_base58_string.return_value = keypair

            mock_earn = MagicMock()
            mock_earn.get_positions = AsyncMock(
                return_value={"success": True, "positions": []}
            )
            MockEarn.return_value = mock_earn

            result = await earn_tool.execute(action="positions")

        assert result["status"] == "success"
        mock_earn.get_positions.assert_awaited_once_with([str(keypair.pubkey())])

    @pytest.mark.asyncio
    async def test_positions_missing_private_key(self, earn_tool):
        earn_tool._private_key = None

        result = await earn_tool.execute(action="positions")

        assert result["status"] == "error"
        assert "private key" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_positions_error(self, earn_tool):
        with patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn:
            mock_earn = MagicMock()
            mock_earn.get_positions = AsyncMock(
                return_value={"success": False, "error": "no positions"}
            )
            MockEarn.return_value = mock_earn

            result = await earn_tool.execute(action="positions", users="UserA")

        assert result["status"] == "error"
        assert "no positions" in result["message"]

    @pytest.mark.asyncio
    async def test_earnings_requires_positions(self, earn_tool):
        result = await earn_tool.execute(action="earnings", user="UserA")

        assert result["status"] == "error"
        assert "positions" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_earnings_missing_user_without_private_key(self, earn_tool):
        earn_tool._private_key = None

        result = await earn_tool.execute(action="earnings", positions="Pos1")

        assert result["status"] == "error"
        assert "user is required" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_earnings_empty_positions_list(self, earn_tool):
        with patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair:
            MockKeypair.from_base58_string.return_value = Keypair()

            result = await earn_tool.execute(action="earnings", positions=",")

        assert result["status"] == "error"
        assert "positions" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_withdraw_success(self, earn_tool):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }
        with (
            patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn,
            patch(
                "solana_agent.tools.jupiter_earn.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.jupiter_earn.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
            patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair,
        ):
            MockKeypair.from_base58_string.return_value = Keypair()

            mock_earn = MagicMock()
            mock_earn.get_withdraw_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=instruction
                )
            )
            MockEarn.return_value = mock_earn
            mock_blockhash.return_value = {"blockhash": Hash.default()}
            mock_send.return_value = {"success": True, "signature": "sig"}

            result = await earn_tool.execute(
                action="withdraw",
                asset=SOL_MINT,
                amount="1",
            )

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_redeem_success(self, earn_tool):
        instruction = {
            "programId": SOL_MINT,
            "accounts": [],
            "data": base64.b64encode(b"\x01").decode("utf-8"),
        }
        with (
            patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn,
            patch(
                "solana_agent.tools.jupiter_earn.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.jupiter_earn.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
            patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair,
        ):
            MockKeypair.from_base58_string.return_value = Keypair()

            mock_earn = MagicMock()
            mock_earn.get_redeem_instructions = AsyncMock(
                return_value=MagicMock(
                    success=True, error=None, instruction=instruction
                )
            )
            MockEarn.return_value = mock_earn
            mock_blockhash.return_value = {"blockhash": Hash.default()}
            mock_send.return_value = {"success": True, "signature": "sig"}

            result = await earn_tool.execute(
                action="redeem",
                asset=SOL_MINT,
                shares="1",
            )

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_earnings_default_user(self, earn_tool):
        with (
            patch("solana_agent.tools.jupiter_earn.Keypair") as MockKeypair,
            patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn,
        ):
            keypair = Keypair()
            MockKeypair.from_base58_string.return_value = keypair

            mock_earn = MagicMock()
            mock_earn.get_earnings = AsyncMock(
                return_value={"success": True, "earnings": []}
            )
            MockEarn.return_value = mock_earn

            result = await earn_tool.execute(
                action="earnings",
                positions="Pos1",
            )

        assert result["status"] == "success"
        mock_earn.get_earnings.assert_awaited_once_with(str(keypair.pubkey()), ["Pos1"])

    @pytest.mark.asyncio
    async def test_earnings_error(self, earn_tool):
        with patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn:
            mock_earn = MagicMock()
            mock_earn.get_earnings = AsyncMock(
                return_value={"success": False, "error": "no earnings"}
            )
            MockEarn.return_value = mock_earn

            result = await earn_tool.execute(
                action="earnings",
                user="UserA",
                positions="Pos1",
            )

        assert result["status"] == "error"
        assert "no earnings" in result["message"]

    @pytest.mark.asyncio
    async def test_tokens_error(self, earn_tool):
        with patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn:
            mock_earn = MagicMock()
            mock_earn.get_tokens = AsyncMock(
                return_value={"success": False, "error": "token error"}
            )
            MockEarn.return_value = mock_earn

            result = await earn_tool.execute(action="tokens")

        assert result["status"] == "error"
        assert "token error" in result["message"]

    @pytest.mark.asyncio
    async def test_unknown_action(self, earn_tool):
        result = await earn_tool.execute(action="unknown")

        assert result["status"] == "error"
        assert "unknown" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_tokens_action_filters(self, earn_tool):
        mock_tokens = [
            {"asset": {"address": SOL_MINT}},
            {"asset": {"address": USDC_MINT}},
            {"asset": {"address": "OtherMint"}},
        ]

        with patch("solana_agent.tools.jupiter_earn.JupiterEarn") as MockEarn:
            mock_earn = MagicMock()
            mock_earn.get_tokens = AsyncMock(
                return_value={"success": True, "tokens": mock_tokens}
            )
            MockEarn.return_value = mock_earn

            result = await earn_tool.execute(action="tokens")

        assert result["status"] == "success"
        assert len(result["tokens"]) == 2


class TestJupiterEarnPlugin:
    def test_plugin_name(self):
        plugin = JupiterEarnPlugin()
        assert plugin.name == "jupiter_earn"

    def test_plugin_description(self):
        plugin = JupiterEarnPlugin()
        assert "jupiter" in plugin.description.lower()

    def test_get_plugin(self):
        plugin = get_plugin()
        assert plugin.name == "jupiter_earn"


class TestJupiterEarnHelpers:
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


class TestJupiterEarnApiClient:
    @pytest.mark.asyncio
    async def test_instruction_success(self):
        class DummyResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"programId": SOL_MINT, "accounts": [], "data": "AQ=="}

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def post(self, *_args, **_kwargs):
                return DummyResponse()

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_deposit_instructions(SOL_MINT, "Signer", "1")

        assert result.success is True
        assert result.instruction is not None

    @pytest.mark.asyncio
    async def test_instruction_non_200(self):
        class DummyResponse:
            status_code = 500
            text = "nope"

            @staticmethod
            def json():
                return {}

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def post(self, *_args, **_kwargs):
                return DummyResponse()

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_withdraw_instructions(SOL_MINT, "Signer", "1")

        assert result.success is False
        assert "Failed to fetch" in result.error

    @pytest.mark.asyncio
    async def test_instruction_exception(self):
        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def post(self, *_args, **_kwargs):
                raise RuntimeError("boom")

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_mint_instructions(SOL_MINT, "Signer", "1")

        assert result.success is False
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_tokens_success(self):
        class DummyResponse:
            status_code = 200

            @staticmethod
            def json():
                return [{"asset": {"address": SOL_MINT}}]

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, *_args, **_kwargs):
                return DummyResponse()

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_tokens()

        assert result["success"] is True
        assert result["tokens"]

    @pytest.mark.asyncio
    async def test_tokens_non_200(self):
        class DummyResponse:
            status_code = 500
            text = "nope"

            @staticmethod
            def json():
                return []

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, *_args, **_kwargs):
                return DummyResponse()

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_tokens()

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_positions_success(self):
        class DummyResponse:
            status_code = 200

            @staticmethod
            def json():
                return [{"token": {"assetAddress": SOL_MINT}}]

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, *_args, **_kwargs):
                return DummyResponse()

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_positions(["User"])

        assert result["success"] is True
        assert result["positions"]

    @pytest.mark.asyncio
    async def test_earnings_success(self):
        class DummyResponse:
            status_code = 200

            @staticmethod
            def json():
                return [{"address": SOL_MINT}]

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, *_args, **_kwargs):
                return DummyResponse()

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_earnings("User", ["Pos1"])

        assert result["success"] is True
        assert result["earnings"]

    @pytest.mark.asyncio
    async def test_redeem_instructions_success(self):
        class DummyResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"programId": SOL_MINT, "accounts": [], "data": "AQ=="}

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def post(self, *_args, **_kwargs):
                return DummyResponse()

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_redeem_instructions(SOL_MINT, "Signer", "1")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_tokens_exception(self):
        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, *_args, **_kwargs):
                raise RuntimeError("boom")

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_tokens()

        assert result["success"] is False
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_positions_error(self):
        class DummyResponse:
            status_code = 503
            text = "down"

            @staticmethod
            def json():
                return {}

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, *_args, **_kwargs):
                return DummyResponse()

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_positions(["User"])

        assert result["success"] is False
        assert "Failed to fetch" in result["error"]

    @pytest.mark.asyncio
    async def test_positions_exception(self):
        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, *_args, **_kwargs):
                raise RuntimeError("boom")

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_positions(["User"])

        assert result["success"] is False
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_earnings_error(self):
        class DummyResponse:
            status_code = 400
            text = "bad"

            @staticmethod
            def json():
                return {}

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, *_args, **_kwargs):
                return DummyResponse()

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_earnings("User", ["Pos1"])

        assert result["success"] is False
        assert "Failed to fetch" in result["error"]

    @pytest.mark.asyncio
    async def test_earnings_exception(self):
        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, *_args, **_kwargs):
                raise RuntimeError("boom")

        with patch(
            "solana_agent.tools.utils.earn.httpx.AsyncClient",
            return_value=DummyClient(),
        ):
            earn = JupiterEarn(api_key="key")
            result = await earn.get_earnings("User", ["Pos1"])

        assert result["success"] is False
        assert "boom" in result["error"]

"""Tests for the Kamino tool."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.transaction import VersionedTransaction

from solana_agent.tools.kamino import KaminoPlugin, KaminoTool, get_plugin


def _make_unsigned_transaction_base64(signer: Keypair) -> str:
    message = MessageV0.try_compile(
        payer=signer.pubkey(),
        instructions=[],
        address_lookup_table_accounts=[],
        recent_blockhash=Hash.default(),
    )
    unsigned_transaction = VersionedTransaction.populate(
        message,
        [signer.sign_message(bytes(message))],
    )
    return base64.b64encode(bytes(unsigned_transaction)).decode("utf-8")


@pytest.fixture
def kamino_tool():
    referrer_signer = Keypair()
    tool = KaminoTool()
    tool.configure(
        {
            "tools": {
                "kamino": {
                    "private_key": "5jGR...base58privatekey",
                    "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                    "referrer_private_key": str(referrer_signer),
                }
            }
        }
    )
    return tool


class TestKaminoToolSchema:
    def test_tool_name(self, kamino_tool):
        assert kamino_tool.name == "kamino"

    def test_schema_has_key_actions(self, kamino_tool):
        schema = kamino_tool.get_schema()
        assert "borrow_deposit" in schema["properties"]["action"]["enum"]
        assert "api_post" in schema["properties"]["action"]["enum"]
        assert "setup_referrer" not in schema["properties"]["action"]["enum"]
        assert "withdraw_referrer_fees" not in schema["properties"]["action"]["enum"]


class TestKaminoToolExecute:
    @pytest.mark.asyncio
    async def test_missing_private_key_for_transaction_action(self):
        tool = KaminoTool()
        tool.configure({"tools": {"kamino": {"rpc_url": "https://rpc.example.com"}}})

        result = await tool.execute(action="earn_deposit", kvault="vault", amount="1.0")

        assert result["status"] == "error"
        assert "private" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_missing_rpc_url_for_transaction_action(self):
        tool = KaminoTool()
        tool.configure({"tools": {"kamino": {"private_key": str(Keypair())}}})

        result = await tool.execute(action="earn_deposit", kvault="vault", amount="1.0")

        assert result == {"status": "error", "message": "RPC URL not configured."}

    @pytest.mark.asyncio
    async def test_list_vaults_success(self, kamino_tool):
        with patch("solana_agent.tools.kamino.KaminoAPI") as MockKamino:
            mock_api = MagicMock()
            mock_api.list_vaults = AsyncMock(
                return_value={"success": True, "data": [{"vault": "A"}]}
            )
            MockKamino.return_value = mock_api

            result = await kamino_tool.execute(action="list_vaults")

        assert result["status"] == "success"
        assert result["action"] == "list_vaults"
        assert result["data"][0]["vault"] == "A"

    @pytest.mark.asyncio
    async def test_user_obligations_requires_market_and_user(self, kamino_tool):
        result = await kamino_tool.execute(
            action="user_obligations", market="", user_pubkey=""
        )

        assert result["status"] == "error"
        assert "market" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_api_get_invalid_params_json(self, kamino_tool):
        result = await kamino_tool.execute(
            action="api_get",
            path="/oracles/prices",
            params_json="[]",
        )

        assert result["status"] == "error"
        assert "json object" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_read_actions_and_path_validation(self, kamino_tool):
        with patch("solana_agent.tools.kamino.KaminoAPI") as MockKamino:
            mock_api = MagicMock()
            mock_api.list_markets = AsyncMock(
                return_value={"success": True, "data": [{"m": 1}]}
            )
            mock_api.get_oracle_prices = AsyncMock(
                return_value={"success": False, "error": "oracle down"}
            )
            mock_api.get_user_vault_positions = AsyncMock(
                return_value={"success": True, "data": [{"p": 1}]}
            )
            mock_api.get_user_obligations = AsyncMock(
                return_value={"success": True, "data": [{"o": 1}]}
            )
            mock_api.api_get = AsyncMock(
                return_value={"success": True, "data": {"ok": True}}
            )
            mock_api.api_post = AsyncMock(
                return_value={"success": True, "data": {"posted": True}}
            )
            MockKamino.return_value = mock_api

            markets = await kamino_tool.execute(action="list_markets")
            oracle_failure = await kamino_tool.execute(
                action="oracle_prices", params_json='{"symbol":"SOL"}'
            )
            vault_positions = await kamino_tool.execute(
                action="vault_positions", user_pubkey="user-1"
            )
            obligations = await kamino_tool.execute(
                action="user_obligations", market="market-1", user_pubkey="user-1"
            )
            api_get = await kamino_tool.execute(
                action="api_get", path="/ping", params_json='{"a":1}'
            )
            api_post = await kamino_tool.execute(
                action="api_post", path="/ping", body_json='{"b":2}'
            )

        assert markets["status"] == "success"
        assert oracle_failure["status"] == "error"
        assert vault_positions["data"] == [{"p": 1}]
        assert obligations["data"] == [{"o": 1}]
        assert api_get["path"] == "/ping"
        assert api_post["data"] == {"posted": True}

        assert (await kamino_tool.execute(action="vault_positions", user_pubkey=""))[
            "status"
        ] == "error"
        assert (await kamino_tool.execute(action="api_get", path="", params_json=""))[
            "status"
        ] == "error"
        assert (await kamino_tool.execute(action="api_post", path="", body_json=""))[
            "status"
        ] == "error"
        assert (
            await kamino_tool.execute(action="api_post", path="/x", body_json="[]")
        )["status"] == "error"

    @pytest.mark.asyncio
    async def test_oracle_prices_invalid_json(self, kamino_tool):
        result = await kamino_tool.execute(action="oracle_prices", params_json="[]")

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_borrow_deposit_success(self, kamino_tool):
        signer = Keypair()
        transaction_base64 = _make_unsigned_transaction_base64(signer)

        with (
            patch("solana_agent.tools.kamino.Keypair") as MockKeypair,
            patch("solana_agent.tools.kamino.KaminoAPI") as MockKamino,
            patch(
                "solana_agent.tools.kamino.get_fresh_blockhash", new_callable=AsyncMock
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.kamino.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
            patch(
                "solana_agent.tools.kamino.replace_blockhash_in_transaction"
            ) as mock_replace,
        ):
            MockKeypair.from_base58_string.return_value = signer
            mock_api = MagicMock()
            mock_api.build_borrow_deposit = AsyncMock(
                return_value=MagicMock(
                    success=True,
                    transaction=transaction_base64,
                    request_id="req-1",
                    raw_response={"transaction": transaction_base64},
                )
            )
            MockKamino.return_value = mock_api
            mock_blockhash.return_value = {"blockhash": str(Hash.default())}
            mock_replace.return_value = transaction_base64
            mock_send.return_value = {"success": True, "signature": "sig-123"}

            result = await kamino_tool.execute(
                action="borrow_deposit",
                market="market-1",
                reserve="reserve-1",
                amount="2.0",
                referrer="referrer-wallet",
                referral_code="ref-abc",
            )

        assert result["status"] == "success"
        assert result["signature"] == "sig-123"
        assert result["request_id"] == "req-1"
        assert result["referrer"] == "referrer-wallet"
        assert result["referral_code"] == "ref-abc"

        call = mock_api.build_borrow_deposit.await_args
        assert call.kwargs["referrer"] == "referrer-wallet"
        assert call.kwargs["referral_code"] == "ref-abc"

    @pytest.mark.asyncio
    async def test_borrow_deposit_uses_config_referral_defaults(self, kamino_tool):
        signer = Keypair()
        transaction_base64 = _make_unsigned_transaction_base64(signer)

        with (
            patch("solana_agent.tools.kamino.Keypair") as MockKeypair,
            patch("solana_agent.tools.kamino.KaminoAPI") as MockKamino,
            patch(
                "solana_agent.tools.kamino.get_fresh_blockhash", new_callable=AsyncMock
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.kamino.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
            patch(
                "solana_agent.tools.kamino.replace_blockhash_in_transaction"
            ) as mock_replace,
        ):
            MockKeypair.from_base58_string.return_value = signer
            mock_api = MagicMock()
            mock_api.build_borrow_deposit = AsyncMock(
                return_value=MagicMock(
                    success=True,
                    transaction=transaction_base64,
                    request_id="req-defaults",
                    raw_response={"transaction": transaction_base64},
                )
            )
            MockKamino.return_value = mock_api
            mock_blockhash.return_value = {"blockhash": str(Hash.default())}
            mock_replace.return_value = transaction_base64
            mock_send.return_value = {"success": True, "signature": "sig-defaults"}

            result = await kamino_tool.execute(
                action="borrow_deposit",
                market="market-1",
                reserve="reserve-1",
                amount="2.0",
            )

        assert result["status"] == "success"
        assert result["referrer"] == kamino_tool._managed_referrer
        assert result["referral_code"] is None
        call = mock_api.build_borrow_deposit.await_args
        assert call.kwargs["referrer"] == result["referrer"]
        assert call.kwargs["referral_code"] is None

    @pytest.mark.asyncio
    async def test_transaction_action_bubbles_kamino_error(self, kamino_tool):
        signer = Keypair()

        with (
            patch("solana_agent.tools.kamino.Keypair") as MockKeypair,
            patch("solana_agent.tools.kamino.KaminoAPI") as MockKamino,
        ):
            MockKeypair.from_base58_string.return_value = signer
            mock_api = MagicMock()
            mock_api.build_earn_deposit = AsyncMock(
                return_value=MagicMock(
                    success=False,
                    transaction=None,
                    error="kamino rejected request",
                    raw_response={},
                )
            )
            MockKamino.return_value = mock_api

            result = await kamino_tool.execute(
                action="earn_deposit",
                kvault="vault-1",
                amount="1.0",
            )

        assert result["status"] == "error"
        assert "kamino rejected request" in result["message"]

    @pytest.mark.asyncio
    async def test_unknown_action(self, kamino_tool):
        result = await kamino_tool.execute(action="not_real")

        assert result["status"] == "error"
        assert "Unknown action" in result["message"]

    @pytest.mark.asyncio
    async def test_setup_referrer_success(self, kamino_tool):
        signer = Keypair()

        with (
            patch("solana_agent.tools.kamino.Keypair") as MockKeypair,
            patch(
                "solana_agent.tools.kamino.build_kamino_referrer_setup_transaction",
                new=AsyncMock(
                    return_value={
                        "status": "success",
                        "transaction": "tx-base64",
                        "referrer_state": "state-1",
                        "referrer_short_url": "short-1",
                    }
                ),
            ) as mock_setup,
            patch.object(
                kamino_tool,
                "_sign_and_send",
                new=AsyncMock(
                    return_value={"status": "success", "signature": "sig-setup"}
                ),
            ),
        ):
            MockKeypair.from_base58_string.return_value = signer

            result = await kamino_tool.execute(
                action="setup_referrer",
                short_url="kamino_ref",
                user_lookup_table="11111111111111111111111111111116",
            )

        assert result["status"] == "success"
        assert result["signature"] == "sig-setup"
        assert result["referrer_state"] == "state-1"
        assert result["short_url"] == "kamino_ref"
        assert mock_setup.await_args.kwargs["short_url"] == "kamino_ref"

    @pytest.mark.asyncio
    async def test_create_user_lookup_table_success(self, kamino_tool):
        signer = Keypair()

        with (
            patch("solana_agent.tools.kamino.Keypair") as MockKeypair,
            patch(
                "solana_agent.tools.kamino.build_kamino_create_lookup_table_transaction",
                new=AsyncMock(
                    return_value={
                        "status": "success",
                        "transaction": "tx-base64",
                        "user_lookup_table": "lut-1",
                    }
                ),
            ) as mock_create,
            patch.object(
                kamino_tool,
                "_sign_and_send",
                new=AsyncMock(
                    return_value={"status": "success", "signature": "sig-lut"}
                ),
            ),
        ):
            MockKeypair.from_base58_string.return_value = signer
            result = await kamino_tool.execute(action="create_user_lookup_table")

        assert result["status"] == "success"
        assert result["signature"] == "sig-lut"
        assert result["user_lookup_table"] == "lut-1"
        assert mock_create.await_count == 1

    @pytest.mark.asyncio
    async def test_init_user_metadata_uses_config_default_referrer(self, kamino_tool):
        signer = Keypair()

        with (
            patch("solana_agent.tools.kamino.Keypair") as MockKeypair,
            patch(
                "solana_agent.tools.kamino.build_kamino_init_user_metadata_transaction",
                new=AsyncMock(
                    return_value={
                        "status": "success",
                        "transaction": "tx-base64",
                        "user_lookup_table": "lut-1",
                    }
                ),
            ) as mock_init,
            patch.object(
                kamino_tool,
                "_sign_and_send",
                new=AsyncMock(
                    return_value={"status": "success", "signature": "sig-meta"}
                ),
            ),
        ):
            MockKeypair.from_base58_string.return_value = signer
            result = await kamino_tool.execute(action="init_user_metadata")

        assert result["status"] == "success"
        assert result["referrer"] == kamino_tool._managed_referrer
        assert mock_init.await_args.kwargs["referrer"] == kamino_tool._managed_referrer

    @pytest.mark.asyncio
    async def test_borrow_deposit_runs_internal_referral_automation_when_configured(
        self,
    ):
        user_signer = Keypair()
        referrer_signer = Keypair()
        tool = KaminoTool()
        tool.configure(
            {
                "tools": {
                    "kamino": {
                        "private_key": "user-private-key",
                        "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                        "referrer_private_key": str(referrer_signer),
                    }
                }
            }
        )

        with (
            patch("solana_agent.tools.kamino.Keypair") as MockKeypair,
            patch("solana_agent.tools.kamino.KaminoAPI") as MockKamino,
            patch(
                "solana_agent.tools.kamino.fetch_kamino_reserve_metadata",
                new=AsyncMock(
                    return_value=MagicMock(
                        lending_market="market-1",
                        reserve="reserve-1",
                        reserve_liquidity_mint="mint-1",
                        pyth_oracle="",
                        switchboard_price_oracle="",
                        switchboard_twap_oracle="",
                        scope_prices="",
                        token_program_id="token-program-1",
                    )
                ),
            ) as mock_metadata,
            patch(
                "solana_agent.tools.kamino.build_kamino_withdraw_referrer_fees_transaction",
                new=AsyncMock(
                    return_value={
                        "status": "success",
                        "transaction": "withdraw-tx",
                    }
                ),
            ) as mock_withdraw,
            patch.object(
                tool,
                "_sign_and_send",
                new=AsyncMock(
                    side_effect=[
                        {"status": "success", "signature": "sig-user"},
                        {"status": "success", "signature": "sig-withdraw"},
                    ]
                ),
            ),
        ):
            MockKeypair.from_base58_string.side_effect = lambda value: {
                "user-private-key": user_signer,
                str(referrer_signer): referrer_signer,
            }[value]
            mock_api = MagicMock()
            mock_api.build_borrow_deposit = AsyncMock(
                return_value=MagicMock(
                    success=True,
                    transaction="user-tx",
                    request_id="req-auto",
                    raw_response={"transaction": "user-tx"},
                )
            )
            MockKamino.return_value = mock_api

            result = await tool.execute(
                action="borrow_deposit",
                market="market-1",
                reserve="reserve-1",
                amount="2.0",
            )

        assert result["status"] == "success"
        assert result["signature"] == "sig-user"
        assert result["referral_automation"]["setup"]["status"] == "skipped"
        assert result["referral_automation"]["withdrawals"][0]["status"] == "success"
        assert mock_metadata.await_count == 1
        assert mock_withdraw.await_args.kwargs["reserve_liquidity_mint"] == "mint-1"


class TestKaminoPlugin:
    def test_plugin_name(self):
        plugin = KaminoPlugin()
        assert plugin.name == "kamino"

    def test_plugin_get_tools_empty_before_init(self):
        plugin = KaminoPlugin()
        assert plugin.get_tools() == []

    def test_plugin_description_and_configure(self):
        plugin = KaminoPlugin()
        assert plugin.description == "Plugin for Kamino Earn and K-Lend operations."

        tool = MagicMock()
        plugin._tool = tool
        plugin.configure({"x": 1})

        assert plugin.config == {"x": 1}
        tool.configure.assert_called_once_with({"x": 1})

    def test_plugin_initialize(self):
        plugin = KaminoPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        tools = plugin.get_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], KaminoTool)


class TestGetPlugin:
    def test_get_plugin_returns_instance(self):
        plugin = get_plugin()
        assert isinstance(plugin, KaminoPlugin)


class TestKaminoToolInternals:
    @pytest.mark.asyncio
    async def test_build_transaction_and_internal_helpers(self, kamino_tool):
        kamino = MagicMock()
        kamino.build_earn_withdraw = AsyncMock(
            return_value=MagicMock(
                success=True,
                transaction="tx",
                request_id="req",
                raw_response={"ok": True},
            )
        )
        kamino.build_borrow_deposit = AsyncMock(
            return_value=MagicMock(
                success=True,
                transaction="borrow-deposit-tx",
                request_id="req2",
                raw_response={"deposit": True},
            )
        )
        kamino.build_borrow_borrow = AsyncMock(
            return_value=MagicMock(
                success=False,
                transaction=None,
                error="borrow failed",
                raw_response={"raw": True},
            )
        )
        kamino.build_borrow_withdraw = AsyncMock(
            return_value=MagicMock(
                success=True,
                transaction="borrow-withdraw-tx",
                request_id="req3",
                raw_response={"withdraw": True},
            )
        )

        earn_missing = await kamino_tool._build_transaction(
            kamino,
            "earn_deposit",
            "wallet",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )
        borrow_missing = await kamino_tool._build_transaction(
            kamino,
            "borrow_borrow",
            "wallet",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )
        setup_missing = await kamino_tool._build_transaction(
            kamino,
            "setup_referrer",
            "wallet",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )
        withdraw_missing = await kamino_tool._build_transaction(
            kamino,
            "withdraw_referrer_fees",
            "wallet",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )
        earn_success = await kamino_tool._build_transaction(
            kamino,
            "earn_withdraw",
            "wallet",
            "vault",
            "",
            "",
            "1",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )
        borrow_failure = await kamino_tool._build_transaction(
            kamino,
            "borrow_borrow",
            "wallet",
            "",
            "market",
            "reserve",
            "1",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )
        borrow_deposit = await kamino_tool._build_transaction(
            kamino,
            "borrow_deposit",
            "wallet",
            "",
            "market",
            "reserve",
            "1",
            "ref",
            "code",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )
        borrow_withdraw = await kamino_tool._build_transaction(
            kamino,
            "borrow_withdraw",
            "wallet",
            "",
            "market",
            "reserve",
            "1",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )

        assert earn_missing["status"] == "error"
        assert borrow_missing["status"] == "error"
        assert setup_missing["status"] == "error"
        assert withdraw_missing["status"] == "error"
        assert earn_success["status"] == "success"
        assert borrow_failure["raw_response"] == {"raw": True}
        assert borrow_deposit["status"] == "success"
        assert borrow_withdraw["status"] == "success"
        assert kamino.build_borrow_deposit.await_args.kwargs["referral_code"] == "code"

    @pytest.mark.asyncio
    async def test_build_transaction_withdraw_referrer_fees_branch(self, kamino_tool):
        with patch(
            "solana_agent.tools.kamino.build_kamino_withdraw_referrer_fees_transaction",
            new=AsyncMock(
                return_value={"status": "success", "transaction": "tx", "extra": True}
            ),
        ) as mock_withdraw:
            result = await kamino_tool._build_transaction(
                MagicMock(),
                "withdraw_referrer_fees",
                "wallet",
                "",
                "market",
                "reserve",
                "",
                "",
                "",
                "",
                "",
                "mint",
                "pyth",
                "switch",
                "twap",
                "scope",
                "token-program",
            )

        assert result["status"] == "success"
        assert mock_withdraw.await_count == 1

    @pytest.mark.asyncio
    async def test_build_transaction_borrow_repay_branch(self, kamino_tool):
        kamino = MagicMock()
        kamino.build_borrow_repay = AsyncMock(
            return_value=MagicMock(
                success=True,
                transaction="repay-tx",
                request_id="req",
                raw_response={"repay": True},
            )
        )

        result = await kamino_tool._build_transaction(
            kamino,
            "borrow_repay",
            "wallet",
            "",
            "market",
            "reserve",
            "1",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_early_error_returns(self, kamino_tool):
        signer = Keypair()

        with (
            patch("solana_agent.tools.kamino.Keypair") as MockKeypair,
            patch.object(
                kamino_tool,
                "_run_pre_transaction_referral_automation",
                new=AsyncMock(
                    return_value={"status": "error", "message": "pre failed"}
                ),
            ),
        ):
            MockKeypair.from_base58_string.return_value = signer
            pre_result = await kamino_tool.execute(
                action="earn_deposit", kvault="vault", amount="1"
            )

        assert pre_result == {"status": "error", "message": "pre failed"}

        with (
            patch("solana_agent.tools.kamino.Keypair") as MockKeypair,
            patch.object(
                kamino_tool,
                "_build_transaction",
                new=AsyncMock(
                    return_value={"status": "error", "message": "build failed"}
                ),
            ),
        ):
            MockKeypair.from_base58_string.return_value = signer
            build_result = await kamino_tool.execute(
                action="earn_deposit", kvault="vault", amount="1"
            )

        assert build_result == {"status": "error", "message": "build failed"}

        with (
            patch("solana_agent.tools.kamino.Keypair") as MockKeypair,
            patch.object(
                kamino_tool,
                "_build_transaction",
                new=AsyncMock(return_value={"status": "success", "transaction": "tx"}),
            ),
            patch.object(
                kamino_tool,
                "_sign_and_send",
                new=AsyncMock(
                    return_value={"status": "error", "message": "sign failed"}
                ),
            ),
        ):
            MockKeypair.from_base58_string.return_value = signer
            sign_result = await kamino_tool.execute(
                action="earn_deposit", kvault="vault", amount="1"
            )

        assert sign_result == {"status": "error", "message": "sign failed"}

    @pytest.mark.asyncio
    async def test_post_transaction_referral_branches(self, kamino_tool):
        assert (
            await kamino_tool._run_post_transaction_referral_automation(
                "earn_deposit", "m", "r", "x"
            )
            == []
        )
        assert (
            await kamino_tool._run_post_transaction_referral_automation(
                "borrow_deposit", "m", "r", "other"
            )
            == []
        )

        with patch.object(
            kamino_tool, "_get_internal_referrer_keypair", return_value=None
        ):
            assert (
                await kamino_tool._run_post_transaction_referral_automation(
                    "borrow_deposit", "m", "r", kamino_tool._managed_referrer
                )
                == []
            )

        wrong_signer = Keypair()
        with patch.object(
            kamino_tool, "_get_internal_referrer_keypair", return_value=wrong_signer
        ):
            assert (
                await kamino_tool._run_post_transaction_referral_automation(
                    "borrow_deposit", "m", "r", kamino_tool._managed_referrer
                )
                == []
            )

        right_signer = Keypair.from_base58_string(kamino_tool._referrer_private_key)
        with (
            patch.object(
                kamino_tool, "_get_internal_referrer_keypair", return_value=right_signer
            ),
            patch(
                "solana_agent.tools.kamino.fetch_kamino_reserve_metadata",
                new=AsyncMock(side_effect=RuntimeError("fetch failed")),
            ),
        ):
            fetch_error = await kamino_tool._run_post_transaction_referral_automation(
                "borrow_deposit", "m", "r", kamino_tool._managed_referrer
            )

        assert fetch_error[0]["status"] == "error"

        with (
            patch.object(
                kamino_tool, "_get_internal_referrer_keypair", return_value=right_signer
            ),
            patch(
                "solana_agent.tools.kamino.fetch_kamino_reserve_metadata",
                new=AsyncMock(
                    return_value=MagicMock(lending_market="other", reserve="r")
                ),
            ),
        ):
            mismatch = await kamino_tool._run_post_transaction_referral_automation(
                "borrow_deposit", "m", "r", kamino_tool._managed_referrer
            )

        assert "does not match" in mismatch[0]["message"]

        metadata = MagicMock(
            lending_market="m",
            reserve="r",
            reserve_liquidity_mint="mint",
            pyth_oracle="",
            switchboard_price_oracle="",
            switchboard_twap_oracle="",
            scope_prices="",
            token_program_id="token-program",
        )
        with (
            patch.object(
                kamino_tool, "_get_internal_referrer_keypair", return_value=right_signer
            ),
            patch(
                "solana_agent.tools.kamino.fetch_kamino_reserve_metadata",
                new=AsyncMock(return_value=metadata),
            ),
            patch(
                "solana_agent.tools.kamino.build_kamino_withdraw_referrer_fees_transaction",
                new=AsyncMock(
                    return_value={"status": "error", "message": "withdraw failed"}
                ),
            ),
        ):
            withdraw_error = (
                await kamino_tool._run_post_transaction_referral_automation(
                    "borrow_deposit", "m", "r", kamino_tool._managed_referrer
                )
            )

        assert withdraw_error[0]["message"] == "withdraw failed"

    @pytest.mark.asyncio
    async def test_sign_send_and_format_helpers(self, kamino_tool):
        signer = Keypair()
        tx_base64 = _make_unsigned_transaction_base64(signer)

        with patch(
            "solana_agent.tools.kamino.get_fresh_blockhash",
            new=AsyncMock(return_value={"error": "rpc down"}),
        ):
            blockhash_error = await kamino_tool._sign_and_send(signer, tx_base64)
        assert blockhash_error["status"] == "error"

        with (
            patch(
                "solana_agent.tools.kamino.get_fresh_blockhash",
                new=AsyncMock(return_value={"blockhash": str(Hash.default())}),
            ),
            patch(
                "solana_agent.tools.kamino.replace_blockhash_in_transaction",
                return_value=_make_unsigned_transaction_base64(Keypair()),
            ),
        ):
            signer_missing = await kamino_tool._sign_and_send(signer, tx_base64)
        assert "not found" in signer_missing["message"]

        with (
            patch(
                "solana_agent.tools.kamino.get_fresh_blockhash",
                new=AsyncMock(return_value={"blockhash": str(Hash.default())}),
            ),
            patch(
                "solana_agent.tools.kamino.replace_blockhash_in_transaction",
                return_value=tx_base64,
            ),
            patch(
                "solana_agent.tools.kamino.send_raw_transaction_with_priority",
                new=AsyncMock(return_value={"success": False, "error": "send failed"}),
            ),
        ):
            send_failure = await kamino_tool._sign_and_send(signer, tx_base64)
        assert send_failure == {"status": "error", "message": "send failed"}

        with patch(
            "solana_agent.tools.kamino.get_fresh_blockhash",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            exception_result = await kamino_tool._sign_and_send(signer, tx_base64)
        assert exception_result["status"] == "error"

        assert kamino_tool._get_internal_referrer_keypair("") is not None
        tool_without_referrer = KaminoTool()
        assert tool_without_referrer._get_internal_referrer_keypair("") is None

        assert kamino_tool._parse_json_object("", "params_json") is None
        assert kamino_tool._parse_json_object('{"x":1}', "params_json") == {"x": 1}
        assert kamino_tool._parse_json_object("bad", "params_json")["status"] == "error"
        assert kamino_tool._parse_json_object("[]", "params_json")["status"] == "error"

        assert kamino_tool._format_read_response(
            "x", {"success": False, "error": "nope"}
        ) == {
            "status": "error",
            "message": "nope",
            "path": None,
        }
        assert kamino_tool._format_read_response(
            "x", {"success": True, "data": 1}, path="/p"
        ) == {
            "status": "success",
            "action": "x",
            "data": 1,
            "path": "/p",
        }

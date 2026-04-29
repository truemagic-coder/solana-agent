"""Tests for the Privy Kamino tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from solders.hash import Hash
from solders.keypair import Keypair

from solana_agent.tools.privy_kamino import (
    PrivyKaminoPlugin,
    PrivyKaminoTool,
    get_plugin,
)


@pytest.fixture
def privy_kamino_tool():
    referrer_signer = Keypair()
    tool = PrivyKaminoTool()
    tool.configure(
        {
            "tools": {
                "privy_kamino": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                    "signing_key": "wallet-auth:test-signing-key",
                    "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                    "referrer_private_key": str(referrer_signer),
                }
            }
        }
    )
    return tool


class TestPrivyKaminoToolSchema:
    def test_tool_name(self, privy_kamino_tool):
        assert privy_kamino_tool.name == "privy_kamino"

    def test_schema_has_referral_actions(self, privy_kamino_tool):
        schema = privy_kamino_tool.get_schema()
        assert "borrow_deposit" in schema["properties"]["action"]["enum"]
        assert "setup_referrer" not in schema["properties"]["action"]["enum"]
        assert "withdraw_referrer_fees" not in schema["properties"]["action"]["enum"]


class TestPrivyKaminoToolExecute:
    @pytest.mark.asyncio
    async def test_execute_requires_wallet_params(self, privy_kamino_tool):
        result = await privy_kamino_tool.execute(
            wallet_id="",
            wallet_public_key="",
            action="list_markets",
        )

        assert result["status"] == "error"
        assert "wallet_id" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_requires_privy_and_rpc_config(self):
        tool = PrivyKaminoTool()

        missing_privy = await tool.execute(
            wallet_id="wallet-1",
            wallet_public_key="WalletPubkey123",
            action="list_markets",
        )
        assert missing_privy == {"status": "error", "message": "Privy config missing."}

        tool.configure(
            {
                "tools": {
                    "privy_kamino": {
                        "app_id": "a",
                        "app_secret": "b",
                        "signing_key": "wallet-auth:k",
                    }
                }
            }
        )
        missing_rpc = await tool.execute(
            wallet_id="wallet-1",
            wallet_public_key="WalletPubkey123",
            action="earn_deposit",
            kvault="vault",
            amount="1",
        )
        assert missing_rpc == {"status": "error", "message": "RPC URL not configured."}

    @pytest.mark.asyncio
    async def test_list_markets_success(self, privy_kamino_tool):
        with patch("solana_agent.tools.privy_kamino.KaminoAPI") as MockKamino:
            mock_api = MagicMock()
            mock_api.list_markets = AsyncMock(
                return_value={"success": True, "data": [{"market": "A"}]}
            )
            MockKamino.return_value = mock_api

            result = await privy_kamino_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="list_markets",
            )

        assert result["status"] == "success"
        assert result["data"][0]["market"] == "A"

    @pytest.mark.asyncio
    async def test_borrow_deposit_success(self, privy_kamino_tool):
        with (
            patch("solana_agent.tools.privy_kamino.KaminoAPI") as MockKamino,
            patch("solana_agent.tools.privy_kamino.AsyncPrivyAPI") as MockPrivy,
            patch.object(
                privy_kamino_tool,
                "_sign_and_send",
                new=AsyncMock(
                    return_value={"status": "success", "signature": "sig-123"}
                ),
            ),
        ):
            mock_api = MagicMock()
            mock_api.build_borrow_deposit = AsyncMock(
                return_value=MagicMock(
                    success=True,
                    transaction="tx-base64",
                    request_id="req-1",
                    raw_response={"transaction": "tx-base64"},
                )
            )
            MockKamino.return_value = mock_api
            mock_privy_client = MagicMock()
            mock_privy_client.close = AsyncMock()
            MockPrivy.return_value = mock_privy_client

            result = await privy_kamino_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
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
        call = mock_api.build_borrow_deposit.await_args
        assert call.kwargs["referrer"] == "referrer-wallet"
        assert call.kwargs["referral_code"] == "ref-abc"

    @pytest.mark.asyncio
    async def test_borrow_deposit_uses_config_referral_defaults(
        self, privy_kamino_tool
    ):
        with (
            patch("solana_agent.tools.privy_kamino.KaminoAPI") as MockKamino,
            patch("solana_agent.tools.privy_kamino.AsyncPrivyAPI") as MockPrivy,
            patch.object(
                privy_kamino_tool,
                "_sign_and_send",
                new=AsyncMock(
                    return_value={"status": "success", "signature": "sig-defaults"}
                ),
            ),
        ):
            mock_api = MagicMock()
            mock_api.build_borrow_deposit = AsyncMock(
                return_value=MagicMock(
                    success=True,
                    transaction="tx-base64",
                    request_id="req-defaults",
                    raw_response={"transaction": "tx-base64"},
                )
            )
            MockKamino.return_value = mock_api
            mock_privy_client = MagicMock()
            mock_privy_client.close = AsyncMock()
            MockPrivy.return_value = mock_privy_client

            result = await privy_kamino_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="borrow_deposit",
                market="market-1",
                reserve="reserve-1",
                amount="2.0",
            )

        assert result["status"] == "success"
        assert result["referrer"] == privy_kamino_tool._managed_referrer
        assert result["referral_code"] is None
        call = mock_api.build_borrow_deposit.await_args
        assert call.kwargs["referrer"] == privy_kamino_tool._managed_referrer
        assert call.kwargs["referral_code"] is None

    @pytest.mark.asyncio
    async def test_read_actions_and_unknown_action(self, privy_kamino_tool):
        with patch("solana_agent.tools.privy_kamino.KaminoAPI") as MockKamino:
            mock_api = MagicMock()
            mock_api.list_vaults = AsyncMock(
                return_value={"success": True, "data": [{"vault": "A"}]}
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

            vaults = await privy_kamino_tool.execute(
                wallet_id="w", wallet_public_key="p", action="list_vaults"
            )
            oracle_failure = await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="oracle_prices",
                params_json='{"symbol":"SOL"}',
            )
            positions = await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="vault_positions",
                user_pubkey="u",
            )
            obligations = await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="user_obligations",
                market="m",
                user_pubkey="u",
            )
            api_get = await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="api_get",
                path="/x",
                params_json='{"a":1}',
            )
            api_post = await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="api_post",
                path="/x",
                body_json='{"b":2}',
            )

        assert vaults["status"] == "success"
        assert oracle_failure["status"] == "error"
        assert positions["data"] == [{"p": 1}]
        assert obligations["data"] == [{"o": 1}]
        assert api_get["path"] == "/x"
        assert api_post["data"] == {"posted": True}
        assert (
            await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="vault_positions",
                user_pubkey="",
            )
        )["status"] == "error"
        assert (
            await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="user_obligations",
                market="",
                user_pubkey="",
            )
        )["status"] == "error"
        assert (
            await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="api_get",
                path="",
                params_json="",
            )
        )["status"] == "error"
        assert (
            await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="api_post",
                path="",
                body_json="",
            )
        )["status"] == "error"
        assert (
            await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="api_get",
                path="/x",
                params_json="[]",
            )
        )["status"] == "error"
        assert (
            await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="api_post",
                path="/x",
                body_json="[]",
            )
        )["status"] == "error"
        assert (
            await privy_kamino_tool.execute(
                wallet_id="w", wallet_public_key="p", action="not_real"
            )
        )["status"] == "error"

    @pytest.mark.asyncio
    async def test_oracle_prices_invalid_json(self, privy_kamino_tool):
        result = await privy_kamino_tool.execute(
            wallet_id="w",
            wallet_public_key="p",
            action="oracle_prices",
            params_json="[]",
        )

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_setup_referrer_success(self, privy_kamino_tool):
        with (
            patch("solana_agent.tools.privy_kamino.AsyncPrivyAPI") as MockPrivy,
            patch(
                "solana_agent.tools.privy_kamino.build_kamino_referrer_setup_transaction",
                new=AsyncMock(
                    return_value={
                        "status": "success",
                        "transaction": "tx-base64",
                        "referrer_state": "state-1",
                    }
                ),
            ) as mock_setup,
            patch.object(
                privy_kamino_tool,
                "_sign_and_send",
                new=AsyncMock(
                    return_value={"status": "success", "signature": "sig-setup"}
                ),
            ),
        ):
            mock_privy_client = MagicMock()
            mock_privy_client.close = AsyncMock()
            MockPrivy.return_value = mock_privy_client

            result = await privy_kamino_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="setup_referrer",
                short_url="kamino_ref",
                user_lookup_table="11111111111111111111111111111116",
            )

        assert result["status"] == "success"
        assert result["signature"] == "sig-setup"
        assert result["referrer_state"] == "state-1"
        assert mock_setup.await_args.kwargs["short_url"] == "kamino_ref"

    @pytest.mark.asyncio
    async def test_create_user_lookup_table_success(self, privy_kamino_tool):
        with (
            patch("solana_agent.tools.privy_kamino.AsyncPrivyAPI") as MockPrivy,
            patch(
                "solana_agent.tools.privy_kamino.build_kamino_create_lookup_table_transaction",
                new=AsyncMock(
                    return_value={
                        "status": "success",
                        "transaction": "tx-base64",
                        "user_lookup_table": "lut-1",
                    }
                ),
            ) as mock_create,
            patch.object(
                privy_kamino_tool,
                "_sign_and_send",
                new=AsyncMock(
                    return_value={"status": "success", "signature": "sig-lut"}
                ),
            ),
        ):
            mock_privy_client = MagicMock()
            mock_privy_client.close = AsyncMock()
            MockPrivy.return_value = mock_privy_client

            result = await privy_kamino_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="create_user_lookup_table",
            )

        assert result["status"] == "success"
        assert result["signature"] == "sig-lut"
        assert result["user_lookup_table"] == "lut-1"
        assert mock_create.await_count == 1

    @pytest.mark.asyncio
    async def test_init_user_metadata_uses_config_default_referrer(
        self, privy_kamino_tool
    ):
        with (
            patch("solana_agent.tools.privy_kamino.AsyncPrivyAPI") as MockPrivy,
            patch(
                "solana_agent.tools.privy_kamino.build_kamino_init_user_metadata_transaction",
                new=AsyncMock(
                    return_value={
                        "status": "success",
                        "transaction": "tx-base64",
                        "user_lookup_table": "lut-1",
                    }
                ),
            ) as mock_init,
            patch.object(
                privy_kamino_tool,
                "_sign_and_send",
                new=AsyncMock(
                    return_value={"status": "success", "signature": "sig-meta"}
                ),
            ),
        ):
            mock_privy_client = MagicMock()
            mock_privy_client.close = AsyncMock()
            MockPrivy.return_value = mock_privy_client

            result = await privy_kamino_tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
                action="init_user_metadata",
            )

        assert result["status"] == "success"
        assert result["referrer"] == privy_kamino_tool._managed_referrer
        assert (
            mock_init.await_args.kwargs["referrer"]
            == privy_kamino_tool._managed_referrer
        )

    @pytest.mark.asyncio
    async def test_borrow_deposit_runs_internal_referral_automation_when_configured(
        self,
    ):
        referrer_signer = Keypair()
        tool = PrivyKaminoTool()
        tool.configure(
            {
                "tools": {
                    "privy_kamino": {
                        "app_id": "test-app-id",
                        "app_secret": "test-app-secret",
                        "signing_key": "wallet-auth:test-signing-key",
                        "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                        "referrer_private_key": str(referrer_signer),
                    }
                }
            }
        )

        with (
            patch("solana_agent.tools.privy_kamino.Keypair") as MockKeypair,
            patch("solana_agent.tools.privy_kamino.KaminoAPI") as MockKamino,
            patch("solana_agent.tools.privy_kamino.AsyncPrivyAPI") as MockPrivy,
            patch(
                "solana_agent.tools.privy_kamino.fetch_kamino_reserve_metadata",
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
                "solana_agent.tools.privy_kamino.build_kamino_withdraw_referrer_fees_transaction",
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
                    return_value={"status": "success", "signature": "sig-user"}
                ),
            ),
            patch.object(
                tool,
                "_sign_and_send_with_keypair",
                new=AsyncMock(
                    side_effect=[
                        {"status": "success", "signature": "sig-withdraw"},
                    ]
                ),
            ),
        ):
            MockKeypair.from_base58_string.return_value = referrer_signer
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
            mock_privy_client = MagicMock()
            mock_privy_client.close = AsyncMock()
            MockPrivy.return_value = mock_privy_client

            result = await tool.execute(
                wallet_id="wallet-123",
                wallet_public_key="WalletPubkey123",
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


class TestPrivyKaminoPlugin:
    def test_plugin_initialize(self):
        plugin = PrivyKaminoPlugin()
        registry = MagicMock()

        plugin.initialize(registry)

        tools = plugin.get_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], PrivyKaminoTool)

    def test_plugin_description_and_configure(self):
        plugin = PrivyKaminoPlugin()
        assert (
            plugin.description
            == "Plugin for Kamino operations using Privy delegated wallets."
        )
        assert plugin.get_tools() == []

        tool = MagicMock()
        plugin._tool = tool
        plugin.configure({"x": 1})

        assert plugin.config == {"x": 1}
        tool.configure.assert_called_once_with({"x": 1})

    def test_get_plugin(self):
        assert isinstance(get_plugin(), PrivyKaminoPlugin)


class TestPrivyKaminoInternals:
    @pytest.mark.asyncio
    async def test_build_transaction_and_internal_helpers(self, privy_kamino_tool):
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
                transaction="deposit-tx",
                request_id="req2",
                raw_response={"deposit": True},
            )
        )
        kamino.build_borrow_borrow = AsyncMock(
            return_value=MagicMock(
                success=True,
                transaction="borrow-tx",
                request_id="req3",
                raw_response={"borrow": True},
            )
        )
        kamino.build_borrow_repay = AsyncMock(
            return_value=MagicMock(
                success=False,
                transaction=None,
                error="repay failed",
                raw_response={"raw": True},
            )
        )

        earn_missing = await privy_kamino_tool._build_transaction(
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
        borrow_missing = await privy_kamino_tool._build_transaction(
            kamino,
            "borrow_repay",
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
        setup_missing = await privy_kamino_tool._build_transaction(
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
        withdraw_missing = await privy_kamino_tool._build_transaction(
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
        earn_success = await privy_kamino_tool._build_transaction(
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
        borrow_failure = await privy_kamino_tool._build_transaction(
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
        borrow_deposit = await privy_kamino_tool._build_transaction(
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
        borrow_borrow = await privy_kamino_tool._build_transaction(
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

        assert earn_missing["status"] == "error"
        assert borrow_missing["status"] == "error"
        assert setup_missing["status"] == "error"
        assert withdraw_missing["status"] == "error"
        assert earn_success["status"] == "success"
        assert borrow_failure["raw_response"] == {"raw": True}
        assert borrow_deposit["status"] == "success"
        assert borrow_borrow["status"] == "success"
        assert kamino.build_borrow_deposit.await_args.kwargs["referral_code"] == "code"

        assert await privy_kamino_tool._run_pre_transaction_referral_automation(
            "borrow_deposit", "x"
        ) == {
            "status": "skipped",
            "reason": "no_pre_transaction_steps_required",
        }

    @pytest.mark.asyncio
    async def test_execute_early_error_returns(self, privy_kamino_tool):
        with (
            patch("solana_agent.tools.privy_kamino.AsyncPrivyAPI") as MockPrivy,
            patch.object(
                privy_kamino_tool,
                "_run_pre_transaction_referral_automation",
                new=AsyncMock(
                    return_value={"status": "error", "message": "pre failed"}
                ),
            ),
        ):
            mock_privy_client = MagicMock()
            mock_privy_client.close = AsyncMock()
            MockPrivy.return_value = mock_privy_client
            pre_result = await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="earn_deposit",
                kvault="vault",
                amount="1",
            )

        assert pre_result == {"status": "error", "message": "pre failed"}

        with (
            patch("solana_agent.tools.privy_kamino.AsyncPrivyAPI") as MockPrivy,
            patch.object(
                privy_kamino_tool,
                "_build_transaction",
                new=AsyncMock(
                    return_value={"status": "error", "message": "build failed"}
                ),
            ),
        ):
            mock_privy_client = MagicMock()
            mock_privy_client.close = AsyncMock()
            MockPrivy.return_value = mock_privy_client
            build_result = await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="earn_deposit",
                kvault="vault",
                amount="1",
            )

        assert build_result == {"status": "error", "message": "build failed"}

        with (
            patch("solana_agent.tools.privy_kamino.AsyncPrivyAPI") as MockPrivy,
            patch.object(
                privy_kamino_tool,
                "_build_transaction",
                new=AsyncMock(return_value={"status": "success", "transaction": "tx"}),
            ),
            patch.object(
                privy_kamino_tool,
                "_sign_and_send",
                new=AsyncMock(
                    return_value={"status": "error", "message": "sign failed"}
                ),
            ),
        ):
            mock_privy_client = MagicMock()
            mock_privy_client.close = AsyncMock()
            MockPrivy.return_value = mock_privy_client
            sign_result = await privy_kamino_tool.execute(
                wallet_id="w",
                wallet_public_key="p",
                action="earn_deposit",
                kvault="vault",
                amount="1",
            )

        assert sign_result == {"status": "error", "message": "sign failed"}

    @pytest.mark.asyncio
    async def test_post_transaction_referral_and_format_helpers(
        self, privy_kamino_tool
    ):
        assert (
            await privy_kamino_tool._run_post_transaction_referral_automation(
                "earn_deposit", "m", "r", "x"
            )
            == []
        )
        assert (
            await privy_kamino_tool._run_post_transaction_referral_automation(
                "borrow_deposit", "m", "r", "other"
            )
            == []
        )

        with patch.object(
            privy_kamino_tool, "_get_internal_referrer_keypair", return_value=None
        ):
            assert (
                await privy_kamino_tool._run_post_transaction_referral_automation(
                    "borrow_deposit", "m", "r", privy_kamino_tool._managed_referrer
                )
                == []
            )

        wrong_signer = Keypair()
        with patch.object(
            privy_kamino_tool,
            "_get_internal_referrer_keypair",
            return_value=wrong_signer,
        ):
            assert (
                await privy_kamino_tool._run_post_transaction_referral_automation(
                    "borrow_deposit", "m", "r", privy_kamino_tool._managed_referrer
                )
                == []
            )

        right_signer = Keypair.from_base58_string(
            privy_kamino_tool._referrer_private_key
        )
        with (
            patch.object(
                privy_kamino_tool,
                "_get_internal_referrer_keypair",
                return_value=right_signer,
            ),
            patch(
                "solana_agent.tools.privy_kamino.fetch_kamino_reserve_metadata",
                new=AsyncMock(side_effect=RuntimeError("fetch failed")),
            ),
        ):
            fetch_error = (
                await privy_kamino_tool._run_post_transaction_referral_automation(
                    "borrow_deposit", "m", "r", privy_kamino_tool._managed_referrer
                )
            )
        assert fetch_error[0]["status"] == "error"

        with (
            patch.object(
                privy_kamino_tool,
                "_get_internal_referrer_keypair",
                return_value=right_signer,
            ),
            patch(
                "solana_agent.tools.privy_kamino.fetch_kamino_reserve_metadata",
                new=AsyncMock(
                    return_value=MagicMock(lending_market="other", reserve="r")
                ),
            ),
        ):
            mismatch = (
                await privy_kamino_tool._run_post_transaction_referral_automation(
                    "borrow_deposit", "m", "r", privy_kamino_tool._managed_referrer
                )
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
                privy_kamino_tool,
                "_get_internal_referrer_keypair",
                return_value=right_signer,
            ),
            patch(
                "solana_agent.tools.privy_kamino.fetch_kamino_reserve_metadata",
                new=AsyncMock(return_value=metadata),
            ),
            patch(
                "solana_agent.tools.privy_kamino.build_kamino_withdraw_referrer_fees_transaction",
                new=AsyncMock(
                    return_value={"status": "error", "message": "withdraw failed"}
                ),
            ),
        ):
            withdraw_error = (
                await privy_kamino_tool._run_post_transaction_referral_automation(
                    "borrow_deposit", "m", "r", privy_kamino_tool._managed_referrer
                )
            )
        assert withdraw_error[0]["message"] == "withdraw failed"

        assert privy_kamino_tool._get_internal_referrer_keypair() is not None
        tool_without_referrer = PrivyKaminoTool()
        assert tool_without_referrer._get_internal_referrer_keypair() is None

        assert privy_kamino_tool._parse_json_object("", "params_json") is None
        assert privy_kamino_tool._parse_json_object('{"x":1}', "params_json") == {
            "x": 1
        }
        assert (
            privy_kamino_tool._parse_json_object("bad", "params_json")["status"]
            == "error"
        )
        assert (
            privy_kamino_tool._parse_json_object("[]", "params_json")["status"]
            == "error"
        )

        assert privy_kamino_tool._format_read_response(
            "x", {"success": False, "error": "nope"}
        ) == {
            "status": "error",
            "message": "nope",
            "path": None,
        }
        assert privy_kamino_tool._format_read_response(
            "x", {"success": True, "data": 1}, path="/p"
        ) == {
            "status": "success",
            "action": "x",
            "data": 1,
            "path": "/p",
        }

    @pytest.mark.asyncio
    async def test_sign_and_send_with_keypair_branches(self, privy_kamino_tool):
        signer = Keypair()
        from .test_kamino_tool import _make_unsigned_transaction_base64

        tx_base64 = _make_unsigned_transaction_base64(signer)

        with patch(
            "solana_agent.tools.privy_kamino.get_fresh_blockhash",
            new=AsyncMock(return_value={"error": "rpc down"}),
        ):
            blockhash_error = await privy_kamino_tool._sign_and_send_with_keypair(
                signer, tx_base64
            )
        assert blockhash_error["status"] == "error"

        with (
            patch(
                "solana_agent.tools.privy_kamino.get_fresh_blockhash",
                new=AsyncMock(return_value={"blockhash": str(Keypair().pubkey())}),
            ),
            patch(
                "solana_agent.tools.privy_kamino.replace_blockhash_in_transaction",
                return_value=_make_unsigned_transaction_base64(Keypair()),
            ),
        ):
            signer_missing = await privy_kamino_tool._sign_and_send_with_keypair(
                signer, tx_base64
            )
        assert "not found" in signer_missing["message"]

        with (
            patch(
                "solana_agent.tools.privy_kamino.get_fresh_blockhash",
                new=AsyncMock(return_value={"blockhash": str(Keypair().pubkey())}),
            ),
            patch(
                "solana_agent.tools.privy_kamino.replace_blockhash_in_transaction",
                return_value=tx_base64,
            ),
            patch(
                "solana_agent.tools.privy_kamino.send_raw_transaction_with_priority",
                new=AsyncMock(return_value={"success": False, "error": "send failed"}),
            ),
        ):
            send_failure = await privy_kamino_tool._sign_and_send_with_keypair(
                signer, tx_base64
            )
        assert send_failure == {"status": "error", "message": "send failed"}

        with patch(
            "solana_agent.tools.privy_kamino.get_fresh_blockhash",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            exception_result = await privy_kamino_tool._sign_and_send_with_keypair(
                signer, tx_base64
            )
        assert exception_result["status"] == "error"

        with (
            patch(
                "solana_agent.tools.privy_kamino.get_fresh_blockhash",
                new=AsyncMock(return_value={"blockhash": str(Hash.default())}),
            ),
            patch(
                "solana_agent.tools.privy_kamino.replace_blockhash_in_transaction",
                return_value=tx_base64,
            ),
            patch(
                "solana_agent.tools.privy_kamino.send_raw_transaction_with_priority",
                new=AsyncMock(return_value={"success": True, "signature": "sig-1"}),
            ),
        ):
            success = await privy_kamino_tool._sign_and_send_with_keypair(
                signer, tx_base64
            )
        assert success == {"status": "success", "signature": "sig-1"}

    @pytest.mark.asyncio
    async def test_build_transaction_remaining_branches(self, privy_kamino_tool):
        with patch(
            "solana_agent.tools.privy_kamino.build_kamino_withdraw_referrer_fees_transaction",
            new=AsyncMock(
                return_value={"status": "success", "transaction": "tx", "extra": True}
            ),
        ) as mock_withdraw:
            withdraw_result = await privy_kamino_tool._build_transaction(
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

        assert withdraw_result["status"] == "success"
        assert mock_withdraw.await_count == 1

        kamino = MagicMock()
        kamino.build_borrow_withdraw = AsyncMock(
            return_value=MagicMock(
                success=True,
                transaction="withdraw-tx",
                request_id="req",
                raw_response={"withdraw": True},
            )
        )

        borrow_withdraw = await privy_kamino_tool._build_transaction(
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

        assert borrow_withdraw["status"] == "success"

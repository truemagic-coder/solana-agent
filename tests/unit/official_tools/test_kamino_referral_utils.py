import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from solders.keypair import Keypair
from solders.pubkey import Pubkey

from solana_agent.tools.utils.kamino import (
    INIT_REFERRER_STATE_AND_SHORT_URL_DISCRIMINATOR,
    INIT_USER_METADATA_DISCRIMINATOR,
    KAMINO_RESERVE_ACCOUNT_DISCRIMINATOR,
    KAMINO_LEND_PROGRAM_ID,
    SPL_TOKEN_PROGRAM_ID,
    TOKEN_2022_PROGRAM_ID,
    WITHDRAW_REFERRER_FEES_DISCRIMINATOR,
    KaminoAPI,
    KaminoTransactionResponse,
    _account_exists,
    _borsh_string,
    _build_create_lookup_table_instruction,
    _compile_placeholder_transaction,
    _optional_pubkey,
    _read_pubkey,
    _resolve_token_program_pubkey,
    build_kamino_create_lookup_table_transaction,
    fetch_kamino_reserve_metadata,
    build_kamino_init_referrer_token_state_instruction,
    build_kamino_refresh_reserve_instruction,
    build_kamino_init_referrer_state_and_short_url_instruction,
    build_kamino_init_user_metadata_instruction,
    build_kamino_init_user_metadata_transaction,
    build_kamino_referrer_setup_transaction,
    build_kamino_withdraw_referrer_fees_instruction,
    build_kamino_withdraw_referrer_fees_transaction,
    derive_kamino_lending_market_authority_pda,
    derive_kamino_referrer_state_pda,
    derive_kamino_referrer_token_state_pda,
    derive_kamino_reserve_liquidity_supply_pda,
    derive_kamino_short_url_pda,
    derive_kamino_user_metadata_pda,
    parse_kamino_reserve_metadata,
    validate_kamino_short_url,
)


WALLET = "11111111111111111111111111111112"
REFERRER = "11111111111111111111111111111113"
MARKET = "11111111111111111111111111111114"
RESERVE = "11111111111111111111111111111115"
MINT = "So11111111111111111111111111111111111111112"
LOOKUP_TABLE = "11111111111111111111111111111116"
PYTH = "11111111111111111111111111111117"
SWITCHBOARD = "11111111111111111111111111111118"
TWAP = "11111111111111111111111111111119"
SCOPE = "1111111111111111111111111111111A"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"


def _build_reserve_account_data() -> bytes:
    data = bytearray(4048)
    data[:8] = KAMINO_RESERVE_ACCOUNT_DISCRIMINATOR
    data[32:64] = bytes(Pubkey.from_string(MARKET))
    data[128:160] = bytes(Pubkey.from_string(MINT))
    data[400:432] = bytes(Pubkey.from_string(TOKEN_PROGRAM))
    data[3904:3936] = bytes(Pubkey.from_string(SCOPE))
    data[3952:3984] = bytes(Pubkey.from_string(SWITCHBOARD))
    data[3984:4016] = bytes(Pubkey.from_string(TWAP))
    data[4016:4048] = bytes(Pubkey.from_string(PYTH))
    return bytes(data)


class TestKaminoReferralInstructionBuilders:
    def test_validate_short_url(self):
        assert validate_kamino_short_url("ref_code-1") is None
        assert validate_kamino_short_url("")
        assert validate_kamino_short_url("bad space")
        assert validate_kamino_short_url("x" * 33)

    def test_init_user_metadata_instruction(self):
        instruction = build_kamino_init_user_metadata_instruction(
            owner=WALLET,
            user_lookup_table=LOOKUP_TABLE,
            referrer=REFERRER,
        )

        assert bytes(instruction.data[:8]) == INIT_USER_METADATA_DISCRIMINATOR
        assert instruction.accounts[0].pubkey == Pubkey.from_string(WALLET)
        assert instruction.accounts[2].pubkey == derive_kamino_user_metadata_pda(WALLET)
        assert instruction.accounts[3].pubkey == derive_kamino_user_metadata_pda(
            REFERRER
        )
        assert bytes(instruction.data[8:]) == bytes(Pubkey.from_string(LOOKUP_TABLE))

    def test_init_referrer_state_instruction(self):
        instruction = build_kamino_init_referrer_state_and_short_url_instruction(
            referrer=WALLET,
            short_url="kamino_ref",
        )

        assert (
            bytes(instruction.data[:8])
            == INIT_REFERRER_STATE_AND_SHORT_URL_DISCRIMINATOR
        )
        assert instruction.accounts[1].pubkey == derive_kamino_referrer_state_pda(
            WALLET
        )
        assert instruction.accounts[2].pubkey == derive_kamino_short_url_pda(
            "kamino_ref"
        )
        assert instruction.accounts[3].pubkey == derive_kamino_user_metadata_pda(WALLET)

    def test_withdraw_referrer_fees_instruction(self):
        referrer_token_account = "1111111111111111111111111111111B"
        instruction = build_kamino_withdraw_referrer_fees_instruction(
            referrer=WALLET,
            lending_market=MARKET,
            reserve=RESERVE,
            reserve_liquidity_mint=MINT,
            referrer_token_account=referrer_token_account,
        )

        assert bytes(instruction.data) == WITHDRAW_REFERRER_FEES_DISCRIMINATOR
        assert instruction.accounts[1].pubkey == derive_kamino_referrer_token_state_pda(
            WALLET, RESERVE
        )
        assert instruction.accounts[
            4
        ].pubkey == derive_kamino_reserve_liquidity_supply_pda(RESERVE)
        assert instruction.accounts[6].pubkey == Pubkey.from_string(MARKET)
        assert instruction.accounts[
            7
        ].pubkey == derive_kamino_lending_market_authority_pda(MARKET)

    def test_init_user_metadata_instruction_defaults_fee_payer_and_missing_referrer(
        self,
    ):
        instruction = build_kamino_init_user_metadata_instruction(
            owner=WALLET,
            user_lookup_table=LOOKUP_TABLE,
        )

        assert instruction.accounts[1].pubkey == Pubkey.from_string(WALLET)
        assert instruction.accounts[3].pubkey == KAMINO_LEND_PROGRAM_ID

    def test_init_referrer_state_instruction_rejects_invalid_short_url(self):
        with pytest.raises(ValueError, match="short_url"):
            build_kamino_init_referrer_state_and_short_url_instruction(
                referrer=WALLET,
                short_url="bad space",
            )

    def test_init_referrer_token_state_instruction_defaults_referrer_to_payer(self):
        instruction = build_kamino_init_referrer_token_state_instruction(
            payer=WALLET,
            lending_market=MARKET,
            reserve=RESERVE,
        )

        assert instruction.accounts[3].pubkey == Pubkey.from_string(WALLET)

    def test_refresh_reserve_instruction_defaults_optional_oracles(self):
        instruction = build_kamino_refresh_reserve_instruction(
            lending_market=MARKET,
            reserve=RESERVE,
        )

        assert instruction.accounts[2].pubkey == KAMINO_LEND_PROGRAM_ID
        assert instruction.accounts[5].pubkey == KAMINO_LEND_PROGRAM_ID

    def test_withdraw_referrer_fees_instruction_uses_custom_token_program(self):
        instruction = build_kamino_withdraw_referrer_fees_instruction(
            referrer=WALLET,
            lending_market=MARKET,
            reserve=RESERVE,
            reserve_liquidity_mint=MINT,
            referrer_token_account="1111111111111111111111111111111B",
            token_program_id=str(TOKEN_2022_PROGRAM_ID),
        )

        assert instruction.accounts[8].pubkey == TOKEN_2022_PROGRAM_ID


class TestKaminoApiHelpers:
    @pytest.mark.asyncio
    async def test_get_success_and_http_error_and_exception(self):
        api = KaminoAPI(base_url="https://kamino.example/")
        success_response = MagicMock(status_code=200)
        success_response.json.return_value = {"ok": True}
        success_response.text = ""
        error_response = MagicMock(status_code=404)
        error_response.json.return_value = {"error": "missing"}
        error_response.text = "missing"

        success_client = AsyncMock()
        success_client.__aenter__.return_value = success_client
        success_client.__aexit__.return_value = False
        success_client.get = AsyncMock(return_value=success_response)

        error_client = AsyncMock()
        error_client.__aenter__.return_value = error_client
        error_client.__aexit__.return_value = False
        error_client.get = AsyncMock(return_value=error_response)

        with patch(
            "solana_agent.tools.utils.kamino.httpx.AsyncClient",
            side_effect=[success_client, error_client, RuntimeError("boom")],
        ):
            success = await api._get("kvaults/vaults", params={"page": 1})
            failure = await api._get("/missing")
            exception = await api._get("/explode")

        assert api._url("path") == "https://kamino.example/path"
        assert success == {"success": True, "data": {"ok": True}}
        assert failure["success"] is False
        assert "Kamino GET /missing failed: 404 - missing" == failure["error"]
        assert exception["success"] is False
        assert exception["data"] == {}

    @pytest.mark.asyncio
    async def test_post_success_and_http_error_and_exception(self):
        api = KaminoAPI()
        success_response = MagicMock(status_code=200)
        success_response.json.return_value = {"requestId": "req-1", "transaction": "tx"}
        success_response.text = ""
        error_response = MagicMock(status_code=400)
        error_response.json.return_value = {"message": "bad request"}
        error_response.text = "bad request"

        success_client = AsyncMock()
        success_client.__aenter__.return_value = success_client
        success_client.__aexit__.return_value = False
        success_client.post = AsyncMock(return_value=success_response)

        error_client = AsyncMock()
        error_client.__aenter__.return_value = error_client
        error_client.__aexit__.return_value = False
        error_client.post = AsyncMock(return_value=error_response)

        with patch(
            "solana_agent.tools.utils.kamino.httpx.AsyncClient",
            side_effect=[success_client, error_client, RuntimeError("explode")],
        ):
            success = await api._post("/ok", body={"x": 1})
            failure = await api._post("/bad")
            exception = await api._post("/explode")

        assert success["success"] is True
        assert failure["error"] == "Kamino POST /bad failed: 400 - bad request"
        assert exception["error"] == "explode"

    def test_parse_response_and_transaction_response_helpers(self):
        api = KaminoAPI()
        json_response = MagicMock(spec=httpx.Response)
        json_response.json.return_value = [1, 2, 3]
        json_response.text = "[1,2,3]"
        text_response = MagicMock(spec=httpx.Response)
        text_response.json.side_effect = ValueError("bad")
        text_response.text = "plain-text"

        assert api._parse_response(json_response) == {"result": [1, 2, 3]}
        assert api._parse_response(text_response) == {"text": "plain-text"}
        assert (
            api._format_error(
                "GET", "/x", MagicMock(status_code=500, text="oops"), {"error": "kaput"}
            )
            == "Kamino GET /x failed: 500 - kaput"
        )

        success = api._as_transaction_response(
            {"success": True, "data": {"transaction": "tx", "requestId": "req"}}
        )
        failure = api._as_transaction_response(
            {"success": False, "error": "nope", "data": {"raw": True}}
        )

        assert success == KaminoTransactionResponse(
            success=True,
            transaction="tx",
            request_id="req",
            raw_response={"transaction": "tx", "requestId": "req"},
        )
        assert failure == KaminoTransactionResponse(
            success=False,
            error="nope",
            raw_response={"raw": True},
        )

    @pytest.mark.asyncio
    async def test_api_methods_forward_to_get_and_post(self):
        api = KaminoAPI()
        with (
            patch.object(
                api, "_get", new=AsyncMock(return_value={"success": True, "data": {}})
            ) as mock_get,
            patch.object(
                api,
                "_post",
                new=AsyncMock(
                    return_value={"success": True, "data": {"transaction": "tx"}}
                ),
            ) as mock_post,
        ):
            await api.list_vaults()
            await api.list_markets()
            await api.get_oracle_prices({"symbol": "SOL"})
            await api.get_user_vault_positions(WALLET)
            await api.get_user_obligations(MARKET, WALLET)
            await api.api_get("/custom", params={"a": 1})
            await api.api_post("/custom", body={"b": 2})
            await api.build_earn_deposit(WALLET, "vault", "1")
            await api.build_earn_withdraw(WALLET, "vault", "1")
            await api.build_borrow_deposit(
                WALLET, MARKET, RESERVE, "1", referrer=REFERRER, referral_code="code"
            )
            await api.build_borrow_borrow(WALLET, MARKET, RESERVE, "1")
            await api.build_borrow_repay(WALLET, MARKET, RESERVE, "1")
            await api.build_borrow_withdraw(WALLET, MARKET, RESERVE, "1")

        assert mock_get.await_count == 6
        assert mock_post.await_count == 7
        deposit_body = next(
            call.args[1]
            for call in mock_post.await_args_list
            if call.args[0] == "/ktx/klend/deposit"
        )
        assert deposit_body["referrer"] == REFERRER
        assert deposit_body["shortUrl"] == "code"

    @pytest.mark.asyncio
    async def test_build_borrow_deposit_omits_optional_referral_fields_when_empty(self):
        api = KaminoAPI()
        with patch.object(
            api,
            "_post",
            new=AsyncMock(
                return_value={"success": True, "data": {"transaction": "tx"}}
            ),
        ) as mock_post:
            await api.build_borrow_deposit(
                WALLET, MARKET, RESERVE, "1", referrer="", referral_code=""
            )

        assert mock_post.await_args.args[0] == "/ktx/klend/deposit"
        assert mock_post.await_args.args[1] == {
            "wallet": WALLET,
            "market": MARKET,
            "reserve": RESERVE,
            "amount": "1",
        }


class TestKaminoUtilityHelpers:
    @pytest.mark.asyncio
    async def test_account_exists_and_resolve_token_program(self):
        client = AsyncMock()
        client.get_account_info = AsyncMock(
            side_effect=[
                MagicMock(value=MagicMock()),
                MagicMock(value=MagicMock(owner=str(TOKEN_2022_PROGRAM_ID))),
                MagicMock(value=MagicMock(owner=str(SPL_TOKEN_PROGRAM_ID))),
            ]
        )

        assert await _account_exists(client, Pubkey.from_string(WALLET)) is True
        assert (
            await _resolve_token_program_pubkey(client, Pubkey.from_string(MINT))
            == TOKEN_2022_PROGRAM_ID
        )
        assert (
            await _resolve_token_program_pubkey(client, Pubkey.from_string(MINT))
            == SPL_TOKEN_PROGRAM_ID
        )

    @pytest.mark.asyncio
    async def test_compile_placeholder_transaction_branches(self):
        signer = Keypair()
        instruction = build_kamino_refresh_reserve_instruction(MARKET, RESERVE)

        assert await _compile_placeholder_transaction(
            "https://rpc.example.com", signer.pubkey(), []
        ) == {
            "status": "error",
            "message": "No instructions to compile.",
        }

        with patch(
            "solana_agent.tools.utils.kamino.get_fresh_blockhash",
            new=AsyncMock(return_value={"error": "rpc down"}),
        ):
            failure = await _compile_placeholder_transaction(
                "https://rpc.example.com", signer.pubkey(), [instruction]
            )

        assert failure == {
            "status": "error",
            "message": "Failed to get blockhash: rpc down",
        }

        with patch(
            "solana_agent.tools.utils.kamino.get_fresh_blockhash",
            new=AsyncMock(return_value={"blockhash": str(Pubkey.default())}),
        ):
            success = await _compile_placeholder_transaction(
                "https://rpc.example.com", signer.pubkey(), [instruction]
            )

        assert success["status"] == "success"
        assert isinstance(success["transaction"], str)

    @pytest.mark.asyncio
    async def test_build_create_lookup_table_instruction_branches(self):
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.get_slot = AsyncMock(return_value=MagicMock(value=123))

        with patch(
            "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
        ):
            instruction, address = await _build_create_lookup_table_instruction(
                "https://rpc.example.com", WALLET
            )

        assert address is not None
        assert instruction is not None

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.get_slot = AsyncMock(return_value=MagicMock(value=None))

        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            pytest.raises(ValueError, match="recent slot"),
        ):
            await _build_create_lookup_table_instruction(
                "https://rpc.example.com", WALLET
            )

    def test_small_helper_functions(self):
        assert _borsh_string("ab") == b"\x02\x00\x00\x00ab"
        assert _optional_pubkey("") == KAMINO_LEND_PROGRAM_ID
        assert _optional_pubkey(WALLET) == Pubkey.from_string(WALLET)
        assert _read_pubkey(bytes(Pubkey.from_string(WALLET)), 0) == WALLET


class TestKaminoReferralTransactionBuilders:
    def test_parse_reserve_metadata(self):
        metadata = parse_kamino_reserve_metadata(
            reserve=RESERVE,
            account_data=_build_reserve_account_data(),
        )

        assert metadata.lending_market == MARKET
        assert metadata.reserve == RESERVE
        assert metadata.reserve_liquidity_mint == MINT
        assert metadata.token_program_id == TOKEN_PROGRAM
        assert metadata.scope_prices == SCOPE
        assert metadata.switchboard_price_oracle == SWITCHBOARD
        assert metadata.switchboard_twap_oracle == TWAP
        assert metadata.pyth_oracle == PYTH

    def test_parse_reserve_metadata_without_discriminator_and_short_data(self):
        raw_data = bytearray(4048)
        raw_data[24:56] = bytes(Pubkey.from_string(MARKET))
        raw_data[120:152] = bytes(Pubkey.from_string(MINT))
        raw_data[392:424] = bytes(Pubkey.from_string(TOKEN_PROGRAM))
        raw_data[3896:3928] = bytes(Pubkey.from_string(SCOPE))
        raw_data[3944:3976] = bytes(Pubkey.from_string(SWITCHBOARD))
        raw_data[3976:4008] = bytes(Pubkey.from_string(TWAP))
        raw_data[4008:4040] = bytes(Pubkey.from_string(PYTH))

        metadata = parse_kamino_reserve_metadata(RESERVE, bytes(raw_data))

        assert metadata.reserve_liquidity_mint == MINT

        with pytest.raises(ValueError, match="shorter than expected"):
            parse_kamino_reserve_metadata(RESERVE, b"short")

    @pytest.mark.asyncio
    async def test_fetch_reserve_metadata(self):
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.get_account_info = AsyncMock(
            return_value=MagicMock(value=MagicMock(data=_build_reserve_account_data()))
        )

        with patch(
            "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
        ):
            metadata = await fetch_kamino_reserve_metadata(
                rpc_url="https://rpc.example.com",
                reserve=RESERVE,
            )

        assert metadata.reserve_liquidity_mint == MINT
        assert metadata.token_program_id == TOKEN_PROGRAM

    @pytest.mark.asyncio
    async def test_fetch_reserve_metadata_missing_account(self):
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.get_account_info = AsyncMock(return_value=MagicMock(value=None))

        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            pytest.raises(ValueError, match="was not found"),
        ):
            await fetch_kamino_reserve_metadata(
                rpc_url="https://rpc.example.com",
                reserve=RESERVE,
            )

    @pytest.mark.asyncio
    async def test_create_lookup_table_transaction(self):
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            patch(
                "solana_agent.tools.utils.kamino._build_create_lookup_table_instruction",
                new=AsyncMock(
                    return_value=(
                        MagicMock(name="create_lut_ix"),
                        Pubkey.from_string(LOOKUP_TABLE),
                    )
                ),
            ),
            patch(
                "solana_agent.tools.utils.kamino._compile_placeholder_transaction",
                new=AsyncMock(
                    return_value={"status": "success", "transaction": "tx-base64"}
                ),
            ) as mock_compile,
        ):
            result = await build_kamino_create_lookup_table_transaction(
                rpc_url="https://rpc.example.com",
                wallet_public_key=WALLET,
            )

        assert result["status"] == "success"
        assert result["user_lookup_table"] == LOOKUP_TABLE
        assert len(mock_compile.await_args.args[2]) == 1

    @pytest.mark.asyncio
    async def test_create_lookup_table_transaction_compile_error(self):
        with (
            patch(
                "solana_agent.tools.utils.kamino._build_create_lookup_table_instruction",
                new=AsyncMock(
                    return_value=(
                        MagicMock(name="create_lut_ix"),
                        Pubkey.from_string(LOOKUP_TABLE),
                    )
                ),
            ),
            patch(
                "solana_agent.tools.utils.kamino._compile_placeholder_transaction",
                new=AsyncMock(
                    return_value={"status": "error", "message": "compile failed"}
                ),
            ),
        ):
            result = await build_kamino_create_lookup_table_transaction(
                rpc_url="https://rpc.example.com",
                wallet_public_key=WALLET,
            )

        assert result == {"status": "error", "message": "compile failed"}

    @pytest.mark.asyncio
    async def test_init_user_metadata_auto_creates_lookup_table(self):
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            patch(
                "solana_agent.tools.utils.kamino._account_exists",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "solana_agent.tools.utils.kamino._build_create_lookup_table_instruction",
                new=AsyncMock(
                    return_value=(
                        MagicMock(name="create_lut_ix"),
                        Pubkey.from_string(LOOKUP_TABLE),
                    )
                ),
            ),
            patch(
                "solana_agent.tools.utils.kamino._compile_placeholder_transaction",
                new=AsyncMock(
                    return_value={"status": "success", "transaction": "tx-base64"}
                ),
            ) as mock_compile,
        ):
            result = await build_kamino_init_user_metadata_transaction(
                rpc_url="https://rpc.example.com",
                wallet_public_key=WALLET,
            )

        assert result["status"] == "success"
        assert result["user_lookup_table"] == LOOKUP_TABLE
        instructions = mock_compile.await_args.args[2]
        assert len(instructions) == 2
        assert bytes(instructions[1].data[:8]) == INIT_USER_METADATA_DISCRIMINATOR

    @pytest.mark.asyncio
    async def test_init_user_metadata_existing_and_compile_error(self):
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            patch(
                "solana_agent.tools.utils.kamino._account_exists",
                new=AsyncMock(return_value=True),
            ),
        ):
            existing = await build_kamino_init_user_metadata_transaction(
                rpc_url="https://rpc.example.com",
                wallet_public_key=WALLET,
            )

        assert existing["status"] == "error"
        assert "already exists" in existing["message"]

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            patch(
                "solana_agent.tools.utils.kamino._account_exists",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "solana_agent.tools.utils.kamino._compile_placeholder_transaction",
                new=AsyncMock(
                    return_value={"status": "error", "message": "compile failed"}
                ),
            ),
        ):
            compile_error = await build_kamino_init_user_metadata_transaction(
                rpc_url="https://rpc.example.com",
                wallet_public_key=WALLET,
                user_lookup_table=LOOKUP_TABLE,
            )

        assert compile_error == {"status": "error", "message": "compile failed"}

    @pytest.mark.asyncio
    async def test_referrer_setup_prepends_user_metadata_init(self):
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            patch(
                "solana_agent.tools.utils.kamino._account_exists",
                new=AsyncMock(side_effect=[False, False]),
            ),
            patch(
                "solana_agent.tools.utils.kamino._build_create_lookup_table_instruction",
                new=AsyncMock(
                    return_value=(
                        MagicMock(name="create_lut_ix"),
                        Pubkey.from_string(LOOKUP_TABLE),
                    )
                ),
            ),
            patch(
                "solana_agent.tools.utils.kamino._compile_placeholder_transaction",
                new=AsyncMock(
                    return_value={"status": "success", "transaction": "tx-base64"}
                ),
            ) as mock_compile,
        ):
            result = await build_kamino_referrer_setup_transaction(
                rpc_url="https://rpc.example.com",
                wallet_public_key=WALLET,
                short_url="kamino_ref",
            )

        assert result["status"] == "success"
        assert result["user_lookup_table"] == LOOKUP_TABLE
        instructions = mock_compile.await_args.args[2]
        assert len(instructions) == 3
        assert bytes(instructions[1].data[:8]) == INIT_USER_METADATA_DISCRIMINATOR
        assert (
            bytes(instructions[2].data[:8])
            == INIT_REFERRER_STATE_AND_SHORT_URL_DISCRIMINATOR
        )

    @pytest.mark.asyncio
    async def test_referrer_setup_uses_existing_lookup_table_without_creating_one(self):
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            patch(
                "solana_agent.tools.utils.kamino._account_exists",
                new=AsyncMock(side_effect=[False, False]),
            ),
            patch(
                "solana_agent.tools.utils.kamino._build_create_lookup_table_instruction",
                new=AsyncMock(),
            ) as mock_create,
            patch(
                "solana_agent.tools.utils.kamino._compile_placeholder_transaction",
                new=AsyncMock(
                    return_value={"status": "success", "transaction": "tx-base64"}
                ),
            ) as mock_compile,
        ):
            result = await build_kamino_referrer_setup_transaction(
                rpc_url="https://rpc.example.com",
                wallet_public_key=WALLET,
                short_url="kamino_ref",
                user_lookup_table=LOOKUP_TABLE,
            )

        assert result["status"] == "success"
        assert result["user_lookup_table"] == LOOKUP_TABLE
        mock_create.assert_not_awaited()
        instructions = mock_compile.await_args.args[2]
        assert len(instructions) == 2

    @pytest.mark.asyncio
    async def test_referrer_setup_validation_and_existing_state_and_compile_error(self):
        invalid = await build_kamino_referrer_setup_transaction(
            rpc_url="https://rpc.example.com",
            wallet_public_key=WALLET,
            short_url="bad space",
        )
        assert invalid["status"] == "error"

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            patch(
                "solana_agent.tools.utils.kamino._account_exists",
                new=AsyncMock(side_effect=[True, True]),
            ),
        ):
            existing = await build_kamino_referrer_setup_transaction(
                rpc_url="https://rpc.example.com",
                wallet_public_key=WALLET,
                short_url="kamino_ref",
                user_lookup_table=LOOKUP_TABLE,
            )

        assert existing["status"] == "error"
        assert "Referrer state already exists" in existing["message"]

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            patch(
                "solana_agent.tools.utils.kamino._account_exists",
                new=AsyncMock(side_effect=[True, False]),
            ),
            patch(
                "solana_agent.tools.utils.kamino._compile_placeholder_transaction",
                new=AsyncMock(
                    return_value={"status": "error", "message": "compile failed"}
                ),
            ),
        ):
            compile_error = await build_kamino_referrer_setup_transaction(
                rpc_url="https://rpc.example.com",
                wallet_public_key=WALLET,
                short_url="kamino_ref",
                user_lookup_table=LOOKUP_TABLE,
            )

        assert compile_error == {"status": "error", "message": "compile failed"}

    @pytest.mark.asyncio
    async def test_withdraw_referrer_fees_prepends_setup_instructions(self):
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            patch(
                "solana_agent.tools.utils.kamino._resolve_token_program_pubkey",
                new=AsyncMock(
                    return_value=Pubkey.from_string(
                        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
                    )
                ),
            ),
            patch(
                "solana_agent.tools.utils.kamino._account_exists",
                new=AsyncMock(side_effect=[False, False]),
            ),
            patch(
                "solana_agent.tools.utils.kamino._compile_placeholder_transaction",
                new=AsyncMock(
                    return_value={"status": "success", "transaction": "tx-base64"}
                ),
            ) as mock_compile,
        ):
            result = await build_kamino_withdraw_referrer_fees_transaction(
                rpc_url="https://rpc.example.com",
                wallet_public_key=WALLET,
                lending_market=MARKET,
                reserve=RESERVE,
                reserve_liquidity_mint=MINT,
                pyth_oracle=PYTH,
                switchboard_price_oracle=SWITCHBOARD,
                switchboard_twap_oracle=TWAP,
                scope_prices=SCOPE,
            )

        assert result["status"] == "success"
        instructions = mock_compile.await_args.args[2]
        assert len(instructions) == 4
        assert bytes(instructions[1].data[:8]) != b""
        assert bytes(instructions[3].data) == WITHDRAW_REFERRER_FEES_DISCRIMINATOR

    @pytest.mark.asyncio
    async def test_withdraw_referrer_fees_existing_accounts_and_compile_error(self):
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch(
                "solana_agent.tools.utils.kamino.AsyncClient", return_value=mock_client
            ),
            patch(
                "solana_agent.tools.utils.kamino._account_exists",
                new=AsyncMock(side_effect=[True, True]),
            ),
            patch(
                "solana_agent.tools.utils.kamino._compile_placeholder_transaction",
                new=AsyncMock(
                    return_value={"status": "error", "message": "compile failed"}
                ),
            ),
        ):
            result = await build_kamino_withdraw_referrer_fees_transaction(
                rpc_url="https://rpc.example.com",
                wallet_public_key=WALLET,
                lending_market=MARKET,
                reserve=RESERVE,
                reserve_liquidity_mint=MINT,
                token_program_id=str(TOKEN_2022_PROGRAM_ID),
            )

        assert result == {"status": "error", "message": "compile failed"}

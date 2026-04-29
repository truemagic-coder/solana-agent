"""Kamino REST API utility functions.

Provides a small async client for Kamino's public read endpoints and
transaction-building endpoints for Earn and K-Lend.
"""

import base64
import logging
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx
from solana.rpc.async_api import AsyncClient
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.message import MessageV0, to_bytes_versioned
from solders.null_signer import NullSigner
from solders.pubkey import Pubkey
from solders.system_program import (
    CreateLookupTableParams,
    create_lookup_table,
)
from solders.transaction import VersionedTransaction
from spl.token.instructions import (
    create_associated_token_account,
    get_associated_token_address,
)

from solana_agent.tools.utils.trigger import get_fresh_blockhash

logger = logging.getLogger(__name__)

KAMINO_API_BASE_URL = "https://api.kamino.finance"
KAMINO_LEND_PROGRAM_ID = Pubkey.from_string(
    "KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD"
)
KAMINO_RESERVE_ACCOUNT_DISCRIMINATOR = bytes([43, 242, 204, 202, 26, 247, 59, 127])
SYSTEM_PROGRAM_ID = Pubkey.from_string("11111111111111111111111111111111")
SYSVAR_RENT_ID = Pubkey.from_string("SysvarRent111111111111111111111111111111111")
SPL_TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
TOKEN_2022_PROGRAM_ID = Pubkey.from_string(
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
)

INIT_USER_METADATA_DISCRIMINATOR = bytes([117, 169, 176, 69, 197, 23, 15, 162])
INIT_REFERRER_TOKEN_STATE_DISCRIMINATOR = bytes([116, 45, 66, 148, 58, 13, 218, 115])
WITHDRAW_REFERRER_FEES_DISCRIMINATOR = bytes([171, 118, 121, 201, 233, 140, 23, 228])
INIT_REFERRER_STATE_AND_SHORT_URL_DISCRIMINATOR = bytes(
    [165, 19, 25, 127, 100, 55, 31, 90]
)
REFRESH_RESERVE_DISCRIMINATOR = bytes([2, 218, 138, 235, 79, 201, 25, 102])

USER_METADATA_SEED = b"user_meta"
REFERRER_TOKEN_STATE_SEED = b"referrer_acc"
REFERRER_STATE_SEED = b"ref_state"
SHORT_URL_SEED = b"short_url"
LENDING_MARKET_AUTHORITY_SEED = b"lma"
RESERVE_LIQUIDITY_SUPPLY_SEED = b"reserve_liq_supply"

SHORT_URL_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


@dataclass
class KaminoTransactionResponse:
    """Response from a Kamino transaction-building endpoint."""

    success: bool
    transaction: Optional[str] = None
    request_id: Optional[str] = None
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KaminoReserveMetadata:
    """Reserve fields needed for referrer fee withdrawal automation."""

    lending_market: str
    reserve: str
    reserve_liquidity_mint: str
    token_program_id: str
    pyth_oracle: str
    switchboard_price_oracle: str
    switchboard_twap_oracle: str
    scope_prices: str


class KaminoAPI:
    """Kamino REST API client."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or KAMINO_API_BASE_URL).rstrip("/")
        self._headers = {"Content-Type": "application/json"}

    def _url(self, path: str) -> str:
        normalized = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{normalized}"

    async def _get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    self._url(path),
                    params=params,
                    headers=self._headers,
                )

            data = self._parse_response(response)
            if response.status_code >= 400:
                return {
                    "success": False,
                    "error": self._format_error("GET", path, response, data),
                    "data": data,
                }

            return {"success": True, "data": data}
        except Exception as e:
            logger.exception("Kamino GET request failed")
            return {"success": False, "error": str(e), "data": {}}

    async def _post(
        self, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self._url(path),
                    json=body or {},
                    headers=self._headers,
                )

            data = self._parse_response(response)
            if response.status_code >= 400:
                return {
                    "success": False,
                    "error": self._format_error("POST", path, response, data),
                    "data": data,
                }

            return {"success": True, "data": data}
        except Exception as e:
            logger.exception("Kamino POST request failed")
            return {"success": False, "error": str(e), "data": {}}

    def _parse_response(self, response: httpx.Response) -> Dict[str, Any]:
        try:
            data = response.json()
            return data if isinstance(data, dict) else {"result": data}
        except Exception:
            return {"text": response.text}

    def _format_error(
        self,
        method: str,
        path: str,
        response: httpx.Response,
        data: Dict[str, Any],
    ) -> str:
        message = data.get("message") or data.get("error") or response.text
        return f"Kamino {method} {path} failed: {response.status_code} - {message}"

    async def build_earn_deposit(
        self, wallet: str, kvault: str, amount: str
    ) -> KaminoTransactionResponse:
        result = await self._post(
            "/ktx/kvault/deposit",
            {"wallet": wallet, "kvault": kvault, "amount": amount},
        )
        return self._as_transaction_response(result)

    async def build_earn_withdraw(
        self, wallet: str, kvault: str, amount: str
    ) -> KaminoTransactionResponse:
        result = await self._post(
            "/ktx/kvault/withdraw",
            {"wallet": wallet, "kvault": kvault, "amount": amount},
        )
        return self._as_transaction_response(result)

    async def build_borrow_deposit(
        self,
        wallet: str,
        market: str,
        reserve: str,
        amount: str,
        referrer: Optional[str] = None,
        referral_code: Optional[str] = None,
    ) -> KaminoTransactionResponse:
        body: Dict[str, Any] = {
            "wallet": wallet,
            "market": market,
            "reserve": reserve,
            "amount": amount,
        }
        if referrer:
            body["referrer"] = referrer
        if referral_code:
            body["shortUrl"] = referral_code

        result = await self._post("/ktx/klend/deposit", body)
        return self._as_transaction_response(result)

    async def build_borrow_borrow(
        self, wallet: str, market: str, reserve: str, amount: str
    ) -> KaminoTransactionResponse:
        result = await self._post(
            "/ktx/klend/borrow",
            {"wallet": wallet, "market": market, "reserve": reserve, "amount": amount},
        )
        return self._as_transaction_response(result)

    async def build_borrow_repay(
        self, wallet: str, market: str, reserve: str, amount: str
    ) -> KaminoTransactionResponse:
        result = await self._post(
            "/ktx/klend/repay",
            {"wallet": wallet, "market": market, "reserve": reserve, "amount": amount},
        )
        return self._as_transaction_response(result)

    async def build_borrow_withdraw(
        self, wallet: str, market: str, reserve: str, amount: str
    ) -> KaminoTransactionResponse:
        result = await self._post(
            "/ktx/klend/withdraw",
            {"wallet": wallet, "market": market, "reserve": reserve, "amount": amount},
        )
        return self._as_transaction_response(result)

    async def list_vaults(self) -> Dict[str, Any]:
        return await self._get("/kvaults/vaults")

    async def list_markets(self) -> Dict[str, Any]:
        return await self._get("/v2/kamino-market")

    async def get_oracle_prices(
        self, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return await self._get("/oracles/prices", params=params)

    async def get_user_vault_positions(self, pubkey: str) -> Dict[str, Any]:
        return await self._get(f"/kvaults/users/{pubkey}/positions")

    async def get_user_obligations(
        self, market_pubkey: str, user_pubkey: str
    ) -> Dict[str, Any]:
        return await self._get(
            f"/kamino-market/{market_pubkey}/users/{user_pubkey}/obligations"
        )

    async def api_get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return await self._get(path, params=params)

    async def api_post(
        self, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return await self._post(path, body=body)

    def _as_transaction_response(
        self, result: Dict[str, Any]
    ) -> KaminoTransactionResponse:
        if not result.get("success"):
            return KaminoTransactionResponse(
                success=False,
                error=result.get("error"),
                raw_response=result.get("data") or {},
            )

        data = result.get("data") or {}
        return KaminoTransactionResponse(
            success=True,
            transaction=data.get("transaction"),
            request_id=data.get("requestId"),
            raw_response=data,
        )


def validate_kamino_short_url(short_url: str) -> Optional[str]:
    """Validate Kamino referral short URLs."""

    if not short_url:
        return "short_url is required."
    if not SHORT_URL_PATTERN.fullmatch(short_url):
        return "short_url must be 1-32 characters using only ASCII letters, digits, '-' or '_'."
    return None


def derive_kamino_user_metadata_pda(owner: str) -> Pubkey:
    return Pubkey.find_program_address(
        [USER_METADATA_SEED, bytes(Pubkey.from_string(owner))],
        KAMINO_LEND_PROGRAM_ID,
    )[0]


def derive_kamino_referrer_token_state_pda(referrer: str, reserve: str) -> Pubkey:
    return Pubkey.find_program_address(
        [
            REFERRER_TOKEN_STATE_SEED,
            bytes(Pubkey.from_string(referrer)),
            bytes(Pubkey.from_string(reserve)),
        ],
        KAMINO_LEND_PROGRAM_ID,
    )[0]


def derive_kamino_referrer_state_pda(referrer: str) -> Pubkey:
    return Pubkey.find_program_address(
        [REFERRER_STATE_SEED, bytes(Pubkey.from_string(referrer))],
        KAMINO_LEND_PROGRAM_ID,
    )[0]


def derive_kamino_short_url_pda(short_url: str) -> Pubkey:
    return Pubkey.find_program_address(
        [SHORT_URL_SEED, short_url.encode("ascii")],
        KAMINO_LEND_PROGRAM_ID,
    )[0]


def derive_kamino_lending_market_authority_pda(lending_market: str) -> Pubkey:
    return Pubkey.find_program_address(
        [LENDING_MARKET_AUTHORITY_SEED, bytes(Pubkey.from_string(lending_market))],
        KAMINO_LEND_PROGRAM_ID,
    )[0]


def derive_kamino_reserve_liquidity_supply_pda(reserve: str) -> Pubkey:
    return Pubkey.find_program_address(
        [RESERVE_LIQUIDITY_SUPPLY_SEED, bytes(Pubkey.from_string(reserve))],
        KAMINO_LEND_PROGRAM_ID,
    )[0]


def _borsh_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _optional_pubkey(address: str) -> Pubkey:
    return Pubkey.from_string(address) if address else KAMINO_LEND_PROGRAM_ID


def _read_pubkey(data: bytes, offset: int) -> str:
    return str(Pubkey.from_bytes(data[offset : offset + 32]))


def parse_kamino_reserve_metadata(
    reserve: str, account_data: bytes
) -> KaminoReserveMetadata:
    """Decode the minimal reserve metadata needed for referral fee withdrawals."""

    if account_data.startswith(KAMINO_RESERVE_ACCOUNT_DISCRIMINATOR):
        base_offset = len(KAMINO_RESERVE_ACCOUNT_DISCRIMINATOR)
    else:
        base_offset = 0

    minimum_size = 4048
    if len(account_data) < minimum_size:
        raise ValueError("Kamino reserve account data is shorter than expected.")

    liquidity_offset = base_offset + 120
    config_offset = base_offset + 3640
    token_info_offset = config_offset + 176

    return KaminoReserveMetadata(
        lending_market=_read_pubkey(account_data, base_offset + 24),
        reserve=reserve,
        reserve_liquidity_mint=_read_pubkey(account_data, liquidity_offset),
        token_program_id=_read_pubkey(account_data, liquidity_offset + 272),
        scope_prices=_read_pubkey(account_data, token_info_offset + 80),
        switchboard_price_oracle=_read_pubkey(account_data, token_info_offset + 128),
        switchboard_twap_oracle=_read_pubkey(account_data, token_info_offset + 160),
        pyth_oracle=_read_pubkey(account_data, token_info_offset + 192),
    )


async def fetch_kamino_reserve_metadata(
    rpc_url: str,
    reserve: str,
) -> KaminoReserveMetadata:
    """Fetch and decode reserve metadata directly from the reserve account."""

    client = AsyncClient(rpc_url)
    try:
        response = await client.get_account_info(Pubkey.from_string(reserve))
        account = getattr(response, "value", None)
        if not account or not getattr(account, "data", b""):
            raise ValueError(f"Reserve account {reserve} was not found.")
        return parse_kamino_reserve_metadata(
            reserve=reserve, account_data=bytes(account.data)
        )
    finally:
        await client.close()


def build_kamino_init_user_metadata_instruction(
    owner: str,
    user_lookup_table: str,
    referrer: str = "",
    fee_payer: str = "",
) -> Instruction:
    owner_pubkey = Pubkey.from_string(owner)
    fee_payer_pubkey = Pubkey.from_string(fee_payer or owner)
    user_metadata = derive_kamino_user_metadata_pda(owner)
    referrer_user_metadata = (
        derive_kamino_user_metadata_pda(referrer)
        if referrer
        else KAMINO_LEND_PROGRAM_ID
    )

    return Instruction(
        program_id=KAMINO_LEND_PROGRAM_ID,
        accounts=[
            AccountMeta(owner_pubkey, True, False),
            AccountMeta(fee_payer_pubkey, True, True),
            AccountMeta(user_metadata, False, True),
            AccountMeta(referrer_user_metadata, False, False),
            AccountMeta(SYSVAR_RENT_ID, False, False),
            AccountMeta(SYSTEM_PROGRAM_ID, False, False),
        ],
        data=INIT_USER_METADATA_DISCRIMINATOR
        + bytes(Pubkey.from_string(user_lookup_table)),
    )


def build_kamino_init_referrer_state_and_short_url_instruction(
    referrer: str, short_url: str
) -> Instruction:
    error = validate_kamino_short_url(short_url)
    if error:
        raise ValueError(error)

    referrer_pubkey = Pubkey.from_string(referrer)
    return Instruction(
        program_id=KAMINO_LEND_PROGRAM_ID,
        accounts=[
            AccountMeta(referrer_pubkey, True, True),
            AccountMeta(derive_kamino_referrer_state_pda(referrer), False, True),
            AccountMeta(derive_kamino_short_url_pda(short_url), False, True),
            AccountMeta(derive_kamino_user_metadata_pda(referrer), False, False),
            AccountMeta(SYSVAR_RENT_ID, False, False),
            AccountMeta(SYSTEM_PROGRAM_ID, False, False),
        ],
        data=INIT_REFERRER_STATE_AND_SHORT_URL_DISCRIMINATOR + _borsh_string(short_url),
    )


def build_kamino_init_referrer_token_state_instruction(
    payer: str,
    lending_market: str,
    reserve: str,
    referrer: str = "",
) -> Instruction:
    payer_pubkey = Pubkey.from_string(payer)
    referrer_pubkey = Pubkey.from_string(referrer or payer)
    return Instruction(
        program_id=KAMINO_LEND_PROGRAM_ID,
        accounts=[
            AccountMeta(payer_pubkey, True, True),
            AccountMeta(Pubkey.from_string(lending_market), False, False),
            AccountMeta(Pubkey.from_string(reserve), False, False),
            AccountMeta(referrer_pubkey, False, False),
            AccountMeta(
                derive_kamino_referrer_token_state_pda(str(referrer_pubkey), reserve),
                False,
                True,
            ),
            AccountMeta(SYSVAR_RENT_ID, False, False),
            AccountMeta(SYSTEM_PROGRAM_ID, False, False),
        ],
        data=INIT_REFERRER_TOKEN_STATE_DISCRIMINATOR,
    )


def build_kamino_refresh_reserve_instruction(
    lending_market: str,
    reserve: str,
    pyth_oracle: str = "",
    switchboard_price_oracle: str = "",
    switchboard_twap_oracle: str = "",
    scope_prices: str = "",
) -> Instruction:
    return Instruction(
        program_id=KAMINO_LEND_PROGRAM_ID,
        accounts=[
            AccountMeta(Pubkey.from_string(reserve), False, True),
            AccountMeta(Pubkey.from_string(lending_market), False, False),
            AccountMeta(_optional_pubkey(pyth_oracle), False, False),
            AccountMeta(_optional_pubkey(switchboard_price_oracle), False, False),
            AccountMeta(_optional_pubkey(switchboard_twap_oracle), False, False),
            AccountMeta(_optional_pubkey(scope_prices), False, False),
        ],
        data=REFRESH_RESERVE_DISCRIMINATOR,
    )


def build_kamino_withdraw_referrer_fees_instruction(
    referrer: str,
    lending_market: str,
    reserve: str,
    reserve_liquidity_mint: str,
    referrer_token_account: str,
    token_program_id: str = "",
) -> Instruction:
    referrer_pubkey = Pubkey.from_string(referrer)
    return Instruction(
        program_id=KAMINO_LEND_PROGRAM_ID,
        accounts=[
            AccountMeta(referrer_pubkey, True, True),
            AccountMeta(
                derive_kamino_referrer_token_state_pda(referrer, reserve),
                False,
                True,
            ),
            AccountMeta(Pubkey.from_string(reserve), False, True),
            AccountMeta(Pubkey.from_string(reserve_liquidity_mint), False, False),
            AccountMeta(
                derive_kamino_reserve_liquidity_supply_pda(reserve),
                False,
                True,
            ),
            AccountMeta(Pubkey.from_string(referrer_token_account), False, True),
            AccountMeta(Pubkey.from_string(lending_market), False, False),
            AccountMeta(
                derive_kamino_lending_market_authority_pda(lending_market),
                False,
                False,
            ),
            AccountMeta(
                Pubkey.from_string(token_program_id)
                if token_program_id
                else SPL_TOKEN_PROGRAM_ID,
                False,
                False,
            ),
        ],
        data=WITHDRAW_REFERRER_FEES_DISCRIMINATOR,
    )


async def _account_exists(client: AsyncClient, address: Pubkey) -> bool:
    response = await client.get_account_info(address)
    return bool(getattr(response, "value", None))


async def _resolve_token_program_pubkey(
    client: AsyncClient, mint_pubkey: Pubkey
) -> Pubkey:
    response = await client.get_account_info(mint_pubkey)
    owner = str(getattr(getattr(response, "value", None), "owner", ""))
    if owner == str(TOKEN_2022_PROGRAM_ID):
        return TOKEN_2022_PROGRAM_ID
    return SPL_TOKEN_PROGRAM_ID


async def _compile_placeholder_transaction(
    rpc_url: str,
    payer: Pubkey,
    instructions: list[Instruction],
) -> Dict[str, Any]:
    if not instructions:
        return {"status": "error", "message": "No instructions to compile."}

    blockhash_result = await get_fresh_blockhash(rpc_url)
    if "error" in blockhash_result:
        return {
            "status": "error",
            "message": f"Failed to get blockhash: {blockhash_result['error']}",
        }

    message = MessageV0.try_compile(
        payer=payer,
        instructions=instructions,
        address_lookup_table_accounts=[],
        recent_blockhash=Hash.from_string(blockhash_result["blockhash"]),
    )
    placeholder_signature = NullSigner(payer).sign_message(to_bytes_versioned(message))
    transaction = VersionedTransaction.populate(message, [placeholder_signature])
    return {
        "status": "success",
        "transaction": base64.b64encode(bytes(transaction)).decode("utf-8"),
    }


async def _build_create_lookup_table_instruction(
    rpc_url: str,
    authority: str,
) -> tuple[Instruction, Pubkey]:
    client = AsyncClient(rpc_url)
    try:
        slot_response = await client.get_slot()
        recent_slot = getattr(slot_response, "value", None)
        if recent_slot is None:
            raise ValueError(
                "RPC did not return a recent slot for lookup table creation."
            )
    finally:
        await client.close()

    instruction, lookup_table_address = create_lookup_table(
        CreateLookupTableParams(
            authority_address=Pubkey.from_string(authority),
            payer_address=Pubkey.from_string(authority),
            recent_slot=int(recent_slot),
        )
    )
    return instruction, lookup_table_address


async def build_kamino_create_lookup_table_transaction(
    rpc_url: str,
    wallet_public_key: str,
) -> Dict[str, Any]:
    (
        create_instruction,
        lookup_table_address,
    ) = await _build_create_lookup_table_instruction(
        rpc_url,
        wallet_public_key,
    )
    compiled = await _compile_placeholder_transaction(
        rpc_url,
        Pubkey.from_string(wallet_public_key),
        [create_instruction],
    )
    if compiled.get("status") == "error":
        return compiled
    return {
        "status": "success",
        "transaction": compiled["transaction"],
        "user_lookup_table": str(lookup_table_address),
    }


async def build_kamino_init_user_metadata_transaction(
    rpc_url: str,
    wallet_public_key: str,
    user_lookup_table: str = "",
    referrer: str = "",
) -> Dict[str, Any]:
    user_metadata = derive_kamino_user_metadata_pda(wallet_public_key)
    client = AsyncClient(rpc_url)
    try:
        if await _account_exists(client, user_metadata):
            return {
                "status": "error",
                "message": "User metadata already exists for this wallet.",
                "user_metadata": str(user_metadata),
            }
    finally:
        await client.close()

    instructions: list[Instruction] = []
    lookup_table_address = user_lookup_table
    if not lookup_table_address:
        (
            create_lookup_table_instruction,
            derived_lookup_table,
        ) = await _build_create_lookup_table_instruction(rpc_url, wallet_public_key)
        instructions.append(create_lookup_table_instruction)
        lookup_table_address = str(derived_lookup_table)

    compiled = await _compile_placeholder_transaction(
        rpc_url,
        Pubkey.from_string(wallet_public_key),
        instructions
        + [
            build_kamino_init_user_metadata_instruction(
                owner=wallet_public_key,
                user_lookup_table=lookup_table_address,
                referrer=referrer,
            )
        ],
    )
    if compiled.get("status") == "error":
        return compiled
    return {
        "status": "success",
        "transaction": compiled["transaction"],
        "user_metadata": str(user_metadata),
        "user_lookup_table": lookup_table_address,
        "referrer_user_metadata": (
            str(derive_kamino_user_metadata_pda(referrer)) if referrer else None
        ),
    }


async def build_kamino_referrer_setup_transaction(
    rpc_url: str,
    wallet_public_key: str,
    short_url: str,
    user_lookup_table: str = "",
) -> Dict[str, Any]:
    error = validate_kamino_short_url(short_url)
    if error:
        return {"status": "error", "message": error}

    user_metadata = derive_kamino_user_metadata_pda(wallet_public_key)
    referrer_state = derive_kamino_referrer_state_pda(wallet_public_key)
    referrer_short_url = derive_kamino_short_url_pda(short_url)
    instructions: list[Instruction] = []

    client = AsyncClient(rpc_url)
    try:
        has_user_metadata = await _account_exists(client, user_metadata)
        has_referrer_state = await _account_exists(client, referrer_state)
    finally:
        await client.close()

    if has_referrer_state:
        return {
            "status": "error",
            "message": "Referrer state already exists for this wallet.",
            "referrer_state": str(referrer_state),
        }

    if not has_user_metadata:
        lookup_table_address = user_lookup_table
        if not lookup_table_address:
            (
                create_lookup_table_instruction,
                derived_lookup_table,
            ) = await _build_create_lookup_table_instruction(rpc_url, wallet_public_key)
            instructions.append(create_lookup_table_instruction)
            lookup_table_address = str(derived_lookup_table)
        instructions.append(
            build_kamino_init_user_metadata_instruction(
                owner=wallet_public_key,
                user_lookup_table=lookup_table_address,
            )
        )
        user_lookup_table = lookup_table_address

    instructions.append(
        build_kamino_init_referrer_state_and_short_url_instruction(
            referrer=wallet_public_key,
            short_url=short_url,
        )
    )

    compiled = await _compile_placeholder_transaction(
        rpc_url,
        Pubkey.from_string(wallet_public_key),
        instructions,
    )
    if compiled.get("status") == "error":
        return compiled
    return {
        "status": "success",
        "transaction": compiled["transaction"],
        "user_metadata": str(user_metadata),
        "user_lookup_table": user_lookup_table or None,
        "referrer_state": str(referrer_state),
        "referrer_short_url": str(referrer_short_url),
    }


async def build_kamino_withdraw_referrer_fees_transaction(
    rpc_url: str,
    wallet_public_key: str,
    lending_market: str,
    reserve: str,
    reserve_liquidity_mint: str,
    pyth_oracle: str = "",
    switchboard_price_oracle: str = "",
    switchboard_twap_oracle: str = "",
    scope_prices: str = "",
    token_program_id: str = "",
) -> Dict[str, Any]:
    referrer_pubkey = Pubkey.from_string(wallet_public_key)
    mint_pubkey = Pubkey.from_string(reserve_liquidity_mint)
    client = AsyncClient(rpc_url)
    try:
        token_program_pubkey = (
            Pubkey.from_string(token_program_id)
            if token_program_id
            else await _resolve_token_program_pubkey(client, mint_pubkey)
        )
        referrer_token_account = get_associated_token_address(
            referrer_pubkey,
            mint_pubkey,
            token_program_id=token_program_pubkey,
        )
        referrer_token_state = derive_kamino_referrer_token_state_pda(
            wallet_public_key, reserve
        )
        has_referrer_token_account = await _account_exists(
            client, referrer_token_account
        )
        has_referrer_token_state = await _account_exists(client, referrer_token_state)
    finally:
        await client.close()

    instructions: list[Instruction] = []
    if not has_referrer_token_account:
        instructions.append(
            create_associated_token_account(
                payer=referrer_pubkey,
                owner=referrer_pubkey,
                mint=mint_pubkey,
                token_program_id=token_program_pubkey,
            )
        )
    if not has_referrer_token_state:
        instructions.append(
            build_kamino_init_referrer_token_state_instruction(
                payer=wallet_public_key,
                lending_market=lending_market,
                reserve=reserve,
            )
        )
    instructions.append(
        build_kamino_refresh_reserve_instruction(
            lending_market=lending_market,
            reserve=reserve,
            pyth_oracle=pyth_oracle,
            switchboard_price_oracle=switchboard_price_oracle,
            switchboard_twap_oracle=switchboard_twap_oracle,
            scope_prices=scope_prices,
        )
    )
    instructions.append(
        build_kamino_withdraw_referrer_fees_instruction(
            referrer=wallet_public_key,
            lending_market=lending_market,
            reserve=reserve,
            reserve_liquidity_mint=reserve_liquidity_mint,
            referrer_token_account=str(referrer_token_account),
            token_program_id=str(token_program_pubkey),
        )
    )

    compiled = await _compile_placeholder_transaction(
        rpc_url,
        referrer_pubkey,
        instructions,
    )
    if compiled.get("status") == "error":
        return compiled
    return {
        "status": "success",
        "transaction": compiled["transaction"],
        "referrer_token_account": str(referrer_token_account),
        "referrer_token_state": str(referrer_token_state),
    }

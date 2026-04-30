"""Kamino tool for Privy delegated wallets."""

import base64
import json
import logging
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives import serialization
from privy import AsyncPrivyAPI
from privy.lib.authorization_signatures import get_authorization_signature
from solana_agent import AutoTool, ToolRegistry
from solders.keypair import Keypair
from solders.message import to_bytes_versioned
from solders.transaction import VersionedTransaction

from solana_agent.tools.utils.kamino import (
    KaminoAPI,
    build_kamino_create_lookup_table_transaction,
    build_kamino_init_user_metadata_transaction,
    build_kamino_referrer_setup_transaction,
    build_kamino_withdraw_referrer_fees_transaction,
    fetch_kamino_reserve_metadata,
)
from solana_agent.tools.utils.trigger import (
    get_fresh_blockhash,
    replace_blockhash_in_transaction,
)
from solana_agent.tools.utils.wallet import send_raw_transaction_with_priority

logger = logging.getLogger(__name__)


def _convert_key_to_pkcs8_pem(key_string: str) -> str:  # pragma: no cover
    private_key_string = key_string.replace("wallet-auth:", "")

    try:
        private_key_pem = (
            "-----BEGIN PRIVATE KEY-----\n"
            f"{private_key_string}\n"
            "-----END PRIVATE KEY-----"
        )
        serialization.load_pem_private_key(
            private_key_pem.encode("utf-8"), password=None
        )
        return private_key_string
    except (ValueError, TypeError):
        pass

    try:
        ec_key_pem = (
            "-----BEGIN EC PRIVATE KEY-----\n"
            f"{private_key_string}\n"
            "-----END EC PRIVATE KEY-----"
        )
        private_key = serialization.load_pem_private_key(
            ec_key_pem.encode("utf-8"), password=None
        )
        pkcs8_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pkcs8_pem = pkcs8_bytes.decode("utf-8")
        return "".join(pkcs8_pem.strip().split("\n")[1:-1])
    except (ValueError, TypeError):
        pass

    try:
        der_bytes = base64.b64decode(private_key_string)
        try:
            private_key = serialization.load_der_private_key(der_bytes, password=None)
        except (ValueError, TypeError):
            from cryptography.hazmat.primitives.asymmetric import ec

            private_key = ec.derive_private_key(
                int.from_bytes(der_bytes, "big"), ec.SECP256R1()
            )
        pkcs8_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pkcs8_pem = pkcs8_bytes.decode("utf-8")
        return "".join(pkcs8_pem.strip().split("\n")[1:-1])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not load private key: {e}")


async def _privy_sign_transaction(  # pragma: no cover
    privy_client: AsyncPrivyAPI,
    wallet_id: str,
    encoded_tx: str,
    signing_key: str,
) -> Optional[str]:
    try:
        pkcs8_key = _convert_key_to_pkcs8_pem(signing_key)
        url = f"https://api.privy.io/v1/wallets/{wallet_id}/rpc"
        body = {
            "method": "signTransaction",
            "params": {"transaction": encoded_tx, "encoding": "base64"},
            "chain_type": "solana",
        }
        auth_signature = get_authorization_signature(
            url=url,
            body=body,
            method="POST",
            app_id=privy_client.app_id,
            private_key=pkcs8_key,
        )
        result = await privy_client.wallets.rpc(
            wallet_id=wallet_id,
            method="signTransaction",
            params={"transaction": encoded_tx, "encoding": "base64"},
            chain_type="solana",
            privy_authorization_signature=auth_signature,
        )
        return result.data.signed_transaction if result.data else None
    except Exception as e:
        logger.error(f"Privy API error signing transaction: {e}")
        return None


class PrivyKaminoTool(AutoTool):
    """Use Kamino with Privy delegated wallets."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="privy_kamino",
            description=(
                "Use Kamino Finance with Privy delegated wallets for Earn, K-Lend, and Kamino read endpoints. "
                "Referral attribution is handled automatically when referrer_private_key is configured."
            ),
            registry=registry,
        )
        self._app_id: Optional[str] = None
        self._app_secret: Optional[str] = None
        self._signing_key: Optional[str] = None
        self._rpc_url: Optional[str] = None
        self._base_url: Optional[str] = None
        self._referrer_private_key: Optional[str] = None
        self._managed_referrer: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "wallet_id": {
                    "type": "string",
                    "description": "Privy wallet ID. REQUIRED.",
                },
                "wallet_public_key": {
                    "type": "string",
                    "description": "Solana public key of the delegated wallet. REQUIRED.",
                },
                "action": {
                    "type": "string",
                    "enum": [
                        "earn_deposit",
                        "earn_withdraw",
                        "borrow_deposit",
                        "borrow_borrow",
                        "borrow_repay",
                        "borrow_withdraw",
                        "list_vaults",
                        "list_markets",
                        "oracle_prices",
                        "vault_positions",
                        "user_obligations",
                        "api_get",
                        "api_post",
                    ],
                    "description": "Kamino action to perform.",
                },
                "kvault": {"type": "string", "default": ""},
                "market": {"type": "string", "default": ""},
                "reserve": {"type": "string", "default": ""},
                "amount": {"type": "string", "default": ""},
                "user_pubkey": {"type": "string", "default": ""},
                "path": {"type": "string", "default": ""},
                "params_json": {"type": "string", "default": ""},
                "body_json": {"type": "string", "default": ""},
                "referrer": {"type": "string", "default": ""},
                "referral_code": {"type": "string", "default": ""},
            },
            "required": [
                "wallet_id",
                "wallet_public_key",
                "action",
                "kvault",
                "market",
                "reserve",
                "amount",
                "user_pubkey",
                "path",
                "params_json",
                "body_json",
                "referrer",
                "referral_code",
            ],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("privy_kamino", {})
        self._app_id = tool_cfg.get("app_id")
        self._app_secret = tool_cfg.get("app_secret")
        self._signing_key = tool_cfg.get("signing_key")
        self._rpc_url = tool_cfg.get("rpc_url")
        self._base_url = tool_cfg.get("base_url")
        self._referrer_private_key = tool_cfg.get("referrer_private_key")
        self._managed_referrer = None
        if self._referrer_private_key:
            self._managed_referrer = str(
                Keypair.from_base58_string(self._referrer_private_key).pubkey()
            )

    async def execute(
        self,
        wallet_id: str,
        wallet_public_key: str,
        action: str,
        kvault: str = "",
        market: str = "",
        reserve: str = "",
        amount: str = "",
        user_pubkey: str = "",
        path: str = "",
        params_json: str = "",
        body_json: str = "",
        referrer: str = "",
        referral_code: str = "",
        short_url: str = "",
        user_lookup_table: str = "",
        reserve_liquidity_mint: str = "",
        pyth_oracle: str = "",
        switchboard_price_oracle: str = "",
        switchboard_twap_oracle: str = "",
        scope_prices: str = "",
        token_program_id: str = "",
    ) -> Dict[str, Any]:
        if not wallet_id or not wallet_public_key:
            return {
                "status": "error",
                "message": "wallet_id and wallet_public_key are required.",
            }
        if not all([self._app_id, self._app_secret, self._signing_key]):
            return {"status": "error", "message": "Privy config missing."}

        action = action.strip().lower()
        kamino = KaminoAPI(base_url=self._base_url)
        effective_referrer = referrer or self._managed_referrer or ""
        effective_referral_code = referral_code or ""

        if action in {
            "earn_deposit",
            "earn_withdraw",
            "borrow_deposit",
            "borrow_borrow",
            "borrow_repay",
            "borrow_withdraw",
            "create_user_lookup_table",
            "init_user_metadata",
            "setup_referrer",
            "withdraw_referrer_fees",
        }:
            if not self._rpc_url:
                return {"status": "error", "message": "RPC URL not configured."}

            privy_client = AsyncPrivyAPI(
                app_id=self._app_id, app_secret=self._app_secret
            )
            try:
                referral_automation = (
                    await self._run_pre_transaction_referral_automation(
                        action=action,
                        effective_referrer=effective_referrer,
                    )
                )
                if referral_automation.get("status") == "error":
                    return referral_automation

                tx_result = await self._build_transaction(
                    kamino=kamino,
                    action=action,
                    wallet_public_key=wallet_public_key,
                    kvault=kvault,
                    market=market,
                    reserve=reserve,
                    amount=amount,
                    referrer=effective_referrer,
                    referral_code=effective_referral_code,
                    short_url=short_url,
                    user_lookup_table=user_lookup_table,
                    reserve_liquidity_mint=reserve_liquidity_mint,
                    pyth_oracle=pyth_oracle,
                    switchboard_price_oracle=switchboard_price_oracle,
                    switchboard_twap_oracle=switchboard_twap_oracle,
                    scope_prices=scope_prices,
                    token_program_id=token_program_id,
                )
                if tx_result.get("status") == "error":
                    return tx_result

                execution = await self._sign_and_send(
                    privy_client=privy_client,
                    wallet_id=wallet_id,
                    transaction_base64=tx_result["transaction"],
                )
                if execution.get("status") == "error":
                    return execution

                withdrawal_results = (
                    await self._run_post_transaction_referral_automation(
                        action=action,
                        market=market,
                        reserve=reserve,
                        effective_referrer=effective_referrer,
                    )
                )

                response = {
                    "status": "success",
                    "action": action,
                    "wallet": wallet_public_key,
                    "signature": execution.get("signature"),
                    "referrer": effective_referrer or None,
                    "referral_code": effective_referral_code or None,
                    "referral_automation": {
                        "setup": referral_automation,
                        "withdrawals": withdrawal_results,
                    },
                }
                for key, value in tx_result.items():
                    if key not in {"status", "transaction"}:
                        response[key] = value
                return response
            finally:
                await privy_client.close()

        if action == "list_vaults":
            return self._format_read_response(action, await kamino.list_vaults())
        if action == "list_markets":
            return self._format_read_response(action, await kamino.list_markets())
        if action == "oracle_prices":
            params = self._parse_json_object(params_json, "params_json")
            if isinstance(params, dict) and params.get("status") == "error":
                return params
            return self._format_read_response(
                action, await kamino.get_oracle_prices(params=params or None)
            )
        if action == "vault_positions":
            if not user_pubkey:
                return {
                    "status": "error",
                    "message": "user_pubkey is required for vault_positions.",
                }
            return self._format_read_response(
                action, await kamino.get_user_vault_positions(user_pubkey)
            )
        if action == "user_obligations":
            if not market or not user_pubkey:
                return {
                    "status": "error",
                    "message": "market and user_pubkey are required for user_obligations.",
                }
            return self._format_read_response(
                action, await kamino.get_user_obligations(market, user_pubkey)
            )
        if action == "api_get":
            if not path:
                return {"status": "error", "message": "path is required for api_get."}
            params = self._parse_json_object(params_json, "params_json")
            if isinstance(params, dict) and params.get("status") == "error":
                return params
            return self._format_read_response(
                action,
                await kamino.api_get(path=path, params=params or None),
                path=path,
            )
        if action == "api_post":
            if not path:
                return {"status": "error", "message": "path is required for api_post."}
            body = self._parse_json_object(body_json, "body_json")
            if isinstance(body, dict) and body.get("status") == "error":
                return body
            return self._format_read_response(
                action, await kamino.api_post(path=path, body=body or None), path=path
            )

        return {
            "status": "error",
            "message": (
                "Unknown action. Valid actions: earn_deposit, earn_withdraw, borrow_deposit, "
                "borrow_borrow, borrow_repay, borrow_withdraw, list_vaults, list_markets, "
                "oracle_prices, vault_positions, user_obligations, api_get, api_post"
            ),
        }

    async def _run_pre_transaction_referral_automation(
        self,
        action: str,
        effective_referrer: str,
    ) -> Dict[str, Any]:
        return {"status": "skipped", "reason": "no_pre_transaction_steps_required"}

    async def _run_post_transaction_referral_automation(
        self,
        action: str,
        market: str,
        reserve: str,
        effective_referrer: str,
    ) -> List[Dict[str, Any]]:
        if action != "borrow_deposit":
            return []
        if not self._managed_referrer or effective_referrer != self._managed_referrer:
            return []

        referrer_keypair = self._get_internal_referrer_keypair()
        if (
            not referrer_keypair
            or str(referrer_keypair.pubkey()) != self._managed_referrer
        ):
            return []

        try:
            target = await fetch_kamino_reserve_metadata(self._rpc_url, reserve)
        except Exception as e:
            return [
                {
                    "status": "error",
                    "market": market,
                    "reserve": reserve,
                    "message": str(e),
                }
            ]

        if target.lending_market != market:
            return [
                {
                    "status": "error",
                    "market": market,
                    "reserve": reserve,
                    "message": "Reserve metadata market does not match requested market.",
                }
            ]

        results: List[Dict[str, Any]] = []
        for target_metadata in [target]:
            tx_result = await build_kamino_withdraw_referrer_fees_transaction(
                rpc_url=self._rpc_url,
                wallet_public_key=self._managed_referrer,
                lending_market=target_metadata.lending_market,
                reserve=target_metadata.reserve,
                reserve_liquidity_mint=target_metadata.reserve_liquidity_mint,
                pyth_oracle=target_metadata.pyth_oracle,
                switchboard_price_oracle=target_metadata.switchboard_price_oracle,
                switchboard_twap_oracle=target_metadata.switchboard_twap_oracle,
                scope_prices=target_metadata.scope_prices,
                token_program_id=target_metadata.token_program_id,
            )
            if tx_result.get("status") == "error":
                results.append(
                    {
                        "status": "error",
                        "market": target_metadata.lending_market,
                        "reserve": target_metadata.reserve,
                        "message": tx_result.get("message"),
                    }
                )
                continue

            execution = await self._sign_and_send_with_keypair(
                keypair=referrer_keypair,
                transaction_base64=tx_result["transaction"],
            )
            results.append(
                {
                    "status": execution.get("status", "error"),
                    "market": target_metadata.lending_market,
                    "reserve": target_metadata.reserve,
                    "signature": execution.get("signature"),
                    "message": execution.get("message"),
                }
            )
        return results

    def _get_internal_referrer_keypair(self) -> Optional[Keypair]:
        if not self._referrer_private_key:
            return None
        return Keypair.from_base58_string(self._referrer_private_key)

    async def _build_transaction(
        self,
        kamino: KaminoAPI,
        action: str,
        wallet_public_key: str,
        kvault: str,
        market: str,
        reserve: str,
        amount: str,
        referrer: str,
        referral_code: str,
        short_url: str,
        user_lookup_table: str,
        reserve_liquidity_mint: str,
        pyth_oracle: str,
        switchboard_price_oracle: str,
        switchboard_twap_oracle: str,
        scope_prices: str,
        token_program_id: str,
    ) -> Dict[str, Any]:
        if action == "create_user_lookup_table":
            return await build_kamino_create_lookup_table_transaction(
                rpc_url=self._rpc_url,
                wallet_public_key=wallet_public_key,
            )

        if action == "init_user_metadata":
            return await build_kamino_init_user_metadata_transaction(
                rpc_url=self._rpc_url,
                wallet_public_key=wallet_public_key,
                user_lookup_table=user_lookup_table,
                referrer=referrer,
            )
        if action == "setup_referrer":
            if not short_url:
                return {
                    "status": "error",
                    "message": "short_url is required for setup_referrer.",
                }
            return await build_kamino_referrer_setup_transaction(
                rpc_url=self._rpc_url,
                wallet_public_key=wallet_public_key,
                short_url=short_url,
                user_lookup_table=user_lookup_table,
            )
        if action == "withdraw_referrer_fees":
            if not market or not reserve or not reserve_liquidity_mint:
                return {
                    "status": "error",
                    "message": "market, reserve, and reserve_liquidity_mint are required for withdraw_referrer_fees.",
                }
            return await build_kamino_withdraw_referrer_fees_transaction(
                rpc_url=self._rpc_url,
                wallet_public_key=wallet_public_key,
                lending_market=market,
                reserve=reserve,
                reserve_liquidity_mint=reserve_liquidity_mint,
                pyth_oracle=pyth_oracle,
                switchboard_price_oracle=switchboard_price_oracle,
                switchboard_twap_oracle=switchboard_twap_oracle,
                scope_prices=scope_prices,
                token_program_id=token_program_id,
            )
        if action in {"earn_deposit", "earn_withdraw"}:
            if not kvault or not amount:
                return {
                    "status": "error",
                    "message": "kvault and amount are required for earn actions.",
                }
            result = (
                await kamino.build_earn_deposit(wallet_public_key, kvault, amount)
                if action == "earn_deposit"
                else await kamino.build_earn_withdraw(wallet_public_key, kvault, amount)
            )
        else:
            if not market or not reserve or not amount:
                return {
                    "status": "error",
                    "message": "market, reserve, and amount are required for borrow actions.",
                }
            if action == "borrow_deposit":
                result = await kamino.build_borrow_deposit(
                    wallet=wallet_public_key,
                    market=market,
                    reserve=reserve,
                    amount=amount,
                    referrer=referrer or None,
                    referral_code=referral_code or None,
                )
            elif action == "borrow_borrow":
                result = await kamino.build_borrow_borrow(
                    wallet_public_key, market, reserve, amount
                )
            elif action == "borrow_repay":
                result = await kamino.build_borrow_repay(
                    wallet_public_key, market, reserve, amount
                )
            else:
                result = await kamino.build_borrow_withdraw(
                    wallet_public_key, market, reserve, amount
                )

        if not result.success or not result.transaction:
            return {
                "status": "error",
                "message": result.error or "Kamino did not return a transaction.",
                "raw_response": result.raw_response,
            }
        return {
            "status": "success",
            "transaction": result.transaction,
            "request_id": result.request_id,
            "raw_response": result.raw_response,
        }

    async def _sign_and_send(  # pragma: no cover
        self,
        privy_client: AsyncPrivyAPI,
        wallet_id: str,
        transaction_base64: str,
    ) -> Dict[str, Any]:
        try:
            blockhash_result = await get_fresh_blockhash(self._rpc_url)
            if "error" in blockhash_result:
                return {
                    "status": "error",
                    "message": f"Failed to get blockhash: {blockhash_result['error']}",
                }
            tx_with_new_blockhash = replace_blockhash_in_transaction(
                transaction_base64, blockhash_result["blockhash"]
            )
            signed_tx = await _privy_sign_transaction(
                privy_client,
                wallet_id,
                tx_with_new_blockhash,
                self._signing_key,
            )
            if not signed_tx:
                return {
                    "status": "error",
                    "message": "Failed to sign transaction via Privy.",
                }
            send_result = await send_raw_transaction_with_priority(
                rpc_url=self._rpc_url,
                tx_bytes=base64.b64decode(signed_tx),
                skip_preflight=True,
                skip_confirmation=False,
                confirm_timeout=30.0,
            )
            if not send_result.get("success"):
                return {
                    "status": "error",
                    "message": send_result.get("error", "Failed to send transaction"),
                }
            return {"status": "success", "signature": send_result.get("signature")}
        except Exception as e:
            logger.exception("Failed to sign and send Privy Kamino transaction")
            return {"status": "error", "message": str(e)}

    async def _sign_and_send_with_keypair(
        self,
        keypair: Keypair,
        transaction_base64: str,
    ) -> Dict[str, Any]:
        try:
            blockhash_result = await get_fresh_blockhash(self._rpc_url)
            if "error" in blockhash_result:
                return {
                    "status": "error",
                    "message": f"Failed to get blockhash: {blockhash_result['error']}",
                }

            tx_with_new_blockhash = replace_blockhash_in_transaction(
                transaction_base64, blockhash_result["blockhash"]
            )
            tx_bytes = base64.b64decode(tx_with_new_blockhash)
            transaction = VersionedTransaction.from_bytes(tx_bytes)
            message_bytes = to_bytes_versioned(transaction.message)

            num_signers = transaction.message.header.num_required_signatures
            account_keys = transaction.message.account_keys
            signer_index = None
            for index in range(num_signers):
                if account_keys[index] == keypair.pubkey():
                    signer_index = index
                    break

            if signer_index is None:
                return {
                    "status": "error",
                    "message": f"Signer pubkey {keypair.pubkey()} not found in Kamino transaction signers.",
                }

            new_signatures = list(transaction.signatures)
            new_signatures[signer_index] = keypair.sign_message(message_bytes)
            signed_transaction = VersionedTransaction.populate(
                transaction.message,
                new_signatures,
            )

            send_result = await send_raw_transaction_with_priority(
                rpc_url=self._rpc_url,
                tx_bytes=bytes(signed_transaction),
                skip_preflight=True,
                skip_confirmation=False,
                confirm_timeout=30.0,
            )
            if not send_result.get("success"):
                return {
                    "status": "error",
                    "message": send_result.get("error", "Failed to send transaction"),
                }

            return {"status": "success", "signature": send_result.get("signature")}
        except Exception as e:
            logger.exception("Failed to sign and send internal Kamino transaction")
            return {"status": "error", "message": str(e)}

    def _parse_json_object(
        self, raw_value: str, field_name: str
    ) -> Optional[Dict[str, Any]]:
        if not raw_value:
            return None
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "message": f"{field_name} must be valid JSON: {e}",
            }
        if not isinstance(parsed, dict):
            return {
                "status": "error",
                "message": f"{field_name} must decode to a JSON object.",
            }
        return parsed

    def _format_read_response(
        self, action: str, result: Dict[str, Any], path: Optional[str] = None
    ) -> Dict[str, Any]:
        if not result.get("success"):
            return {
                "status": "error",
                "message": result.get("error", "Kamino request failed."),
                "path": path,
            }
        response: Dict[str, Any] = {
            "status": "success",
            "action": action,
            "data": result.get("data"),
        }
        if path:
            response["path"] = path
        return response


class PrivyKaminoPlugin:
    def __init__(self) -> None:
        self.name = "privy_kamino"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self) -> str:
        return "Plugin for Kamino operations using Privy delegated wallets."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = PrivyKaminoTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return PrivyKaminoPlugin()

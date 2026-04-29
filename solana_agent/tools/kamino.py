"""Kamino tool for Solana Agent Kit.

Supports Kamino Earn and K-Lend transaction endpoints plus the main public read
endpoints documented in Kamino's API reference.
"""

import base64
import json
import logging
from typing import Any, Dict, List, Optional

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


class KaminoTool(AutoTool):
    """Kamino Earn and Borrow tool."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="kamino",
            description=(
                "Use Kamino Finance for Earn vault actions, K-Lend deposit/borrow/repay/withdraw, "
                "and Kamino vault, market, oracle, and user position reads. Referral attribution is handled automatically "
                "when referrer_private_key is configured. Actions: earn_deposit, earn_withdraw, borrow_deposit, "
                "borrow_borrow, borrow_repay, borrow_withdraw, list_vaults, list_markets, oracle_prices, "
                "vault_positions, user_obligations, api_get, api_post."
            ),
            registry=registry,
        )
        self._private_key: Optional[str] = None
        self._rpc_url: Optional[str] = None
        self._base_url: Optional[str] = None
        self._referrer_private_key: Optional[str] = None
        self._managed_referrer: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
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
                "kvault": {
                    "type": "string",
                    "description": "Kamino vault address for earn_deposit or earn_withdraw. Pass empty string if not needed.",
                    "default": "",
                },
                "market": {
                    "type": "string",
                    "description": "Kamino market address for K-Lend operations or user_obligations. Pass empty string if not needed.",
                    "default": "",
                },
                "reserve": {
                    "type": "string",
                    "description": "Kamino reserve address for K-Lend operations. Pass empty string if not needed.",
                    "default": "",
                },
                "amount": {
                    "type": "string",
                    "description": "Human-readable token amount expected by Kamino REST transaction endpoints, such as '1.0'. Pass empty string if not needed.",
                    "default": "",
                },
                "user_pubkey": {
                    "type": "string",
                    "description": "User public key for vault_positions or user_obligations. Pass empty string if not needed.",
                    "default": "",
                },
                "path": {
                    "type": "string",
                    "description": "Kamino API path for api_get or api_post, such as '/oracles/prices'. Pass empty string if not needed.",
                    "default": "",
                },
                "params_json": {
                    "type": "string",
                    "description": "JSON object string of query parameters for oracle_prices or api_get. Pass empty string if not needed.",
                    "default": "",
                },
                "body_json": {
                    "type": "string",
                    "description": "JSON object string for api_post. Pass empty string if not needed.",
                    "default": "",
                },
                "referrer": {
                    "type": "string",
                    "description": "Optional referrer wallet override for borrow_deposit. If omitted, the configured referrer_private_key is used when present. Pass empty string if not needed.",
                    "default": "",
                },
                "referral_code": {
                    "type": "string",
                    "description": "Optional Kamino referral short URL override for borrow_deposit. Pass empty string if not needed.",
                    "default": "",
                },
            },
            "required": [
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
        tool_cfg = config.get("tools", {}).get("kamino", {})
        self._private_key = tool_cfg.get("private_key")
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
            if not self._private_key:
                return {"status": "error", "message": "Private key not configured."}
            if not self._rpc_url:
                return {"status": "error", "message": "RPC URL not configured."}

            keypair = Keypair.from_base58_string(self._private_key)
            wallet = str(keypair.pubkey())

            referral_automation = await self._run_pre_transaction_referral_automation(
                action=action,
                wallet=wallet,
                effective_referrer=effective_referrer,
            )
            if referral_automation.get("status") == "error":
                return referral_automation

            tx_result = await self._build_transaction(
                kamino=kamino,
                action=action,
                wallet=wallet,
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
                keypair=keypair,
                transaction_base64=tx_result["transaction"],
            )
            if execution.get("status") == "error":
                return execution

            withdrawal_results = await self._run_post_transaction_referral_automation(
                action=action,
                market=market,
                reserve=reserve,
                effective_referrer=effective_referrer,
            )

            response = {
                "status": "success",
                "action": action,
                "wallet": wallet,
                "signature": execution.get("signature"),
                "kvault": kvault or None,
                "market": market or None,
                "reserve": reserve or None,
                "amount": amount or None,
                "referrer": effective_referrer or None,
                "referral_code": effective_referral_code or None,
                "short_url": short_url or None,
                "user_lookup_table": user_lookup_table or None,
                "reserve_liquidity_mint": reserve_liquidity_mint or None,
                "referral_automation": {
                    "setup": referral_automation,
                    "withdrawals": withdrawal_results,
                },
            }
            for key, value in tx_result.items():
                if key not in {"status", "transaction"}:
                    response[key] = value
            return response

        if action == "list_vaults":
            result = await kamino.list_vaults()
            return self._format_read_response(action, result)

        if action == "list_markets":
            result = await kamino.list_markets()
            return self._format_read_response(action, result)

        if action == "oracle_prices":
            params = self._parse_json_object(params_json, "params_json")
            if isinstance(params, dict) and params.get("status") == "error":
                return params
            result = await kamino.get_oracle_prices(params=params or None)
            return self._format_read_response(action, result)

        if action == "vault_positions":
            if not user_pubkey:
                return {
                    "status": "error",
                    "message": "user_pubkey is required for vault_positions.",
                }
            result = await kamino.get_user_vault_positions(user_pubkey)
            return self._format_read_response(action, result)

        if action == "user_obligations":
            if not market or not user_pubkey:
                return {
                    "status": "error",
                    "message": "market and user_pubkey are required for user_obligations.",
                }
            result = await kamino.get_user_obligations(market, user_pubkey)
            return self._format_read_response(action, result)

        if action == "api_get":
            if not path:
                return {"status": "error", "message": "path is required for api_get."}
            params = self._parse_json_object(params_json, "params_json")
            if isinstance(params, dict) and params.get("status") == "error":
                return params
            result = await kamino.api_get(path=path, params=params or None)
            return self._format_read_response(action, result, path=path)

        if action == "api_post":
            if not path:
                return {"status": "error", "message": "path is required for api_post."}
            body = self._parse_json_object(body_json, "body_json")
            if isinstance(body, dict) and body.get("status") == "error":
                return body
            result = await kamino.api_post(path=path, body=body or None)
            return self._format_read_response(action, result, path=path)

        return {
            "status": "error",
            "message": (
                "Unknown action. Valid actions: earn_deposit, earn_withdraw, borrow_deposit, "
                "borrow_borrow, borrow_repay, borrow_withdraw, list_vaults, list_markets, "
                "oracle_prices, vault_positions, user_obligations, api_get, api_post"
            ),
        }

    async def _build_transaction(
        self,
        kamino: KaminoAPI,
        action: str,
        wallet: str,
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
                wallet_public_key=wallet,
            )

        if action == "init_user_metadata":
            return await build_kamino_init_user_metadata_transaction(
                rpc_url=self._rpc_url,
                wallet_public_key=wallet,
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
                wallet_public_key=wallet,
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
                wallet_public_key=wallet,
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
                await kamino.build_earn_deposit(wallet, kvault, amount)
                if action == "earn_deposit"
                else await kamino.build_earn_withdraw(wallet, kvault, amount)
            )
        else:
            if not market or not reserve or not amount:
                return {
                    "status": "error",
                    "message": "market, reserve, and amount are required for borrow actions.",
                }
            if action == "borrow_deposit":
                result = await kamino.build_borrow_deposit(
                    wallet=wallet,
                    market=market,
                    reserve=reserve,
                    amount=amount,
                    referrer=referrer or None,
                    referral_code=referral_code or None,
                )
            elif action == "borrow_borrow":
                result = await kamino.build_borrow_borrow(
                    wallet, market, reserve, amount
                )
            elif action == "borrow_repay":
                result = await kamino.build_borrow_repay(
                    wallet, market, reserve, amount
                )
            else:
                result = await kamino.build_borrow_withdraw(
                    wallet, market, reserve, amount
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

    async def _run_pre_transaction_referral_automation(
        self,
        action: str,
        wallet: str,
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

        referrer_keypair = self._get_internal_referrer_keypair("")
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

        results: List[Dict[str, Any]] = []
        if target.lending_market != market:
            return [
                {
                    "status": "error",
                    "market": market,
                    "reserve": reserve,
                    "message": "Reserve metadata market does not match requested market.",
                }
            ]

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

            execution = await self._sign_and_send(
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

    def _get_internal_referrer_keypair(self, wallet: str) -> Optional[Keypair]:
        if not self._referrer_private_key:
            return None
        return Keypair.from_base58_string(self._referrer_private_key)

    async def _sign_and_send(
        self, keypair: Keypair, transaction_base64: str
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
            logger.exception("Failed to sign and send Kamino transaction")
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


class KaminoPlugin:
    """Plugin for Kamino Finance operations."""

    def __init__(self) -> None:
        self.name = "kamino"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self) -> str:
        return "Plugin for Kamino Earn and K-Lend operations."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = KaminoTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return KaminoPlugin()

"""
DFlow Prediction Market tool for Solana Agent Kit.

Enables prediction market trading with safety-first approach:
- Quality filters applied by default (volume, liquidity, age)
- Safety scoring on all market queries
- Risk warnings for low-quality markets
- Blocking async execution for agent compatibility

⚠️ PREDICTION MARKET RISKS:
Unlike token swaps, prediction markets carry unique risks that cannot be fully detected:
- Insider trading: Market creators may have privileged information
- Resolution risk: Markets may resolve unfairly or not at all
- Liquidity traps: You may not be able to exit your position

This tool applies safety filters and provides warnings, but cannot guarantee market legitimacy.
Only bet what you can afford to lose. Prefer established series (major elections, sports).
"""

import base64
import logging
from typing import Any, Dict, List, Optional

from solana_agent import AutoTool, ToolRegistry
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.message import to_bytes_versioned

from solana_agent.tools.utils.dflow import DFlowPredictionClient
from solana_agent.tools.utils.trigger import (
    replace_blockhash_in_transaction,
    get_fresh_blockhash,
)
from solana_agent.tools.utils.wallet import send_raw_transaction_with_priority

logger = logging.getLogger(__name__)

# Common USDC mint on Solana
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


class DFlowPredictionTool(AutoTool):
    """
    Prediction market tool with safety-first approach.

    Provides discovery and trading of prediction markets with:
    - Quality filters (min volume, liquidity, age)
    - Safety scoring (HIGH/MEDIUM/LOW)
    - Risk warnings on all queries
    - Blocking async execution
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="dflow_prediction",
            description=(
                "Trade prediction markets on Solana. Search for markets, check safety scores, "
                "and buy/sell outcome tokens (YES/NO). "
                "⚠️ WARNING: Prediction markets carry risks including insider trading, "
                "resolution manipulation, and liquidity traps. Safety scores are provided "
                "but cannot guarantee market legitimacy."
            ),
            registry=registry,
        )
        self._private_key: Optional[str] = None
        self._rpc_url: Optional[str] = None
        self._platform_fee_bps: Optional[int] = None
        self._platform_fee_scale: Optional[int] = None
        self._fee_account: Optional[str] = None
        self._min_volume_usd: int = 1000
        self._min_liquidity_usd: int = 500
        self._include_risky: bool = False
        self._payer_private_key: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "search",
                        "list_events",
                        "get_event",
                        "list_markets",
                        "get_market",
                        "buy",
                        "sell",
                        "positions",
                    ],
                    "description": (
                        "Action to perform: "
                        "'search' - Search markets by text query. "
                        "'list_events' - List active prediction events. "
                        "'get_event' - Get specific event details. "
                        "'list_markets' - List active markets. "
                        "'get_market' - Get specific market details. "
                        "'buy' - Buy outcome tokens (YES/NO). "
                        "'sell' - Sell outcome tokens. "
                        "'positions' - Get user's prediction market positions."
                    ),
                },
                "query": {
                    "type": ["string", "null"],
                    "description": "Search query text (for 'search' action). Pass null if not needed.",
                },
                "event_id": {
                    "type": ["string", "null"],
                    "description": "Event ticker/ID (for 'get_event' action). Pass null if not needed.",
                },
                "market_id": {
                    "type": ["string", "null"],
                    "description": "Market ticker (for 'get_market', 'buy', 'sell' actions). Pass null if not needed.",
                },
                "mint_address": {
                    "type": ["string", "null"],
                    "description": "Outcome token mint address (alternative to market_id). Pass null if not needed.",
                },
                "side": {
                    "type": ["string", "null"],
                    "enum": ["YES", "NO", None],
                    "description": "Which side to buy/sell (for 'buy', 'sell' actions). Pass null if not needed.",
                },
                "amount": {
                    "type": ["number", "null"],
                    "description": "Amount in USDC to spend (for 'buy') or tokens to sell (for 'sell'). Pass null if not needed.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 20).",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "closed", "determined"],
                    "description": "Filter by market status (default 'active').",
                },
                "sort": {
                    "type": "string",
                    "enum": ["volume", "volume24h", "liquidity", "openInterest"],
                    "description": "Sort field (default 'volume').",
                },
                "include_risky": {
                    "type": ["boolean", "null"],
                    "description": "Include low-quality markets (with warnings). Default false. Pass null to use default.",
                },
            },
            "required": [
                "action",
                "query",
                "event_id",
                "market_id",
                "mint_address",
                "side",
                "amount",
                "limit",
                "status",
                "sort",
                "include_risky",
            ],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("dflow_prediction", {})
        self._private_key = tool_cfg.get("private_key")
        self._rpc_url = tool_cfg.get("rpc_url")
        self._platform_fee_bps = tool_cfg.get("platform_fee_bps")
        self._platform_fee_scale = tool_cfg.get("platform_fee_scale")
        self._fee_account = tool_cfg.get("fee_account")
        self._min_volume_usd = tool_cfg.get("min_volume_usd", 1000)
        self._min_liquidity_usd = tool_cfg.get("min_liquidity_usd", 500)
        self._include_risky = tool_cfg.get("include_risky", False)
        self._payer_private_key = tool_cfg.get("payer_private_key")

    def _get_client(
        self, include_risky: Optional[bool] = None
    ) -> DFlowPredictionClient:
        """Get a configured DFlow prediction client."""
        return DFlowPredictionClient(
            min_volume_usd=self._min_volume_usd,
            min_liquidity_usd=self._min_liquidity_usd,
            include_risky=include_risky
            if include_risky is not None
            else self._include_risky,
        )

    def _get_keypair(self) -> Keypair:
        """Get the keypair from private key config."""
        if not self._private_key:
            raise ValueError("private_key not configured")
        return Keypair.from_base58_string(self._private_key)

    async def _sign_and_send(
        self,
        keypair: Keypair,
        transaction_base64: str,
    ) -> str:
        """
        Sign and send a transaction via RPC.

        Returns the transaction signature.
        Raises Exception on failure.
        """
        if not self._rpc_url:
            raise Exception("rpc_url must be configured for prediction market trades")

        # Get fresh blockhash
        blockhash_result = await get_fresh_blockhash(self._rpc_url)
        if "error" in blockhash_result:
            raise Exception(f"Failed to get blockhash: {blockhash_result['error']}")

        fresh_blockhash = blockhash_result["blockhash"]

        # Replace blockhash in transaction
        tx_with_new_blockhash = replace_blockhash_in_transaction(
            transaction_base64, fresh_blockhash
        )

        # Decode and sign
        tx_bytes = base64.b64decode(tx_with_new_blockhash)
        transaction = VersionedTransaction.from_bytes(tx_bytes)
        message_bytes = to_bytes_versioned(transaction.message)

        # Get signing info
        num_signers = transaction.message.header.num_required_signatures
        account_keys = transaction.message.account_keys
        new_signatures = list(transaction.signatures)

        # Sign with payer if configured
        if self._payer_private_key:
            payer_keypair = Keypair.from_base58_string(self._payer_private_key)
            payer_pubkey = payer_keypair.pubkey()
            payer_signature = payer_keypair.sign_message(message_bytes)

            for i in range(num_signers):
                if account_keys[i] == payer_pubkey:
                    new_signatures[i] = payer_signature
                    break

        # Sign with main keypair
        taker_pubkey = keypair.pubkey()
        taker_signature = keypair.sign_message(message_bytes)

        taker_index = None
        for i in range(num_signers):
            if account_keys[i] == taker_pubkey:
                taker_index = i
                break

        if taker_index is None:
            raise Exception(
                f"Taker pubkey {taker_pubkey} not found in transaction signers"
            )

        new_signatures[taker_index] = taker_signature

        # Create signed transaction
        signed_transaction = VersionedTransaction.populate(
            transaction.message,
            new_signatures,
        )
        signed_tx_bytes = bytes(signed_transaction)

        # Send via RPC
        send_result = await send_raw_transaction_with_priority(
            rpc_url=self._rpc_url,
            tx_bytes=signed_tx_bytes,
            skip_preflight=True,
            skip_confirmation=False,
            confirm_timeout=30.0,
        )

        if not send_result.get("success"):
            raise Exception(send_result.get("error", "Failed to send transaction"))

        return send_result.get("signature")

    async def _get_market_with_mints(
        self,
        client: DFlowPredictionClient,
        market_id: Optional[str],
        mint_address: Optional[str],
    ) -> Dict[str, Any]:
        """Get market data including YES/NO mint addresses."""
        market = await client.get_market(market_id=market_id, mint_address=mint_address)

        # Extract YES/NO mints from accounts
        accounts = market.get("accounts", {})
        for settlement_mint, account_data in accounts.items():
            market["yes_mint"] = account_data.get("yesMint")
            market["no_mint"] = account_data.get("noMint")
            market["settlement_mint"] = settlement_mint
            break

        return market

    async def execute(
        self,
        action: str,
        query: Optional[str] = None,
        event_id: Optional[str] = None,
        market_id: Optional[str] = None,
        mint_address: Optional[str] = None,
        side: Optional[str] = None,
        amount: Optional[float] = None,
        limit: int = 20,
        status: str = "active",
        sort: str = "volume",
        include_risky: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Execute a prediction market action."""

        client = self._get_client(include_risky)

        try:
            # =====================================================================
            # DISCOVERY ACTIONS
            # =====================================================================

            if action == "search":
                if not query:
                    return {
                        "status": "error",
                        "message": "query is required for search action",
                    }
                result = await client.search(query=query, limit=limit)
                return {"status": "success", **result}

            elif action == "list_events":
                result = await client.list_events(
                    limit=limit,
                    status=status,
                    sort=sort,
                    with_nested_markets=True,
                )
                return {"status": "success", **result}

            elif action == "get_event":
                if not event_id:
                    return {
                        "status": "error",
                        "message": "event_id is required for get_event action",
                    }
                event = await client.get_event(event_id)
                return {"status": "success", "event": event}

            elif action == "list_markets":
                result = await client.list_markets(
                    limit=limit,
                    status=status,
                    sort=sort,
                )
                return {"status": "success", **result}

            elif action == "get_market":
                if not market_id and not mint_address:
                    return {
                        "status": "error",
                        "message": "market_id or mint_address is required for get_market action",
                    }
                market = await self._get_market_with_mints(
                    client, market_id, mint_address
                )
                return {"status": "success", "market": market}

            # =====================================================================
            # TRADING ACTIONS
            # =====================================================================

            elif action == "buy":
                if not self._private_key:
                    return {"status": "error", "message": "private_key not configured"}
                if not self._rpc_url:
                    return {"status": "error", "message": "rpc_url not configured"}
                if not market_id and not mint_address:
                    return {
                        "status": "error",
                        "message": "market_id or mint_address required for buy",
                    }
                if not side:
                    return {
                        "status": "error",
                        "message": "side (YES/NO) required for buy",
                    }
                if not amount:
                    return {"status": "error", "message": "amount required for buy"}

                keypair = Keypair.from_base58_string(self._private_key)
                user_pubkey = str(keypair.pubkey())

                # Get market to find the outcome mint
                market = await self._get_market_with_mints(
                    client, market_id, mint_address
                )

                # Check safety
                safety = market.get("safety", {})
                if safety.get("recommendation") == "AVOID" and not include_risky:
                    return {
                        "status": "error",
                        "message": "Market safety score is LOW. Use include_risky=true to proceed.",
                        "safety": safety,
                    }

                # Get the appropriate outcome mint
                output_mint = (
                    market.get("yes_mint")
                    if side.upper() == "YES"
                    else market.get("no_mint")
                )
                if not output_mint:
                    return {
                        "status": "error",
                        "message": f"Could not find {side} outcome mint for market",
                    }

                # Convert amount to scaled integer (USDC has 6 decimals)
                amount_scaled = int(amount * 1_000_000)

                # Get order
                order = await client.get_prediction_order(
                    input_mint=USDC_MINT,
                    output_mint=output_mint,
                    amount=amount_scaled,
                    user_public_key=user_pubkey,
                    platform_fee_bps=self._platform_fee_bps,
                    platform_fee_scale=self._platform_fee_scale,
                    fee_account=self._fee_account,
                )

                # Execute with blocking wait
                async def sign_and_send(tx_b64: str) -> str:
                    return await self._sign_and_send(keypair, tx_b64)

                result = await client.execute_prediction_order_blocking(
                    order_response=order,
                    sign_and_send_func=sign_and_send,
                )

                if result.success:
                    return {
                        "status": "success",
                        "action": "buy",
                        "market": market.get("ticker"),
                        "side": side.upper(),
                        "amount_in": f"{amount} USDC",
                        "tokens_received": result.out_amount,
                        "signature": result.signature,
                        "tx_signature": result.signature,
                        "execution_mode": result.execution_mode,
                        "safety": safety,
                    }
                else:
                    return {
                        "status": "error",
                        "message": result.error,
                        "signature": result.signature,
                    }

            elif action == "sell":
                if not self._private_key:
                    return {"status": "error", "message": "private_key not configured"}
                if not self._rpc_url:
                    return {"status": "error", "message": "rpc_url not configured"}
                if not market_id and not mint_address:
                    return {
                        "status": "error",
                        "message": "market_id or mint_address required for sell",
                    }
                if not side:
                    return {
                        "status": "error",
                        "message": "side (YES/NO) required for sell",
                    }
                if not amount:
                    return {"status": "error", "message": "amount required for sell"}

                keypair = Keypair.from_base58_string(self._private_key)
                user_pubkey = str(keypair.pubkey())

                # Get market to find the outcome mint
                market = await self._get_market_with_mints(
                    client, market_id, mint_address
                )

                # Get the appropriate outcome mint (this is our input for selling)
                input_mint = (
                    market.get("yes_mint")
                    if side.upper() == "YES"
                    else market.get("no_mint")
                )
                if not input_mint:
                    return {
                        "status": "error",
                        "message": f"Could not find {side} outcome mint for market",
                    }

                # Outcome tokens typically have 6 decimals like USDC
                amount_scaled = int(amount * 1_000_000)

                # Get order - selling outcome tokens for USDC
                order = await client.get_prediction_order(
                    input_mint=input_mint,
                    output_mint=USDC_MINT,
                    amount=amount_scaled,
                    user_public_key=user_pubkey,
                    platform_fee_bps=self._platform_fee_bps,
                    platform_fee_scale=self._platform_fee_scale,
                    fee_account=self._fee_account,
                )

                # Execute with blocking wait
                async def sign_and_send(tx_b64: str) -> str:
                    return await self._sign_and_send(keypair, tx_b64)

                result = await client.execute_prediction_order_blocking(
                    order_response=order,
                    sign_and_send_func=sign_and_send,
                )

                if result.success:
                    return {
                        "status": "success",
                        "action": "sell",
                        "market": market.get("ticker"),
                        "side": side.upper(),
                        "tokens_sold": f"{amount} {side.upper()}",
                        "usdc_received": result.out_amount,
                        "signature": result.signature,
                        "tx_signature": result.signature,
                        "execution_mode": result.execution_mode,
                    }
                else:
                    return {
                        "status": "error",
                        "message": result.error,
                        "signature": result.signature,
                    }

            elif action == "positions":
                # For positions, we need to check the user's token balances
                if not self._private_key:
                    return {"status": "error", "message": "private_key not configured"}

                if not self._rpc_url:
                    return {
                        "status": "error",
                        "message": "rpc_url must be configured to query positions",
                    }

                keypair = Keypair.from_base58_string(self._private_key)
                user_pubkey = str(keypair.pubkey())

                # Query positions via RPC + DFlow outcome mints
                positions_result = await client.get_positions(
                    wallet_address=user_pubkey,
                    rpc_url=self._rpc_url,
                )

                return positions_result

            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception(f"DFlow prediction action failed: {e}")
            return {"status": "error", "message": str(e)}


class DFlowPredictionPlugin:
    """Plugin for DFlow prediction market trading."""

    def __init__(self):
        self.name = "dflow_prediction"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for prediction market trading with safety scoring."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = DFlowPredictionTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return DFlowPredictionPlugin()

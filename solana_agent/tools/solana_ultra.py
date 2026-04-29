"""
Jupiter Ultra swap tool for Solana wallets.

Enables swapping tokens using Jupiter Ultra API with a Solana keypair.

NOTE: We bypass Jupiter's /execute endpoint and send transactions directly via
our RPC (Helius). This is because Jupiter's execute can have reliability issues.
This matches the pattern used in privy_ultra.py and privy_trigger.py.
"""

import base64
import logging
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.message import to_bytes_versioned

from solana_agent.tools.utils.ultra import JupiterUltra
from solana_agent.tools.utils.trigger import (
    replace_blockhash_in_transaction,
    get_fresh_blockhash,
)
from solana_agent.tools.utils.wallet import send_raw_transaction_with_priority

logger = logging.getLogger(__name__)


class SolanaUltraTool(AutoTool):
    """Swap tokens using Jupiter Ultra with a Solana keypair."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="solana_ultra",
            description="Swap tokens using Jupiter Ultra API. Provides the best trading experience with MEV protection, optimal slippage, and fast execution.",
            registry=registry,
        )
        self._private_key: Optional[str] = None
        self._jupiter_api_key: Optional[str] = None
        self._referral_account: Optional[str] = None
        self._referral_fee: Optional[int] = None
        self._payer_private_key: Optional[str] = None
        self._rpc_url: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_mint": {
                    "type": "string",
                    "description": "The mint address of the token to swap from.",
                },
                "output_mint": {
                    "type": "string",
                    "description": "The mint address of the token to receive.",
                },
                "amount": {
                    "type": "integer",
                    "description": "The amount of input token to swap (in smallest units, e.g., lamports for SOL).",
                },
            },
            "required": ["input_mint", "output_mint", "amount"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("solana_ultra", {})
        self._private_key = tool_cfg.get("private_key")
        self._jupiter_api_key = tool_cfg.get("jupiter_api_key")
        self._referral_account = tool_cfg.get("referral_account")
        self._referral_fee = tool_cfg.get("referral_fee")
        self._payer_private_key = tool_cfg.get("payer_private_key")
        self._rpc_url = tool_cfg.get("rpc_url")

    async def _sign_and_execute(  # pragma: no cover
        self,
        keypair: Keypair,
        transaction_base64: str,
    ) -> Dict[str, Any]:
        """
        Replace blockhash, sign with keypair (and payer if configured), then send.

        Jupiter's /execute endpoint can have reliability issues, so we:
        1. Get a fresh blockhash from our RPC
        2. Replace the blockhash in Jupiter's transaction
        3. Sign with our keys
        4. Send directly via our RPC

        This matches the pattern used in privy_ultra.py and privy_trigger.py.
        """
        try:
            # RPC URL is required - Jupiter's execute has reliability issues
            if not self._rpc_url:
                return {
                    "status": "error",
                    "message": "rpc_url must be configured for Ultra swaps. Jupiter's execute endpoint has reliability issues.",
                }

            # Step 1: Get fresh blockhash from our RPC
            blockhash_result = await get_fresh_blockhash(self._rpc_url)
            if "error" in blockhash_result:
                return {
                    "status": "error",
                    "message": f"Failed to get blockhash: {blockhash_result['error']}",
                }

            fresh_blockhash = blockhash_result["blockhash"]
            logger.info(f"Got fresh blockhash: {fresh_blockhash}")

            # Step 2: Replace blockhash in the transaction
            tx_with_new_blockhash = replace_blockhash_in_transaction(
                transaction_base64, fresh_blockhash
            )

            # Step 3: Sign with keypair (and payer if configured)
            tx_bytes = base64.b64decode(tx_with_new_blockhash)
            transaction = VersionedTransaction.from_bytes(tx_bytes)
            message_bytes = to_bytes_versioned(transaction.message)

            # Get the number of required signatures
            num_signers = transaction.message.header.num_required_signatures
            account_keys = transaction.message.account_keys
            new_signatures = list(transaction.signatures)

            # If payer is configured, sign with payer first
            if self._payer_private_key:
                payer_keypair = Keypair.from_base58_string(self._payer_private_key)
                payer_pubkey = payer_keypair.pubkey()
                payer_signature = payer_keypair.sign_message(message_bytes)

                # Find payer's position in signers
                payer_index = None
                for i in range(num_signers):
                    if account_keys[i] == payer_pubkey:
                        payer_index = i
                        break

                if payer_index is not None:
                    new_signatures[payer_index] = payer_signature
                    logger.info(f"Payer signed at index {payer_index}")

            # Sign with main keypair (taker)
            taker_pubkey = keypair.pubkey()
            taker_signature = keypair.sign_message(message_bytes)

            # Find taker's position in signers
            taker_index = None
            for i in range(num_signers):
                if account_keys[i] == taker_pubkey:
                    taker_index = i
                    break

            if taker_index is None:
                return {
                    "status": "error",
                    "message": f"Taker pubkey {taker_pubkey} not found in transaction signers.",
                }

            new_signatures[taker_index] = taker_signature
            logger.info(f"Taker signed at index {taker_index}")

            # Create signed transaction
            signed_transaction = VersionedTransaction.populate(
                transaction.message,
                new_signatures,
            )
            signed_tx_bytes = bytes(signed_transaction)

            # Step 4: Send via our RPC
            send_result = await send_raw_transaction_with_priority(
                rpc_url=self._rpc_url,
                tx_bytes=signed_tx_bytes,
                skip_preflight=True,  # Some RPCs don't support preflight simulation
                skip_confirmation=False,  # Wait for confirmation - blockhash is from our RPC
                confirm_timeout=30.0,
            )

            if not send_result.get("success"):
                return {
                    "status": "error",
                    "message": send_result.get("error", "Failed to send transaction"),
                }

            return {
                "status": "success",
                "signature": send_result.get("signature"),
            }

        except Exception as e:
            logger.exception(f"Failed to sign and execute: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def execute(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
    ) -> Dict[str, Any]:
        if not self._private_key:
            return {"status": "error", "message": "Private key not configured."}

        try:
            keypair = Keypair.from_base58_string(self._private_key)
            taker = str(keypair.pubkey())

            # Check if integrator payer is configured for gasless transactions
            payer_pubkey = None
            if self._payer_private_key:  # pragma: no cover
                payer_keypair = Keypair.from_base58_string(self._payer_private_key)
                payer_pubkey = str(payer_keypair.pubkey())

            # Initialize Jupiter Ultra client
            ultra = JupiterUltra(api_key=self._jupiter_api_key)

            # Get swap order
            order = await ultra.get_order(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount,
                taker=taker,
                referral_account=self._referral_account,
                referral_fee=self._referral_fee,
                payer=payer_pubkey,
                close_authority=taker if payer_pubkey else None,
            )

            if not order.transaction:
                return {
                    "status": "error",
                    "message": "No transaction returned from Jupiter Ultra.",
                }

            # Sign and execute via RPC (bypasses Jupiter's /execute endpoint)
            exec_result = await self._sign_and_execute(
                keypair=keypair,
                transaction_base64=order.transaction,
            )

            if exec_result.get("status") == "success":
                # Format price impact as percentage with sign
                price_impact_str = ""
                if order.price_impact is not None:
                    price_impact_str = f"{order.price_impact:.2f}%"

                return {
                    "status": "success",
                    "signature": exec_result.get("signature"),
                    "swap_type": order.swap_type,
                    "gasless": order.gasless,
                    "in_amount": order.in_amount,
                    "out_amount": order.out_amount,
                    "in_usd_value": order.in_usd_value,
                    "out_usd_value": order.out_usd_value,
                    "slippage_bps": order.slippage_bps,
                    "price_impact_pct": price_impact_str,
                }
            else:
                return exec_result

        except Exception as e:
            logger.exception(f"Solana Ultra swap failed: {str(e)}")
            return {"status": "error", "message": str(e)}


class SolanaUltraPlugin:
    """Plugin for swapping tokens using Jupiter Ultra."""

    def __init__(self):
        self.name = "solana_ultra"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for swapping tokens using Jupiter Ultra API."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = SolanaUltraTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return SolanaUltraPlugin()

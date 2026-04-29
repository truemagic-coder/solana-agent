"""DFlow Swap tool for Solana wallets.

Enables fast token swaps using DFlow's Swap API with a Solana keypair.
DFlow offers faster swaps compared to Jupiter Ultra with similar liquidity.

Note: Platform fees are not supported. DFlow requires a specific token account
(ATA) for each output token to collect fees, which is impractical for a
general-purpose swap tool. Use Jupiter Ultra (solana_ultra) if you need fees.
"""

import base64
import logging
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry
from solders.keypair import Keypair  # type: ignore
from solders.transaction import VersionedTransaction  # type: ignore
from solders.message import to_bytes_versioned  # type: ignore
from solana.rpc.async_api import AsyncClient  # type: ignore
from solana.rpc.commitment import Confirmed  # type: ignore

from solana_agent.tools.utils.dflow import DFlowSwap

logger = logging.getLogger(__name__)

# Default Solana RPC endpoint
DEFAULT_RPC_URL = "https://api.mainnet-beta.solana.com"


def _sign_dflow_transaction(
    transaction_base64: str,
    sign_message_func,
    payer_sign_func=None,
) -> str:
    """Sign a DFlow transaction with the user's keypair (and optionally a payer).

    Args:
        transaction_base64: Base64-encoded transaction from DFlow
        sign_message_func: Function to sign message bytes (user's keypair.sign_message)
        payer_sign_func: Optional function to sign message bytes (payer's keypair.sign_message)

    Returns:
        Base64-encoded signed transaction
    """
    tx_bytes = base64.b64decode(transaction_base64)
    transaction = VersionedTransaction.from_bytes(tx_bytes)
    message_bytes = to_bytes_versioned(transaction.message)

    if payer_sign_func:
        # Payer signs first, then user
        payer_signature = payer_sign_func(message_bytes)
        user_signature = sign_message_func(message_bytes)
        signed_tx = VersionedTransaction.populate(
            transaction.message,
            [payer_signature, user_signature],
        )
    else:
        # Just user signs
        user_signature = sign_message_func(message_bytes)
        signed_tx = VersionedTransaction.populate(
            transaction.message,
            [user_signature],
        )

    return base64.b64encode(bytes(signed_tx)).decode("utf-8")


class SolanaDFlowSwapTool(AutoTool):
    """Fast token swaps using DFlow API with a Solana keypair."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="solana_dflow_swap",
            description=(
                "Fast token swap on Solana using DFlow API. "
                "Swaps tokens instantly with competitive rates and low slippage. "
                "Supports SOL and all SPL tokens. Faster alternative to Jupiter Ultra."
            ),
            registry=registry,
        )
        self._private_key: Optional[str] = None
        self._payer_private_key: Optional[str] = None
        self._rpc_url: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_mint": {
                    "type": "string",
                    "description": "Token mint address to sell/swap from. Use 'So11111111111111111111111111111111111111112' for native SOL.",
                },
                "output_mint": {
                    "type": "string",
                    "description": "Token mint address to buy/swap to. Use 'So11111111111111111111111111111111111111112' for native SOL.",
                },
                "amount": {
                    "type": "integer",
                    "description": "Amount to swap in the smallest unit (lamports for SOL, base units for tokens). Example: 1000000000 for 1 SOL.",
                },
                "slippage_bps": {
                    "type": "integer",
                    "description": "Maximum slippage tolerance in basis points (100 = 1%). Default is auto.",
                    "default": 0,
                },
            },
            "required": ["input_mint", "output_mint", "amount"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("solana_dflow_swap", {})
        self._private_key = tool_cfg.get("private_key")
        self._payer_private_key = tool_cfg.get("payer_private_key")
        self._rpc_url = tool_cfg.get("rpc_url") or DEFAULT_RPC_URL

    async def execute(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 0,
    ) -> Dict[str, Any]:
        if not self._private_key:
            return {"status": "error", "message": "Private key not configured."}

        try:
            keypair = Keypair.from_base58_string(self._private_key)
            user_pubkey = str(keypair.pubkey())

            # Check if payer is configured for gasless transactions
            payer_keypair = None
            sponsor = None
            if self._payer_private_key:
                payer_keypair = Keypair.from_base58_string(self._payer_private_key)
                sponsor = str(payer_keypair.pubkey())

            # Initialize DFlow client
            dflow = DFlowSwap()

            # Get order from DFlow
            order_result = await dflow.get_order(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount,
                user_public_key=user_pubkey,
                slippage_bps=slippage_bps if slippage_bps > 0 else None,
                sponsor=sponsor,
            )

            if not order_result.success:
                return {"status": "error", "message": order_result.error}

            if not order_result.transaction:
                return {
                    "status": "error",
                    "message": "No transaction returned from DFlow.",
                }

            # Sign the transaction
            signed_tx = _sign_dflow_transaction(
                transaction_base64=order_result.transaction,
                sign_message_func=keypair.sign_message,
                payer_sign_func=payer_keypair.sign_message if payer_keypair else None,
            )

            # Send the transaction to Solana
            signature = await self._send_transaction(signed_tx)

            if not signature:
                return {
                    "status": "error",
                    "message": "Failed to send transaction to Solana.",
                }

            return {
                "status": "success",
                "signature": signature,
                "input_amount": order_result.in_amount,
                "output_amount": order_result.out_amount,
                "min_output_amount": order_result.min_out_amount,
                "input_mint": order_result.input_mint,
                "output_mint": order_result.output_mint,
                "price_impact": order_result.price_impact_pct,
                "execution_mode": order_result.execution_mode,
                "message": f"Swap successful! Signature: {signature}",
            }

        except Exception as e:
            logger.exception(f"DFlow swap failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _send_transaction(self, signed_tx_base64: str) -> Optional[str]:
        """Send a signed transaction to Solana."""
        try:
            tx_bytes = base64.b64decode(signed_tx_base64)

            async with AsyncClient(self._rpc_url) as client:
                result = await client.send_raw_transaction(
                    tx_bytes,
                    opts={
                        "skip_preflight": True,
                        "preflight_commitment": Confirmed,
                        "max_retries": 3,
                    },
                )

                if result.value:
                    return str(result.value)

                return None
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
            return None


class SolanaDFlowSwapPlugin:
    """Plugin for fast token swaps using DFlow API."""

    def __init__(self):
        self.name = "solana_dflow_swap"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for fast token swaps using DFlow API with a Solana keypair."

    def initialize(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._tool = SolanaDFlowSwapTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:
        return [self._tool] if self._tool else []


def get_plugin():
    return SolanaDFlowSwapPlugin()

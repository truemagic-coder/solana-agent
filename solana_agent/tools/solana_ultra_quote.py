"""
Jupiter Ultra swap quote tool for Solana wallets.

Provides a preview of swap details (slippage, price impact, amounts) without executing.
"""

import logging
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry
from solders.keypair import Keypair

from solana_agent.tools.utils.ultra import JupiterUltra

logger = logging.getLogger(__name__)


class SolanaUltraQuoteTool(AutoTool):
    """Get a swap quote from Jupiter Ultra with a Solana keypair."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="solana_ultra_quote",
            description="Get a quote for swapping tokens using Jupiter Ultra API. Shows slippage, price impact, and amounts before executing the swap.",
            registry=registry,
        )
        self._private_key: Optional[str] = None
        self._jupiter_api_key: Optional[str] = None
        self._referral_account: Optional[str] = None
        self._referral_fee: Optional[int] = None
        self._payer_private_key: Optional[str] = None

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
        tool_cfg = config.get("tools", {}).get("solana_ultra_quote", {})
        self._private_key = tool_cfg.get("private_key")
        self._jupiter_api_key = tool_cfg.get("jupiter_api_key")
        self._referral_account = tool_cfg.get("referral_account")
        self._referral_fee = tool_cfg.get("referral_fee")
        self._payer_private_key = tool_cfg.get("payer_private_key")

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

            # Get swap quote (doesn't execute, just shows details)
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

            # Format price impact as percentage with sign
            price_impact_str = ""
            if order.price_impact is not None:
                price_impact_str = f"{order.price_impact:.2f}%"

            # Format USD values
            in_usd_str = f"${order.in_usd_value:.2f}" if order.in_usd_value else None
            out_usd_str = f"${order.out_usd_value:.2f}" if order.out_usd_value else None

            # Return quote details without executing
            return {
                "status": "success",
                "input_mint": order.input_mint,
                "output_mint": order.output_mint,
                "in_amount": order.in_amount,
                "out_amount": order.out_amount,
                "in_usd_value": in_usd_str,
                "out_usd_value": out_usd_str,
                "slippage_bps": order.slippage_bps,
                "price_impact_pct": price_impact_str,
                "swap_type": order.swap_type,
                "gasless": order.gasless,
                "message": "Preview only - no transaction executed. Call solana_ultra to execute the swap.",
            }

        except Exception as e:
            logger.exception(f"Solana Ultra quote failed: {str(e)}")
            return {"status": "error", "message": str(e)}


class SolanaUltraQuotePlugin:
    """Plugin for getting swap quotes using Jupiter Ultra."""

    def __init__(self):
        self.name = "solana_ultra_quote"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for getting swap quotes using Jupiter Ultra API."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = SolanaUltraQuoteTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return SolanaUltraQuotePlugin()

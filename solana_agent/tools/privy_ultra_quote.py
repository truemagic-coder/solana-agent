"""
Jupiter Ultra swap quote tool for Privy embedded wallets.

Provides a preview of swap details (slippage, price impact, amounts) without executing.
"""

import logging
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry
from solders.keypair import Keypair

from solana_agent.tools.utils.ultra import JupiterUltra


logger = logging.getLogger(__name__)


class PrivyUltraQuoteTool(AutoTool):
    """Get a swap quote from Jupiter Ultra via Privy delegated wallet."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="privy_ultra_quote",
            description="Get a quote for swapping tokens using Jupiter Ultra API via Privy delegated wallet. Shows slippage, price impact, and amounts before executing the swap.",
            registry=registry,
        )
        self.jupiter_api_key = None
        self.referral_account = None
        self.referral_fee = None
        self.payer_private_key = None

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
                    "description": "Solana public key of the wallet. REQUIRED.",
                },
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
            "required": [
                "wallet_id",
                "wallet_public_key",
                "input_mint",
                "output_mint",
                "amount",
            ],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        tool_cfg = config.get("tools", {}).get("privy_ultra_quote", {})
        self.jupiter_api_key = tool_cfg.get("jupiter_api_key")
        self.referral_account = tool_cfg.get("referral_account")
        self.referral_fee = tool_cfg.get("referral_fee")
        self.payer_private_key = tool_cfg.get("payer_private_key")

    async def execute(  # pragma: no cover
        self,
        wallet_id: str,
        wallet_public_key: str,
        input_mint: str,
        output_mint: str,
        amount: int,
    ) -> Dict[str, Any]:
        if not wallet_id or not wallet_public_key:
            return {
                "status": "error",
                "message": "wallet_id and wallet_public_key are required.",
            }

        public_key = wallet_public_key

        # Check if integrator payer is configured for gasless transactions
        payer_pubkey = None
        if self.payer_private_key:
            payer_keypair = Keypair.from_base58_string(self.payer_private_key)
            payer_pubkey = str(payer_keypair.pubkey())

        # Initialize Jupiter Ultra client
        ultra = JupiterUltra(api_key=self.jupiter_api_key)

        try:
            # Get swap quote (doesn't execute, just shows details)
            order = await ultra.get_order(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount,
                taker=public_key,
                referral_account=self.referral_account,
                referral_fee=self.referral_fee,
                payer=payer_pubkey,
                close_authority=public_key if payer_pubkey else None,
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
                "message": "Preview only - no transaction executed. Call privy_ultra to execute the swap.",
            }

        except Exception as e:
            logger.exception(f"Privy Ultra quote failed: {str(e)}")
            return {"status": "error", "message": str(e)}


class PrivyUltraQuotePlugin:
    """Plugin for getting swap quotes via Privy using Jupiter Ultra."""

    def __init__(self):
        self.name = "privy_ultra_quote"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for getting swap quotes using Jupiter Ultra API via Privy."

    def initialize(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._tool = PrivyUltraQuoteTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:
        return [self._tool] if self._tool else []


def get_plugin():
    return PrivyUltraQuotePlugin()

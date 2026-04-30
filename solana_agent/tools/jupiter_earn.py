"""
Jupiter Earn tool for Solana Agent Kit.

Supports deposit/withdraw/mint/redeem instructions using Jupiter Earn API,
then sends transactions via our RPC (Helius recommended).
"""

import base64
import logging
from typing import Any, Dict, List, Optional

from solana_agent import AutoTool, ToolRegistry
from solders.instruction import Instruction, AccountMeta
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.transaction import Transaction

from solana_agent.tools.utils.earn import JupiterEarn
from solana_agent.tools.utils.trigger import get_fresh_blockhash
from solana_agent.tools.utils.wallet import send_raw_transaction_with_priority

logger = logging.getLogger(__name__)

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
ALLOWED_EARN_ASSETS = {SOL_MINT, USDC_MINT}


def _normalize_asset(asset: Optional[str]) -> Optional[str]:
    if not asset:
        return None
    asset = asset.strip()
    upper = asset.upper()
    if upper == "SOL":
        return SOL_MINT
    if upper == "USDC":
        return USDC_MINT
    return asset


def _build_instruction(instruction: Dict[str, Any]) -> Instruction:
    program_id = Pubkey.from_string(instruction.get("programId", ""))
    accounts = []
    for acct in instruction.get("accounts", []):
        accounts.append(
            AccountMeta(
                Pubkey.from_string(acct.get("pubkey", "")),
                acct.get("isSigner", False),
                acct.get("isWritable", False),
            )
        )

    data_str = instruction.get("data", "")
    try:
        data = base64.b64decode(data_str)
    except Exception as e:
        raise ValueError(f"Invalid instruction data: {e}")

    return Instruction(program_id=program_id, accounts=accounts, data=data)


class JupiterEarnTool(AutoTool):
    """Use Jupiter Earn API to deposit/withdraw/mint/redeem and query earn data."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="jupiter_earn",
            description=(
                "Use Jupiter Earn API to deposit/withdraw assets or mint/redeem shares, "
                "and to query tokens/positions/earnings. "
                "Actions: 'deposit', 'withdraw', 'mint', 'redeem', 'tokens', 'positions', 'earnings'. "
                "Only SOL and USDC are supported for earn transactions."
            ),
            registry=registry,
        )
        self._private_key: Optional[str] = None
        self._jupiter_api_key: Optional[str] = None
        self._rpc_url: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "deposit",
                        "withdraw",
                        "mint",
                        "redeem",
                        "tokens",
                        "positions",
                        "earnings",
                    ],
                    "description": (
                        "Action to perform: deposit/withdraw assets, mint/redeem shares, "
                        "or query tokens/positions/earnings."
                    ),
                },
                "asset": {
                    "type": "string",
                    "description": "Asset mint address (SOL or USDC only). Pass empty string if not needed.",
                    "default": "",
                },
                "amount": {
                    "type": "string",
                    "description": "Amount to deposit/withdraw (base units). Pass empty string if not needed.",
                    "default": "",
                },
                "shares": {
                    "type": "string",
                    "description": "Shares to mint/redeem (base units). Pass empty string if not needed.",
                    "default": "",
                },
                "users": {
                    "type": "string",
                    "description": "Comma-separated user wallet addresses for positions. Pass empty string if not needed.",
                    "default": "",
                },
                "user": {
                    "type": "string",
                    "description": "User wallet address for earnings. Pass empty string if not needed.",
                    "default": "",
                },
                "positions": {
                    "type": "string",
                    "description": "Comma-separated position token addresses for earnings. Pass empty string if not needed.",
                    "default": "",
                },
            },
            "required": [
                "action",
                "asset",
                "amount",
                "shares",
                "users",
                "user",
                "positions",
            ],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("jupiter_earn", {})
        self._private_key = tool_cfg.get("private_key")
        self._jupiter_api_key = tool_cfg.get("jupiter_api_key")
        self._rpc_url = tool_cfg.get("rpc_url")

    async def execute(
        self,
        action: str,
        asset: Optional[str] = None,
        amount: Optional[str] = None,
        shares: Optional[str] = None,
        users: Optional[str] = None,
        user: Optional[str] = None,
        positions: Optional[str] = None,
    ) -> Dict[str, Any]:
        action = action.lower().strip()

        if not self._jupiter_api_key:
            return {"status": "error", "message": "Jupiter API key not configured."}

        earn = JupiterEarn(api_key=self._jupiter_api_key)

        if action in {"deposit", "withdraw", "mint", "redeem"}:
            if not self._private_key:
                return {"status": "error", "message": "Private key not configured."}
            if not self._rpc_url:
                return {"status": "error", "message": "RPC URL not configured."}

            normalized_asset = _normalize_asset(asset)
            if not normalized_asset or normalized_asset not in ALLOWED_EARN_ASSETS:
                return {
                    "status": "error",
                    "message": "Only SOL and USDC are supported for earn transactions.",
                }

            keypair = Keypair.from_base58_string(self._private_key)
            signer = str(keypair.pubkey())

            if action in {"deposit", "withdraw"}:
                if not amount:
                    return {
                        "status": "error",
                        "message": "Missing required parameter: amount.",
                    }
                instruction_result = (
                    await earn.get_deposit_instructions(
                        normalized_asset, signer, str(amount)
                    )
                    if action == "deposit"
                    else await earn.get_withdraw_instructions(
                        normalized_asset, signer, str(amount)
                    )
                )
            else:
                if not shares:
                    return {
                        "status": "error",
                        "message": "Missing required parameter: shares.",
                    }
                instruction_result = (
                    await earn.get_mint_instructions(
                        normalized_asset, signer, str(shares)
                    )
                    if action == "mint"
                    else await earn.get_redeem_instructions(
                        normalized_asset, signer, str(shares)
                    )
                )

            if not instruction_result.success or not instruction_result.instruction:
                return {
                    "status": "error",
                    "message": instruction_result.error
                    or "Failed to fetch earn instructions.",
                }

            try:
                instruction = _build_instruction(instruction_result.instruction)
            except Exception as e:
                return {"status": "error", "message": str(e)}

            blockhash_result = await get_fresh_blockhash(self._rpc_url)
            if "error" in blockhash_result:
                return {
                    "status": "error",
                    "message": f"Failed to get blockhash: {blockhash_result['error']}",
                }

            msg = Message(
                instructions=[instruction],
                payer=keypair.pubkey(),
            )
            tx = Transaction(
                from_keypairs=[keypair],
                message=msg,
                recent_blockhash=blockhash_result.get("blockhash"),
            )

            send_result = await send_raw_transaction_with_priority(
                rpc_url=self._rpc_url,
                tx_bytes=bytes(tx),
                skip_preflight=True,
                skip_confirmation=False,
                confirm_timeout=30.0,
            )

            if not send_result.get("success"):
                return {
                    "status": "error",
                    "message": send_result.get("error", "Failed to send transaction"),
                }

            return {
                "status": "success",
                "action": action,
                "signature": send_result.get("signature"),
                "asset": normalized_asset,
                "amount": str(amount) if amount else None,
                "shares": str(shares) if shares else None,
            }

        if action == "tokens":
            tokens_result = await earn.get_tokens()
            if not tokens_result.get("success"):
                return {"status": "error", "message": tokens_result.get("error")}

            tokens = tokens_result.get("tokens", [])
            filtered = [
                token
                for token in tokens
                if token.get("asset", {}).get("address") in ALLOWED_EARN_ASSETS
                or token.get("assetAddress") in ALLOWED_EARN_ASSETS
            ]
            return {"status": "success", "tokens": filtered}

        if action == "positions":
            users_list: List[str] = []
            if users:
                users_list = [u.strip() for u in users.split(",") if u.strip()]
            if not users_list:
                if not self._private_key:
                    return {
                        "status": "error",
                        "message": "Private key not configured for default positions lookup.",
                    }
                keypair = Keypair.from_base58_string(self._private_key)
                users_list = [str(keypair.pubkey())]

            positions_result = await earn.get_positions(users_list)
            if not positions_result.get("success"):
                return {"status": "error", "message": positions_result.get("error")}

            positions_data = positions_result.get("positions", [])
            filtered = [
                position
                for position in positions_data
                if position.get("token", {}).get("assetAddress") in ALLOWED_EARN_ASSETS
            ]
            return {"status": "success", "positions": filtered}

        if action == "earnings":
            if not user:
                if not self._private_key:
                    return {
                        "status": "error",
                        "message": "user is required for earnings when private key is not configured.",
                    }
                keypair = Keypair.from_base58_string(self._private_key)
                user = str(keypair.pubkey())

            if not positions:
                return {
                    "status": "error",
                    "message": "positions is required for earnings.",
                }

            positions_list = [p.strip() for p in positions.split(",") if p.strip()]
            if not positions_list:
                return {
                    "status": "error",
                    "message": "positions is required for earnings.",
                }

            earnings_result = await earn.get_earnings(user, positions_list)
            if not earnings_result.get("success"):
                return {"status": "error", "message": earnings_result.get("error")}

            earnings_data = earnings_result.get("earnings", [])
            filtered = [
                entry
                for entry in earnings_data
                if entry.get("address") in ALLOWED_EARN_ASSETS
            ]
            return {"status": "success", "earnings": filtered}

        return {
            "status": "error",
            "message": (
                "Unknown action. Valid actions: deposit, withdraw, mint, redeem, "
                "tokens, positions, earnings"
            ),
        }


class JupiterEarnPlugin:
    """Plugin for Jupiter Earn tool."""

    def __init__(self) -> None:
        self.name = "jupiter_earn"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self) -> str:
        return "Plugin for Jupiter Earn (lend) operations using Solana wallets."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = JupiterEarnTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return JupiterEarnPlugin()

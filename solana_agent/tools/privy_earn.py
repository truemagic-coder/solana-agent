"""
Jupiter Earn tool for Privy delegated wallets.

Uses Jupiter Earn API instructions and sends transactions via our RPC
(Helius recommended). Uses the official Privy Python SDK for signing.
"""

import base64
import logging
from typing import Any, Dict, List, Optional

from solana_agent import AutoTool, ToolRegistry
from privy import AsyncPrivyAPI
from privy.lib.authorization_signatures import get_authorization_signature
from cryptography.hazmat.primitives import serialization
from solders.instruction import Instruction, AccountMeta
from solders.message import Message, to_bytes_versioned
from solders.pubkey import Pubkey
from solders.null_signer import NullSigner
from solders.transaction import VersionedTransaction

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


def _convert_key_to_pkcs8_pem(key_string: str) -> str:  # pragma: no cover
    """Convert a private key to PKCS#8 PEM format for the Privy SDK."""
    private_key_string = key_string.replace("wallet-auth:", "")

    # Try loading as PKCS#8 PEM format first
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

    # Try as EC PRIVATE KEY (SEC1) format
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
        lines = pkcs8_pem.strip().split("\n")
        return "".join(lines[1:-1])
    except (ValueError, TypeError):
        pass

    # Try loading as raw DER bytes
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
        lines = pkcs8_pem.strip().split("\n")
        return "".join(lines[1:-1])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not load private key: {e}")


async def _privy_sign_transaction(  # pragma: no cover
    privy_client: AsyncPrivyAPI,
    wallet_id: str,
    encoded_tx: str,
    signing_key: str,
) -> Optional[str]:
    """Sign a Solana transaction via Privy using the official SDK."""
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


class PrivyEarnTool(AutoTool):
    """Use Jupiter Earn API with Privy delegated wallets."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="privy_earn",
            description=(
                "Use Jupiter Earn API with Privy delegated wallets. "
                "Actions: 'deposit', 'withdraw', 'mint', 'redeem', 'tokens', "
                "'positions', 'earnings'. Only SOL and USDC are supported for earn transactions."
            ),
            registry=registry,
        )
        self._app_id: Optional[str] = None
        self._app_secret: Optional[str] = None
        self._signing_key: Optional[str] = None
        self._jupiter_api_key: Optional[str] = None
        self._rpc_url: Optional[str] = None

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
                "wallet_id",
                "wallet_public_key",
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
        tool_cfg = config.get("tools", {}).get("privy_earn", {})
        self._app_id = tool_cfg.get("app_id")
        self._app_secret = tool_cfg.get("app_secret")
        self._signing_key = tool_cfg.get("signing_key")
        self._jupiter_api_key = tool_cfg.get("jupiter_api_key")
        self._rpc_url = tool_cfg.get("rpc_url")

    async def execute(
        self,
        wallet_id: str,
        wallet_public_key: str,
        action: str,
        asset: Optional[str] = None,
        amount: Optional[str] = None,
        shares: Optional[str] = None,
        users: Optional[str] = None,
        user: Optional[str] = None,
        positions: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not wallet_id or not wallet_public_key:
            return {
                "status": "error",
                "message": "wallet_id and wallet_public_key are required.",
            }

        if not all([self._app_id, self._app_secret, self._signing_key]):
            return {"status": "error", "message": "Privy config missing."}

        if not self._jupiter_api_key:
            return {"status": "error", "message": "Jupiter API key not configured."}

        action = action.lower().strip()
        earn = JupiterEarn(api_key=self._jupiter_api_key)

        privy_client = AsyncPrivyAPI(app_id=self._app_id, app_secret=self._app_secret)
        try:
            if action in {"deposit", "withdraw", "mint", "redeem"}:
                if not self._rpc_url:
                    return {"status": "error", "message": "RPC URL not configured."}

                normalized_asset = _normalize_asset(asset)
                if not normalized_asset or normalized_asset not in ALLOWED_EARN_ASSETS:
                    return {
                        "status": "error",
                        "message": "Only SOL and USDC are supported for earn transactions.",
                    }

                signer = wallet_public_key

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

                payer_pubkey = Pubkey.from_string(wallet_public_key)
                msg = Message.new_with_blockhash(
                    instructions=[instruction],
                    payer=payer_pubkey,
                    blockhash=blockhash_result.get("blockhash"),
                )

                placeholder_sig = NullSigner(payer_pubkey).sign_message(
                    to_bytes_versioned(msg)
                )
                unsigned_tx = VersionedTransaction.populate(
                    message=msg,
                    signatures=[placeholder_sig],
                )
                encoded_tx = base64.b64encode(bytes(unsigned_tx)).decode("utf-8")

                signed_tx = await _privy_sign_transaction(
                    privy_client,
                    wallet_id,
                    encoded_tx,
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
                        "message": send_result.get(
                            "error", "Failed to send transaction"
                        ),
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
                    users_list = [wallet_public_key]

                positions_result = await earn.get_positions(users_list)
                if not positions_result.get("success"):
                    return {"status": "error", "message": positions_result.get("error")}

                positions_data = positions_result.get("positions", [])
                filtered = [
                    position
                    for position in positions_data
                    if position.get("token", {}).get("assetAddress")
                    in ALLOWED_EARN_ASSETS
                ]
                return {"status": "success", "positions": filtered}

            if action == "earnings":
                if not user:
                    user = wallet_public_key

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

        finally:
            await privy_client.close()


class PrivyEarnPlugin:
    """Plugin for Jupiter Earn with Privy wallets."""

    def __init__(self) -> None:
        self.name = "privy_earn"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self) -> str:
        return "Plugin for Jupiter Earn (lend) operations using Privy wallets."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = PrivyEarnTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return PrivyEarnPlugin()

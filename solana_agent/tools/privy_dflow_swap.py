"""DFlow Swap tool for Privy embedded wallets.

Enables fast token swaps using DFlow's Swap API via Privy delegated wallets.
Uses the official Privy Python SDK for wallet operations.

DFlow offers faster swaps compared to Jupiter Ultra with similar liquidity.

Transactions are signed via Privy's signTransaction and sent via Helius RPC
for priority fees and reliable blockhash handling.

Note: Platform fees are not supported. DFlow requires a specific token account
(ATA) for each output token to collect fees, which is impractical for a
general-purpose swap tool. Use Jupiter Ultra (privy_ultra) if you need fees.
"""

import base64
import logging
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry
from privy import AsyncPrivyAPI
from privy.lib.authorization_signatures import get_authorization_signature
from cryptography.hazmat.primitives import serialization
from solders.keypair import Keypair  # type: ignore
from solders.transaction import VersionedTransaction  # type: ignore
from solders.message import to_bytes_versioned  # type: ignore

from solana_agent.tools.utils.dflow import DFlowSwap
from solana_agent.tools.utils.wallet import send_raw_transaction_with_priority

logger = logging.getLogger(__name__)


def _convert_key_to_pkcs8_pem(key_string: str) -> str:  # pragma: no cover
    """Convert a private key to PKCS#8 PEM format for the Privy SDK."""
    private_key_string = key_string.replace("wallet-auth:", "")

    # Try loading as PKCS#8 PEM format first
    try:
        private_key_pem = f"-----BEGIN PRIVATE KEY-----\n{private_key_string}\n-----END PRIVATE KEY-----"
        serialization.load_pem_private_key(
            private_key_pem.encode("utf-8"), password=None
        )
        return private_key_string
    except (ValueError, TypeError):
        pass

    # Try as EC PRIVATE KEY (SEC1) format
    try:
        ec_key_pem = f"-----BEGIN EC PRIVATE KEY-----\n{private_key_string}\n-----END EC PRIVATE KEY-----"
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
) -> Dict[str, Any]:
    """Sign a Solana transaction via Privy using the official SDK.

    Uses Privy's signTransaction RPC method to sign the transaction without
    broadcasting. The signed transaction should then be sent via Helius or
    another RPC provider for reliable blockhash handling.

    Returns:
        Dict with 'signed_transaction' (base64 encoded) on success, or error details.
    """
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

        # Extract the signed transaction from the result
        if result.data:
            signed_tx = getattr(result.data, "signed_transaction", None)
            if signed_tx:
                return {"success": True, "signed_transaction": signed_tx}
            # Some SDK versions return the transaction directly
            tx = getattr(result.data, "transaction", None)
            if tx:
                return {"success": True, "signed_transaction": tx}
        return {"success": False, "error": "No signed transaction returned from Privy"}
    except Exception as e:
        logger.error(f"Privy API error signing transaction: {e}")
        return {"success": False, "error": str(e)}


# Default RPC URL if none configured
DEFAULT_RPC_URL = "https://api.mainnet-beta.solana.com"


class PrivyDFlowSwapTool(AutoTool):
    """Fast token swaps using DFlow API with Privy embedded wallets."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="privy_dflow_swap",
            description=(
                "Fast token swap on Solana using DFlow API with Privy delegated wallets. "
                "Swaps tokens instantly with competitive rates and low slippage. "
                "Supports SOL and all SPL tokens. Faster alternative to Jupiter Ultra."
            ),
            registry=registry,
        )
        self._app_id: Optional[str] = None
        self._app_secret: Optional[str] = None
        self._signing_key: Optional[str] = None
        self._payer_private_key: Optional[str] = None
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
                "input_mint": {
                    "type": "string",
                    "description": "Token mint address to sell/swap from. Use 'So11111111111111111111111111111111111111112' for native SOL.",
                },
                "output_mint": {
                    "type": "string",
                    "description": "Token mint address to buy/swap to. Use 'So11111111111111111111111111111111111111112' for native SOL.",
                },
                "amount": {
                    "type": "string",
                    "description": "Amount to swap in the smallest unit (lamports for SOL, base units for tokens). Example: '1000000000' for 1 SOL.",
                },
                "slippage_bps": {
                    "type": "integer",
                    "description": "Maximum slippage tolerance in basis points (100 = 1%). Default is auto.",
                    "default": 0,
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
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("privy_dflow_swap", {})
        self._app_id = tool_cfg.get("app_id")
        self._app_secret = tool_cfg.get("app_secret")
        self._signing_key = tool_cfg.get("signing_key")
        self._payer_private_key = tool_cfg.get("payer_private_key")
        # RPC URL for sending transactions (Helius recommended for priority fees)
        self._rpc_url = tool_cfg.get("rpc_url")

    async def execute(  # pragma: no cover
        self,
        wallet_id: str,
        wallet_public_key: str,
        input_mint: str,
        output_mint: str,
        amount: str,
        slippage_bps: int = 0,
    ) -> Dict[str, Any]:
        if not wallet_id or not wallet_public_key:
            return {
                "status": "error",
                "message": "wallet_id and wallet_public_key are required.",
            }

        if not all([self._app_id, self._app_secret, self._signing_key]):
            return {"status": "error", "message": "Privy config missing."}

        # Create Privy client
        privy_client = AsyncPrivyAPI(
            app_id=self._app_id,
            app_secret=self._app_secret,
        )

        try:
            public_key = wallet_public_key

            # Initialize DFlow client
            dflow = DFlowSwap()

            # Get sponsor pubkey if gasless is configured
            sponsor = None
            if self._payer_private_key:
                payer_keypair = Keypair.from_base58_string(self._payer_private_key)
                sponsor = str(payer_keypair.pubkey())

            # Retry logic for blockhash expiration
            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                # Get fresh order from DFlow (includes fresh blockhash)
                order_result = await dflow.get_order(
                    input_mint=input_mint,
                    output_mint=output_mint,
                    amount=int(amount),
                    user_public_key=public_key,
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

                # Sign transaction
                tx_to_sign = order_result.transaction

                # If gasless, sign with payer first
                if self._payer_private_key:
                    payer_keypair = Keypair.from_base58_string(self._payer_private_key)
                    tx_bytes = base64.b64decode(order_result.transaction)
                    transaction = VersionedTransaction.from_bytes(tx_bytes)
                    message_bytes = to_bytes_versioned(transaction.message)
                    payer_signature = payer_keypair.sign_message(message_bytes)

                    # Create partially signed transaction
                    partially_signed = VersionedTransaction.populate(
                        transaction.message,
                        [payer_signature, transaction.signatures[1]],
                    )
                    tx_to_sign = base64.b64encode(bytes(partially_signed)).decode(
                        "utf-8"
                    )

                # Sign via Privy (signTransaction only, not signAndSendTransaction)
                sign_result = await _privy_sign_transaction(
                    privy_client,
                    wallet_id,
                    tx_to_sign,
                    self._signing_key,
                )

                if not sign_result.get("success"):
                    error_msg = sign_result.get("error", "Failed to sign transaction")
                    return {"status": "error", "message": error_msg}

                signed_tx_base64 = sign_result.get("signed_transaction")
                if not signed_tx_base64:
                    return {
                        "status": "error",
                        "message": "No signed transaction returned from Privy.",
                    }

                # Send via RPC (Helius recommended for priority fees)
                rpc_url = self._rpc_url or DEFAULT_RPC_URL
                tx_bytes = base64.b64decode(signed_tx_base64)
                send_result = await send_raw_transaction_with_priority(
                    rpc_url=rpc_url,
                    tx_bytes=tx_bytes,
                )

                if send_result.get("success"):
                    signature = send_result.get("signature")
                    if not signature:
                        return {
                            "status": "error",
                            "message": "No transaction signature returned.",
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

                # Check if it's a blockhash error - retry with fresh transaction
                error_msg = send_result.get("error", "")
                if (
                    "blockhash" in error_msg.lower()
                    or "Blockhash not found" in error_msg
                ):
                    last_error = error_msg
                    logger.warning(
                        f"Blockhash expired on attempt {attempt + 1}/{max_retries}, retrying with fresh transaction..."
                    )
                    continue
                else:
                    # Non-blockhash error, don't retry
                    return {
                        "status": "error",
                        "message": send_result.get(
                            "error", "Failed to send transaction."
                        ),
                    }

            # All retries exhausted
            return {
                "status": "error",
                "message": f"Transaction failed after {max_retries} attempts. Last error: {last_error}",
            }

        except Exception as e:
            logger.exception(f"DFlow swap failed: {str(e)}")
            return {"status": "error", "message": str(e)}
        finally:
            await privy_client.close()


class PrivyDFlowSwapPlugin:
    """Plugin for fast token swaps using DFlow API with Privy wallets."""

    def __init__(self):
        self.name = "privy_dflow_swap"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return (
            "Plugin for fast token swaps using DFlow API with Privy delegated wallets."
        )

    def initialize(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._tool = PrivyDFlowSwapTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:
        return [self._tool] if self._tool else []


def get_plugin():
    return PrivyDFlowSwapPlugin()

"""
Jupiter Ultra swap tool for Privy embedded wallets.

Enables swapping tokens using Jupiter Ultra API via Privy delegated wallets.
Uses the official Privy Python SDK for wallet operations.

NOTE: We bypass Jupiter's /execute endpoint and send transactions directly via
our RPC (Helius). This is because Jupiter's execute can have reliability issues.
This matches the pattern used in privy_trigger.py.
"""

import base64
import logging
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry
from privy import AsyncPrivyAPI
from privy.lib.authorization_signatures import get_authorization_signature
from cryptography.hazmat.primitives import serialization
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


def convert_key_to_pkcs8_pem(key_string: str) -> str:  # pragma: no cover
    """Convert a private key to PKCS#8 PEM format for the Privy SDK.

    The SDK expects keys in PKCS#8 PEM format (-----BEGIN PRIVATE KEY-----).
    This function handles keys in various formats:
    - Already in PKCS#8 PEM base64
    - SEC1 EC format (-----BEGIN EC PRIVATE KEY-----)
    - Raw DER bytes (PKCS#8 or SEC1)

    Returns the base64 content (without headers) in PKCS#8 format.
    """
    # Strip wallet-auth: prefix if present
    private_key_string = key_string.replace("wallet-auth:", "")

    # Try loading as PKCS#8 PEM format first
    try:
        private_key_pem = f"-----BEGIN PRIVATE KEY-----\n{private_key_string}\n-----END PRIVATE KEY-----"
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode("utf-8"), password=None
        )
        # Already in correct format, return as-is
        return private_key_string
    except (ValueError, TypeError):
        pass

    # Try as EC PRIVATE KEY (SEC1) format
    try:
        ec_key_pem = f"-----BEGIN EC PRIVATE KEY-----\n{private_key_string}\n-----END EC PRIVATE KEY-----"
        private_key = serialization.load_pem_private_key(
            ec_key_pem.encode("utf-8"), password=None
        )
        # Convert to PKCS#8 format
        pkcs8_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        # Extract base64 content (remove headers and newlines)
        pkcs8_pem = pkcs8_bytes.decode("utf-8")
        lines = pkcs8_pem.strip().split("\n")
        base64_content = "".join(lines[1:-1])
        return base64_content
    except (ValueError, TypeError):
        pass

    # Try loading as raw DER bytes
    try:
        der_bytes = base64.b64decode(private_key_string)
        # Try PKCS#8 DER
        try:
            private_key = serialization.load_der_private_key(der_bytes, password=None)
        except (ValueError, TypeError):
            # Try SEC1 DER
            from cryptography.hazmat.primitives.asymmetric import ec

            private_key = ec.derive_private_key(
                int.from_bytes(der_bytes, "big"), ec.SECP256R1()
            )
        # Convert to PKCS#8 PEM format
        pkcs8_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pkcs8_pem = pkcs8_bytes.decode("utf-8")
        lines = pkcs8_pem.strip().split("\n")
        base64_content = "".join(lines[1:-1])
        return base64_content
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Could not load private key. Expected base64-encoded PKCS#8 or SEC1 format. "
            f"Generate with: openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256. Error: {e}"
        )


async def privy_sign_transaction(  # pragma: no cover
    privy_client: AsyncPrivyAPI,
    wallet_id: str,
    encoded_tx: str,
    signing_key: str,
) -> Dict[str, Any]:
    """Sign a Solana transaction via Privy using the official SDK.

    Uses the wallets.rpc method with method="signTransaction" for Solana.
    The SDK handles authorization signature generation automatically when provided.
    """
    # Convert the key to PKCS#8 format expected by the SDK
    pkcs8_key = convert_key_to_pkcs8_pem(signing_key)

    # Generate the authorization signature using the SDK's utility
    # IMPORTANT: The body must match exactly what the SDK sends to the API
    url = f"https://api.privy.io/v1/wallets/{wallet_id}/rpc"
    body = {
        "method": "signTransaction",
        "params": {"transaction": encoded_tx, "encoding": "base64"},
        "chain_type": "solana",
    }

    # Use SDK's authorization signature helper
    auth_signature = get_authorization_signature(
        url=url,
        body=body,
        method="POST",
        app_id=privy_client.app_id,
        private_key=pkcs8_key,
    )

    # Call the SDK's rpc method for Solana signTransaction
    result = await privy_client.wallets.rpc(
        wallet_id=wallet_id,
        method="signTransaction",
        params={"transaction": encoded_tx, "encoding": "base64"},
        chain_type="solana",
        privy_authorization_signature=auth_signature,
    )

    return {
        "data": {
            "signedTransaction": result.data.signed_transaction if result.data else None
        }
    }


async def _privy_sign_transaction(  # pragma: no cover
    privy_client: AsyncPrivyAPI,
    wallet_id: str,
    encoded_tx: str,
    signing_key: str,
) -> Optional[str]:
    """Sign a Solana transaction via Privy using the official SDK.

    Returns just the signed transaction string (or None on failure).
    Used by _sign_and_execute method.
    """
    try:
        # Convert the key to PKCS#8 format expected by the SDK
        pkcs8_key = convert_key_to_pkcs8_pem(signing_key)

        # IMPORTANT: The body must match exactly what the SDK sends to the API
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


class PrivyUltraTool(AutoTool):
    """Swap tokens using Jupiter Ultra via Privy delegated wallet."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="privy_ultra",
            description="Swap tokens using Jupiter Ultra API via Privy delegated wallet. Provides the best trading experience with MEV protection, optimal slippage, and fast execution.",
            registry=registry,
        )
        self.app_id = None
        self.app_secret = None
        self.signing_key = None
        self.jupiter_api_key = None
        self.referral_account = None
        self.referral_fee = None
        self.payer_private_key = None
        self._rpc_url = None
        self._signing_key = None

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
        tool_cfg = config.get("tools", {}).get("privy_ultra", {})
        self.app_id = tool_cfg.get("app_id")
        self.app_secret = tool_cfg.get("app_secret")
        self.signing_key = tool_cfg.get("signing_key")
        self._signing_key = self.signing_key  # For _sign_and_execute
        self.jupiter_api_key = tool_cfg.get("jupiter_api_key")
        self.referral_account = tool_cfg.get("referral_account")
        self.referral_fee = tool_cfg.get("referral_fee")
        self.payer_private_key = tool_cfg.get("payer_private_key")
        self._payer_private_key = self.payer_private_key  # For _sign_and_execute
        self._rpc_url = tool_cfg.get("rpc_url")

    async def _sign_and_execute(  # pragma: no cover
        self,
        privy_client: AsyncPrivyAPI,
        wallet_id: str,
        transaction_base64: str,
    ) -> Dict[str, Any]:
        """
        Replace blockhash, sign with payer (if configured) and Privy, then send.

        Jupiter's /execute endpoint can have reliability issues, so we:
        1. Get a fresh blockhash from our RPC
        2. Replace the blockhash in Jupiter's transaction
        3. Sign with our keys
        4. Send directly via our RPC

        This matches the pattern used in privy_trigger.py.
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

            tx_to_sign = tx_with_new_blockhash

            # Step 3: If payer is configured, sign with payer first
            if self._payer_private_key:
                payer_keypair = Keypair.from_base58_string(self._payer_private_key)
                tx_bytes = base64.b64decode(tx_with_new_blockhash)
                transaction = VersionedTransaction.from_bytes(tx_bytes)
                message_bytes = to_bytes_versioned(transaction.message)
                payer_signature = payer_keypair.sign_message(message_bytes)

                # Find the payer's position in the account keys
                # The first N accounts (where N = num_required_signatures) are signers
                payer_pubkey = payer_keypair.pubkey()
                num_signers = transaction.message.header.num_required_signatures
                account_keys = transaction.message.account_keys

                payer_index = None
                for i in range(num_signers):
                    if account_keys[i] == payer_pubkey:
                        payer_index = i
                        break

                if payer_index is None:
                    logger.warning(
                        f"Payer pubkey {payer_pubkey} not found in signers. "
                        f"Signers: {[str(account_keys[i]) for i in range(num_signers)]}"
                    )
                    # Payer not in transaction - this might be a non-gasless transaction
                    # Just pass through to Privy signing
                else:
                    # Create signature list with payer signature in correct position
                    new_signatures = list(transaction.signatures)
                    new_signatures[payer_index] = payer_signature
                    logger.info(f"Payer signed at index {payer_index}")

                    partially_signed = VersionedTransaction.populate(
                        transaction.message,
                        new_signatures,
                    )
                    tx_to_sign = base64.b64encode(bytes(partially_signed)).decode(
                        "utf-8"
                    )

            # Step 4: Sign with Privy using the official SDK
            signed_tx = await _privy_sign_transaction(
                privy_client,
                wallet_id,
                tx_to_sign,
                self._signing_key,
            )

            if not signed_tx:
                return {
                    "status": "error",
                    "message": "Failed to sign transaction via Privy.",
                }

            # Step 5: Send via our RPC
            tx_bytes = base64.b64decode(signed_tx)
            send_result = await send_raw_transaction_with_priority(
                rpc_url=self._rpc_url,
                tx_bytes=tx_bytes,
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

        if not all([self.app_id, self.app_secret, self.signing_key]):
            return {"status": "error", "message": "Privy config missing."}

        # Create Privy client using the official SDK
        privy_client = AsyncPrivyAPI(
            app_id=self.app_id,
            app_secret=self.app_secret,
        )

        try:
            public_key = wallet_public_key

            # Check if integrator payer is configured for gasless transactions
            payer_pubkey = None
            if self.payer_private_key:
                payer_keypair = Keypair.from_base58_string(self.payer_private_key)
                payer_pubkey = str(payer_keypair.pubkey())

            # Initialize Jupiter Ultra client
            ultra = JupiterUltra(api_key=self.jupiter_api_key)

            # Get swap order
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

            if not order.transaction:
                return {
                    "status": "error",
                    "message": "No transaction returned from Jupiter Ultra.",
                }

            # Sign and execute via RPC (bypasses Jupiter's /execute endpoint)
            exec_result = await self._sign_and_execute(
                privy_client=privy_client,
                wallet_id=wallet_id,
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
            logger.exception(f"Privy Ultra swap failed: {str(e)}")
            return {"status": "error", "message": str(e)}
        finally:
            await privy_client.close()


class PrivyUltraPlugin:
    """Plugin for swapping tokens using Jupiter Ultra via Privy."""

    def __init__(self):
        self.name = "privy_ultra"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for swapping tokens using Jupiter Ultra API via Privy delegated wallet."

    def initialize(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._tool = PrivyUltraTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:
        return [self._tool] if self._tool else []


def get_plugin():
    return PrivyUltraPlugin()

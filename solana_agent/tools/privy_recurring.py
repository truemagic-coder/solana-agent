"""
Jupiter Recurring (DCA) tool for Privy delegated wallets.

Enables creating, canceling, and managing DCA orders using Jupiter Recurring API
with Privy embedded wallets. Uses the official Privy Python SDK.
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

from solana_agent.tools.utils.recurring import JupiterRecurring

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
) -> Optional[str]:
    """Sign a Solana transaction via Privy using the official SDK."""
    try:
        # Convert the key to PKCS#8 format expected by the SDK
        pkcs8_key = _convert_key_to_pkcs8_pem(signing_key)

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


class PrivyRecurringTool(AutoTool):
    """Create and manage DCA orders using Jupiter Recurring API with Privy wallets."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="privy_recurring",
            description=(
                "Create and manage DCA (Dollar Cost Averaging) orders on Solana using Jupiter Recurring API "
                "with Privy delegated wallets. "
                "Actions: 'create' (new DCA order), 'cancel' (cancel specific order), 'list' (view orders). "
                "For cancel, first use 'list' action to get the order public key."
            ),
            registry=registry,
        )
        self._app_id: Optional[str] = None
        self._app_secret: Optional[str] = None
        self._signing_key: Optional[str] = None
        self._jupiter_api_key: Optional[str] = None
        self._payer_private_key: Optional[str] = None

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
                    "enum": ["create", "cancel", "list"],
                    "description": (
                        "Action to perform: 'create' (new DCA order), "
                        "'cancel' (cancel specific order by pubkey), "
                        "'list' (view active orders)"
                    ),
                },
                "input_mint": {
                    "type": "string",
                    "description": "Token mint to sell (required for 'create'). Pass empty string if not needed.",
                    "default": "",
                },
                "output_mint": {
                    "type": "string",
                    "description": "Token mint to buy (required for 'create'). Pass empty string if not needed.",
                    "default": "",
                },
                "in_amount": {
                    "type": "string",
                    "description": "Total amount of input token to DCA in base units (required for 'create'). Pass empty string if not needed.",
                    "default": "",
                },
                "order_count": {
                    "type": "integer",
                    "description": "Number of orders to split into (required for 'create'). Pass 0 if not needed.",
                    "default": 0,
                },
                "frequency": {
                    "type": "string",
                    "description": "Interval between orders in seconds, e.g., '3600' for hourly, '86400' for daily (required for 'create'). Pass empty string if not needed.",
                    "default": "",
                },
                "min_out_amount": {
                    "type": "string",
                    "description": "Minimum output amount per order (optional for 'create'). Pass empty string if not needed.",
                    "default": "",
                },
                "max_out_amount": {
                    "type": "string",
                    "description": "Maximum output amount per order (optional for 'create'). Pass empty string if not needed.",
                    "default": "",
                },
                "start_at": {
                    "type": "string",
                    "description": "Unix timestamp when to start DCA (optional for 'create', defaults to now). Pass empty string if not needed.",
                    "default": "",
                },
                "order_pubkey": {
                    "type": "string",
                    "description": "Order public key to cancel (required for 'cancel'). Get this from 'list' action. Pass empty string if not needed.",
                    "default": "",
                },
            },
            "required": [
                "wallet_id",
                "wallet_public_key",
                "action",
                "input_mint",
                "output_mint",
                "in_amount",
                "order_count",
                "frequency",
                "min_out_amount",
                "max_out_amount",
                "start_at",
                "order_pubkey",
            ],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("privy_recurring", {})
        self._app_id = tool_cfg.get("app_id")
        self._app_secret = tool_cfg.get("app_secret")
        self._signing_key = tool_cfg.get("signing_key")
        self._jupiter_api_key = tool_cfg.get("jupiter_api_key")
        self._payer_private_key = tool_cfg.get("payer_private_key")

    async def execute(
        self,
        wallet_id: str,
        wallet_public_key: str,
        action: str,
        input_mint: Optional[str] = None,
        output_mint: Optional[str] = None,
        in_amount: Optional[str] = None,
        order_count: Optional[int] = None,
        frequency: Optional[str] = None,
        min_out_amount: Optional[str] = None,
        max_out_amount: Optional[str] = None,
        start_at: Optional[str] = None,
        order_pubkey: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not wallet_id or not wallet_public_key:
            return {
                "status": "error",
                "message": "wallet_id and wallet_public_key are required.",
            }

        if not all([self._app_id, self._app_secret, self._signing_key]):
            return {"status": "error", "message": "Privy config missing."}

        public_key = wallet_public_key

        action = action.lower().strip()
        recurring = JupiterRecurring(api_key=self._jupiter_api_key)

        if action == "create":
            return await self._create_order(
                recurring,
                wallet_id,
                public_key,
                input_mint,
                output_mint,
                in_amount,
                order_count,
                frequency,
                min_out_amount,
                max_out_amount,
                start_at,
            )
        elif action == "cancel":
            return await self._cancel_order(
                recurring, wallet_id, public_key, order_pubkey
            )
        elif action == "list":
            return await self._list_orders(recurring, public_key)
        else:
            return {
                "status": "error",
                "message": f"Unknown action: {action}. Valid actions: create, cancel, list",
            }

    async def _sign_and_execute(  # pragma: no cover
        self,
        recurring: JupiterRecurring,
        wallet_id: str,
        transaction_base64: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """Sign with payer (if configured) and Privy, then execute."""
        try:
            tx_to_sign = transaction_base64

            # If payer is configured, sign with payer first
            if self._payer_private_key:
                payer_keypair = Keypair.from_base58_string(self._payer_private_key)
                tx_bytes = base64.b64decode(transaction_base64)
                transaction = VersionedTransaction.from_bytes(tx_bytes)
                message_bytes = to_bytes_versioned(transaction.message)
                payer_signature = payer_keypair.sign_message(message_bytes)

                # Create partially signed transaction
                partially_signed = VersionedTransaction.populate(
                    transaction.message,
                    [payer_signature, transaction.signatures[1]],
                )
                tx_to_sign = base64.b64encode(bytes(partially_signed)).decode("utf-8")

            # Sign with Privy
            signed_tx = await _privy_sign_transaction(
                wallet_id,
                tx_to_sign,
                self._app_id,
                self._app_secret,
                self._signing_key,
            )

            if not signed_tx:
                return {
                    "status": "error",
                    "message": "Failed to sign transaction via Privy.",
                }

            # Execute
            exec_result = await recurring.execute(signed_tx, request_id)

            if not exec_result.success:
                return {"status": "error", "message": exec_result.error}

            return {"status": "success", "signature": exec_result.signature}

        except Exception as e:
            logger.exception(f"Failed to sign and execute: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _create_order(  # pragma: no cover
        self,
        recurring: JupiterRecurring,
        wallet_id: str,
        public_key: str,
        input_mint: Optional[str],
        output_mint: Optional[str],
        in_amount: Optional[str],
        order_count: Optional[int],
        frequency: Optional[str],
        min_out_amount: Optional[str],
        max_out_amount: Optional[str],
        start_at: Optional[str],
    ) -> Dict[str, Any]:
        if not all([input_mint, output_mint, in_amount, order_count, frequency]):
            return {
                "status": "error",
                "message": "Missing required parameters: input_mint, output_mint, in_amount, order_count, frequency",
            }

        try:
            payer_pubkey = None
            if self._payer_private_key:
                payer_keypair = Keypair.from_base58_string(self._payer_private_key)
                payer_pubkey = str(payer_keypair.pubkey())

            result = await recurring.create_order(
                input_mint=input_mint,
                output_mint=output_mint,
                user=public_key,
                in_amount=in_amount,
                order_count=order_count,
                frequency=frequency,
                payer=payer_pubkey,
                min_out_amount=min_out_amount,
                max_out_amount=max_out_amount,
                start_at=start_at,
            )

            if not result.success:
                return {"status": "error", "message": result.error}

            if not result.transaction:
                return {
                    "status": "error",
                    "message": "No transaction returned from Jupiter.",
                }

            exec_result = await self._sign_and_execute(
                recurring, wallet_id, result.transaction, result.request_id
            )

            if exec_result["status"] != "success":
                return exec_result

            return {
                "status": "success",
                "action": "create",
                "order_pubkey": result.order,
                "signature": exec_result["signature"],
                "message": f"DCA order created. Order pubkey: {result.order}. Will execute {order_count} orders.",
            }

        except Exception as e:
            logger.exception(f"Failed to create recurring order: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _cancel_order(  # pragma: no cover
        self,
        recurring: JupiterRecurring,
        wallet_id: str,
        public_key: str,
        order_pubkey: Optional[str],
    ) -> Dict[str, Any]:
        if not order_pubkey:
            return {
                "status": "error",
                "message": "Missing required parameter: order_pubkey. Use 'list' action first to get order pubkeys.",
            }

        try:
            # Verify the order belongs to this user before attempting to cancel
            orders_result = await recurring.get_orders(
                user=public_key, order_status="active"
            )
            if orders_result.get("success", False):
                user_orders = [
                    o.get("order") or o.get("orderPubkey")
                    for o in orders_result.get("orders", [])
                ]
                if order_pubkey not in user_orders:
                    return {
                        "status": "error",
                        "message": f"Order {order_pubkey} does not belong to this user or is not active.",
                    }

            payer_pubkey = None
            if self._payer_private_key:
                payer_keypair = Keypair.from_base58_string(self._payer_private_key)
                payer_pubkey = str(payer_keypair.pubkey())

            result = await recurring.cancel_order(
                user=public_key,
                order=order_pubkey,
                payer=payer_pubkey,
            )

            if not result.success:
                return {"status": "error", "message": result.error}

            if not result.transaction:
                return {
                    "status": "error",
                    "message": "No transaction returned from Jupiter.",
                }

            exec_result = await self._sign_and_execute(
                recurring, wallet_id, result.transaction, result.request_id
            )

            if exec_result["status"] != "success":
                return exec_result

            return {
                "status": "success",
                "action": "cancel",
                "order_pubkey": order_pubkey,
                "signature": exec_result["signature"],
                "message": f"DCA order {order_pubkey} cancelled. Remaining funds returned.",
            }

        except Exception as e:
            logger.exception(f"Failed to cancel recurring order: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _list_orders(  # pragma: no cover
        self,
        recurring: JupiterRecurring,
        public_key: str,
    ) -> Dict[str, Any]:
        try:
            result = await recurring.get_orders(user=public_key, order_status="active")

            if not result.get("success", False):
                return {
                    "status": "error",
                    "message": result.get("error", "Failed to fetch orders"),
                }

            orders = result.get("orders", [])

            formatted_orders = []
            for order in orders:
                formatted_orders.append(
                    {
                        "order_pubkey": order.get("order") or order.get("orderPubkey"),
                        "input_mint": order.get("inputMint"),
                        "output_mint": order.get("outputMint"),
                        "deposit_amount": order.get("depositAmount"),
                        "order_count": order.get("orderCount"),
                        "executed_count": order.get("executedCount"),
                        "frequency": order.get("frequency"),
                        "next_execution": order.get("nextExecution"),
                        "created_at": order.get("createdAt"),
                    }
                )

            return {
                "status": "success",
                "action": "list",
                "wallet": public_key,
                "order_count": len(formatted_orders),
                "orders": formatted_orders,
                "message": f"Found {len(formatted_orders)} active DCA orders."
                if formatted_orders
                else "No active DCA orders.",
            }

        except Exception as e:
            logger.exception(f"Failed to list recurring orders: {str(e)}")
            return {"status": "error", "message": str(e)}


class PrivyRecurringPlugin:
    """Plugin for creating and managing DCA orders using Jupiter Recurring with Privy wallets."""

    def __init__(self):
        self.name = "privy_recurring"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for creating and managing DCA orders using Jupiter Recurring API with Privy delegated wallets."

    def initialize(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._tool = PrivyRecurringTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:
        return [self._tool] if self._tool else []


def get_plugin():
    return PrivyRecurringPlugin()

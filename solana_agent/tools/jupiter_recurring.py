"""
Jupiter Recurring (DCA) tool for Solana Agent Kit.

Enables creating, canceling, and managing DCA orders using Jupiter Recurring API.
"""

import logging
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry
from solders.keypair import Keypair  # type: ignore

from solana_agent.tools.utils.recurring import (
    JupiterRecurring,
    sign_recurring_transaction,
)

logger = logging.getLogger(__name__)


class JupiterRecurringTool(AutoTool):
    """Create and manage DCA orders using Jupiter Recurring API."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="jupiter_recurring",
            description=(
                "Create and manage DCA (Dollar Cost Averaging) orders on Solana using Jupiter Recurring API. "
                "Actions: 'create' (new DCA order), 'cancel' (cancel specific order), 'list' (view orders). "
                "For cancel, first use 'list' action to get the order public key."
            ),
            registry=registry,
        )
        self._private_key: Optional[str] = None
        self._jupiter_api_key: Optional[str] = None
        self._payer_private_key: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
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
                "wallet_address": {
                    "type": "string",
                    "description": "Wallet address to query orders for (optional for 'list', defaults to configured wallet). Pass empty string if not needed.",
                    "default": "",
                },
            },
            "required": [
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
                "wallet_address",
            ],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("jupiter_recurring", {})
        self._private_key = tool_cfg.get("private_key")
        self._jupiter_api_key = tool_cfg.get("jupiter_api_key")
        self._payer_private_key = tool_cfg.get("payer_private_key")

    async def execute(
        self,
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
        wallet_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        action = action.lower().strip()
        recurring = JupiterRecurring(api_key=self._jupiter_api_key)

        if action == "create":
            return await self._create_order(
                recurring,
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
            return await self._cancel_order(recurring, order_pubkey)
        elif action == "list":
            return await self._list_orders(recurring, wallet_address)
        else:
            return {
                "status": "error",
                "message": f"Unknown action: {action}. Valid actions: create, cancel, list",
            }

    async def _create_order(  # pragma: no cover
        self,
        recurring: JupiterRecurring,
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

        if not self._private_key:
            return {"status": "error", "message": "Private key not configured."}

        try:
            keypair = Keypair.from_base58_string(self._private_key)
            user = str(keypair.pubkey())

            # Check if integrator payer is configured for gasless transactions
            payer_keypair = None
            payer_pubkey = None
            if self._payer_private_key:
                payer_keypair = Keypair.from_base58_string(self._payer_private_key)
                payer_pubkey = str(payer_keypair.pubkey())

            # Create order
            result = await recurring.create_order(
                input_mint=input_mint,
                output_mint=output_mint,
                user=user,
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

            # Sign the transaction
            signed_tx = sign_recurring_transaction(
                transaction_base64=result.transaction,
                sign_message_func=keypair.sign_message,
                payer_sign_func=payer_keypair.sign_message if payer_keypair else None,
            )

            # Execute
            exec_result = await recurring.execute(signed_tx, result.request_id)

            if not exec_result.success:
                return {"status": "error", "message": exec_result.error}

            return {
                "status": "success",
                "action": "create",
                "order_pubkey": result.order,
                "signature": exec_result.signature,
                "message": f"DCA order created. Order pubkey: {result.order}. Will execute {order_count} orders.",
            }

        except Exception as e:
            logger.exception(f"Failed to create recurring order: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _cancel_order(  # pragma: no cover
        self,
        recurring: JupiterRecurring,
        order_pubkey: Optional[str],
    ) -> Dict[str, Any]:
        if not order_pubkey:
            return {
                "status": "error",
                "message": "Missing required parameter: order_pubkey. Use 'list' action first to get order pubkeys.",
            }

        if not self._private_key:
            return {"status": "error", "message": "Private key not configured."}

        try:
            keypair = Keypair.from_base58_string(self._private_key)
            user = str(keypair.pubkey())

            # Verify the order belongs to this user before attempting to cancel
            orders_result = await recurring.get_orders(user=user, order_status="active")
            if orders_result.get("success", False):
                user_orders = [
                    o.get("order") or o.get("orderPubkey")
                    for o in orders_result.get("orders", [])
                ]
                if order_pubkey not in user_orders:
                    return {
                        "status": "error",
                        "message": f"Order {order_pubkey} does not belong to this wallet or is not active.",
                    }

            payer_keypair = None
            payer_pubkey = None
            if self._payer_private_key:
                payer_keypair = Keypair.from_base58_string(self._payer_private_key)
                payer_pubkey = str(payer_keypair.pubkey())

            result = await recurring.cancel_order(
                user=user,
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

            signed_tx = sign_recurring_transaction(
                transaction_base64=result.transaction,
                sign_message_func=keypair.sign_message,
                payer_sign_func=payer_keypair.sign_message if payer_keypair else None,
            )

            exec_result = await recurring.execute(signed_tx, result.request_id)

            if not exec_result.success:
                return {"status": "error", "message": exec_result.error}

            return {
                "status": "success",
                "action": "cancel",
                "order_pubkey": order_pubkey,
                "signature": exec_result.signature,
                "message": f"DCA order {order_pubkey} cancelled. Remaining funds returned.",
            }

        except Exception as e:
            logger.exception(f"Failed to cancel recurring order: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _list_orders(  # pragma: no cover
        self,
        recurring: JupiterRecurring,
        wallet_address: Optional[str],
    ) -> Dict[str, Any]:
        try:
            if wallet_address:
                wallet = wallet_address
            elif self._private_key:
                keypair = Keypair.from_base58_string(self._private_key)
                wallet = str(keypair.pubkey())
            else:
                return {
                    "status": "error",
                    "message": "Either wallet_address parameter or private_key config required.",
                }

            result = await recurring.get_orders(user=wallet, order_status="active")

            if not result.get("success", False):
                return {
                    "status": "error",
                    "message": result.get("error", "Failed to fetch orders"),
                }

            orders = result.get("orders", [])

            # Format orders for easy reading
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
                "wallet": wallet,
                "order_count": len(formatted_orders),
                "orders": formatted_orders,
                "message": f"Found {len(formatted_orders)} active DCA orders."
                if formatted_orders
                else "No active DCA orders.",
            }

        except Exception as e:
            logger.exception(f"Failed to list recurring orders: {str(e)}")
            return {"status": "error", "message": str(e)}


class JupiterRecurringPlugin:
    """Plugin for creating and managing DCA orders using Jupiter Recurring."""

    def __init__(self):
        self.name = "jupiter_recurring"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):  # pragma: no cover
        return (
            "Plugin for creating and managing DCA orders using Jupiter Recurring API."
        )

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = JupiterRecurringTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return JupiterRecurringPlugin()

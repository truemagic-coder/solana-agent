"""
Jupiter Trigger API utility functions.

Provides a simplified interface to Jupiter's Trigger API for creating,
canceling, and managing limit orders.
"""

import logging
import base64
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import httpx

from solders.transaction import VersionedTransaction  # type: ignore
from solders.message import to_bytes_versioned, MessageV0  # type: ignore
from solders.hash import Hash  # type: ignore

logger = logging.getLogger(__name__)

# Jupiter Trigger API base URL (API key required, free tier available at portal.jup.ag)
JUPITER_TRIGGER_API = "https://api.jup.ag/trigger/v1"


@dataclass
class TriggerOrderResponse:
    """Response from Jupiter Trigger /createOrder endpoint."""

    success: bool
    order: Optional[str] = None  # Order account public key
    transaction: Optional[str] = None  # Base64 encoded transaction
    request_id: Optional[str] = None
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerExecuteResponse:
    """Response from Jupiter Trigger /execute endpoint."""

    success: bool
    status: Optional[str] = None
    signature: Optional[str] = None
    error: Optional[str] = None
    code: Optional[int] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerCancelResponse:
    """Response from Jupiter Trigger /cancelOrder endpoint."""

    success: bool
    transaction: Optional[str] = None  # Base64 encoded transaction
    request_id: Optional[str] = None
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerCancelMultipleResponse:
    """Response from Jupiter Trigger /cancelOrders endpoint."""

    success: bool
    transactions: List[str] = field(default_factory=list)  # Base64 encoded transactions
    request_id: Optional[str] = None
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


class JupiterTrigger:
    """Jupiter Trigger API client for limit orders."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize Jupiter Trigger client.

        Args:
            api_key: Jupiter API key (required, get free key at portal.jup.ag)
            base_url: Optional custom base URL (defaults to api.jup.ag)
        """
        self.base_url = base_url or JUPITER_TRIGGER_API
        self.api_key = api_key
        self._headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
        }

    async def create_order(  # pragma: no cover
        self,
        input_mint: str,
        output_mint: str,
        maker: str,
        making_amount: str,
        taking_amount: str,
        payer: Optional[str] = None,
        expired_at: Optional[str] = None,
        fee_bps: Optional[int] = None,
        fee_account: Optional[str] = None,
        compute_unit_price: str = "auto",
    ) -> TriggerOrderResponse:
        """
        Create a new trigger (limit) order.

        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            maker: Maker wallet address (who is placing the order)
            making_amount: Amount of input token to sell (in base units as string)
            taking_amount: Amount of output token to receive (in base units as string)
            payer: Optional payer wallet address for gasless (defaults to maker)
            expired_at: Optional expiry time in unix seconds (as string)
            fee_bps: Optional integrator fee in basis points
            fee_account: Optional referral token account for collecting fees
            compute_unit_price: Compute unit price ("auto" recommended)

        Returns:
            TriggerOrderResponse with transaction to sign
        """
        # Validate expired_at is in the future if provided
        if expired_at:
            try:
                exp_timestamp = int(expired_at)
                current_timestamp = int(time.time())
                if exp_timestamp <= current_timestamp:
                    return TriggerOrderResponse(
                        success=False,
                        error=f"expired_at timestamp ({exp_timestamp}) must be in the future. Current time is {current_timestamp}.",
                    )
            except (ValueError, TypeError):
                return TriggerOrderResponse(
                    success=False,
                    error=f"Invalid expired_at value: {expired_at}. Must be a unix timestamp.",
                )

        body = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "maker": maker,
            "payer": payer or maker,
            "params": {
                "makingAmount": making_amount,
                "takingAmount": taking_amount,
            },
            "computeUnitPrice": compute_unit_price,
        }

        if expired_at:
            body["params"]["expiredAt"] = expired_at
        if fee_bps is not None:
            body["params"]["feeBps"] = str(fee_bps)
        if fee_account:
            body["feeAccount"] = fee_account

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/createOrder",
                    json=body,
                    headers=self._headers,
                )

                if response.status_code != 200:
                    return TriggerOrderResponse(
                        success=False,
                        error=f"Failed to create trigger order: {response.status_code} - {response.text}",
                    )

                data = response.json()

                return TriggerOrderResponse(
                    success=True,
                    order=data.get("order", ""),
                    transaction=data.get("transaction", ""),
                    request_id=data.get("requestId", ""),
                    raw_response=data,
                )
        except Exception as e:
            logger.exception("Failed to create trigger order")
            return TriggerOrderResponse(success=False, error=str(e))

    async def cancel_order(  # pragma: no cover
        self,
        maker: str,
        order: str,
        payer: Optional[str] = None,
        compute_unit_price: str = "auto",
    ) -> TriggerCancelResponse:
        """
        Cancel a single trigger order.

        Args:
            maker: Maker wallet address
            order: Order account public key to cancel
            payer: Optional payer wallet address for gasless
            compute_unit_price: Compute unit price ("auto" recommended)

        Returns:
            TriggerCancelResponse with transaction to sign
        """
        body = {
            "maker": maker,
            "order": order,
            "computeUnitPrice": compute_unit_price,
        }
        if payer:
            body["payer"] = payer

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/cancelOrder",
                    json=body,
                    headers=self._headers,
                )

                if response.status_code != 200:
                    return TriggerCancelResponse(
                        success=False,
                        error=f"Failed to cancel trigger order: {response.status_code} - {response.text}",
                    )

                data = response.json()

                return TriggerCancelResponse(
                    success=True,
                    transaction=data.get("transaction", ""),
                    request_id=data.get("requestId", ""),
                    raw_response=data,
                )
        except Exception as e:
            logger.exception("Failed to cancel trigger order")
            return TriggerCancelResponse(success=False, error=str(e))

    async def cancel_orders(  # pragma: no cover
        self,
        maker: str,
        orders: Optional[List[str]] = None,
        payer: Optional[str] = None,
        compute_unit_price: str = "auto",
    ) -> TriggerCancelMultipleResponse:
        """
        Cancel multiple trigger orders (or all if orders not specified).

        Args:
            maker: Maker wallet address
            orders: Optional list of order account public keys (if None, cancels ALL orders)
            payer: Optional payer wallet address for gasless
            compute_unit_price: Compute unit price ("auto" recommended)

        Returns:
            TriggerCancelMultipleResponse with transactions to sign (batched in groups of 5)
        """
        body = {
            "maker": maker,
            "computeUnitPrice": compute_unit_price,
        }
        if orders:
            body["orders"] = orders
        if payer:
            body["payer"] = payer

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/cancelOrders",
                    json=body,
                    headers=self._headers,
                )

                if response.status_code != 200:
                    return TriggerCancelMultipleResponse(
                        success=False,
                        error=f"Failed to cancel trigger orders: {response.status_code} - {response.text}",
                    )

                data = response.json()

                return TriggerCancelMultipleResponse(
                    success=True,
                    transactions=data.get("transactions", []),
                    request_id=data.get("requestId", ""),
                    raw_response=data,
                )
        except Exception as e:
            logger.exception("Failed to cancel trigger orders")
            return TriggerCancelMultipleResponse(success=False, error=str(e))

    async def execute(  # pragma: no cover
        self,
        signed_transaction: str,
        request_id: str,
        max_retries: int = 3,
    ) -> TriggerExecuteResponse:
        """
        Execute a signed trigger order transaction.

        Args:
            signed_transaction: Base64 encoded signed transaction
            request_id: Request ID from create/cancel response
            max_retries: Number of retries on 504 timeout (default 3)

        Returns:
            TriggerExecuteResponse with execution result
        """
        import asyncio

        payload = {
            "signedTransaction": signed_transaction,
            "requestId": request_id,
        }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Use longer timeout for execute - Jupiter waits for tx confirmation
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.base_url}/execute",
                        json=payload,
                        headers=self._headers,
                    )

                    # Retry on 504 Gateway Timeout
                    if response.status_code == 504:
                        last_error = f"504 Gateway Timeout (attempt {attempt + 1}/{max_retries + 1})"
                        logger.warning(f"Jupiter execute timed out: {last_error}")
                        if attempt < max_retries:
                            await asyncio.sleep(2**attempt)  # Exponential backoff
                            continue
                        return TriggerExecuteResponse(
                            success=False,
                            error=f"Jupiter execute endpoint timed out after {max_retries + 1} attempts",
                        )

                    if response.status_code != 200:
                        return TriggerExecuteResponse(
                            success=False,
                            error=f"Failed to execute trigger order: {response.status_code} - {response.text}",
                        )

                    data = response.json()
                    status = data.get("status", "")

                    return TriggerExecuteResponse(
                        success=status.lower() == "success",
                        status=status,
                        signature=data.get("signature"),
                        error=data.get("error"),
                        code=data.get("code", 0),
                        raw_response=data,
                    )
            except httpx.TimeoutException as e:
                last_error = (
                    f"Timeout: {str(e)} (attempt {attempt + 1}/{max_retries + 1})"
                )
                logger.warning(f"Jupiter execute request timed out: {last_error}")
                if attempt < max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
            except Exception as e:
                logger.exception("Failed to execute trigger order")
                return TriggerExecuteResponse(success=False, error=str(e))

        return TriggerExecuteResponse(
            success=False,
            error=last_error or "Failed after all retries",
        )

    async def get_orders(  # pragma: no cover
        self,
        user: str,
        order_status: str = "active",
        input_mint: Optional[str] = None,
        output_mint: Optional[str] = None,
        page: int = 1,
    ) -> Dict[str, Any]:
        """
        Get trigger orders for a user.

        Args:
            user: User wallet address
            order_status: "active" or "history"
            input_mint: Optional filter by input token mint
            output_mint: Optional filter by output token mint
            page: Page number for pagination (10 orders per page)

        Returns:
            Orders data with pagination info
        """
        params = {
            "user": user,
            "orderStatus": order_status,
            "page": str(page),
        }
        if input_mint:
            params["inputMint"] = input_mint
        if output_mint:
            params["outputMint"] = output_mint

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/getTriggerOrders",
                    params=params,
                    headers=self._headers,
                )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Failed to get trigger orders: {response.status_code} - {response.text}",
                        "orders": [],
                    }

                data = response.json()
                return {
                    "success": True,
                    "orders": data.get("orders", []),
                    "total": data.get("total", 0),
                    "page": data.get("page", 1),
                }
        except Exception as e:
            logger.exception("Failed to get trigger orders")
            return {"success": False, "error": str(e), "orders": []}


def replace_blockhash_in_transaction(  # pragma: no cover
    transaction_base64: str,
    new_blockhash: str,
) -> str:
    """
    Replace the blockhash in a versioned transaction with a fresh one.

    This is necessary when the transaction was created by an external service
    (like Jupiter) using their RPC, but we need to send it via a different RPC.
    The blockhash must be recent and recognized by the sending RPC.

    Args:
        transaction_base64: Base64 encoded unsigned transaction
        new_blockhash: Fresh blockhash string from our RPC

    Returns:
        Base64 encoded transaction with replaced blockhash (still unsigned)
    """
    transaction_bytes = base64.b64decode(transaction_base64)
    transaction = VersionedTransaction.from_bytes(transaction_bytes)

    # Get the message and replace blockhash
    old_message = transaction.message

    # Create new message with updated blockhash
    new_message = MessageV0(
        header=old_message.header,
        account_keys=old_message.account_keys,
        recent_blockhash=Hash.from_string(new_blockhash),
        instructions=old_message.instructions,
        address_table_lookups=old_message.address_table_lookups,
    )

    # Create new unsigned transaction with new message
    # Use default signatures (all zeros) since it's unsigned
    new_transaction = VersionedTransaction.populate(
        new_message,
        transaction.signatures,  # Keep placeholder signatures
    )

    return base64.b64encode(bytes(new_transaction)).decode("utf-8")


async def get_fresh_blockhash(rpc_url: str) -> dict:  # pragma: no cover
    """
    Get a fresh blockhash from the RPC.

    Args:
        rpc_url: The RPC endpoint URL

    Returns:
        Dict with 'blockhash' and 'lastValidBlockHeight' on success,
        or 'error' on failure.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getLatestBlockhash",
        "params": [{"commitment": "confirmed"}],
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(rpc_url, json=payload)
            if response.status_code != 200:
                return {"error": f"RPC error: {response.status_code}"}

            data = response.json()
            if "error" in data:
                return {"error": f"RPC error: {data['error']}"}

            result = data.get("result", {}).get("value", {})
            return {
                "blockhash": result.get("blockhash"),
                "lastValidBlockHeight": result.get("lastValidBlockHeight"),
            }
    except Exception as e:
        logger.exception("Failed to get fresh blockhash")
        return {"error": str(e)}


def sign_trigger_transaction(  # pragma: no cover
    transaction_base64: str,
    sign_message_func,
    payer_sign_func=None,
) -> str:
    """
    Sign a Jupiter Trigger transaction.

    Args:
        transaction_base64: Base64 encoded transaction
        sign_message_func: Function that signs a message and returns signature (maker)
        payer_sign_func: Optional function for payer signature (for gasless)

    Returns:
        Base64 encoded signed transaction
    """
    transaction_bytes = base64.b64decode(transaction_base64)
    transaction = VersionedTransaction.from_bytes(transaction_bytes)

    message_bytes = to_bytes_versioned(transaction.message)

    if payer_sign_func:
        # Gasless: payer signs first, then maker
        payer_signature = payer_sign_func(message_bytes)
        maker_signature = sign_message_func(message_bytes)
        signed_transaction = VersionedTransaction.populate(
            transaction.message, [payer_signature, maker_signature]
        )
    else:
        # Normal: only maker signs
        signature = sign_message_func(message_bytes)
        signed_transaction = VersionedTransaction.populate(
            transaction.message, [signature]
        )

    return base64.b64encode(bytes(signed_transaction)).decode("utf-8")

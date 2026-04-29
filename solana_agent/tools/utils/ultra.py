"""
Jupiter Ultra API utility functions.

Provides a simplified interface to Jupiter's Ultra API for swapping tokens,
getting holdings, checking token security, and searching tokens.
"""

import logging
import base64
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import httpx

from solders.transaction import VersionedTransaction
from solders.message import to_bytes_versioned

logger = logging.getLogger(__name__)

# Jupiter Ultra API base URL (API key required, free tier available at portal.jup.ag)
JUPITER_ULTRA_API = "https://api.jup.ag/ultra/v1"


@dataclass
class UltraOrderResponse:
    """Response from Jupiter Ultra /order endpoint."""

    request_id: str
    transaction: str  # Base64 encoded transaction
    in_amount: str
    out_amount: str
    input_mint: str
    output_mint: str
    slippage_bps: int
    swap_type: str
    price_impact: Optional[
        float
    ]  # Price impact as percentage (negative = cost, positive = benefit)
    in_usd_value: Optional[float]  # Input USD value
    out_usd_value: Optional[float]  # Output USD value
    gasless: bool
    raw_response: Dict[str, Any]


@dataclass
class UltraExecuteResponse:
    """Response from Jupiter Ultra /execute endpoint."""

    status: str
    signature: Optional[str]
    input_amount_result: Optional[str]
    output_amount_result: Optional[str]
    error: Optional[str]
    code: int
    raw_response: Dict[str, Any]


class JupiterUltra:
    """Jupiter Ultra API client."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize Jupiter Ultra client.

        Args:
            api_key: Jupiter API key (required, get free key at portal.jup.ag)
            base_url: Optional custom base URL (defaults to api.jup.ag)
        """
        self.base_url = base_url or JUPITER_ULTRA_API
        self.api_key = api_key
        self._headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
        }

    async def get_order(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        taker: str,
        referral_account: Optional[str] = None,
        referral_fee: Optional[int] = None,
        payer: Optional[str] = None,
        close_authority: Optional[str] = None,
    ) -> UltraOrderResponse:
        """
        Get a swap order from Jupiter Ultra.

        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount of input token (in smallest units)
            taker: User's wallet address
            referral_account: Optional referral account for fees
            referral_fee: Optional referral fee in basis points (50-255)
            payer: Optional integrator payer public key for gasless transactions
            close_authority: Optional close authority for token accounts (required with payer)

        Returns:
            UltraOrderResponse with transaction to sign

        Raises:
            Exception: If the order request fails
        """
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "taker": taker,
        }

        if referral_account:
            params["referralAccount"] = referral_account
        if referral_fee is not None:
            params["referralFee"] = str(referral_fee)
        if payer:
            params["payer"] = payer
        if close_authority:
            params["closeAuthority"] = close_authority

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/order",
                params=params,
                headers=self._headers,
            )

            if response.status_code != 200:
                raise Exception(
                    f"Failed to get order: {response.status_code} - {response.text}"
                )

            data = response.json()

            return UltraOrderResponse(
                request_id=data.get("requestId", ""),
                transaction=data.get("transaction", ""),
                in_amount=data.get("inAmount", ""),
                out_amount=data.get("outAmount", ""),
                input_mint=data.get("inputMint", ""),
                output_mint=data.get("outputMint", ""),
                slippage_bps=data.get("slippageBps", 0),
                swap_type=data.get("swapType", ""),
                price_impact=data.get("priceImpact"),
                in_usd_value=data.get("inUsdValue"),
                out_usd_value=data.get("outUsdValue"),
                gasless=data.get("gasless", False),
                raw_response=data,
            )

    async def execute_order(
        self,
        signed_transaction: str,
        request_id: str,
    ) -> UltraExecuteResponse:
        """
        Execute a signed swap order.

        Args:
            signed_transaction: Base64 encoded signed transaction
            request_id: Request ID from get_order response

        Returns:
            UltraExecuteResponse with execution result

        Raises:
            Exception: If the execute request fails
        """
        payload = {
            "signedTransaction": signed_transaction,
            "requestId": request_id,
        }

        # Use longer timeout for execute - Jupiter waits for tx confirmation
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/execute",
                json=payload,
                headers=self._headers,
            )

            if response.status_code != 200:  # pragma: no cover
                raise Exception(
                    f"Failed to execute order: {response.status_code} - {response.text}"
                )

            data = response.json()

            return UltraExecuteResponse(
                status=data.get("status", ""),
                signature=data.get("signature"),
                input_amount_result=data.get("inputAmountResult"),
                output_amount_result=data.get("outputAmountResult"),
                error=data.get("error"),
                code=data.get("code", 0),
                raw_response=data,
            )

    async def get_holdings(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get token holdings for a wallet.

        Args:
            wallet_address: Solana wallet address

        Returns:
            Holdings data including native SOL and all token balances
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/holdings/{wallet_address}",
                headers=self._headers,
            )

            if response.status_code != 200:
                raise Exception(
                    f"Failed to get holdings: {response.status_code} - {response.text}"
                )

            return response.json()

    async def get_native_holdings(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get native SOL balance for a wallet.

        Args:
            wallet_address: Solana wallet address

        Returns:
            Native SOL balance data
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/holdings/{wallet_address}/native",
                headers=self._headers,
            )

            if response.status_code != 200:  # pragma: no cover
                raise Exception(
                    f"Failed to get native holdings: {response.status_code} - {response.text}"
                )

            return response.json()

    async def get_shield(self, mints: List[str]) -> Dict[str, Any]:
        """
        Get security warnings for token mints.

        Args:
            mints: List of token mint addresses to check

        Returns:
            Shield data with warnings for each mint
        """
        mints_param = ",".join(mints)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/shield",
                params={"mints": mints_param},
                headers=self._headers,
            )

            if response.status_code != 200:  # pragma: no cover
                raise Exception(
                    f"Failed to get shield: {response.status_code} - {response.text}"
                )

            return response.json()

    async def search_tokens(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for tokens by symbol, name, or mint address.

        Args:
            query: Search query (symbol, name, or mint address)
                   Can be comma-separated for multiple searches

        Returns:
            List of matching tokens with metadata
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/search",
                params={"query": query},
                headers=self._headers,
            )

            if response.status_code != 200:  # pragma: no cover
                raise Exception(
                    f"Failed to search tokens: {response.status_code} - {response.text}"
                )

            return response.json()


def sign_ultra_transaction(
    transaction_base64: str,
    sign_message_func,
    payer_sign_func=None,
) -> str:
    """
    Sign a Jupiter Ultra transaction.

    Args:
        transaction_base64: Base64 encoded transaction from get_order
        sign_message_func: Function that signs a message and returns signature (taker)
        payer_sign_func: Optional function that signs for the payer (integrator gas payer)

    Returns:
        Base64 encoded signed transaction
    """
    transaction_bytes = base64.b64decode(transaction_base64)
    transaction = VersionedTransaction.from_bytes(transaction_bytes)

    message_bytes = to_bytes_versioned(transaction.message)

    # Get taker signature
    taker_signature = sign_message_func(message_bytes)

    if payer_sign_func:
        # With integrator payer: transaction needs both taker and payer signatures
        payer_signature = payer_sign_func(message_bytes)
        signed_transaction = VersionedTransaction.populate(
            transaction.message, [payer_signature, taker_signature]
        )
    else:
        # Standard flow: only taker signature
        signed_transaction = VersionedTransaction.populate(
            transaction.message, [taker_signature]
        )

    return base64.b64encode(bytes(signed_transaction)).decode("utf-8")

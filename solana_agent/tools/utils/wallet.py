from typing import Dict, List, Optional
import httpx
import nacl.signing
import logging
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts
from solders.transaction import Transaction
from solders.instruction import Instruction
from solders.keypair import Keypair
from solders.message import Message
from solders.signature import Signature
from solders.pubkey import Pubkey

logger = logging.getLogger(__name__)


def sanitize_privy_user_id(user_id: Optional[str]) -> Optional[str]:
    """
    Sanitize and normalize privy user_id to handle LLM formatting errors.

    LLMs sometimes mangle the user ID by:
    - Adding quotes: "did:privy:xxx" or 'did:privy:xxx'
    - Changing case: Did:privy:xxx or DID:PRIVY:xxx
    - Adding whitespace

    This function normalizes to the correct format: did:privy:xxx

    Args:
        user_id: The raw user_id string from LLM

    Returns:
        Sanitized user_id or None if input was None/empty
    """
    if not user_id:
        return None

    # Strip whitespace and quotes
    cleaned = user_id.strip().strip('"').strip("'").strip()

    if not cleaned:
        return None

    # Normalize the did:privy: prefix to lowercase
    if cleaned.lower().startswith("did:privy:"):
        # Keep the unique ID part as-is, but normalize prefix
        unique_part = cleaned[10:]  # Everything after "did:privy:"
        return f"did:privy:{unique_part}"

    return cleaned


class SolanaTransaction:
    """Transaction parameters for Solana."""

    def __init__(
        self,
        instructions: List[Instruction],
        accounts_to_sign: Optional[List[Keypair]] = None,
    ):
        self.instructions = instructions
        self.accounts_to_sign = accounts_to_sign


class SolanaWalletClient:
    """Solana wallet implementation."""

    def __init__(
        self,
        rpc_url: str,
        keypair: Optional[Keypair] = None,
        pubkey: Optional[str] = None,
        fee_payer: Optional[str] = None,
    ):
        self.client = AsyncClient(rpc_url)
        self.rpc_url = rpc_url
        self.keypair = keypair
        self.pubkey = pubkey
        self.fee_payer: Optional[Keypair] = None
        if pubkey:
            self.pubkey = Pubkey.from_string(pubkey)
        elif keypair:
            self.pubkey = keypair.pubkey()
        if fee_payer:  # pragma: no cover
            self.fee_payer = Keypair.from_base58_string(fee_payer)

    def sign_message(self, message: bytes) -> Signature:  # pragma: no cover
        signed = nacl.signing.SigningKey(self.keypair.secret()).sign(message)
        return Signature.from_bytes(signed.signature)

    def send_transaction(
        self, transaction: SolanaTransaction
    ) -> Dict[str, str]:  # pragma: no cover
        recent_blockhash = self.client.get_latest_blockhash().value.blockhash
        payer = self.keypair.pubkey()

        ixs = []
        for instruction in transaction.instructions:
            ixs.append(instruction)

        signers = [self.keypair]
        if transaction.accounts_to_sign:
            signers.extend(transaction.accounts_to_sign)

        message = Message(
            instructions=ixs,
            recent_blockhash=recent_blockhash,
            payer=payer,
        )
        tx = Transaction(
            from_keypairs=signers,
            message=message,
            recent_blockhash=recent_blockhash,
        )
        result = self.client.send_transaction(
            tx,
            opts=TxOpts(
                skip_preflight=False, max_retries=10, preflight_commitment=Confirmed
            ),
        )
        self.client.confirm_transaction(result.value, commitment=Confirmed)
        return {"hash": str(result.value)}

    async def get_priority_fee_estimate_helius(  # pragma: no cover
        self,
        serialized_transaction: str,
    ) -> int:
        """
        Get the priority fee estimate from Helius.

        :param serialized_transaction: The base64-encoded serialized transaction.
        :return: The estimated priority fee (int).
        """
        payload = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "getPriorityFeeEstimate",
            "params": [
                {
                    "transaction": serialized_transaction,
                    "options": {"recommended": True},
                }
            ],
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.rpc_url, json=payload)
            response.raise_for_status()
            result = response.json()
            if "error" in result:
                raise RuntimeError(f"Fee estimation failed: {result['error']}")
            return int(result["result"]["priorityFeeEstimate"])


async def send_raw_transaction_with_priority(  # pragma: no cover
    rpc_url: str,
    tx_bytes: bytes,
    skip_preflight: bool = True,
    max_retries: int = 5,
    confirm_timeout: float = 30.0,
    skip_confirmation: bool = False,
) -> Dict[str, any]:
    """
    Send a raw transaction to Solana RPC, with Helius priority fee logging if applicable.

    This is the standard way to send pre-signed transactions (e.g., from DFlow, Jupiter)
    through Helius or any Solana RPC endpoint.

    Args:
        rpc_url: The RPC endpoint URL (Helius recommended for priority fees)
        tx_bytes: The serialized signed transaction bytes
        skip_preflight: Skip preflight simulation (default True for pre-signed txs)
        max_retries: Number of retries for the RPC call
        confirm_timeout: Max seconds to wait for confirmation (default 30s)
        skip_confirmation: Skip waiting for confirmation entirely (default False)

    Returns:
        Dict with 'success' and 'signature' on success, or 'error' on failure.
    """
    import asyncio

    try:
        client = AsyncClient(rpc_url)
        try:
            # Get priority fee estimate if using Helius (for logging)
            if "helius" in rpc_url.lower():
                try:
                    import base64

                    tx_base64 = base64.b64encode(tx_bytes).decode("utf-8")
                    async with httpx.AsyncClient() as http_client:
                        payload = {
                            "jsonrpc": "2.0",
                            "id": "1",
                            "method": "getPriorityFeeEstimate",
                            "params": [
                                {
                                    "transaction": tx_base64,
                                    "options": {"recommended": True},
                                }
                            ],
                        }
                        response = await http_client.post(rpc_url, json=payload)
                        if response.status_code == 200:
                            result = response.json()
                            if "result" in result:
                                priority_fee = result["result"].get(
                                    "priorityFeeEstimate", 0
                                )
                                logger.info(
                                    f"Helius priority fee estimate: {priority_fee}"
                                )
                except Exception as fee_error:
                    logger.debug(f"Could not get priority fee estimate: {fee_error}")

            # Send the transaction
            logger.info(f"Sending transaction with skip_preflight={skip_preflight}")
            result = await client.send_raw_transaction(
                tx_bytes,
                opts=TxOpts(
                    skip_preflight=skip_preflight,
                    preflight_commitment=Confirmed,
                    max_retries=max_retries,
                ),
            )

            signature = str(result.value)
            logger.info(f"Transaction sent: {signature}")

            # Skip confirmation if requested (useful for pre-signed txs with external blockhashes)
            if skip_confirmation:
                return {"success": True, "signature": signature}

            # Confirm the transaction with timeout
            try:
                confirmation = await asyncio.wait_for(
                    client.confirm_transaction(
                        result.value,
                        commitment=Confirmed,
                        sleep_seconds=0.5,
                        last_valid_block_height=None,
                    ),
                    timeout=confirm_timeout,
                )
                if confirmation.value and confirmation.value[0].err:
                    return {
                        "success": False,
                        "error": f"Transaction failed: {confirmation.value[0].err}",
                    }
            except asyncio.TimeoutError:
                logger.warning(
                    f"Transaction confirmation timed out after {confirm_timeout}s. "
                    f"Transaction may still land. Signature: {signature}"
                )
                # Return success anyway - tx was sent, just not confirmed in time
            except Exception as confirm_error:
                logger.debug(f"Could not confirm transaction: {confirm_error}")
                # Still return success since transaction was sent

            return {"success": True, "signature": signature}

        finally:
            await client.close()

    except Exception as e:
        logger.error(f"RPC error sending transaction: {e}")
        return {"success": False, "error": str(e)}

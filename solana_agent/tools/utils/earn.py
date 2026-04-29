"""
Jupiter Earn (Lend) API utility functions.

Provides a simplified interface to Jupiter's Earn endpoints for deposits,
withdrawals, minting shares, redeeming shares, and querying tokens/positions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Jupiter Earn API base URL (API key required, free tier available at portal.jup.ag)
JUPITER_EARN_API = "https://api.jup.ag/lend/v1"


@dataclass
class EarnInstructionResponse:
    """Response from Jupiter Earn instruction endpoints."""

    success: bool
    instruction: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


class JupiterEarn:
    """Jupiter Earn API client."""

    def __init__(self, api_key: str, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or JUPITER_EARN_API
        self.api_key = api_key
        self._headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
        }

    async def _post_instruction(
        self, path: str, body: Dict[str, Any]
    ) -> EarnInstructionResponse:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}{path}",
                    json=body,
                    headers=self._headers,
                )

                if response.status_code != 200:
                    return EarnInstructionResponse(
                        success=False,
                        error=(
                            f"Failed to fetch earn instructions: {response.status_code} - "
                            f"{response.text}"
                        ),
                    )

                data = response.json()
                return EarnInstructionResponse(
                    success=True, instruction=data, raw_response=data
                )
        except Exception as e:
            logger.exception("Failed to fetch earn instructions")
            return EarnInstructionResponse(success=False, error=str(e))

    async def get_deposit_instructions(
        self, asset: str, signer: str, amount: str
    ) -> EarnInstructionResponse:
        body = {"asset": asset, "signer": signer, "amount": amount}
        return await self._post_instruction("/earn/deposit-instructions", body)

    async def get_withdraw_instructions(
        self, asset: str, signer: str, amount: str
    ) -> EarnInstructionResponse:
        body = {"asset": asset, "signer": signer, "amount": amount}
        return await self._post_instruction("/earn/withdraw-instructions", body)

    async def get_mint_instructions(
        self, asset: str, signer: str, shares: str
    ) -> EarnInstructionResponse:
        body = {"asset": asset, "signer": signer, "shares": shares}
        return await self._post_instruction("/earn/mint-instructions", body)

    async def get_redeem_instructions(
        self, asset: str, signer: str, shares: str
    ) -> EarnInstructionResponse:
        body = {"asset": asset, "signer": signer, "shares": shares}
        return await self._post_instruction("/earn/redeem-instructions", body)

    async def get_tokens(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/earn/tokens",
                    headers=self._headers,
                )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": (
                            f"Failed to fetch earn tokens: {response.status_code} - "
                            f"{response.text}"
                        ),
                        "tokens": [],
                    }

                data = response.json()
                return {"success": True, "tokens": data}
        except Exception as e:
            logger.exception("Failed to fetch earn tokens")
            return {"success": False, "error": str(e), "tokens": []}

    async def get_positions(self, users: List[str]) -> Dict[str, Any]:
        try:
            params = {"users": ",".join(users)}
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/earn/positions",
                    params=params,
                    headers=self._headers,
                )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": (
                            f"Failed to fetch earn positions: {response.status_code} - "
                            f"{response.text}"
                        ),
                        "positions": [],
                    }

                data = response.json()
                return {"success": True, "positions": data}
        except Exception as e:
            logger.exception("Failed to fetch earn positions")
            return {"success": False, "error": str(e), "positions": []}

    async def get_earnings(self, user: str, positions: List[str]) -> Dict[str, Any]:
        try:
            params = {"user": user, "positions": ",".join(positions)}
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/earn/earnings",
                    params=params,
                    headers=self._headers,
                )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": (
                            f"Failed to fetch earn earnings: {response.status_code} - "
                            f"{response.text}"
                        ),
                        "earnings": [],
                    }

                data = response.json()
                return {"success": True, "earnings": data}
        except Exception as e:
            logger.exception("Failed to fetch earn earnings")
            return {"success": False, "error": str(e), "earnings": []}

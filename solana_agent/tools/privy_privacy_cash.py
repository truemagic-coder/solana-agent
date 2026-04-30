"""Privy PrivacyCash tool for SOL/USDC privacy transactions via cash.solana-agent.com."""

import logging
from typing import Any, Dict, List, Optional

import httpx
from solana_agent import AutoTool, ToolRegistry

logger = logging.getLogger(__name__)


class PrivyPrivacyCashTool(AutoTool):
    """PrivacyCash operations via cash.solana-agent.com."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="privy_privacy_cash",
            description=(
                "PrivacyCash operations for Privy wallets via cash.solana-agent.com. "
                "Actions: transfer, deposit, withdraw, balance."
            ),
            registry=registry,
        )
        self.base_url = "https://cash.solana-agent.com"
        self.api_key = ""

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["transfer", "deposit", "withdraw", "balance"],
                    "description": (
                        "Action to perform: "
                        "transfer (deposit + withdraw), deposit, withdraw, "
                        "balance."
                    ),
                },
                "wallet_id": {
                    "type": "string",
                    "description": "Privy wallet id to operate on. Pass empty string if not needed.",
                    "default": "",
                },
                "amount": {
                    "type": "number",
                    "description": "Amount of token (human units). Pass 0 if not needed.",
                    "default": 0,
                },
                "recipient": {
                    "type": "string",
                    "description": "Recipient wallet address. Pass empty string if not needed.",
                    "default": "",
                },
                "token": {
                    "type": "string",
                    "description": "Token symbol: SOL or USDC. Pass empty string if not needed.",
                    "default": "",
                },
            },
            "required": ["action", "wallet_id", "amount", "recipient", "token"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the tool with API key from config."""
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("privy_privacy_cash", {})
        if isinstance(tool_cfg, dict):
            self.api_key = tool_cfg.get("api_key", "")
            if tool_cfg.get("base_url"):
                self.base_url = tool_cfg.get("base_url")

    async def _request(
        self, method: str, endpoint: str, json_data: Optional[dict] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "success": False,
                "error": (
                    "Privy Cash API key not configured. "
                    "Set privy_privacy_cash.api_key in config."
                ),
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        url = f"{self.base_url}{endpoint}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                else:
                    response = await client.post(url, headers=headers, json=json_data)

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"API error: {response.status_code}",
                        "details": response.text,
                    }

                data = response.json()
                return {"success": True, "data": data}
            except Exception as e:  # pragma: no cover
                return {"success": False, "error": str(e)}

    async def execute(  # pragma: no cover
        self,
        action: str,
        wallet_id: str = "",
        amount: float = 0,
        recipient: str = "",
        token: str = "",
    ) -> Dict[str, Any]:
        action = action.lower().strip()
        token_symbol = token.upper().strip() if token else ""

        if token_symbol and token_symbol not in {"SOL", "USDC"}:
            return {
                "success": False,
                "error": "token must be 'SOL' or 'USDC'",
            }

        if action == "transfer":
            if not wallet_id or not recipient or not token_symbol:
                return {
                    "success": False,
                    "error": "wallet_id, recipient, token are required",
                }
            if not isinstance(amount, (int, float)) or amount <= 0:
                return {"success": False, "error": "amount must be > 0"}
            return await self._request(
                "POST",
                "/transfer",
                json_data={
                    "walletId": wallet_id,
                    "amount": amount,
                    "recipient": recipient,
                    "token": token_symbol,
                },
            )

        if action == "deposit":
            if not wallet_id or not token_symbol:
                return {"success": False, "error": "wallet_id, token are required"}
            if not isinstance(amount, (int, float)) or amount <= 0:
                return {"success": False, "error": "amount must be > 0"}
            return await self._request(
                "POST",
                "/deposit",
                json_data={
                    "walletId": wallet_id,
                    "amount": amount,
                    "token": token_symbol,
                },
            )

        if action == "withdraw":
            if not wallet_id or not recipient or not token_symbol:
                return {
                    "success": False,
                    "error": "wallet_id, recipient, token are required",
                }
            if not isinstance(amount, (int, float)) or amount <= 0:
                return {"success": False, "error": "amount must be > 0"}
            return await self._request(
                "POST",
                "/withdraw",
                json_data={
                    "walletId": wallet_id,
                    "amount": amount,
                    "recipient": recipient,
                    "token": token_symbol,
                },
            )

        if action == "balance":
            if not wallet_id or not token_symbol:
                return {"success": False, "error": "wallet_id, token are required"}
            return await self._request(
                "POST",
                "/balance",
                json_data={"walletId": wallet_id, "token": token_symbol},
            )

        return {
            "success": False,
            "error": (
                "Unknown action: "
                f"{action}. Valid: transfer, deposit, withdraw, balance."
            ),
        }


class PrivyPrivacyCashPlugin:
    """Plugin for PrivacyCash operations via Privy."""

    def __init__(self):
        self.name = "privy_privacy_cash"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for PrivacyCash operations via cash.solana-agent.com."

    def initialize(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._tool = PrivyPrivacyCashTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:
        return [self._tool] if self._tool else []


def get_plugin():
    return PrivyPrivacyCashPlugin()

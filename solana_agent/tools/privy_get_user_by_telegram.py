"""
Privy Get User By Telegram Tool.

Looks up an existing Privy user by their Telegram user ID.
Used for bot-first Telegram bot flows to check if a user already exists.
"""

import base64
import logging
from typing import Dict, Any, List, Optional
from solana_agent import AutoTool, ToolRegistry
import httpx


async def get_privy_user_by_telegram(  # pragma: no cover
    telegram_user_id: str, app_id: str, app_secret: str
) -> Optional[Dict[str, Any]]:
    """
    Look up a Privy user by their Telegram user ID.

    Args:
        telegram_user_id: The Telegram user ID
        app_id: Privy app ID
        app_secret: Privy app secret

    Returns:
        Dict with user data if found, None if not found
    """
    url = "https://api.privy.io/v1/users/telegram/telegram_user_id"
    auth_string = f"{app_id}:{app_secret}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "privy-app-id": app_id,
        "Content-Type": "application/json",
    }

    body = {"telegram_user_id": telegram_user_id}

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=body, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            logging.error(f"Privy get user by telegram error: {resp.text}")
            resp.raise_for_status()
        return resp.json()


def extract_wallet_info(user_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract wallet information from user's linked accounts.

    Args:
        user_data: Privy user data with linked_accounts

    Returns:
        List of wallet info dicts with id, address, chain_type
    """
    wallets = []
    for account in user_data.get("linked_accounts", []):
        account_type = account.get("type", "")
        # Check for embedded wallet types
        if (
            "embedded_wallet" in account_type.lower()
            or account.get("connector_type") == "embedded"
        ):
            wallets.append(
                {
                    "wallet_id": account.get("id"),
                    "address": account.get("address") or account.get("public_key"),
                    "chain_type": account.get("chain_type") or account_type,
                    "delegated": account.get("delegated", False),
                }
            )
    return wallets


class PrivyGetUserByTelegramTool(AutoTool):
    """Tool for looking up a Privy user by Telegram ID."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="privy_get_user_by_telegram",
            description=(
                "Look up an existing Privy user by their Telegram user ID. "
                "Returns user info including 'user_id' (the Privy DID like 'did:privy:xxx') and linked wallets. "
                "IMPORTANT: Extract 'result.user_id' from the response and pass it to other privy_* tools."
            ),
            registry=registry,
        )
        self.app_id = None
        self.app_secret = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "telegram_user_id": {
                    "type": "string",
                    "description": "The Telegram user ID to look up.",
                }
            },
            "required": ["telegram_user_id"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        tool_cfg = config.get("tools", {}).get("privy_get_user_by_telegram", {})
        self.app_id = tool_cfg.get("app_id")
        self.app_secret = tool_cfg.get("app_secret")

    async def execute(self, telegram_user_id: str) -> Dict[str, Any]:
        if not all([self.app_id, self.app_secret]):
            return {
                "status": "error",
                "message": "Privy config missing (app_id, app_secret).",
            }

        try:
            user_data = await get_privy_user_by_telegram(
                telegram_user_id, self.app_id, self.app_secret
            )

            if user_data is None:
                return {
                    "status": "not_found",
                    "message": f"No Privy user found for Telegram ID: {telegram_user_id}",
                }

            wallets = extract_wallet_info(user_data)

            return {
                "status": "success",
                "result": {
                    "user_id": user_data.get("id"),
                    "created_at": user_data.get("created_at"),
                    "wallets": wallets,
                    "has_wallet": len(wallets) > 0,
                },
            }
        except httpx.HTTPStatusError as e:  # pragma: no cover
            logging.exception(f"Privy get user by telegram failed: {str(e)}")
            return {
                "status": "error",
                "message": f"HTTP error: {e.response.status_code} - {e.response.text}",
            }
        except Exception as e:  # pragma: no cover
            logging.exception(f"Privy get user by telegram failed: {str(e)}")
            return {"status": "error", "message": str(e)}


class PrivyGetUserByTelegramPlugin:
    """Plugin for looking up Privy users by Telegram ID."""

    def __init__(self):
        self.name = "privy_get_user_by_telegram"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for looking up Privy users by their Telegram user ID."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = PrivyGetUserByTelegramTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return PrivyGetUserByTelegramPlugin()

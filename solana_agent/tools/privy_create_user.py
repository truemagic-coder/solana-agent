"""
Privy Create User Tool.

Creates a new Privy user with a linked Telegram account.
Used for bot-first Telegram bot flows where wallets are created server-side.
"""

import base64
import logging
from typing import Dict, Any, List, Optional
from solana_agent import AutoTool, ToolRegistry
import httpx


async def create_privy_user_with_telegram(  # pragma: no cover
    telegram_user_id: str, app_id: str, app_secret: str
) -> Dict[str, Any]:
    """
    Create a new Privy user with a linked Telegram account.

    Args:
        telegram_user_id: The Telegram user ID
        app_id: Privy app ID
        app_secret: Privy app secret

    Returns:
        Dict with user data including id and linked_accounts
    """
    url = "https://api.privy.io/v1/users"
    auth_string = f"{app_id}:{app_secret}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "privy-app-id": app_id,
        "Content-Type": "application/json",
    }

    body = {
        "linked_accounts": [{"type": "telegram", "telegram_user_id": telegram_user_id}]
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            logging.error(f"Privy create user error: {resp.text}")
            resp.raise_for_status()
        return resp.json()


class PrivyCreateUserTool(AutoTool):
    """Tool for creating a new Privy user with Telegram linked account."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="privy_create_user",
            description="Create a new Privy user with a linked Telegram account. Used for bot-first Telegram bot flows.",
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
                    "description": "The Telegram user ID to link to the new Privy user.",
                }
            },
            "required": ["telegram_user_id"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        tool_cfg = config.get("tools", {}).get("privy_create_user", {})
        self.app_id = tool_cfg.get("app_id")
        self.app_secret = tool_cfg.get("app_secret")

    async def execute(self, telegram_user_id: str) -> Dict[str, Any]:
        if not all([self.app_id, self.app_secret]):
            return {
                "status": "error",
                "message": "Privy config missing (app_id, app_secret).",
            }

        try:
            user_data = await create_privy_user_with_telegram(
                telegram_user_id, self.app_id, self.app_secret
            )
            return {
                "status": "success",
                "result": {
                    "user_id": user_data.get("id"),
                    "created_at": user_data.get("created_at"),
                    "linked_accounts": user_data.get("linked_accounts", []),
                },
            }
        except httpx.HTTPStatusError as e:  # pragma: no cover
            logging.exception(f"Privy create user failed: {str(e)}")
            return {
                "status": "error",
                "message": f"HTTP error: {e.response.status_code} - {e.response.text}",
            }
        except Exception as e:  # pragma: no cover
            logging.exception(f"Privy create user failed: {str(e)}")
            return {"status": "error", "message": str(e)}


class PrivyCreateUserPlugin:
    """Plugin for creating Privy users with Telegram accounts."""

    def __init__(self):
        self.name = "privy_create_user"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for creating Privy users with linked Telegram accounts."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = PrivyCreateUserTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return PrivyCreateUserPlugin()

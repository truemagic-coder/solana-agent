"""
Jupiter Holdings tool.

Gets token holdings for a wallet using Jupiter Ultra API.
"""

import logging
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry

from solana_agent.tools.utils.ultra import JupiterUltra

logger = logging.getLogger(__name__)


class JupiterHoldingsTool(AutoTool):
    """Get token holdings for a wallet using Jupiter Ultra API."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="jupiter_holdings",
            description="Get detailed token holdings for a Solana wallet address, including native SOL balance and all token balances with metadata.",
            registry=registry,
        )
        self._jupiter_api_key: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "wallet_address": {
                    "type": "string",
                    "description": "The Solana wallet address to get holdings for.",
                },
                "native_only": {
                    "type": "boolean",
                    "description": "If true, only returns native SOL balance (faster). Default is false.",
                    "default": False,
                },
            },
            "required": ["wallet_address", "native_only"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("jupiter_holdings", {})
        self._jupiter_api_key = tool_cfg.get("jupiter_api_key")

    async def execute(
        self,
        wallet_address: str,
        native_only: bool = False,
    ) -> Dict[str, Any]:
        try:
            ultra = JupiterUltra(api_key=self._jupiter_api_key)

            if native_only:
                holdings = await ultra.get_native_holdings(wallet_address)
            else:
                holdings = await ultra.get_holdings(wallet_address)

            return {
                "status": "success",
                "holdings": holdings,
            }

        except Exception as e:
            logger.exception(f"Failed to get holdings: {str(e)}")
            return {"status": "error", "message": str(e)}


class JupiterHoldingsPlugin:
    """Plugin for getting token holdings via Jupiter Ultra API."""

    def __init__(self):
        self.name = "jupiter_holdings"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for getting token holdings using Jupiter Ultra API."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = JupiterHoldingsTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return JupiterHoldingsPlugin()

"""
Jupiter Shield tool.

Gets security warnings for token mints using Jupiter Ultra API.
"""

import logging
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry

from solana_agent.tools.utils.ultra import JupiterUltra

logger = logging.getLogger(__name__)


class JupiterShieldTool(AutoTool):
    """Get security warnings for tokens using Jupiter Shield API."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="jupiter_shield",
            description="Check token security by getting warnings for token mint addresses. Useful for detecting potentially malicious tokens before trading.",
            registry=registry,
        )
        self._jupiter_api_key: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of token mint addresses to check for security warnings.",
                },
            },
            "required": ["mints"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("jupiter_shield", {})
        self._jupiter_api_key = tool_cfg.get("jupiter_api_key")

    async def execute(
        self,
        mints: List[str],
    ) -> Dict[str, Any]:
        if not mints:
            return {"status": "error", "message": "No mints provided."}

        try:
            ultra = JupiterUltra(api_key=self._jupiter_api_key)
            shield_data = await ultra.get_shield(mints)

            # Process warnings into a more readable format
            warnings_summary = {}
            for mint, warnings in shield_data.get("warnings", {}).items():
                if warnings:
                    warnings_summary[mint] = {
                        "has_warnings": True,
                        "warning_count": len(warnings),
                        "warnings": [
                            {
                                "type": w.get("type"),
                                "message": w.get("message"),
                                "severity": w.get("severity"),
                            }
                            for w in warnings
                        ],
                    }
                else:
                    warnings_summary[mint] = {
                        "has_warnings": False,
                        "warning_count": 0,
                        "warnings": [],
                    }

            return {
                "status": "success",
                "shield": warnings_summary,
                "raw_response": shield_data,
            }

        except Exception as e:
            logger.exception(f"Failed to get shield data: {str(e)}")
            return {"status": "error", "message": str(e)}


class JupiterShieldPlugin:
    """Plugin for checking token security via Jupiter Shield API."""

    def __init__(self):
        self.name = "jupiter_shield"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for checking token security using Jupiter Shield API."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = JupiterShieldTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return JupiterShieldPlugin()

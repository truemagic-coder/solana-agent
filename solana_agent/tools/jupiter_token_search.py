"""
Jupiter Token Search tool.

Searches for tokens by symbol, name, or mint address using Jupiter Ultra API.
"""

import logging
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry

from solana_agent.tools.utils.ultra import JupiterUltra

logger = logging.getLogger(__name__)


class JupiterTokenSearchTool(AutoTool):
    """Search for tokens using Jupiter Ultra API."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="jupiter_token_search",
            description="Search for Solana tokens by symbol, name, or mint address. Returns detailed token information including price, market cap, liquidity, and trading stats.",
            registry=registry,
        )
        self._jupiter_api_key: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query - can be token symbol (e.g., 'SOL'), name (e.g., 'Jupiter'), or mint address. Can be comma-separated for multiple searches.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        tool_cfg = config.get("tools", {}).get("jupiter_token_search", {})
        self._jupiter_api_key = tool_cfg.get("jupiter_api_key")

    async def execute(
        self,
        query: str,
    ) -> Dict[str, Any]:
        if not query:
            return {"status": "error", "message": "No query provided."}

        try:
            ultra = JupiterUltra(api_key=self._jupiter_api_key)
            tokens = await ultra.search_tokens(query)

            # Format results for better readability
            formatted_tokens = []
            for token in tokens:
                formatted_token = {
                    "mint": token.get("id"),
                    "name": token.get("name"),
                    "symbol": token.get("symbol"),
                    "decimals": token.get("decimals"),
                    "icon": token.get("icon"),
                    "is_verified": token.get("isVerified", False),
                    "price_usd": token.get("usdPrice"),
                    "market_cap": token.get("mcap"),
                    "fdv": token.get("fdv"),
                    "liquidity": token.get("liquidity"),
                    "holder_count": token.get("holderCount"),
                    "organic_score": token.get("organicScore"),
                    "organic_score_label": token.get("organicScoreLabel"),
                    "tags": token.get("tags", []),
                    "cexes": token.get("cexes", []),
                }

                # Add audit info if available
                audit = token.get("audit", {})
                if audit:
                    formatted_token["audit"] = {
                        "mint_authority_disabled": audit.get("mintAuthorityDisabled"),
                        "freeze_authority_disabled": audit.get(
                            "freezeAuthorityDisabled"
                        ),
                        "top_holders_percentage": audit.get("topHoldersPercentage"),
                    }

                # Add 24h stats if available
                stats_24h = token.get("stats24h", {})
                if stats_24h:
                    formatted_token["stats_24h"] = {
                        "price_change": stats_24h.get("priceChange"),
                        "volume_change": stats_24h.get("volumeChange"),
                        "buy_volume": stats_24h.get("buyVolume"),
                        "sell_volume": stats_24h.get("sellVolume"),
                        "num_buys": stats_24h.get("numBuys"),
                        "num_sells": stats_24h.get("numSells"),
                        "num_traders": stats_24h.get("numTraders"),
                    }

                formatted_tokens.append(formatted_token)

            return {
                "status": "success",
                "count": len(formatted_tokens),
                "tokens": formatted_tokens,
            }

        except Exception as e:
            logger.exception(f"Failed to search tokens: {str(e)}")
            return {"status": "error", "message": str(e)}


class JupiterTokenSearchPlugin:
    """Plugin for searching tokens via Jupiter Ultra API."""

    def __init__(self):
        self.name = "jupiter_token_search"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for searching tokens using Jupiter Ultra API."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = JupiterTokenSearchTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return JupiterTokenSearchPlugin()

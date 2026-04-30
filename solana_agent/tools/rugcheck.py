import logging
from typing import Dict, Any, List, Optional
import httpx
from solana_agent import AutoTool, ToolRegistry


def summarize_rugcheck(data: dict) -> str:
    """Summarize the most important info for LLM consumption."""
    lines = []
    meta = data.get("tokenMeta", {})
    file_meta = data.get("fileMeta", {})
    lines.append(f"Token Name: {meta.get('name') or file_meta.get('name')}")
    lines.append(f"Symbol: {meta.get('symbol') or file_meta.get('symbol')}")
    lines.append(f"Mint: {data.get('mint')}")
    lines.append(
        f"Score: {data.get('score')} (normalized: {data.get('score_normalised')})"
    )
    lines.append(f"Rugged: {data.get('rugged')}")
    lines.append(f"Total Holders: {data.get('totalHolders')}")
    lines.append(f"Total Market Liquidity (USD): {data.get('totalMarketLiquidity')}")
    lines.append(f"Price: {data.get('price')}")
    lines.append(
        f"Verified on Jupiter: {data.get('verification', {}).get('jup_verified')}"
    )
    lines.append(f"Creator: {data.get('creator')}")
    lines.append("Top 3 Holders:")
    for i, holder in enumerate(data.get("topHolders", [])[:3], 1):
        lines.append(
            f"  {i}. Address: {holder['address']} - {holder['pct']:.2f}% - Insider: {holder['insider']}"
        )
    if data.get("risks"):
        lines.append("Risks:")
        for risk in data["risks"]:
            lines.append(f"  - {risk}")
    else:
        lines.append("Risks: None detected.")
    if data.get("markets"):
        lines.append("Markets:")
        for m in data["markets"]:
            lines.append(f"  - {m.get('marketType')} (pubkey: {m.get('pubkey')})")
    return "\n".join(lines)


class RugCheckTool(AutoTool):
    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="rugcheck",
            description="Check Solana token risk and liquidity using rugcheck.xyz.",
            registry=registry,
        )

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mint": {
                    "type": "string",
                    "description": "The SPL token mint address to check.",
                }
            },
            "required": ["mint"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        # No config needed for rugcheck

    async def execute(self, mint: str) -> Dict[str, Any]:
        url = f"https://api.rugcheck.xyz/v1/tokens/{mint}/report"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logging.error(f"Rugcheck API error: {resp.status_code} {resp.text}")
                    return {
                        "status": "error",
                        "message": f"Rugcheck API error: {resp.status_code}",
                        "details": resp.text,
                    }
                data = resp.json()
                summary = summarize_rugcheck(data)
                return {
                    "status": "success",
                    "result": summary,
                }
        except Exception as e:
            logging.exception(f"Rugcheck error: {e}")
            return {"status": "error", "message": str(e)}


class RugCheckPlugin:
    def __init__(self):
        self.name = "rugcheck"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for checking Solana token risk and liquidity using rugcheck.xyz."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = RugCheckTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return RugCheckPlugin()

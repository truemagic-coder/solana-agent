"""
Vybe Network tool for labeling known Solana accounts.

This tool fetches and caches known accounts from Vybe Network API,
allowing bulk lookup of wallet addresses to identify CEX wallets,
market makers, AMM pools, project treasuries, and influencers.
"""

import logging
import time
from typing import Any

import httpx
from solana_agent import AutoTool, ToolRegistry

# Module-level cache for known accounts (shared across tool instances)
_known_accounts_cache: dict[str, dict[str, Any]] = {}
_cache_timestamp: float = 0
_CACHE_TTL_SECONDS: int = 3600  # 1 hour cache TTL


class VybeTool(AutoTool):
    """Tool for looking up known Solana account labels via Vybe Network."""

    def __init__(self, registry: ToolRegistry | None = None):
        super().__init__(
            name="vybe",
            description=(
                "Look up labels for Solana wallet addresses. Identifies CEX wallets, "
                "market makers, AMM pools, project treasuries, and influencers. "
                "Use this to understand who owns wallets when analyzing top holders or traders."
            ),
            registry=registry,
        )
        self._api_key: str | None = None
        self._base_url = "https://api.vybenetwork.xyz"

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "addresses": {
                    "type": "string",
                    "description": (
                        "Comma-separated list of Solana wallet addresses to look up. "
                        "Example: 'addr1,addr2,addr3'. Returns labels for any known addresses."
                    ),
                },
                "refresh_cache": {
                    "type": "boolean",
                    "description": (
                        "Force refresh the known accounts cache. "
                        "Default is false. Only set to true if you need fresh data."
                    ),
                    "default": False,
                },
            },
            "required": ["addresses", "refresh_cache"],
            "additionalProperties": False,
        }

    def configure(self, config: dict[str, Any]) -> None:
        super().configure(config)
        tool_config = config.get("tools", {}).get("vybe", {})
        self._api_key = tool_config.get("api_key")

    async def _fetch_known_accounts(self) -> dict[str, dict[str, Any]]:
        """Fetch all known accounts from Vybe API and return as address->info dict."""
        global _known_accounts_cache, _cache_timestamp

        headers = {"accept": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self._base_url}/account/known-accounts",
                headers=headers,
            )
            if resp.status_code != 200:
                logging.error(f"Vybe API error: {resp.status_code} {resp.text}")
                raise Exception(f"Vybe API error: {resp.status_code}")

            data = resp.json()

            # Build address lookup dict
            # Response format: {"data": [{"ownerAddress": "...", "name": "...", "labels": [...], ...}]}
            # Or directly a list: [{"ownerAddress": "...", ...}]
            if isinstance(data, list):
                accounts = data
            else:
                accounts = data.get("data", [])

            address_map: dict[str, dict[str, Any]] = {}
            for account in accounts:
                addr = account.get("ownerAddress") or account.get("address")
                if addr:
                    address_map[addr] = {
                        "name": account.get("name") or account.get("entityName"),
                        "labels": account.get("labels", []),
                        "entity_id": account.get("entityId"),
                        "entity_name": account.get("entityName"),
                        "type": account.get("type"),
                    }

            # Update cache
            _known_accounts_cache = address_map
            _cache_timestamp = time.time()

            return address_map

    async def _get_known_accounts(
        self, refresh: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Get known accounts, using cache if available and not expired."""
        global _known_accounts_cache, _cache_timestamp

        now = time.time()
        cache_expired = (now - _cache_timestamp) > _CACHE_TTL_SECONDS

        if refresh or cache_expired or not _known_accounts_cache:
            return await self._fetch_known_accounts()

        return _known_accounts_cache

    async def execute(
        self,
        addresses: str,
        refresh_cache: bool = False,
    ) -> dict[str, Any]:
        """Look up labels for the given wallet addresses."""
        if not self._api_key:
            return {
                "success": False,
                "error": "Vybe API key is required. Configure tools.vybe.api_key",
            }

        # Parse addresses
        address_list = [addr.strip() for addr in addresses.split(",") if addr.strip()]

        if not address_list:
            return {
                "success": False,
                "error": "No valid addresses provided",
            }

        try:
            known_accounts = await self._get_known_accounts(refresh=refresh_cache)

            # Look up each address
            results: list[dict[str, Any]] = []
            known_count = 0
            unknown_count = 0

            for addr in address_list:
                if addr in known_accounts:
                    info = known_accounts[addr]
                    results.append(
                        {
                            "address": addr,
                            "known": True,
                            "name": info["name"],
                            "labels": info["labels"],
                            "entity_name": info["entity_name"],
                            "type": info["type"],
                        }
                    )
                    known_count += 1
                else:
                    results.append(
                        {
                            "address": addr,
                            "known": False,
                            "name": None,
                            "labels": [],
                            "entity_name": None,
                            "type": None,
                        }
                    )
                    unknown_count += 1

            # Build summary for LLM
            summary_lines = [
                f"Looked up {len(address_list)} addresses: {known_count} known, {unknown_count} unknown."
            ]

            for r in results:
                if r["known"]:
                    labels_str = ", ".join(r["labels"]) if r["labels"] else "no labels"
                    name = r["name"] or r["entity_name"] or "Unknown name"
                    summary_lines.append(
                        f"  {r['address'][:8]}...{r['address'][-4:]}: {name} ({labels_str})"
                    )
                else:
                    summary_lines.append(
                        f"  {r['address'][:8]}...{r['address'][-4:]}: Unknown wallet"
                    )

            return {
                "success": True,
                "summary": "\n".join(summary_lines),
                "results": results,
                "known_count": known_count,
                "unknown_count": unknown_count,
                "cache_age_seconds": int(time.time() - _cache_timestamp),
            }

        except Exception as e:
            logging.exception(f"Vybe lookup error: {e}")
            return {
                "success": False,
                "error": str(e),
            }


class VybePlugin:
    """Plugin for Vybe Network known account lookups."""

    def __init__(self):
        self.name = "vybe"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):  # pragma: no cover
        return "Plugin for looking up known Solana account labels via Vybe Network."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = VybeTool(registry=tool_registry)

    def configure(self, config: dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> list[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return VybePlugin()

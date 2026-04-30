"""Birdeye Tool - Comprehensive Solana token analytics and wallet data."""

import logging
from typing import Any, Dict, Optional

import httpx
from solana_agent import AutoTool, ToolRegistry
from solana_agent.tools.utils.x402 import (
    has_x402_auth_config,
    request_with_x402_private_key,
    request_with_x402_privy,
    resolve_x402_private_key,
)

logger = logging.getLogger(__name__)


class BirdeyeTool(AutoTool):
    """
    Comprehensive Birdeye API integration for Solana token analytics.

    Provides 50 actions covering:
    - Price data (single, multi, historical, OHLCV)
    - Token analytics (overview, security, metadata, market data, trade data, holders, trending)
    - Trade history (by token, by pair, with time bounds)
    - Wallet analytics (portfolio, balance, transactions, PNL)
    - Trader analytics (gainers/losers, trade history)
    - Pair data (overview single/multiple)
    - Search functionality
    - Network utilities
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="birdeye",
            description=(
                "Comprehensive Birdeye API for Solana token analytics and wallet data. "
                "Use action parameter to specify what data to fetch. "
                "Actions include: price, multi_price, history_price, ohlcv, token_overview, "
                "token_holder, token_trending, token_security, wallet_pnl_summary, search, and more."
            ),
            registry=registry,
        )
        self.base_url = "https://public-api.birdeye.so"
        self.api_key = ""
        self.auth_mode = "api_key"
        self.private_key = ""
        self.privy_app_id = ""
        self.privy_app_secret = ""
        self.privy_authorization_signature = ""
        self.privy_request_expiry = ""
        self.privy_api_url = ""
        self.x402_rpc_url = None
        self.default_chain = "solana"
        self._runtime_context: Dict[str, Any] = {}

    def set_runtime_context(self, runtime_context: Optional[Dict[str, Any]]) -> None:
        self._runtime_context = dict(runtime_context or {})

    def clear_runtime_context(self) -> None:
        self._runtime_context = {}

    def _get_runtime_privy_wallet_id(self) -> str:
        return str(self._runtime_context.get("privy_wallet_id") or "").strip()

    def _is_x402_mode(self) -> bool:
        """Return whether the configured auth mode uses x402."""
        return self.auth_mode in {"x402_private_key", "x402_privy"}

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "The action to perform. Available actions: "
                        "PRICE: price, multi_price, history_price, historical_price_unix, price_volume_single, price_volume_multi. "
                        "OHLCV: ohlcv, ohlcv_pair, ohlcv_base_quote, ohlcv_v3, ohlcv_pair_v3. "
                        "TRADES: trades_token, trades_pair, trades_token_seek, trades_pair_seek, trades_v3, trades_token_v3. "
                        "TOKEN: token_list, token_list_v3, token_list_scroll, token_overview, token_metadata_single, "
                        "token_metadata_multiple, token_market_data, token_market_data_multiple, token_trade_data_single, "
                        "token_trade_data_multiple, token_holder, token_trending, token_new_listing, token_top_traders, "
                        "token_markets, token_security, token_creation_info, token_mint_burn, token_all_time_trades_single, "
                        "token_all_time_trades_multiple, token_exit_liquidity, token_exit_liquidity_multiple. "
                        "PAIR: pair_overview_single, pair_overview_multiple. "
                        "TRADER: trader_gainers_losers, trader_txs_seek. "
                        "WALLET: wallet_token_list, wallet_token_balance, wallet_tx_list, wallet_balance_change, "
                        "wallet_pnl_summary, wallet_pnl_details, wallet_pnl_multiple, wallet_current_net_worth, "
                        "wallet_net_worth, wallet_net_worth_details. "
                        "SEARCH: search. "
                        "UTILS: latest_block, networks, supported_chains."
                    ),
                },
                "address": {
                    "type": "string",
                    "description": "Token or pair address. Required for most actions. Pass empty string if not needed.",
                    "default": "",
                },
                "wallet": {
                    "type": "string",
                    "description": "Wallet address. Required for wallet_* actions. Pass empty string if not needed.",
                    "default": "",
                },
                "keyword": {
                    "type": "string",
                    "description": "Search keyword for the 'search' action. Pass empty string if not needed.",
                    "default": "",
                },
                "list_address": {
                    "type": "string",
                    "description": "Comma-separated list of addresses for multi_* actions. Pass empty string if not needed.",
                    "default": "",
                },
                "type": {
                    "type": "string",
                    "description": "Time interval type (1m, 5m, 15m, 30m, 1H, 4H, 1D, 1W). Pass empty string if not needed.",
                    "default": "",
                },
                "time_from": {
                    "type": "integer",
                    "description": "Start time as Unix timestamp. Pass 0 if not needed.",
                    "default": 0,
                },
                "time_to": {
                    "type": "integer",
                    "description": "End time as Unix timestamp. Pass 0 if not needed.",
                    "default": 0,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Pass 0 if not needed.",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return. Pass 0 to use default.",
                    "default": 0,
                },
                "chain": {
                    "type": "string",
                    "description": "Blockchain network (default: solana). Pass empty string for default.",
                    "default": "",
                },
                "token_address": {
                    "type": "string",
                    "description": "Token address for wallet_token_balance. Pass empty string if not needed.",
                    "default": "",
                },
                "base_address": {
                    "type": "string",
                    "description": "Base token address for ohlcv_base_quote. Pass empty string if not needed.",
                    "default": "",
                },
                "quote_address": {
                    "type": "string",
                    "description": "Quote token address for ohlcv_base_quote. Pass empty string if not needed.",
                    "default": "",
                },
                "unixtime": {
                    "type": "integer",
                    "description": "Unix timestamp for historical_price_unix. Pass 0 if not needed.",
                    "default": 0,
                },
                "before_time": {
                    "type": "integer",
                    "description": "Before time filter for seek actions. Pass 0 if not needed.",
                    "default": 0,
                },
                "after_time": {
                    "type": "integer",
                    "description": "After time filter for seek actions. Pass 0 if not needed.",
                    "default": 0,
                },
                "tx_type": {
                    "type": "string",
                    "description": "Transaction type filter (buy, sell, all). Pass empty string if not needed.",
                    "default": "",
                },
                "time_frame": {
                    "type": "string",
                    "description": "Time frame for top traders (24h, 7d, 30d). Pass empty string if not needed.",
                    "default": "",
                },
                "owner": {
                    "type": "string",
                    "description": "Owner address for trades_v3. Pass empty string if not needed.",
                    "default": "",
                },
                "min_liquidity": {
                    "type": "integer",
                    "description": "Minimum liquidity filter. Pass 0 if not needed.",
                    "default": 0,
                },
                "wallets": {
                    "type": "string",
                    "description": "Comma-separated wallet addresses for wallet_pnl_multiple. Pass empty string if not needed.",
                    "default": "",
                },
                "tokens": {
                    "type": "string",
                    "description": "Comma-separated token addresses for wallet_pnl_details. Pass empty string if not needed.",
                    "default": "",
                },
                "time": {
                    "type": "string",
                    "description": "ISO 8601 UTC time for wallet_net_worth. Pass empty string if not needed.",
                    "default": "",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of time periods for wallet_net_worth. Pass 0 if not needed.",
                    "default": 0,
                },
                "direction": {
                    "type": "string",
                    "description": "Direction for wallet_net_worth (back, forward). Pass empty string if not needed.",
                    "default": "",
                },
            },
            "required": [
                "action",
                "address",
                "wallet",
                "keyword",
                "list_address",
                "type",
                "time_from",
                "time_to",
                "offset",
                "limit",
                "chain",
                "token_address",
                "base_address",
                "quote_address",
                "unixtime",
                "before_time",
                "after_time",
                "tx_type",
                "time_frame",
                "owner",
                "min_liquidity",
                "wallets",
                "tokens",
                "time",
                "count",
                "direction",
            ],
            "additionalProperties": False,
        }

    def configure(self, config: dict) -> None:
        """Configure the tool with API key from config."""
        super().configure(config)
        # Get API key and chain from tools.birdeye config
        tools_config = config.get("tools", {})
        birdeye_config = tools_config.get("birdeye", {})
        if isinstance(birdeye_config, dict):
            self.api_key = birdeye_config.get("api_key", "")
            self.auth_mode = birdeye_config.get("auth_mode", "api_key")
            self.private_key = birdeye_config.get("private_key", "")
            self.privy_app_id = birdeye_config.get(
                "privy_app_id", ""
            ) or birdeye_config.get("app_id", "")
            self.privy_app_secret = birdeye_config.get(
                "privy_app_secret", ""
            ) or birdeye_config.get("app_secret", "")
            self.privy_authorization_signature = birdeye_config.get(
                "privy_authorization_signature", ""
            )
            self.privy_request_expiry = birdeye_config.get("privy_request_expiry", "")
            self.privy_api_url = birdeye_config.get("privy_api_url", "")
            self.x402_rpc_url = birdeye_config.get("x402_rpc_url")
            if birdeye_config.get("chain"):
                self.default_chain = birdeye_config.get("chain")

    def _resolve_url(self, endpoint: str) -> str:
        """Resolve the final Birdeye URL for the configured auth mode."""
        if self._is_x402_mode():
            if endpoint.startswith("/x402/"):
                return f"{self.base_url}{endpoint}"
            return f"{self.base_url}/x402{endpoint}"
        return f"{self.base_url}{endpoint}"

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict = None,
        json_data: dict = None,
        chain: str = "solana",
    ) -> dict:
        """Make authenticated request to Birdeye API."""
        runtime_privy_wallet_id = self._get_runtime_privy_wallet_id()

        if self._is_x402_mode() and not has_x402_auth_config(
            auth_mode=self.auth_mode,
            private_key=self.private_key,
            privy_wallet_id=runtime_privy_wallet_id,
            privy_app_id=self.privy_app_id,
            privy_app_secret=self.privy_app_secret,
            privy_authorization_signature=self.privy_authorization_signature,
            privy_request_expiry=self.privy_request_expiry,
            privy_api_url=self.privy_api_url or None,
            rpc_url=self.x402_rpc_url,
        ):
            if self.auth_mode == "x402_private_key":
                error_message = "Birdeye x402 private key not configured. Set birdeye.private_key in config."
            else:
                error_message = (
                    "Birdeye x402 signing key not configured. Pass runtime_context.privy_wallet_id "
                    "and set birdeye.privy_app_id and birdeye.privy_app_secret in config."
                )
            return {
                "success": False,
                "error": error_message,
            }

        resolved_private_key = resolve_x402_private_key(
            auth_mode=self.auth_mode,
            private_key=self.private_key,
        )
        if not self._is_x402_mode() and not self.api_key:
            return {
                "success": False,
                "error": "Birdeye API key not configured. Set birdeye.api_key in config.",
            }

        headers = {
            "Accept": "application/json",
            "x-chain": chain,
        }
        if self.auth_mode == "api_key":
            headers["X-API-KEY"] = self.api_key

        url = self._resolve_url(endpoint)

        try:
            if self.auth_mode == "x402_private_key":
                response = await request_with_x402_private_key(
                    method=method,
                    url=url,
                    private_key=resolved_private_key,
                    headers=headers,
                    params=params,
                    json_data=json_data,
                    timeout=30.0,
                    rpc_url=self.x402_rpc_url,
                )
            elif self.auth_mode == "x402_privy":
                response = await request_with_x402_privy(
                    method=method,
                    url=url,
                    privy_wallet_id=runtime_privy_wallet_id,
                    privy_app_id=self.privy_app_id,
                    privy_app_secret=self.privy_app_secret,
                    privy_authorization_signature=self.privy_authorization_signature
                    or None,
                    privy_request_expiry=self.privy_request_expiry or None,
                    privy_api_url=self.privy_api_url or None,
                    headers=headers,
                    params=params,
                    json_data=json_data,
                    timeout=30.0,
                    rpc_url=self.x402_rpc_url,
                )
            else:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    if method.upper() == "GET":
                        response = await client.get(url, headers=headers, params=params)
                    else:
                        response = await client.post(
                            url, headers=headers, params=params, json=json_data
                        )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "details": response.text,
                }

            data = response.json()
            return {
                "success": True,
                "data": data.get("data", data),
                "payment_mode": self.auth_mode,
            }
        except Exception as e:  # pragma: no cover
            return {"success": False, "error": str(e)}

    async def execute(  # pragma: no cover
        self,
        action: str,
        address: str = "",
        wallet: str = "",
        keyword: str = "",
        list_address: str = "",
        type: str = "",
        time_from: int = 0,
        time_to: int = 0,
        offset: int = 0,
        limit: int = 0,
        chain: str = "",
        token_address: str = "",
        base_address: str = "",
        quote_address: str = "",
        unixtime: int = 0,
        before_time: int = 0,
        after_time: int = 0,
        tx_type: str = "",
        time_frame: str = "",
        owner: str = "",
        min_liquidity: int = 0,
        wallets: str = "",
        tokens: str = "",
        time: str = "",
        count: int = 0,
        direction: str = "",
    ) -> Dict[str, Any]:
        """Execute a Birdeye action."""
        # Use default chain if not specified
        chain = chain or self.default_chain

        # Build kwargs from the schema parameters for the internal action handlers
        kwargs: Dict[str, Any] = {}
        if address:
            kwargs["address"] = address
        if wallet:
            kwargs["wallet"] = wallet
        if keyword:
            kwargs["keyword"] = keyword
        if list_address:
            kwargs["list_address"] = list_address
        if type:
            kwargs["type"] = type
        if time_from:
            kwargs["time_from"] = time_from
        if time_to:
            kwargs["time_to"] = time_to
        if offset:
            kwargs["offset"] = offset
        if limit:
            kwargs["limit"] = limit
        if token_address:
            kwargs["token_address"] = token_address
        if base_address:
            kwargs["base_address"] = base_address
        if quote_address:
            kwargs["quote_address"] = quote_address
        if unixtime:
            kwargs["unixtime"] = unixtime
        if before_time:
            kwargs["before_time"] = before_time
        if after_time:
            kwargs["after_time"] = after_time
        if tx_type:
            kwargs["tx_type"] = tx_type
        if time_frame:
            kwargs["time_frame"] = time_frame
        if owner:
            kwargs["owner"] = owner
        if min_liquidity:
            kwargs["min_liquidity"] = min_liquidity
        if wallets:
            kwargs["wallets"] = wallets
        if tokens:
            kwargs["tokens"] = tokens
        if time:
            kwargs["time"] = time
        if count:
            kwargs["count"] = count
        if direction:
            kwargs["direction"] = direction

        # ==================== PRICE ====================
        if action == "price":
            # Get current price of a token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("check_liquidity"):
                params["check_liquidity"] = kwargs["check_liquidity"]
            if kwargs.get("include_liquidity"):
                params["include_liquidity"] = kwargs["include_liquidity"]
            return await self._request("GET", "/defi/price", params=params, chain=chain)

        elif action == "multi_price":
            # Get prices for multiple tokens (GET method)
            addresses = kwargs.get("list_address")
            if not addresses:
                return {"success": False, "error": "list_address is required"}
            params = {"list_address": addresses}
            if kwargs.get("check_liquidity"):
                params["check_liquidity"] = kwargs["check_liquidity"]
            if kwargs.get("include_liquidity"):
                params["include_liquidity"] = kwargs["include_liquidity"]
            return await self._request(
                "GET", "/defi/multi_price", params=params, chain=chain
            )

        elif action == "multi_price_post":
            # Get prices for multiple tokens (POST method)
            addresses = kwargs.get("list_address")
            if not addresses:
                return {"success": False, "error": "list_address is required"}
            json_data = {"list_address": addresses}
            params = {}
            if kwargs.get("check_liquidity"):
                params["check_liquidity"] = kwargs["check_liquidity"]
            if kwargs.get("include_liquidity"):
                params["include_liquidity"] = kwargs["include_liquidity"]
            return await self._request(
                "POST",
                "/defi/multi_price",
                params=params,
                json_data=json_data,
                chain=chain,
            )

        elif action == "history_price":
            # Get historical price data
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {
                "address": address,
                "address_type": kwargs.get("address_type", "token"),
                "type": kwargs.get("type", "1H"),
            }
            if kwargs.get("time_from"):
                params["time_from"] = kwargs["time_from"]
            if kwargs.get("time_to"):
                params["time_to"] = kwargs["time_to"]
            return await self._request(
                "GET", "/defi/history_price", params=params, chain=chain
            )

        elif action == "historical_price_unix":
            # Get historical price at specific unix timestamp
            address = kwargs.get("address")
            unixtime = kwargs.get("unixtime")
            if not address or not unixtime:
                return {"success": False, "error": "address and unixtime are required"}
            params = {"address": address, "unixtime": unixtime}
            return await self._request(
                "GET", "/defi/historical_price_unix", params=params, chain=chain
            )

        elif action == "price_volume_single":
            # Get price and volume for single token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("type"):
                params["type"] = kwargs["type"]
            return await self._request(
                "GET", "/defi/price_volume/single", params=params, chain=chain
            )

        elif action == "price_volume_multi":
            # Get price and volume for multiple tokens (POST)
            addresses = kwargs.get("list_address")
            if not addresses:
                return {"success": False, "error": "list_address is required"}
            json_data = {"list_address": addresses}
            params = {}
            if kwargs.get("type"):
                params["type"] = kwargs["type"]
            return await self._request(
                "POST",
                "/defi/price_volume/multi",
                params=params,
                json_data=json_data,
                chain=chain,
            )

        # ==================== OHLCV ====================
        elif action == "ohlcv":
            # Get OHLCV candlestick data for a token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {
                "address": address,
                "type": kwargs.get("type", "1H"),
            }
            if kwargs.get("time_from"):
                params["time_from"] = kwargs["time_from"]
            if kwargs.get("time_to"):
                params["time_to"] = kwargs["time_to"]
            return await self._request("GET", "/defi/ohlcv", params=params, chain=chain)

        elif action == "ohlcv_pair":
            # Get OHLCV data for a trading pair
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {
                "address": address,
                "type": kwargs.get("type", "1H"),
            }
            if kwargs.get("time_from"):
                params["time_from"] = kwargs["time_from"]
            if kwargs.get("time_to"):
                params["time_to"] = kwargs["time_to"]
            return await self._request(
                "GET", "/defi/ohlcv/pair", params=params, chain=chain
            )

        elif action == "ohlcv_base_quote":
            # Get OHLCV for base/quote pair
            base_address = kwargs.get("base_address")
            quote_address = kwargs.get("quote_address")
            if not base_address or not quote_address:
                return {
                    "success": False,
                    "error": "base_address and quote_address are required",
                }
            params = {
                "base_address": base_address,
                "quote_address": quote_address,
                "type": kwargs.get("type", "1H"),
            }
            if kwargs.get("time_from"):
                params["time_from"] = kwargs["time_from"]
            if kwargs.get("time_to"):
                params["time_to"] = kwargs["time_to"]
            return await self._request(
                "GET", "/defi/ohlcv/base_quote", params=params, chain=chain
            )

        elif action == "ohlcv_v3":
            # Get OHLCV v3 for token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {
                "address": address,
                "type": kwargs.get("type", "1H"),
            }
            if kwargs.get("time_from"):
                params["time_from"] = kwargs["time_from"]
            if kwargs.get("time_to"):
                params["time_to"] = kwargs["time_to"]
            if kwargs.get("currency"):
                params["currency"] = kwargs["currency"]
            return await self._request(
                "GET", "/defi/v3/ohlcv", params=params, chain=chain
            )

        elif action == "ohlcv_pair_v3":
            # Get OHLCV v3 for pair
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {
                "address": address,
                "type": kwargs.get("type", "1H"),
            }
            if kwargs.get("time_from"):
                params["time_from"] = kwargs["time_from"]
            if kwargs.get("time_to"):
                params["time_to"] = kwargs["time_to"]
            return await self._request(
                "GET", "/defi/v3/ohlcv/pair", params=params, chain=chain
            )

        # ==================== TRADES ====================
        elif action == "trades_token":
            # Get recent trades for a token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("tx_type"):
                params["tx_type"] = kwargs["tx_type"]
            if kwargs.get("sort_type"):
                params["sort_type"] = kwargs["sort_type"]
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/defi/txs/token", params=params, chain=chain
            )

        elif action == "trades_pair":
            # Get recent trades for a pair
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("tx_type"):
                params["tx_type"] = kwargs["tx_type"]
            if kwargs.get("sort_type"):
                params["sort_type"] = kwargs["sort_type"]
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/defi/txs/pair", params=params, chain=chain
            )

        elif action == "trades_token_seek":
            # Get trades for token with time bounds
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("tx_type"):
                params["tx_type"] = kwargs["tx_type"]
            if kwargs.get("before_time"):
                params["before_time"] = kwargs["before_time"]
            if kwargs.get("after_time"):
                params["after_time"] = kwargs["after_time"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/defi/txs/token/seek_by_time", params=params, chain=chain
            )

        elif action == "trades_pair_seek":
            # Get trades for pair with time bounds
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("tx_type"):
                params["tx_type"] = kwargs["tx_type"]
            if kwargs.get("before_time"):
                params["before_time"] = kwargs["before_time"]
            if kwargs.get("after_time"):
                params["after_time"] = kwargs["after_time"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/defi/txs/pair/seek_by_time", params=params, chain=chain
            )

        elif action == "trades_v3":
            # Get trades v3 with filters
            params = {}
            if kwargs.get("address"):
                params["address"] = kwargs["address"]
            if kwargs.get("owner"):
                params["owner"] = kwargs["owner"]
            if kwargs.get("tx_type"):
                params["tx_type"] = kwargs["tx_type"]
            if kwargs.get("before_time"):
                params["before_time"] = kwargs["before_time"]
            if kwargs.get("after_time"):
                params["after_time"] = kwargs["after_time"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/defi/v3/txs", params=params, chain=chain
            )

        elif action == "trades_token_v3":
            # Get token trades v3
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("tx_type"):
                params["tx_type"] = kwargs["tx_type"]
            if kwargs.get("before_time"):
                params["before_time"] = kwargs["before_time"]
            if kwargs.get("after_time"):
                params["after_time"] = kwargs["after_time"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            if kwargs.get("cursor"):
                params["cursor"] = kwargs["cursor"]
            return await self._request(
                "GET", "/defi/v3/token/txs", params=params, chain=chain
            )

        # ==================== TOKEN ====================
        elif action == "token_list":
            # Get list of tokens
            params = {}
            if kwargs.get("sort_by"):
                params["sort_by"] = kwargs["sort_by"]
            if kwargs.get("sort_type"):
                params["sort_type"] = kwargs["sort_type"]
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            if kwargs.get("min_liquidity"):
                params["min_liquidity"] = kwargs["min_liquidity"]
            return await self._request(
                "GET", "/defi/tokenlist", params=params, chain=chain
            )

        elif action == "token_list_v3":
            # Get token list v3
            params = {}
            if kwargs.get("sort_by"):
                params["sort_by"] = kwargs["sort_by"]
            if kwargs.get("sort_type"):
                params["sort_type"] = kwargs["sort_type"]
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            if kwargs.get("min_liquidity"):
                params["min_liquidity"] = kwargs["min_liquidity"]
            if kwargs.get("min_volume_24h_usd"):
                params["min_volume_24h_usd"] = kwargs["min_volume_24h_usd"]
            if kwargs.get("min_market_cap"):
                params["min_market_cap"] = kwargs["min_market_cap"]
            return await self._request(
                "GET", "/defi/v3/token/list", params=params, chain=chain
            )

        elif action == "token_list_scroll":
            # Get token list with scroll pagination
            params = {}
            if kwargs.get("sort_by"):
                params["sort_by"] = kwargs["sort_by"]
            if kwargs.get("sort_type"):
                params["sort_type"] = kwargs["sort_type"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            if kwargs.get("min_liquidity"):
                params["min_liquidity"] = kwargs["min_liquidity"]
            if kwargs.get("scroll_id"):
                params["scroll_id"] = kwargs["scroll_id"]
            return await self._request(
                "GET", "/defi/v3/token/list/scroll", params=params, chain=chain
            )

        elif action == "token_overview":
            # Get detailed token overview
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            return await self._request(
                "GET", "/defi/token_overview", params=params, chain=chain
            )

        elif action == "token_metadata_single":
            # Get metadata for single token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            return await self._request(
                "GET", "/defi/v3/token/meta-data/single", params=params, chain=chain
            )

        elif action == "token_metadata_multiple":
            # Get metadata for multiple tokens
            addresses = kwargs.get("list_address")
            if not addresses:
                return {"success": False, "error": "list_address is required"}
            params = {"list_address": addresses}
            return await self._request(
                "GET", "/defi/v3/token/meta-data/multiple", params=params, chain=chain
            )

        elif action == "token_market_data":
            # Get market data for single token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            return await self._request(
                "GET", "/defi/v3/token/market-data", params=params, chain=chain
            )

        elif action == "token_market_data_multiple":
            # Get market data for multiple tokens
            addresses = kwargs.get("list_address")
            if not addresses:
                return {"success": False, "error": "list_address is required"}
            params = {"list_address": addresses}
            return await self._request(
                "GET", "/defi/v3/token/market-data/multiple", params=params, chain=chain
            )

        elif action == "token_trade_data_single":
            # Get trade data for single token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            return await self._request(
                "GET", "/defi/v3/token/trade-data/single", params=params, chain=chain
            )

        elif action == "token_trade_data_multiple":
            # Get trade data for multiple tokens
            addresses = kwargs.get("list_address")
            if not addresses:
                return {"success": False, "error": "list_address is required"}
            params = {"list_address": addresses}
            return await self._request(
                "GET", "/defi/v3/token/trade-data/multiple", params=params, chain=chain
            )

        elif action == "token_holder":
            # Get token holders
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/defi/v3/token/holder", params=params, chain=chain
            )

        elif action == "token_trending":
            # Get trending tokens
            params = {}
            if kwargs.get("sort_by"):
                params["sort_by"] = kwargs["sort_by"]
            if kwargs.get("sort_type"):
                params["sort_type"] = kwargs["sort_type"]
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/defi/token_trending", params=params, chain=chain
            )

        elif action == "token_new_listing":
            # Get newly listed tokens
            params = {}
            if kwargs.get("time_to"):
                params["time_to"] = kwargs["time_to"]
            if kwargs.get("time_from"):
                params["time_from"] = kwargs["time_from"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            if kwargs.get("meme_platform_enabled"):
                params["meme_platform_enabled"] = kwargs["meme_platform_enabled"]
            return await self._request(
                "GET", "/defi/v2/tokens/new_listing", params=params, chain=chain
            )

        elif action == "token_top_traders":
            # Get top traders for a token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("time_frame"):
                params["time_frame"] = kwargs["time_frame"]
            if kwargs.get("sort_type"):
                params["sort_type"] = kwargs["sort_type"]
            if kwargs.get("sort_by"):
                params["sort_by"] = kwargs["sort_by"]
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/defi/v2/tokens/top_traders", params=params, chain=chain
            )

        elif action == "token_markets":
            # Get markets for a token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("sort_by"):
                params["sort_by"] = kwargs["sort_by"]
            if kwargs.get("sort_type"):
                params["sort_type"] = kwargs["sort_type"]
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/defi/v2/markets", params=params, chain=chain
            )

        elif action == "token_security":
            # Get token security analysis
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            return await self._request(
                "GET", "/defi/token_security", params=params, chain=chain
            )

        elif action == "token_creation_info":
            # Get token creation information
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            return await self._request(
                "GET", "/defi/token_creation_info", params=params, chain=chain
            )

        elif action == "token_mint_burn":
            # Get mint/burn transactions
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("tx_type"):
                params["tx_type"] = kwargs["tx_type"]
            if kwargs.get("sort_type"):
                params["sort_type"] = kwargs["sort_type"]
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/defi/v3/token/mint-burn-txs", params=params, chain=chain
            )

        elif action == "token_all_time_trades_single":
            # Get all-time trade stats for single token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            return await self._request(
                "GET", "/defi/v3/all-time-trades/single", params=params, chain=chain
            )

        elif action == "token_all_time_trades_multiple":
            # Get all-time trade stats for multiple tokens
            addresses = kwargs.get("list_address")
            if not addresses:
                return {"success": False, "error": "list_address is required"}
            params = {"list_address": addresses}
            return await self._request(
                "GET", "/defi/v3/all-time-trades/multiple", params=params, chain=chain
            )

        elif action == "token_exit_liquidity":
            # Get exit liquidity for a token
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            return await self._request(
                "GET", "/defi/v3/token/exit-liquidity", params=params, chain=chain
            )

        elif action == "token_exit_liquidity_multiple":
            # Get exit liquidity for multiple tokens (max 50)
            addresses = kwargs.get("list_address")
            if not addresses:
                return {"success": False, "error": "list_address is required"}
            params = {"list_address": addresses}
            return await self._request(
                "GET",
                "/defi/v3/token/exit-liquidity/multiple",
                params=params,
                chain=chain,
            )

        # ==================== PAIR ====================
        elif action == "pair_overview_single":
            # Get overview for single pair
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            return await self._request(
                "GET", "/defi/v3/pair/overview/single", params=params, chain=chain
            )

        elif action == "pair_overview_multiple":
            # Get overview for multiple pairs
            addresses = kwargs.get("list_address")
            if not addresses:
                return {"success": False, "error": "list_address is required"}
            params = {"list_address": addresses}
            return await self._request(
                "GET", "/defi/v3/pair/overview/multiple", params=params, chain=chain
            )

        # ==================== TRADER ====================
        elif action == "trader_gainers_losers":
            # Get top gainers and losers
            params = {}
            if kwargs.get("type"):
                params["type"] = kwargs["type"]
            if kwargs.get("sort_by"):
                params["sort_by"] = kwargs["sort_by"]
            if kwargs.get("sort_type"):
                params["sort_type"] = kwargs["sort_type"]
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/trader/gainers-losers", params=params, chain=chain
            )

        elif action == "trader_txs_seek":
            # Get trader transactions with time bounds
            address = kwargs.get("address")
            if not address:
                return {"success": False, "error": "address is required"}
            params = {"address": address}
            if kwargs.get("tx_type"):
                params["tx_type"] = kwargs["tx_type"]
            if kwargs.get("before_time"):
                params["before_time"] = kwargs["before_time"]
            if kwargs.get("after_time"):
                params["after_time"] = kwargs["after_time"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            return await self._request(
                "GET", "/trader/txs/seek_by_time", params=params, chain=chain
            )

        # ==================== WALLET ====================
        elif action == "wallet_token_list":
            # Get wallet token holdings
            wallet = kwargs.get("wallet")
            if not wallet:
                return {"success": False, "error": "wallet is required"}
            params = {"wallet": wallet}
            return await self._request(
                "GET", "/v1/wallet/token_list", params=params, chain=chain
            )

        elif action == "wallet_token_balance":
            # Get specific token balance in wallet
            wallet = kwargs.get("wallet")
            token_address = kwargs.get("token_address")
            if not wallet or not token_address:
                return {
                    "success": False,
                    "error": "wallet and token_address are required",
                }
            params = {"wallet": wallet, "token_address": token_address}
            return await self._request(
                "GET", "/v1/wallet/token_balance", params=params, chain=chain
            )

        elif action == "wallet_tx_list":
            # Get wallet transaction history
            wallet = kwargs.get("wallet")
            if not wallet:
                return {"success": False, "error": "wallet is required"}
            params = {"wallet": wallet}
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            if kwargs.get("before_time"):
                params["before_time"] = kwargs["before_time"]
            return await self._request(
                "GET", "/v1/wallet/tx_list", params=params, chain=chain
            )

        elif action == "wallet_balance_change":
            # Get wallet balance changes
            wallet = kwargs.get("wallet")
            if not wallet:
                return {"success": False, "error": "wallet is required"}
            params = {"wallet": wallet}
            if kwargs.get("token_address"):
                params["token_address"] = kwargs["token_address"]
            if kwargs.get("time_from"):
                params["time_from"] = kwargs["time_from"]
            if kwargs.get("time_to"):
                params["time_to"] = kwargs["time_to"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            return await self._request(
                "GET", "/wallet/v2/balance-change", params=params, chain=chain
            )

        elif action == "wallet_pnl_summary":
            # Get PNL summary for a wallet
            wallet = kwargs.get("wallet")
            if not wallet:
                return {"success": False, "error": "wallet is required"}
            params = {"wallet": wallet}
            if kwargs.get("tx_type"):
                params["tx_type"] = kwargs["tx_type"]
            return await self._request(
                "GET", "/wallet/v2/pnl/summary", params=params, chain=chain
            )

        elif action == "wallet_pnl_details":
            # Get PNL details broken down by token (POST, max 100 tokens)
            wallet = kwargs.get("wallet")
            if not wallet:
                return {"success": False, "error": "wallet is required"}
            json_data = {"wallet": wallet}
            if kwargs.get("tokens"):
                json_data["tokens"] = kwargs["tokens"]
            return await self._request(
                "POST", "/wallet/v2/pnl/details", json_data=json_data, chain=chain
            )

        elif action == "wallet_pnl_multiple":
            # Get PNL for multiple wallets (max 50)
            wallets = kwargs.get("wallets")
            if not wallets:
                return {"success": False, "error": "wallets list is required"}
            params = {
                "wallets": ",".join(wallets) if isinstance(wallets, list) else wallets
            }
            return await self._request(
                "GET", "/wallet/v2/pnl/multiple", params=params, chain=chain
            )

        elif action == "wallet_current_net_worth":
            # Get current net worth and portfolio
            wallet = kwargs.get("wallet")
            if not wallet:
                return {"success": False, "error": "wallet is required"}
            params = {"wallet": wallet}
            return await self._request(
                "GET", "/wallet/v2/current-net-worth", params=params, chain=chain
            )

        elif action == "wallet_net_worth":
            # Get historical net worth by dates
            wallet = kwargs.get("wallet")
            if not wallet:
                return {"success": False, "error": "wallet is required"}
            params = {"wallet": wallet}
            if kwargs.get("time"):
                params["time"] = kwargs["time"]  # ISO 8601 UTC format
            if kwargs.get("type"):
                params["type"] = kwargs["type"]  # 1h or 1d
            if kwargs.get("count"):
                params["count"] = kwargs["count"]
            if kwargs.get("direction"):
                params["direction"] = kwargs["direction"]  # back or forward
            return await self._request(
                "GET", "/wallet/v2/net-worth", params=params, chain=chain
            )

        elif action == "wallet_net_worth_details":
            # Get asset details on a specific date
            wallet = kwargs.get("wallet")
            if not wallet:
                return {"success": False, "error": "wallet is required"}
            params = {"wallet": wallet}
            if kwargs.get("time"):
                params["time"] = kwargs["time"]  # ISO 8601 UTC format
            if kwargs.get("type"):
                params["type"] = kwargs["type"]  # 1h or 1d
            return await self._request(
                "GET", "/wallet/v2/net-worth-details", params=params, chain=chain
            )

        # ==================== SEARCH ====================
        elif action == "search":
            # Search for tokens/pairs
            keyword = kwargs.get("keyword")
            if not keyword:
                return {"success": False, "error": "keyword is required"}
            params = {"keyword": keyword}
            if kwargs.get("target"):
                params["target"] = kwargs["target"]
            if kwargs.get("sort_by"):
                params["sort_by"] = kwargs["sort_by"]
            if kwargs.get("sort_type"):
                params["sort_type"] = kwargs["sort_type"]
            if kwargs.get("offset"):
                params["offset"] = kwargs["offset"]
            if kwargs.get("limit"):
                params["limit"] = kwargs["limit"]
            if kwargs.get("verify_token"):
                params["verify_token"] = kwargs["verify_token"]
            if kwargs.get("markets"):
                params["markets"] = kwargs["markets"]
            return await self._request(
                "GET", "/defi/v3/search", params=params, chain=chain
            )

        # ==================== UTILS ====================
        elif action == "latest_block":
            # Get latest block info
            return await self._request("GET", "/defi/v3/txs/latest_block", chain=chain)

        elif action == "networks":
            # Get supported networks
            return await self._request("GET", "/defi/networks", chain=chain)

        elif action == "supported_chains":
            # Get supported chains for wallet API
            return await self._request(
                "GET", "/v1/wallet/list_supported_chain", chain=chain
            )

        else:
            return {"success": False, "error": f"Unknown action: {action}"}


class BirdeyePlugin:
    """Plugin for Birdeye API integration."""

    def __init__(self):
        self.name = "birdeye"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for Birdeye Solana token analytics and wallet data."

    def initialize(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._tool = BirdeyeTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> list[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return BirdeyePlugin()

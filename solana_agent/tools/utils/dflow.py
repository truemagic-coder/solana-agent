"""
DFlow API utility functions.

Provides interfaces to DFlow's APIs:
1. Swap API: Fast token swaps on Solana with platform fee support
2. Prediction Market API: Safety-focused prediction market discovery and trading

DFlow Swap offers three modes:
- Order API: Combined quote + transaction (best for gasless/sponsored swaps)
- Imperative: Two-step quote then swap (precise route control)
- Declarative: Intent-based swaps (deferred routing)

Prediction Market features:
- Quality filters applied by default (volume, liquidity, age)
- Safety scoring on all market queries
- Blocking async execution for agent compatibility
"""

import asyncio
import logging
import base64
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import httpx

from solders.transaction import VersionedTransaction  # type: ignore
from solders.message import to_bytes_versioned  # type: ignore

logger = logging.getLogger(__name__)

# DFlow API base URLs
DFLOW_API_URL = "https://quote-api.dflow.net"
DFLOW_METADATA_API = "https://prediction-markets-api.dflow.net/api/v1"

# Known verified series for safety scoring
# These are well-established markets with clear resolution criteria
KNOWN_SERIES = {
    "US-POLITICS",
    "US-ELECTIONS",
    "NFL",
    "NBA",
    "MLB",
    "NHL",
    "SOCCER",
    "CRYPTO",
    "FED",
    "ECONOMICS",
}

# Series prefixes that indicate verified/safe markets (e.g., Kalshi markets)
SAFE_SERIES_PREFIXES = {
    "KX",  # Kalshi Exchange - regulated, clear resolution
    "POLY",  # Polymarket - established platform
}

# Categories that have objectively verifiable outcomes (binary, date-bound)
OBJECTIVE_CATEGORIES = {
    "politics",
    "elections",
    "sports",
    "fed",
    "economics",
    "crypto-price",
}

# Default quality filters
DEFAULT_MIN_VOLUME_USD = 1000
DEFAULT_MIN_LIQUIDITY_USD = 500
DEFAULT_MIN_AGE_HOURS = 24


@dataclass
class DFlowOrderResponse:
    """Response from DFlow /order endpoint."""

    success: bool
    transaction: Optional[str] = None  # Base64 encoded transaction
    in_amount: Optional[str] = None
    out_amount: Optional[str] = None
    min_out_amount: Optional[str] = None
    input_mint: Optional[str] = None
    output_mint: Optional[str] = None
    slippage_bps: Optional[int] = None
    execution_mode: Optional[str] = None  # "sync" or "async"
    price_impact_pct: Optional[str] = None
    platform_fee: Optional[Dict[str, Any]] = None
    context_slot: Optional[int] = None
    last_valid_block_height: Optional[int] = None
    compute_unit_limit: Optional[int] = None
    prioritization_fee_lamports: Optional[int] = None
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DFlowOrderStatusResponse:
    """Response from DFlow /order-status endpoint."""

    success: bool
    status: Optional[str] = (
        None  # "pending", "expired", "failed", "open", "pendingClose", "closed"
    )
    in_amount: Optional[str] = None
    out_amount: Optional[str] = None
    fills: Optional[list] = None
    reverts: Optional[list] = None
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


class DFlowSwap:
    """DFlow Swap API client for fast token swaps."""

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize DFlow Swap client.

        Args:
            base_url: Optional custom base URL (defaults to quote-api.dflow.net)
        """
        self.base_url = base_url or DFLOW_API_URL
        self._headers = {
            "Content-Type": "application/json",
        }

    async def get_order(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        user_public_key: str,
        slippage_bps: Optional[int] = None,
        platform_fee_bps: Optional[int] = None,
        platform_fee_mode: Optional[str] = None,
        fee_account: Optional[str] = None,
        referral_account: Optional[str] = None,
        sponsor: Optional[str] = None,
        destination_wallet: Optional[str] = None,
        wrap_and_unwrap_sol: bool = True,
        prioritization_fee_lamports: Optional[str] = None,
        dynamic_compute_unit_limit: bool = True,
        only_direct_routes: bool = False,
        max_route_length: Optional[int] = None,
    ) -> DFlowOrderResponse:
        """
        Get a swap order from DFlow (combined quote + transaction).

        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount of input token (in smallest units/lamports)
            user_public_key: User's wallet address (required for transaction)
            slippage_bps: Max slippage in basis points (or None for "auto")
            platform_fee_bps: Platform fee in basis points (e.g., 50 = 0.5%)
            platform_fee_mode: "outputMint" (default) or "inputMint"
            fee_account: Token account to receive platform fee (must match fee mint)
            referral_account: Referral account if using Jupiter referral program
            sponsor: Sponsor wallet for gasless swaps (pays tx fees)
            destination_wallet: Wallet to receive output (defaults to user)
            wrap_and_unwrap_sol: Whether to auto wrap/unwrap SOL (default: True)
            prioritization_fee_lamports: Priority fee ("auto", "medium", "high", "veryHigh", or lamports)
            dynamic_compute_unit_limit: Whether to simulate for CU limit (default: True)
            only_direct_routes: Only use single-leg routes (default: False)
            max_route_length: Max number of route legs

        Returns:
            DFlowOrderResponse with transaction to sign
        """
        params: Dict[str, Any] = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": str(wrap_and_unwrap_sol).lower(),
            "dynamicComputeUnitLimit": str(dynamic_compute_unit_limit).lower(),
        }

        # Slippage - use "auto" if not specified
        if slippage_bps is not None:
            params["slippageBps"] = slippage_bps
        else:
            params["slippageBps"] = "auto"

        # Platform fee configuration
        if platform_fee_bps is not None and platform_fee_bps > 0:
            params["platformFeeBps"] = platform_fee_bps
            if fee_account:
                params["feeAccount"] = fee_account
            if platform_fee_mode:
                params["platformFeeMode"] = platform_fee_mode
            if referral_account:
                params["referralAccount"] = referral_account

        # Gasless/sponsored swap
        if sponsor:
            params["sponsor"] = sponsor

        # Custom destination
        if destination_wallet:
            params["destinationWallet"] = destination_wallet

        # Priority fee
        if prioritization_fee_lamports:
            params["prioritizationFeeLamports"] = prioritization_fee_lamports
        else:
            params["prioritizationFeeLamports"] = "auto"

        # Route options
        if only_direct_routes:
            params["onlyDirectRoutes"] = "true"
        if max_route_length is not None:
            params["maxRouteLength"] = max_route_length

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/order",
                    params=params,
                    headers=self._headers,
                )

                if response.status_code != 200:
                    error_text = response.text
                    try:
                        error_data = response.json()
                        error_text = error_data.get("error", error_text)
                    except Exception:
                        pass
                    return DFlowOrderResponse(
                        success=False,
                        error=f"DFlow API error: {response.status_code} - {error_text}",
                    )

                data = response.json()

                return DFlowOrderResponse(
                    success=True,
                    transaction=data.get("transaction"),
                    in_amount=data.get("inAmount"),
                    out_amount=data.get("outAmount"),
                    min_out_amount=data.get("minOutAmount")
                    or data.get("otherAmountThreshold"),
                    input_mint=data.get("inputMint"),
                    output_mint=data.get("outputMint"),
                    slippage_bps=data.get("slippageBps"),
                    execution_mode=data.get("executionMode"),
                    price_impact_pct=data.get("priceImpactPct"),
                    platform_fee=data.get("platformFee"),
                    context_slot=data.get("contextSlot"),
                    last_valid_block_height=data.get("lastValidBlockHeight"),
                    compute_unit_limit=data.get("computeUnitLimit"),
                    prioritization_fee_lamports=data.get("prioritizationFeeLamports"),
                    raw_response=data,
                )
        except httpx.TimeoutException:
            return DFlowOrderResponse(
                success=False,
                error="Request timed out. Please try again.",
            )
        except Exception as e:
            logger.exception("Failed to get DFlow order")
            return DFlowOrderResponse(success=False, error=str(e))

    async def get_order_status(
        self,
        signature: str,
        last_valid_block_height: Optional[int] = None,
    ) -> DFlowOrderStatusResponse:
        """
        Get the status of an order by transaction signature.

        Args:
            signature: Base58-encoded transaction signature
            last_valid_block_height: Optional block height for expiry check

        Returns:
            DFlowOrderStatusResponse with order status
        """
        params: Dict[str, Any] = {
            "signature": signature,
        }

        if last_valid_block_height is not None:
            params["lastValidBlockHeight"] = last_valid_block_height

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/order-status",
                    params=params,
                    headers=self._headers,
                )

                if response.status_code == 404:  # pragma: no cover
                    return DFlowOrderStatusResponse(
                        success=False,
                        error="Order not found",
                    )

                if response.status_code != 200:  # pragma: no cover
                    error_text = response.text
                    try:
                        error_data = response.json()
                        error_text = error_data.get("error", error_text)
                    except Exception:
                        pass
                    return DFlowOrderStatusResponse(
                        success=False,
                        error=f"DFlow API error: {response.status_code} - {error_text}",
                    )

                data = response.json()

                return DFlowOrderStatusResponse(
                    success=True,
                    status=data.get("status"),
                    in_amount=data.get("inAmount"),
                    out_amount=data.get("outAmount"),
                    fills=data.get("fills"),
                    reverts=data.get("reverts"),
                    raw_response=data,
                )
        except Exception as e:
            logger.exception("Failed to get DFlow order status")
            return DFlowOrderStatusResponse(success=False, error=str(e))


def sign_dflow_transaction(  # pragma: no cover
    transaction_base64: str,
    sign_message_func,
    sponsor_sign_func=None,
) -> str:
    """
    Sign a DFlow transaction.

    Args:
        transaction_base64: Base64 encoded transaction from DFlow
        sign_message_func: Function that signs a message and returns signature (user)
        sponsor_sign_func: Optional function for sponsor signature (for gasless)

    Returns:
        Base64 encoded signed transaction
    """
    transaction_bytes = base64.b64decode(transaction_base64)
    transaction = VersionedTransaction.from_bytes(transaction_bytes)

    message_bytes = to_bytes_versioned(transaction.message)

    if sponsor_sign_func:
        # Gasless: sponsor signs first, then user
        sponsor_signature = sponsor_sign_func(message_bytes)
        user_signature = sign_message_func(message_bytes)
        signed_transaction = VersionedTransaction.populate(
            transaction.message, [sponsor_signature, user_signature]
        )
    else:
        # Normal: only user signs
        signature = sign_message_func(message_bytes)
        signed_transaction = VersionedTransaction.populate(
            transaction.message, [signature]
        )

    return base64.b64encode(bytes(signed_transaction)).decode("utf-8")


# =============================================================================
# PREDICTION MARKET API
# =============================================================================


@dataclass
class SafetyResult:
    """Safety assessment for a prediction market."""

    score: str  # HIGH, MEDIUM, LOW, UNKNOWN
    warnings: List[str]
    recommendation: str  # PROCEED, CAUTION, AVOID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
        }


@dataclass
class DFlowPredictionOrderResult:
    """Result of a prediction market order."""

    success: bool
    signature: Optional[str]
    execution_mode: str  # sync, async
    in_amount: Optional[str]
    out_amount: Optional[str]
    min_out_amount: Optional[str]
    price_impact_pct: Optional[str]
    error: Optional[str] = None
    fills: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "execution_mode": self.execution_mode,
        }
        if self.signature:
            result["signature"] = self.signature
        if self.in_amount:
            result["in_amount"] = self.in_amount
        if self.out_amount:
            result["out_amount"] = self.out_amount
        if self.min_out_amount:
            result["min_out_amount"] = self.min_out_amount
        if self.price_impact_pct:
            result["price_impact_pct"] = self.price_impact_pct
        if self.error:
            result["error"] = self.error
        if self.fills:
            result["fills"] = self.fills
        return result


def calculate_safety_score(
    market: Dict[str, Any],
    trades: Optional[List[Dict[str, Any]]] = None,
    current_time: Optional[int] = None,
) -> SafetyResult:
    """
    Calculate safety score for a prediction market.

    Args:
        market: Market data from API
        trades: Optional recent trades for activity check
        current_time: Optional current timestamp (for testing)

    Returns:
        SafetyResult with score, warnings, and recommendation
    """
    warnings: List[str] = []
    score_points = 100
    now = current_time or int(time.time())

    # Get market identifiers
    ticker = market.get("ticker", "") or ""
    series_ticker = market.get("seriesTicker") or market.get("series_ticker", "") or ""
    category = (market.get("category", "") or "").lower()

    # Check if this is a verified/regulated market (Kalshi, Polymarket, etc.)
    is_verified_platform = any(
        ticker.upper().startswith(prefix) or series_ticker.upper().startswith(prefix)
        for prefix in SAFE_SERIES_PREFIXES
    )

    # Check if this is a known established series
    is_known_series = any(
        series_ticker.upper().startswith(known) for known in KNOWN_SERIES
    )

    # Check if the market has objectively verifiable outcomes
    has_clear_resolution_date = bool(
        market.get("closeTime") or market.get("expirationTime") or market.get("endDate")
    )
    is_objective_category = category in OBJECTIVE_CATEGORIES

    # Verified platforms with clear dates are HIGH safety - skip other checks
    if is_verified_platform and has_clear_resolution_date:
        # These are regulated markets with binary outcomes and set dates
        return SafetyResult("HIGH", [], "PROCEED")

    # Known series with clear dates also get a boost
    if is_known_series and has_clear_resolution_date:
        return SafetyResult("HIGH", [], "PROCEED")

    # Age check - less strict for verified platforms
    created_at = market.get("createdAt") or market.get("openTime")
    if created_at:
        age_hours = (now - created_at) / 3600
        if age_hours < 24:
            if not is_verified_platform:
                warnings.append("New market (< 24 hours old)")
                score_points -= 30
        elif age_hours < 168:  # 7 days
            if not is_verified_platform:
                warnings.append("Young market (< 7 days old)")
                score_points -= 15

    # Volume check
    volume = market.get("volume", 0)
    if volume < 1000:
        warnings.append(f"Low volume (${volume:,.0f})")
        score_points -= 25
    elif volume < 10000:
        warnings.append(f"Moderate volume (${volume:,.0f})")
        score_points -= 10

    # Liquidity check (DFlow API returns openInterest instead of liquidity)
    liquidity = market.get("liquidity") or market.get("openInterest", 0)
    if liquidity < 500:
        warnings.append("Low liquidity - may be hard to exit")
        score_points -= 30
    elif liquidity < 2000:
        warnings.append("Moderate liquidity")
        score_points -= 10

    # Activity check (if trades provided)
    if trades is not None:
        recent_cutoff = now - 86400  # 24 hours ago
        recent_trades = [t for t in trades if t.get("createdTime", 0) > recent_cutoff]
        if len(recent_trades) == 0:
            warnings.append("No trades in 24 hours")
            score_points -= 20

    # Series verification - only penalize if not verified platform
    if not is_known_series and not is_verified_platform and series_ticker:
        warnings.append("Unknown/unverified series")
        score_points -= 15

    # Resolution clarity check - verified platforms have clear rules by default
    rules = market.get("rulesPrimary", "")
    if not is_verified_platform and (not rules or len(rules) < 50):
        warnings.append("Unclear resolution criteria")
        score_points -= 20

    # Boost for objective categories with clear dates
    if is_objective_category and has_clear_resolution_date:
        score_points += 15

    # Calculate final score
    if score_points >= 70:
        return SafetyResult("HIGH", warnings, "PROCEED")
    elif score_points >= 40:
        return SafetyResult("MEDIUM", warnings, "CAUTION")
    else:
        return SafetyResult("LOW", warnings, "AVOID")


class DFlowPredictionClient:
    """
    DFlow Prediction Market API client.

    Provides a safety-focused interface for:
    - Discovering markets with quality filters
    - Trading with blocking async execution
    - Platform fee collection
    """

    def __init__(
        self,
        metadata_base_url: Optional[str] = None,
        trade_base_url: Optional[str] = None,
        min_volume_usd: int = DEFAULT_MIN_VOLUME_USD,
        min_liquidity_usd: int = DEFAULT_MIN_LIQUIDITY_USD,
        min_age_hours: int = DEFAULT_MIN_AGE_HOURS,
        include_risky: bool = False,
    ):
        """
        Initialize DFlow Prediction client.

        Args:
            metadata_base_url: Override metadata API URL
            trade_base_url: Override trade API URL
            min_volume_usd: Minimum volume filter (default 1000)
            min_liquidity_usd: Minimum liquidity filter (default 500)
            min_age_hours: Minimum market age filter (default 24)
            include_risky: Include low-quality markets (with warnings)
        """
        self.metadata_url = metadata_base_url or DFLOW_METADATA_API
        self.trade_url = trade_base_url or DFLOW_API_URL
        self.min_volume_usd = min_volume_usd
        self.min_liquidity_usd = min_liquidity_usd
        self.min_age_hours = min_age_hours
        self.include_risky = include_risky
        self._headers = {"Content-Type": "application/json"}

    def _apply_quality_filters(
        self, items: List[Dict[str, Any]], item_type: str = "market"
    ) -> List[Dict[str, Any]]:
        """Apply quality filters to markets/events unless include_risky is True."""
        if self.include_risky:
            return items

        filtered = []
        for item in items:
            volume = item.get("volume", 0)
            # DFlow API returns openInterest instead of liquidity for prediction markets
            liquidity = item.get("liquidity") or item.get("openInterest", 0)

            if volume >= self.min_volume_usd and liquidity >= self.min_liquidity_usd:
                filtered.append(item)

        return filtered

    def _add_safety_scores(
        self,
        items: List[Dict[str, Any]],
        trades_by_ticker: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Add safety scores and formatted resolution dates to markets/events."""
        from datetime import datetime

        for item in items:
            ticker = item.get("ticker", "")
            trades = trades_by_ticker.get(ticker) if trades_by_ticker else None
            safety = calculate_safety_score(item, trades)
            item["safety"] = safety.to_dict()

            # Add human-readable resolution date
            close_time = (
                item.get("closeTime")
                or item.get("expirationTime")
                or item.get("endDate")
            )
            if close_time:
                try:
                    if isinstance(close_time, (int, float)):
                        from datetime import timezone

                        dt = datetime.fromtimestamp(close_time, tz=timezone.utc)
                        item["resolution_date"] = dt.strftime("%Y-%m-%d %H:%M UTC")
                    else:
                        item["resolution_date"] = str(close_time)
                except Exception:
                    item["resolution_date"] = str(close_time)
        return items

    # =========================================================================
    # METADATA API - Discovery
    # =========================================================================

    async def search(
        self,
        query: str,
        limit: int = 20,
        with_nested_markets: bool = True,
    ) -> Dict[str, Any]:
        """
        Search for events by text query.

        Args:
            query: Search text
            limit: Max results (default 20)
            with_nested_markets: Include markets in response

        Returns:
            Dict with events list and safety scores
        """
        params = {
            "q": query,
            "limit": limit,
            "withNestedMarkets": str(with_nested_markets).lower(),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.metadata_url}/search",
                params=params,
                headers=self._headers,
            )

            if response.status_code != 200:
                raise Exception(
                    f"Search failed: {response.status_code} - {response.text}"
                )

            data = response.json()
            events = data.get("events", data) if isinstance(data, dict) else data

            # Apply filters and add safety scores
            events = self._apply_quality_filters(events, "event")
            events = self._add_safety_scores(events)

            return {"events": events, "count": len(events)}

    async def list_events(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
        status: str = "active",
        sort: str = "volume",
        series_tickers: Optional[List[str]] = None,
        with_nested_markets: bool = True,
    ) -> Dict[str, Any]:
        """
        List prediction events with filters.

        Args:
            limit: Max results (default 50)
            cursor: Pagination cursor
            status: Filter by status (active, closed, determined)
            sort: Sort by (volume, volume24h, liquidity, openInterest)
            series_tickers: Filter by series
            with_nested_markets: Include markets

        Returns:
            Dict with events list, cursor, and safety scores
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "status": status,
            "sort": sort,
            "withNestedMarkets": str(with_nested_markets).lower(),
        }
        if cursor:
            params["cursor"] = cursor
        if series_tickers:
            params["seriesTickers"] = ",".join(series_tickers[:25])

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.metadata_url}/events",
                params=params,
                headers=self._headers,
            )

            if response.status_code != 200:
                raise Exception(
                    f"List events failed: {response.status_code} - {response.text}"
                )

            data = response.json()
            events = data.get("events", [])
            next_cursor = data.get("cursor")

            # Apply filters and safety scores
            events = self._apply_quality_filters(events, "event")
            events = self._add_safety_scores(events)

            return {
                "events": events,
                "count": len(events),
                "cursor": next_cursor,
            }

    async def get_event(self, event_id: str) -> Dict[str, Any]:
        """
        Get a specific event by ticker/ID.

        Args:
            event_id: Event ticker

        Returns:
            Event data with safety score
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.metadata_url}/event/{event_id}",
                headers=self._headers,
            )

            if response.status_code != 200:
                raise Exception(
                    f"Get event failed: {response.status_code} - {response.text}"
                )

            event = response.json()

            # Add safety score
            safety = calculate_safety_score(event)
            event["safety"] = safety.to_dict()

            return event

    async def list_markets(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
        status: str = "active",
        sort: str = "volume",
    ) -> Dict[str, Any]:
        """
        List prediction markets with filters.

        Args:
            limit: Max results (default 50)
            cursor: Pagination cursor
            status: Filter by status
            sort: Sort field

        Returns:
            Dict with markets list and safety scores
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "status": status,
            "sort": sort,
        }
        if cursor:
            params["cursor"] = cursor

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.metadata_url}/markets",
                params=params,
                headers=self._headers,
            )

            if response.status_code != 200:
                raise Exception(
                    f"List markets failed: {response.status_code} - {response.text}"
                )

            data = response.json()
            markets = data.get("markets", [])
            next_cursor = data.get("cursor")

            # Apply filters and safety scores
            markets = self._apply_quality_filters(markets, "market")
            markets = self._add_safety_scores(markets)

            return {
                "markets": markets,
                "count": len(markets),
                "cursor": next_cursor,
            }

    async def get_market(
        self, market_id: Optional[str] = None, mint_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get a specific market by ticker or mint address.

        Args:
            market_id: Market ticker
            mint_address: Outcome token mint address

        Returns:
            Market data with safety score
        """
        if not market_id and not mint_address:
            raise ValueError("Either market_id or mint_address must be provided")

        async with httpx.AsyncClient(timeout=30.0) as client:
            if mint_address:
                url = f"{self.metadata_url}/market/by-mint/{mint_address}"
            else:
                url = f"{self.metadata_url}/market/{market_id}"

            response = await client.get(url, headers=self._headers)

            if response.status_code != 200:
                raise Exception(
                    f"Get market failed: {response.status_code} - {response.text}"
                )

            market = response.json()

            # Add safety score
            safety = calculate_safety_score(market)
            market["safety"] = safety.to_dict()

            return market

    async def list_series(
        self,
        category: Optional[str] = None,
        status: str = "active",
    ) -> Dict[str, Any]:
        """
        List prediction series (templates for events).

        Args:
            category: Filter by category
            status: Filter by status

        Returns:
            Dict with series list
        """
        params: Dict[str, Any] = {"status": status}
        if category:
            params["category"] = category

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.metadata_url}/series",
                params=params,
                headers=self._headers,
            )

            if response.status_code != 200:
                raise Exception(
                    f"List series failed: {response.status_code} - {response.text}"
                )

            data = response.json()
            return {"series": data.get("series", data)}

    async def get_categories(self) -> Dict[str, Any]:
        """
        Get available categories and tags.

        Returns:
            Dict with categories and their tags
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.metadata_url}/tags_by_categories",
                headers=self._headers,
            )

            if response.status_code != 200:
                raise Exception(
                    f"Get categories failed: {response.status_code} - {response.text}"
                )

            return response.json()

    async def get_trades(
        self,
        ticker: Optional[str] = None,
        mint_address: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get recent trades for a market.

        Args:
            ticker: Market ticker
            mint_address: Outcome token mint
            limit: Max trades (default 100)

        Returns:
            Dict with trades list
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            if mint_address:
                url = f"{self.metadata_url}/trades/by-mint/{mint_address}"
                params: Dict[str, Any] = {"limit": limit}
            else:
                url = f"{self.metadata_url}/trades"
                params = {"limit": limit}
                if ticker:
                    params["ticker"] = ticker

            response = await client.get(url, params=params, headers=self._headers)

            if response.status_code != 200:
                raise Exception(
                    f"Get trades failed: {response.status_code} - {response.text}"
                )

            data = response.json()
            return {"trades": data.get("trades", data)}

    async def get_outcome_mints(self) -> Dict[str, Any]:
        """
        Get all outcome token mints mapped to markets.

        Returns:
            Dict mapping mints to market info
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.metadata_url}/outcome_mints",
                headers=self._headers,
            )

            if response.status_code != 200:
                raise Exception(
                    f"Get outcome mints failed: {response.status_code} - {response.text}"
                )

            return response.json()

    # =========================================================================
    # TRADE API - Order Execution
    # =========================================================================

    async def get_prediction_order(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        user_public_key: str,
        slippage_bps: int = 100,
        platform_fee_bps: Optional[int] = None,
        platform_fee_scale: Optional[int] = None,
        fee_account: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get an order quote and transaction for prediction market trading.

        Args:
            input_mint: Token to sell
            output_mint: Token to buy (outcome token)
            amount: Amount of input token (scaled integer)
            user_public_key: User wallet address
            slippage_bps: Slippage tolerance in basis points
            platform_fee_bps: Platform fee for sync swaps
            platform_fee_scale: Platform fee for async swaps (3 decimals)
            fee_account: Token account to receive fees

        Returns:
            Order response with transaction to sign
        """
        params: Dict[str, Any] = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "userPublicKey": user_public_key,
            "slippageBps": slippage_bps,
        }

        if platform_fee_bps is not None:
            params["platformFeeBps"] = platform_fee_bps
        if platform_fee_scale is not None:
            params["platformFeeScale"] = platform_fee_scale
        if fee_account:
            params["feeAccount"] = fee_account

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.trade_url}/order",
                params=params,
                headers=self._headers,
            )

            if response.status_code != 200:
                raise Exception(
                    f"Get order failed: {response.status_code} - {response.text}"
                )

            return response.json()

    async def get_prediction_order_status(self, request_id: str) -> Dict[str, Any]:
        """
        Check status of an async prediction order.

        Args:
            request_id: Request ID from get_prediction_order response

        Returns:
            Order status with fills/reverts if complete
        """
        params = {"requestId": request_id}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.trade_url}/order-status",
                params=params,
                headers=self._headers,
            )

            if response.status_code != 200:
                raise Exception(
                    f"Get order status failed: {response.status_code} - {response.text}"
                )

            return response.json()

    async def execute_prediction_order_blocking(
        self,
        order_response: Dict[str, Any],
        sign_and_send_func,
        max_wait_seconds: int = 90,
        poll_interval_seconds: int = 2,
    ) -> DFlowPredictionOrderResult:
        """
        Execute a prediction order with blocking wait for async orders.

        This method handles both sync and async execution modes:
        - Sync: Single transaction, returns immediately after send
        - Async: Polls order-status until complete or timeout

        Args:
            order_response: Response from get_prediction_order
            sign_and_send_func: Async function(transaction_base64) -> signature
            max_wait_seconds: Maximum time to wait for async completion
            poll_interval_seconds: Time between status checks

        Returns:
            DFlowPredictionOrderResult with final status
        """
        execution_mode = order_response.get("executionMode", "sync")
        transaction = order_response.get("transaction")
        request_id = order_response.get("requestId")

        if not transaction:
            return DFlowPredictionOrderResult(
                success=False,
                signature=None,
                execution_mode=execution_mode,
                in_amount=None,
                out_amount=None,
                min_out_amount=None,
                price_impact_pct=None,
                error="No transaction in order response",
            )

        try:
            # Sign and send the transaction
            signature = await sign_and_send_func(transaction)

            if execution_mode == "sync":
                # Sync mode: done after single transaction
                return DFlowPredictionOrderResult(
                    success=True,
                    signature=signature,
                    execution_mode="sync",
                    in_amount=order_response.get("inAmount"),
                    out_amount=order_response.get("outAmount"),
                    min_out_amount=order_response.get("minOutAmount"),
                    price_impact_pct=order_response.get("priceImpactPct"),
                )

            # Async mode: poll for completion
            start_time = time.time()

            while time.time() - start_time < max_wait_seconds:
                await asyncio.sleep(poll_interval_seconds)

                try:
                    status = await self.get_prediction_order_status(request_id)
                except Exception as e:
                    logger.warning(f"Order status check failed: {e}")
                    continue

                order_status = status.get("status", "")

                if order_status == "closed":
                    # Order complete
                    return DFlowPredictionOrderResult(
                        success=True,
                        signature=signature,
                        execution_mode="async",
                        in_amount=status.get("inAmount"),
                        out_amount=status.get("outAmount"),
                        min_out_amount=order_response.get("minOutAmount"),
                        price_impact_pct=order_response.get("priceImpactPct"),
                        fills=status.get("fills"),
                    )
                elif order_status in ("failed", "expired"):
                    return DFlowPredictionOrderResult(
                        success=False,
                        signature=signature,
                        execution_mode="async",
                        in_amount=None,
                        out_amount=None,
                        min_out_amount=None,
                        price_impact_pct=None,
                        error=f"Order {order_status}",
                    )
                elif status.get("nextTransaction"):
                    # Need to send another transaction
                    next_tx = status["nextTransaction"]
                    signature = await sign_and_send_func(next_tx)

            # Timeout
            return DFlowPredictionOrderResult(
                success=False,
                signature=signature,
                execution_mode="async",
                in_amount=None,
                out_amount=None,
                min_out_amount=None,
                price_impact_pct=None,
                error=f"Timeout ({max_wait_seconds}s) waiting for order completion",
            )

        except Exception as e:
            logger.exception(f"Order execution failed: {e}")
            return DFlowPredictionOrderResult(
                success=False,
                signature=None,
                execution_mode=execution_mode,
                in_amount=None,
                out_amount=None,
                min_out_amount=None,
                price_impact_pct=None,
                error=str(e),
            )

    async def get_positions(
        self,
        wallet_address: str,
        rpc_url: str,
    ) -> Dict[str, Any]:
        """
        Get prediction market positions for a wallet.

        Queries token accounts via RPC and cross-references with DFlow outcome mints
        to identify prediction market positions.

        Args:
            wallet_address: Solana wallet address to check
            rpc_url: RPC endpoint URL

        Returns:
            Dict with positions list and summary
        """
        # Get all outcome mints from DFlow
        outcome_mints = await self.get_outcome_mints()

        # Query wallet token accounts via RPC
        token_accounts = await _get_token_accounts(rpc_url, wallet_address)

        if "error" in token_accounts:
            return {
                "status": "error",
                "message": f"Failed to query token accounts: {token_accounts['error']}",
            }

        # Cross-reference to find prediction positions
        positions = []

        for account in token_accounts.get("accounts", []):
            mint = account.get("mint")
            if mint and mint in outcome_mints:
                market_info = outcome_mints[mint]
                amount = account.get("amount", "0")
                ui_amount = account.get("uiAmount", 0)

                # Only include if they have a balance
                if ui_amount > 0:
                    positions.append(
                        {
                            "mint": mint,
                            "ticker": market_info.get(
                                "market", market_info.get("ticker", "unknown")
                            ),
                            "side": market_info.get("side", "unknown").upper(),
                            "amount": amount,
                            "ui_amount": ui_amount,
                            "decimals": account.get("decimals", 6),
                        }
                    )

        return {
            "status": "success",
            "wallet": wallet_address,
            "position_count": len(positions),
            "positions": positions,
            "hint": "Each position represents outcome tokens. Sell to exit or hold until resolution."
            if positions
            else "No prediction market positions found.",
        }


async def _get_token_accounts(rpc_url: str, wallet_address: str) -> Dict[str, Any]:
    """
    Get all SPL token accounts for a wallet via RPC.

    Args:
        rpc_url: RPC endpoint URL
        wallet_address: Wallet public key

    Returns:
        Dict with 'accounts' list or 'error' on failure
    """
    # Token Program ID
    TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            wallet_address,
            {"programId": TOKEN_PROGRAM_ID},
            {"encoding": "jsonParsed"},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(rpc_url, json=payload)
            if response.status_code != 200:
                return {"error": f"RPC error: {response.status_code}"}

            data = response.json()
            if "error" in data:
                return {"error": f"RPC error: {data['error']}"}

            result = data.get("result", {}).get("value", [])

            accounts = []
            for item in result:
                parsed = item.get("account", {}).get("data", {}).get("parsed", {})
                info = parsed.get("info", {})
                token_amount = info.get("tokenAmount", {})

                accounts.append(
                    {
                        "mint": info.get("mint"),
                        "amount": token_amount.get("amount", "0"),
                        "uiAmount": float(token_amount.get("uiAmount", 0) or 0),
                        "decimals": token_amount.get("decimals", 0),
                    }
                )

            return {"accounts": accounts}

    except Exception as e:
        logger.exception("Failed to get token accounts")
        return {"error": str(e)}

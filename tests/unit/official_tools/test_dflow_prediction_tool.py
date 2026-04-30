"""
Tests for DFlow Prediction Market Tool.

Tests the DFlowPredictionTool which provides safety-focused prediction
market discovery and trading with quality filters and risk warnings.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import time

from solana_agent.tools.dflow_prediction import (
    DFlowPredictionTool,
    DFlowPredictionPlugin,
)
from solana_agent.tools.utils.dflow import (
    calculate_safety_score,
    SafetyResult,
    DFlowPredictionClient,
    DFlowPredictionOrderResult,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def prediction_tool():
    """Create a configured DFlowPredictionTool."""
    tool = DFlowPredictionTool()
    tool.configure(
        {
            "tools": {
                "dflow_prediction": {
                    "private_key": "5jGR...base58privatekey",
                    "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                    "platform_fee_bps": 50,
                    "fee_account": "FeeAccountPubkey123",
                    "min_volume_usd": 1000,
                    "min_liquidity_usd": 500,
                }
            }
        }
    )
    return tool


@pytest.fixture
def prediction_tool_no_key():
    """Create an unconfigured DFlowPredictionTool."""
    tool = DFlowPredictionTool()
    tool.configure({"tools": {"dflow_prediction": {}}})
    return tool


@pytest.fixture
def sample_market():
    """Sample market data for testing."""
    return {
        "ticker": "PRES-2028-DEM-HARRIS",
        "eventTicker": "PRES-2028-DEM",
        "title": "Will Kamala Harris be the 2028 Democratic Nominee?",
        "subtitle": "Democratic Primary",
        "status": "active",
        "result": None,
        "marketType": "binary",
        "volume": 125000,
        "openInterest": 45000,
        "liquidity": 20000,
        "openTime": int(time.time()) - 86400 * 30,  # 30 days ago
        "closeTime": int(time.time()) + 86400 * 365,
        "expirationTime": int(time.time()) + 86400 * 400,
        "yesAsk": "0.37",
        "yesBid": "0.35",
        "noAsk": "0.65",
        "noBid": "0.63",
        "rulesPrimary": "This market will resolve YES if Kamala Harris wins the Democratic nomination for President in 2028.",
        "seriesTicker": "US-POLITICS-2028",
        "accounts": {
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {
                "yesMint": "YesMintAddress123",
                "noMint": "NoMintAddress456",
                "marketLedger": "LedgerAddress789",
                "redemptionStatus": "Open",
            }
        },
    }


@pytest.fixture
def low_quality_market():
    """Sample low-quality market for safety testing."""
    return {
        "ticker": "RANDOM-CAT-SLEEP",
        "eventTicker": "RANDOM-CAT",
        "title": "Will my cat sleep today?",
        "subtitle": None,
        "status": "active",
        "result": None,
        "marketType": "binary",
        "volume": 50,
        "openInterest": 20,
        "liquidity": 10,
        "openTime": int(time.time()) - 3600,  # 1 hour ago
        "closeTime": int(time.time()) + 86400,
        "yesAsk": "0.50",
        "yesBid": "0.48",
        "noAsk": "0.52",
        "noBid": "0.50",
        "rulesPrimary": "Cat sleeps",
        "seriesTicker": "RANDOM-STUFF",
        "accounts": {},
    }


@pytest.fixture
def sample_event():
    """Sample event data for testing."""
    return {
        "ticker": "PRES-2028-DEM",
        "title": "2028 Democratic Nominee",
        "subtitle": "Who will win?",
        "seriesTicker": "US-POLITICS-2028",
        "imageUrl": "https://example.com/image.png",
        "volume": 250000,
        "volume24h": 15000,
        "openInterest": 90000,
        "liquidity": 45000,
        "status": "active",
        "strikeDate": int(time.time()) + 86400 * 365,
        "rulesPrimary": "This event resolves based on the outcome of the 2028 Democratic Primary.",
    }


# =============================================================================
# SAFETY SCORING TESTS
# =============================================================================


class TestSafetyScoring:
    """Test the safety scoring algorithm."""

    def test_high_quality_market_gets_high_score(self, sample_market):
        """High volume, liquidity, established series should get HIGH score."""
        result = calculate_safety_score(sample_market)
        assert result.score == "HIGH"
        assert result.recommendation == "PROCEED"
        assert len(result.warnings) == 0

    def test_low_volume_warning(self):
        """Low volume should trigger warning."""
        market = {"volume": 500, "liquidity": 5000, "rulesPrimary": "x" * 100}
        result = calculate_safety_score(market)
        assert any("Low volume" in w for w in result.warnings)

    def test_low_liquidity_warning(self):
        """Low liquidity should trigger warning."""
        market = {"volume": 50000, "liquidity": 200, "rulesPrimary": "x" * 100}
        result = calculate_safety_score(market)
        assert any("Low liquidity" in w for w in result.warnings)

    def test_new_market_warning(self):
        """New market (< 24 hours) should trigger warning."""
        market = {
            "volume": 50000,
            "liquidity": 5000,
            "openTime": int(time.time()) - 3600,  # 1 hour ago
            "rulesPrimary": "x" * 100,
        }
        result = calculate_safety_score(market)
        assert any("< 24 hours" in w for w in result.warnings)

    def test_young_market_warning(self):
        """Young market (< 7 days) should trigger warning."""
        market = {
            "volume": 50000,
            "liquidity": 5000,
            "openTime": int(time.time()) - 86400 * 3,  # 3 days ago
            "rulesPrimary": "x" * 100,
        }
        result = calculate_safety_score(market)
        assert any("< 7 days" in w for w in result.warnings)

    def test_unknown_series_warning(self):
        """Unknown series should trigger warning."""
        market = {
            "volume": 50000,
            "liquidity": 5000,
            "seriesTicker": "RANDOM-UNKNOWN-SERIES",
            "rulesPrimary": "x" * 100,
        }
        result = calculate_safety_score(market)
        assert any("Unknown/unverified series" in w for w in result.warnings)

    def test_known_series_no_warning(self):
        """Known series should not trigger warning."""
        market = {
            "volume": 50000,
            "liquidity": 5000,
            "seriesTicker": "US-POLITICS-2028",
            "rulesPrimary": "x" * 100,
        }
        result = calculate_safety_score(market)
        assert not any("Unknown/unverified series" in w for w in result.warnings)

    def test_unclear_rules_warning(self):
        """Short/missing resolution rules should trigger warning."""
        market = {"volume": 50000, "liquidity": 5000, "rulesPrimary": "yes"}
        result = calculate_safety_score(market)
        assert any("Unclear resolution" in w for w in result.warnings)

    def test_no_recent_trades_warning(self):
        """No recent trades should trigger warning."""
        market = {"volume": 50000, "liquidity": 5000, "rulesPrimary": "x" * 100}
        trades = [{"createdTime": int(time.time()) - 86400 * 2}]  # 2 days ago
        result = calculate_safety_score(market, trades)
        assert any("No trades in 24 hours" in w for w in result.warnings)

    def test_low_quality_market_gets_low_score(self, low_quality_market):
        """Multiple issues should result in LOW score."""
        result = calculate_safety_score(low_quality_market)
        assert result.score == "LOW"
        assert result.recommendation == "AVOID"
        assert len(result.warnings) >= 3

    def test_safety_result_to_dict(self):
        """SafetyResult should serialize to dict."""
        result = SafetyResult("HIGH", ["Warning 1"], "PROCEED")
        d = result.to_dict()
        assert d["score"] == "HIGH"
        assert d["warnings"] == ["Warning 1"]
        assert d["recommendation"] == "PROCEED"

    def test_moderate_liquidity_warning(self):
        """Moderate liquidity (500-2000) should trigger warning."""
        market = {"volume": 50000, "liquidity": 1500, "rulesPrimary": "x" * 100}
        result = calculate_safety_score(market)
        assert any("Moderate liquidity" in w for w in result.warnings)

    def test_moderate_volume_warning(self):
        """Moderate volume (1000-10000) should trigger warning."""
        market = {"volume": 5000, "liquidity": 5000, "rulesPrimary": "x" * 100}
        result = calculate_safety_score(market)
        assert any("Moderate volume" in w for w in result.warnings)

    def test_medium_safety_score(self):
        """Markets with multiple issues should get MEDIUM score."""
        # Market with issues that should result in MEDIUM score (40-69 points)
        # Start at 100, need to get to 40-69 range
        market = {
            "volume": 5000,  # Moderate volume (-10 points) -> 90
            "liquidity": 1500,  # Moderate liquidity (-10 points) -> 80
            "rulesPrimary": "short",  # Unclear rules (-20 points) -> 60
            "seriesTicker": "US-POLITICS-2024",  # Known series (no penalty)
            "openTime": int(time.time()) - 86400 * 30,  # 30 days ago (no penalty)
        }
        result = calculate_safety_score(market)
        # Should be MEDIUM with score around 60
        assert result.score == "MEDIUM"
        assert result.recommendation == "CAUTION"

    def test_kalshi_market_with_clear_date_gets_high_score(self):
        """Kalshi markets (KX prefix) with clear resolution date should get HIGH score."""
        market = {
            "ticker": "KXTRUMPOUT-26-TRUMP",
            "volume": 500,  # Would normally be low
            "liquidity": 100,  # Would normally be low
            "closeTime": int(time.time()) + 86400 * 30,  # 30 days from now
            "rulesPrimary": "short",  # Would normally trigger warning
        }
        result = calculate_safety_score(market)
        assert result.score == "HIGH"
        assert result.recommendation == "PROCEED"
        assert len(result.warnings) == 0

    def test_polymarket_with_clear_date_gets_high_score(self):
        """Polymarket (POLY prefix) with clear resolution date should get HIGH score."""
        market = {
            "ticker": "POLY-ELECTION-2024",
            "volume": 500,
            "liquidity": 100,
            "expirationTime": int(time.time()) + 86400 * 30,
            "rulesPrimary": "short",
        }
        result = calculate_safety_score(market)
        assert result.score == "HIGH"
        assert result.recommendation == "PROCEED"

    def test_verified_platform_without_date_still_checks_other_factors(self):
        """Verified platform without clear date should still check other factors."""
        market = {
            "ticker": "KXSOMETHING",
            "volume": 500,  # Low volume
            "liquidity": 100,  # Low liquidity
            # No closeTime or expirationTime
            "rulesPrimary": "short",
        }
        result = calculate_safety_score(market)
        # Without clear date, should not auto-pass
        assert result.score != "HIGH" or len(result.warnings) > 0

    def test_objective_category_with_date_gets_boost(self):
        """Objective category (e.g., politics) with clear date should get score boost."""
        # This market would normally be MEDIUM but gets boosted
        market = {
            "ticker": "SOMEMARKET",
            "category": "politics",
            "volume": 5000,  # Moderate volume (-10)
            "liquidity": 1500,  # Moderate liquidity (-10)
            "closeTime": int(time.time()) + 86400 * 30,  # Has clear date
            "rulesPrimary": "x" * 100,  # Clear rules
            "seriesTicker": "UNKNOWN-SERIES",  # Unknown series (-15)
            # Without boost: 100 - 10 - 10 - 15 = 65 (MEDIUM)
            # With boost: 65 + 15 = 80 (HIGH)
        }
        result = calculate_safety_score(market)
        assert result.score == "HIGH"
        assert result.recommendation == "PROCEED"


# =============================================================================
# TOOL SCHEMA TESTS
# =============================================================================


class TestDFlowPredictionToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, prediction_tool):
        """Should have correct tool name."""
        assert prediction_tool.name == "dflow_prediction"

    def test_tool_description_includes_warning(self, prediction_tool):
        """Description should include risk warning."""
        assert "WARNING" in prediction_tool.description
        assert "prediction market" in prediction_tool.description.lower()

    def test_schema_has_action(self, prediction_tool):
        """Schema should have action property."""
        schema = prediction_tool.get_schema()
        assert "action" in schema["properties"]
        assert "action" in schema["required"]

    def test_schema_actions_list(self, prediction_tool):
        """Schema should list all available actions."""
        schema = prediction_tool.get_schema()
        actions = schema["properties"]["action"]["enum"]
        expected_actions = [
            "search",
            "list_events",
            "get_event",
            "list_markets",
            "get_market",
            "buy",
            "sell",
            "positions",
        ]
        assert set(actions) == set(expected_actions)

    def test_schema_buy_sell_parameters(self, prediction_tool):
        """Schema should have buy/sell parameters."""
        schema = prediction_tool.get_schema()
        props = schema["properties"]
        assert "market_id" in props
        assert "side" in props
        assert "amount" in props
        # OpenAI strict mode requires nullable types, so enum includes None
        assert props["side"]["enum"] == ["YES", "NO", None]


# =============================================================================
# TOOL CONFIGURATION TESTS
# =============================================================================


class TestDFlowPredictionToolConfigure:
    """Test configuration method."""

    def test_configure_stores_private_key(self, prediction_tool):
        """Should store private key from config."""
        assert prediction_tool._private_key == "5jGR...base58privatekey"

    def test_configure_stores_rpc_url(self, prediction_tool):
        """Should store RPC URL."""
        assert "helius" in prediction_tool._rpc_url

    def test_configure_stores_platform_fee(self, prediction_tool):
        """Should store platform fee settings."""
        assert prediction_tool._platform_fee_bps == 50
        assert prediction_tool._fee_account == "FeeAccountPubkey123"

    def test_configure_stores_quality_filters(self, prediction_tool):
        """Should store quality filter settings."""
        assert prediction_tool._min_volume_usd == 1000
        assert prediction_tool._min_liquidity_usd == 500

    def test_get_client_returns_configured_client(self, prediction_tool):
        """_get_client should return properly configured client."""
        client = prediction_tool._get_client()
        assert client.min_volume_usd == 1000
        assert client.min_liquidity_usd == 500
        assert client.include_risky is False

    def test_get_client_override_include_risky(self, prediction_tool):
        """_get_client should allow overriding include_risky."""
        client = prediction_tool._get_client(include_risky=True)
        assert client.include_risky is True


# =============================================================================
# DISCOVERY ACTION TESTS
# =============================================================================


class TestDiscoveryActions:
    """Test discovery actions (search, list, get)."""

    @pytest.mark.asyncio
    async def test_search_requires_query(self, prediction_tool):
        """Search action should require query parameter."""
        result = await prediction_tool.execute(action="search")
        assert result["status"] == "error"
        assert "query" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_search_success(self, prediction_tool, sample_event):
        """Search should return events with safety scores."""
        with patch.object(
            DFlowPredictionClient,
            "search",
            new_callable=AsyncMock,
            return_value={"events": [sample_event], "count": 1},
        ):
            result = await prediction_tool.execute(action="search", query="harris")
            assert result["status"] == "success"
            assert "events" in result
            assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_list_events_success(self, prediction_tool, sample_event):
        """List events should return filtered events."""
        with patch.object(
            DFlowPredictionClient,
            "list_events",
            new_callable=AsyncMock,
            return_value={"events": [sample_event], "count": 1, "cursor": None},
        ):
            result = await prediction_tool.execute(action="list_events", limit=10)
            assert result["status"] == "success"
            assert "events" in result

    @pytest.mark.asyncio
    async def test_get_event_requires_event_id(self, prediction_tool):
        """Get event should require event_id."""
        result = await prediction_tool.execute(action="get_event")
        assert result["status"] == "error"
        assert "event_id" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_get_event_success(self, prediction_tool, sample_event):
        """Get event should return event with safety score."""
        with patch.object(
            DFlowPredictionClient,
            "get_event",
            new_callable=AsyncMock,
            return_value=sample_event,
        ):
            result = await prediction_tool.execute(
                action="get_event", event_id="PRES-2028-DEM"
            )
            assert result["status"] == "success"
            assert "event" in result

    @pytest.mark.asyncio
    async def test_list_markets_success(self, prediction_tool, sample_market):
        """List markets should return filtered markets."""
        with patch.object(
            DFlowPredictionClient,
            "list_markets",
            new_callable=AsyncMock,
            return_value={"markets": [sample_market], "count": 1, "cursor": None},
        ):
            result = await prediction_tool.execute(action="list_markets")
            assert result["status"] == "success"
            assert "markets" in result

    @pytest.mark.asyncio
    async def test_get_market_requires_id_or_mint(self, prediction_tool):
        """Get market should require market_id or mint_address."""
        result = await prediction_tool.execute(action="get_market")
        assert result["status"] == "error"
        assert (
            "market_id" in result["message"].lower()
            or "mint" in result["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_get_market_by_id_success(self, prediction_tool, sample_market):
        """Get market by ID should return market with mints."""
        with patch.object(
            DFlowPredictionClient,
            "get_market",
            new_callable=AsyncMock,
            return_value=sample_market,
        ):
            result = await prediction_tool.execute(
                action="get_market", market_id="PRES-2028-DEM-HARRIS"
            )
            assert result["status"] == "success"
            assert "market" in result
            # Should have extracted mints
            assert result["market"].get("yes_mint") == "YesMintAddress123"
            assert result["market"].get("no_mint") == "NoMintAddress456"

    @pytest.mark.asyncio
    async def test_get_market_by_mint_success(self, prediction_tool, sample_market):
        """Get market by mint address should work."""
        with patch.object(
            DFlowPredictionClient,
            "get_market",
            new_callable=AsyncMock,
            return_value=sample_market,
        ):
            result = await prediction_tool.execute(
                action="get_market", mint_address="YesMintAddress123"
            )
            assert result["status"] == "success"


# =============================================================================
# BUY ACTION TESTS
# =============================================================================


class TestBuyAction:
    """Test buy action."""

    @pytest.mark.asyncio
    async def test_buy_requires_private_key(self, prediction_tool_no_key):
        """Buy should require private_key configuration."""
        result = await prediction_tool_no_key.execute(
            action="buy",
            market_id="TEST",
            side="YES",
            amount=10,
        )
        assert result["status"] == "error"
        assert "private_key" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_buy_requires_rpc_url(self, prediction_tool):
        """Buy should require rpc_url configuration."""
        prediction_tool._rpc_url = None
        result = await prediction_tool.execute(
            action="buy",
            market_id="TEST",
            side="YES",
            amount=10,
        )
        assert result["status"] == "error"
        assert "rpc_url" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_buy_requires_market_id_or_mint(self, prediction_tool):
        """Buy should require market_id or mint_address."""
        result = await prediction_tool.execute(
            action="buy",
            side="YES",
            amount=10,
        )
        assert result["status"] == "error"
        assert "market_id" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_buy_requires_side(self, prediction_tool):
        """Buy should require side (YES/NO)."""
        result = await prediction_tool.execute(
            action="buy",
            market_id="TEST",
            amount=10,
        )
        assert result["status"] == "error"
        assert "side" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_buy_requires_amount(self, prediction_tool):
        """Buy should require amount."""
        result = await prediction_tool.execute(
            action="buy",
            market_id="TEST",
            side="YES",
        )
        assert result["status"] == "error"
        assert "amount" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_buy_blocks_low_safety_market(
        self, prediction_tool, low_quality_market
    ):
        """Buy should block LOW safety markets unless include_risky=True."""
        low_quality_market["safety"] = {
            "score": "LOW",
            "warnings": [],
            "recommendation": "AVOID",
        }

        with (
            patch.object(
                DFlowPredictionClient,
                "get_market",
                new_callable=AsyncMock,
                return_value=low_quality_market,
            ),
            patch("solana_agent.tools.dflow_prediction.Keypair") as MockKeypair,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            result = await prediction_tool.execute(
                action="buy",
                market_id="RANDOM-CAT-SLEEP",
                side="YES",
                amount=10,
            )
            assert result["status"] == "error"
            assert "LOW" in result["message"]
            assert "include_risky" in result["message"]

    @pytest.mark.asyncio
    async def test_buy_success_sync(self, prediction_tool, sample_market):
        """Buy should execute successfully for sync orders."""
        order_response = {
            "transaction": "base64tx...",
            "executionMode": "sync",
            "requestId": "req123",
            "inAmount": "10000000",
            "outAmount": "28571428",
            "minOutAmount": "28000000",
            "priceImpactPct": "0.1",
        }

        order_result = DFlowPredictionOrderResult(
            success=True,
            signature="sig123",
            execution_mode="sync",
            in_amount="10000000",
            out_amount="28571428",
            min_out_amount="28000000",
            price_impact_pct="0.1",
        )

        with (
            patch.object(
                DFlowPredictionClient,
                "get_market",
                new_callable=AsyncMock,
                return_value=sample_market,
            ),
            patch.object(
                DFlowPredictionClient,
                "get_prediction_order",
                new_callable=AsyncMock,
                return_value=order_response,
            ),
            patch.object(
                DFlowPredictionClient,
                "execute_prediction_order_blocking",
                new_callable=AsyncMock,
                return_value=order_result,
            ),
            patch("solana_agent.tools.dflow_prediction.Keypair") as MockKeypair,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            result = await prediction_tool.execute(
                action="buy",
                market_id="PRES-2028-DEM-HARRIS",
                side="YES",
                amount=10,
            )

            assert result["status"] == "success"
            assert result["action"] == "buy"
            assert result["side"] == "YES"
            assert result["signature"] == "sig123"
            assert result["execution_mode"] == "sync"

    @pytest.mark.asyncio
    async def test_buy_handles_order_failure(self, prediction_tool, sample_market):
        """Buy should handle order execution failure."""
        order_response = {
            "transaction": "base64tx...",
            "executionMode": "sync",
            "requestId": "req123",
        }

        order_result = DFlowPredictionOrderResult(
            success=False,
            signature="sig123",
            execution_mode="sync",
            in_amount=None,
            out_amount=None,
            min_out_amount=None,
            price_impact_pct=None,
            error="Insufficient funds",
        )

        with (
            patch.object(
                DFlowPredictionClient,
                "get_market",
                new_callable=AsyncMock,
                return_value=sample_market,
            ),
            patch.object(
                DFlowPredictionClient,
                "get_prediction_order",
                new_callable=AsyncMock,
                return_value=order_response,
            ),
            patch.object(
                DFlowPredictionClient,
                "execute_prediction_order_blocking",
                new_callable=AsyncMock,
                return_value=order_result,
            ),
            patch("solana_agent.tools.dflow_prediction.Keypair") as MockKeypair,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            result = await prediction_tool.execute(
                action="buy",
                market_id="PRES-2028-DEM-HARRIS",
                side="YES",
                amount=10,
            )

            assert result["status"] == "error"
            assert "Insufficient funds" in result["message"]


# =============================================================================
# SELL ACTION TESTS
# =============================================================================


class TestSellAction:
    """Test sell action."""

    @pytest.mark.asyncio
    async def test_sell_requires_private_key(self, prediction_tool_no_key):
        """Sell should require private_key configuration."""
        result = await prediction_tool_no_key.execute(
            action="sell",
            market_id="TEST",
            side="YES",
            amount=10,
        )
        assert result["status"] == "error"
        assert "private_key" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_sell_requires_side(self, prediction_tool):
        """Sell should require side (YES/NO)."""
        result = await prediction_tool.execute(
            action="sell",
            market_id="TEST",
            amount=10,
        )
        assert result["status"] == "error"
        assert "side" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_sell_success(self, prediction_tool, sample_market):
        """Sell should execute successfully."""
        order_response = {
            "transaction": "base64tx...",
            "executionMode": "sync",
            "requestId": "req123",
            "inAmount": "28571428",
            "outAmount": "9500000",
            "minOutAmount": "9400000",
        }

        order_result = DFlowPredictionOrderResult(
            success=True,
            signature="sig456",
            execution_mode="sync",
            in_amount="28571428",
            out_amount="9500000",
            min_out_amount="9400000",
            price_impact_pct="0.2",
        )

        with (
            patch.object(
                DFlowPredictionClient,
                "get_market",
                new_callable=AsyncMock,
                return_value=sample_market,
            ),
            patch.object(
                DFlowPredictionClient,
                "get_prediction_order",
                new_callable=AsyncMock,
                return_value=order_response,
            ),
            patch.object(
                DFlowPredictionClient,
                "execute_prediction_order_blocking",
                new_callable=AsyncMock,
                return_value=order_result,
            ),
            patch("solana_agent.tools.dflow_prediction.Keypair") as MockKeypair,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            result = await prediction_tool.execute(
                action="sell",
                market_id="PRES-2028-DEM-HARRIS",
                side="YES",
                amount=28.57,
            )

            assert result["status"] == "success"
            assert result["action"] == "sell"
            assert result["side"] == "YES"
            assert result["signature"] == "sig456"


# =============================================================================
# POSITIONS ACTION TESTS
# =============================================================================


class TestPositionsAction:
    """Test positions action."""

    @pytest.mark.asyncio
    async def test_positions_requires_private_key(self, prediction_tool_no_key):
        """Positions should require private_key configuration."""
        result = await prediction_tool_no_key.execute(action="positions")
        assert result["status"] == "error"
        assert "private_key" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_positions_requires_rpc_url(self):
        """Positions should require rpc_url configuration."""
        tool = DFlowPredictionTool()
        tool.configure(
            {
                "tools": {
                    "dflow_prediction": {
                        "private_key": "5jGR...base58privatekey",
                        # No rpc_url
                    }
                }
            }
        )
        with patch("solana_agent.tools.dflow_prediction.Keypair") as MockKeypair:
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            result = await tool.execute(action="positions")
            assert result["status"] == "error"
            assert "rpc_url must be configured" in result["message"]

    @pytest.mark.asyncio
    async def test_positions_returns_positions(self, prediction_tool):
        """Positions should return actual positions via get_positions."""
        with (
            patch.object(
                DFlowPredictionClient,
                "get_positions",
                new_callable=AsyncMock,
                return_value={
                    "status": "success",
                    "wallet": "UserPubkey123",
                    "position_count": 2,
                    "positions": [
                        {
                            "mint": "Mint1",
                            "ticker": "PRES-2028",
                            "side": "YES",
                            "amount": "1000000",
                            "ui_amount": 1.0,
                            "decimals": 6,
                        },
                        {
                            "mint": "Mint2",
                            "ticker": "BTC-100K",
                            "side": "NO",
                            "amount": "2000000",
                            "ui_amount": 2.0,
                            "decimals": 6,
                        },
                    ],
                    "hint": "Each position represents outcome tokens.",
                },
            ),
            patch("solana_agent.tools.dflow_prediction.Keypair") as MockKeypair,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            result = await prediction_tool.execute(action="positions")
            assert result["status"] == "success"
            assert result["wallet"] == "UserPubkey123"
            assert result["position_count"] == 2
            assert len(result["positions"]) == 2

    @pytest.mark.asyncio
    async def test_positions_no_positions(self, prediction_tool):
        """Positions should return empty when no positions found."""
        with (
            patch.object(
                DFlowPredictionClient,
                "get_positions",
                new_callable=AsyncMock,
                return_value={
                    "status": "success",
                    "wallet": "UserPubkey123",
                    "position_count": 0,
                    "positions": [],
                    "hint": "No prediction market positions found.",
                },
            ),
            patch("solana_agent.tools.dflow_prediction.Keypair") as MockKeypair,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            result = await prediction_tool.execute(action="positions")
            assert result["status"] == "success"
            assert result["position_count"] == 0
            assert result["positions"] == []


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_unknown_action_error(self, prediction_tool):
        """Unknown action should return error."""
        result = await prediction_tool.execute(action="unknown_action")
        assert result["status"] == "error"
        assert "unknown" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_api_exception_handled(self, prediction_tool):
        """API exceptions should be caught and returned."""
        with patch.object(
            DFlowPredictionClient,
            "search",
            new_callable=AsyncMock,
            side_effect=Exception("API timeout"),
        ):
            result = await prediction_tool.execute(action="search", query="test")
            assert result["status"] == "error"
            assert "API timeout" in result["message"]


# =============================================================================
# CLIENT TESTS
# =============================================================================


class TestDFlowPredictionClient:
    """Test DFlowPredictionClient methods."""

    def test_apply_quality_filters(self):
        """Quality filters should remove low-quality items."""
        client = DFlowPredictionClient(min_volume_usd=1000, min_liquidity_usd=500)

        items = [
            {"volume": 5000, "liquidity": 2000},  # passes
            {"volume": 500, "liquidity": 2000},  # fails volume
            {"volume": 5000, "liquidity": 200},  # fails liquidity
            {"volume": 100, "liquidity": 50},  # fails both
        ]

        filtered = client._apply_quality_filters(items)
        assert len(filtered) == 1
        assert filtered[0]["volume"] == 5000

    def test_apply_quality_filters_include_risky(self):
        """include_risky should bypass filters."""
        client = DFlowPredictionClient(
            min_volume_usd=1000, min_liquidity_usd=500, include_risky=True
        )

        items = [
            {"volume": 5000, "liquidity": 2000},
            {"volume": 100, "liquidity": 50},
        ]

        filtered = client._apply_quality_filters(items)
        assert len(filtered) == 2

    def test_add_safety_scores(self):
        """Safety scores should be added to items."""
        client = DFlowPredictionClient()

        items = [
            {
                "ticker": "A",
                "volume": 50000,
                "liquidity": 10000,
                "rulesPrimary": "x" * 100,
            },
            {"ticker": "B", "volume": 100, "liquidity": 50, "rulesPrimary": "short"},
        ]

        result = client._add_safety_scores(items)

        assert "safety" in result[0]
        assert result[0]["safety"]["score"] in ["HIGH", "MEDIUM", "LOW"]
        assert "safety" in result[1]

    def test_add_safety_scores_with_resolution_date_int(self):
        """Resolution date should be formatted from int timestamp."""
        client = DFlowPredictionClient()

        items = [
            {
                "ticker": "A",
                "volume": 50000,
                "liquidity": 10000,
                "rulesPrimary": "x" * 100,
                "closeTime": 1735689600,  # 2025-01-01 00:00:00 UTC
            },
        ]

        result = client._add_safety_scores(items)
        assert "resolution_date" in result[0]
        assert "2025-01-01" in result[0]["resolution_date"]
        assert "UTC" in result[0]["resolution_date"]

    def test_add_safety_scores_with_resolution_date_string(self):
        """Resolution date should handle string date."""
        client = DFlowPredictionClient()

        items = [
            {
                "ticker": "A",
                "volume": 50000,
                "liquidity": 10000,
                "rulesPrimary": "x" * 100,
                "closeTime": "2025-01-01T00:00:00Z",  # String format
            },
        ]

        result = client._add_safety_scores(items)
        assert "resolution_date" in result[0]
        assert result[0]["resolution_date"] == "2025-01-01T00:00:00Z"

    def test_add_safety_scores_no_resolution_date(self):
        """Items without resolution date should not have resolution_date field."""
        client = DFlowPredictionClient()

        items = [
            {
                "ticker": "A",
                "volume": 50000,
                "liquidity": 10000,
                "rulesPrimary": "x" * 100,
                # No closeTime, expirationTime, or endDate
            },
        ]

        result = client._add_safety_scores(items)
        assert "resolution_date" not in result[0]

    def test_add_safety_scores_with_invalid_timestamp(self):
        """Invalid timestamp should fallback to string representation."""
        client = DFlowPredictionClient()

        items = [
            {
                "ticker": "A",
                "volume": 50000,
                "liquidity": 10000,
                "rulesPrimary": "x" * 100,
                "closeTime": -99999999999999,  # Invalid timestamp
            },
        ]

        result = client._add_safety_scores(items)
        assert "resolution_date" in result[0]
        # Should fallback to string of the invalid value
        assert "-99999999999999" in result[0]["resolution_date"]


# =============================================================================
# BLOCKING EXECUTION TESTS
# =============================================================================


class TestBlockingExecution:
    """Test blocking async execution."""

    @pytest.mark.asyncio
    async def test_sync_execution_returns_immediately(self):
        """Sync execution should return after single transaction."""
        client = DFlowPredictionClient()

        order_response = {
            "transaction": "base64tx",
            "executionMode": "sync",
            "inAmount": "1000",
            "outAmount": "2000",
            "minOutAmount": "1900",
            "priceImpactPct": "0.1",
        }

        async def mock_sign_send(tx):
            return "sig123"

        result = await client.execute_prediction_order_blocking(
            order_response, mock_sign_send
        )

        assert result.success is True
        assert result.signature == "sig123"
        assert result.execution_mode == "sync"
        assert result.in_amount == "1000"

    @pytest.mark.asyncio
    async def test_async_execution_polls_until_closed(self):
        """Async execution should poll until order is closed."""
        client = DFlowPredictionClient()

        order_response = {
            "transaction": "base64tx",
            "executionMode": "async",
            "requestId": "req123",
            "inAmount": "1000",
            "outAmount": "2000",
            "minOutAmount": "1900",
        }

        poll_count = 0

        async def mock_sign_send(tx):
            return "sig123"

        async def mock_get_status(request_id):
            nonlocal poll_count
            poll_count += 1
            if poll_count >= 2:
                return {
                    "status": "closed",
                    "inAmount": "1000",
                    "outAmount": "2000",
                    "fills": [],
                }
            return {"status": "pending"}

        with patch.object(
            client, "get_prediction_order_status", side_effect=mock_get_status
        ):
            result = await client.execute_prediction_order_blocking(
                order_response,
                mock_sign_send,
                max_wait_seconds=10,
                poll_interval_seconds=0.1,
            )

        assert result.success is True
        assert result.execution_mode == "async"
        assert poll_count >= 2

    @pytest.mark.asyncio
    async def test_async_execution_handles_failed_status(self):
        """Async execution should handle failed orders."""
        client = DFlowPredictionClient()

        order_response = {
            "transaction": "base64tx",
            "executionMode": "async",
            "requestId": "req123",
        }

        async def mock_sign_send(tx):
            return "sig123"

        async def mock_get_status(request_id):
            return {"status": "failed"}

        with patch.object(
            client, "get_prediction_order_status", side_effect=mock_get_status
        ):
            result = await client.execute_prediction_order_blocking(
                order_response,
                mock_sign_send,
                max_wait_seconds=5,
                poll_interval_seconds=0.1,
            )

        assert result.success is False
        assert "failed" in result.error

    @pytest.mark.asyncio
    async def test_async_execution_timeout(self):
        """Async execution should timeout after max_wait_seconds."""
        client = DFlowPredictionClient()

        order_response = {
            "transaction": "base64tx",
            "executionMode": "async",
            "requestId": "req123",
        }

        async def mock_sign_send(tx):
            return "sig123"

        async def mock_get_status(request_id):
            return {"status": "pending"}

        with patch.object(
            client, "get_prediction_order_status", side_effect=mock_get_status
        ):
            result = await client.execute_prediction_order_blocking(
                order_response,
                mock_sign_send,
                max_wait_seconds=0.2,
                poll_interval_seconds=0.05,
            )

        assert result.success is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_no_transaction_error(self):
        """Should return error when no transaction in response."""
        client = DFlowPredictionClient()

        order_response = {"executionMode": "sync"}

        async def mock_sign_send(tx):
            return "sig123"

        result = await client.execute_prediction_order_blocking(
            order_response, mock_sign_send
        )

        assert result.success is False
        assert "No transaction" in result.error


# =============================================================================
# PLUGIN TESTS
# =============================================================================


class TestDFlowPredictionPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Plugin should have correct name."""
        plugin = DFlowPredictionPlugin()
        assert plugin.name == "dflow_prediction"

    def test_plugin_description(self):
        """Plugin should have description."""
        plugin = DFlowPredictionPlugin()
        assert "prediction market" in plugin.description.lower()


# =============================================================================
# ORDER RESULT TESTS
# =============================================================================


class TestDFlowPredictionOrderResult:
    """Test order result dataclass."""

    def test_success_result_to_dict(self):
        """Successful result should serialize correctly."""
        result = DFlowPredictionOrderResult(
            success=True,
            signature="sig123",
            execution_mode="sync",
            in_amount="1000",
            out_amount="2000",
            min_out_amount="1900",
            price_impact_pct="0.1",
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["signature"] == "sig123"
        assert d["in_amount"] == "1000"

    def test_failure_result_to_dict(self):
        """Failed result should serialize correctly."""
        result = DFlowPredictionOrderResult(
            success=False,
            signature=None,
            execution_mode="async",
            in_amount=None,
            out_amount=None,
            min_out_amount=None,
            price_impact_pct=None,
            error="Order failed",
        )

        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "Order failed"
        assert "signature" not in d

    def test_result_with_fills(self):
        """Result with fills should serialize correctly."""
        result = DFlowPredictionOrderResult(
            success=True,
            signature="sig123",
            execution_mode="async",
            in_amount="1000",
            out_amount="2000",
            min_out_amount="1900",
            price_impact_pct="0.1",
            fills=[{"amount": "500"}, {"amount": "500"}],
        )

        d = result.to_dict()
        assert d["fills"] == [{"amount": "500"}, {"amount": "500"}]


# =============================================================================
# SIGN AND SEND TESTS
# =============================================================================


class TestSignAndSend:
    """Test the _sign_and_send method."""

    @pytest.fixture
    def configured_tool(self):
        """Create a tool with valid config for signing tests."""
        # Generate a fresh keypair for each test
        from solders.keypair import Keypair as SoldersKeypair

        test_keypair = SoldersKeypair()

        tool = DFlowPredictionTool()
        tool.configure(
            {
                "tools": {
                    "dflow_prediction": {
                        "private_key": str(test_keypair),
                        "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                    }
                }
            }
        )
        return tool

    def test_get_keypair_no_config(self):
        """Should raise when private_key not configured."""
        tool = DFlowPredictionTool()
        tool.configure({"tools": {"dflow_prediction": {}}})
        with pytest.raises(ValueError, match="not configured"):
            tool._get_keypair()

    def test_get_keypair_success(self, configured_tool):
        """Should return keypair when configured."""
        keypair = configured_tool._get_keypair()
        assert keypair is not None
        # Should be a valid Keypair object
        assert hasattr(keypair, "pubkey")

    @pytest.mark.asyncio
    async def test_sign_and_send_no_rpc_url(self, configured_tool):
        """Should raise when rpc_url not configured."""
        configured_tool._rpc_url = None
        keypair = configured_tool._get_keypair()

        with pytest.raises(Exception, match="rpc_url must be configured"):
            await configured_tool._sign_and_send(keypair, "base64tx")

    @pytest.mark.asyncio
    async def test_sign_and_send_blockhash_failure(self, configured_tool):
        """Should handle blockhash fetch failure."""
        keypair = configured_tool._get_keypair()

        with patch(
            "solana_agent.tools.dflow_prediction.get_fresh_blockhash",
            new_callable=AsyncMock,
        ) as mock_blockhash:
            mock_blockhash.return_value = {"error": "RPC timeout"}

            with pytest.raises(Exception) as exc_info:
                await configured_tool._sign_and_send(keypair, "dummybase64")

            assert "blockhash" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_sign_and_send_success(self, configured_tool):
        """Should successfully sign and send transaction."""
        import base64
        from solders.transaction import VersionedTransaction
        from solders.message import MessageV0
        from solders.hash import Hash
        from solders.signature import Signature

        keypair = configured_tool._get_keypair()

        # Create a minimal valid transaction - use unsigned signature placeholder
        message = MessageV0.try_compile(
            payer=keypair.pubkey(),
            instructions=[],
            address_lookup_table_accounts=[],
            recent_blockhash=Hash.default(),
        )
        # Create with default signature placeholder
        tx = VersionedTransaction.populate(message, [Signature.default()])
        tx_b64 = base64.b64encode(bytes(tx)).decode()

        with (
            patch(
                "solana_agent.tools.dflow_prediction.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.dflow_prediction.replace_blockhash_in_transaction"
            ) as mock_replace,
            patch(
                "solana_agent.tools.dflow_prediction.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_blockhash.return_value = {
                "blockhash": "GHtXQBsoZHVnNFa9YevAzFr17DJjgHXk3ycTKD5xD3Zi",
                "lastValidBlockHeight": 12345678,
            }
            mock_replace.return_value = tx_b64  # Return the same tx for simplicity
            mock_send.return_value = {
                "success": True,
                "signature": "5VERv8NMvzbJMEkV8xnrLkEaWRtSz9CosKDYjCJjBRnbJLgp8uirBgmQpjKhoR4tjF3ZpRzrFmBV6UjKdiSZkQUW",
            }

            signature = await configured_tool._sign_and_send(keypair, tx_b64)
            assert (
                signature
                == "5VERv8NMvzbJMEkV8xnrLkEaWRtSz9CosKDYjCJjBRnbJLgp8uirBgmQpjKhoR4tjF3ZpRzrFmBV6UjKdiSZkQUW"
            )

    @pytest.mark.asyncio
    async def test_sign_and_send_rpc_send_failure(self, configured_tool):
        """Should handle RPC send failure."""
        import base64
        from solders.transaction import VersionedTransaction
        from solders.message import MessageV0
        from solders.hash import Hash
        from solders.signature import Signature

        keypair = configured_tool._get_keypair()

        message = MessageV0.try_compile(
            payer=keypair.pubkey(),
            instructions=[],
            address_lookup_table_accounts=[],
            recent_blockhash=Hash.default(),
        )
        tx = VersionedTransaction.populate(message, [Signature.default()])
        tx_b64 = base64.b64encode(bytes(tx)).decode()

        with (
            patch(
                "solana_agent.tools.dflow_prediction.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.dflow_prediction.replace_blockhash_in_transaction"
            ) as mock_replace,
            patch(
                "solana_agent.tools.dflow_prediction.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_blockhash.return_value = {
                "blockhash": "GHtXQBsoZHVnNFa9YevAzFr17DJjgHXk3ycTKD5xD3Zi",
                "lastValidBlockHeight": 12345678,
            }
            mock_replace.return_value = tx_b64
            mock_send.return_value = {
                "success": False,
                "error": "Transaction simulation failed",
            }

            with pytest.raises(Exception) as exc_info:
                await configured_tool._sign_and_send(keypair, tx_b64)

            assert (
                "simulation failed" in str(exc_info.value).lower()
                or "failed" in str(exc_info.value).lower()
            )

    @pytest.mark.asyncio
    async def test_sign_and_send_taker_not_found(self, configured_tool):
        """Should handle when taker pubkey not in signers."""
        import base64
        from solders.transaction import VersionedTransaction
        from solders.message import MessageV0
        from solders.hash import Hash
        from solders.keypair import Keypair as SoldersKeypair
        from solders.signature import Signature

        # Use a different keypair for the transaction
        different_keypair = SoldersKeypair()

        message = MessageV0.try_compile(
            payer=different_keypair.pubkey(),
            instructions=[],
            address_lookup_table_accounts=[],
            recent_blockhash=Hash.default(),
        )
        tx = VersionedTransaction.populate(message, [Signature.default()])
        tx_b64 = base64.b64encode(bytes(tx)).decode()

        # But try to sign with configured keypair
        configured_keypair = configured_tool._get_keypair()

        with (
            patch(
                "solana_agent.tools.dflow_prediction.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.dflow_prediction.replace_blockhash_in_transaction"
            ) as mock_replace,
        ):
            mock_blockhash.return_value = {
                "blockhash": "GHtXQBsoZHVnNFa9YevAzFr17DJjgHXk3ycTKD5xD3Zi",
                "lastValidBlockHeight": 12345678,
            }
            mock_replace.return_value = tx_b64

            with pytest.raises(Exception) as exc_info:
                await configured_tool._sign_and_send(configured_keypair, tx_b64)

            assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_sign_and_send_with_payer(self, configured_tool):
        """Should sign with payer when configured."""
        import base64
        from solders.transaction import VersionedTransaction
        from solders.message import MessageV0
        from solders.hash import Hash
        from solders.keypair import Keypair as SoldersKeypair
        from solders.signature import Signature

        keypair = configured_tool._get_keypair()
        payer_keypair = SoldersKeypair()
        configured_tool._payer_private_key = str(payer_keypair)

        # Create a transaction with both payer and keypair as signers
        # For simplicity, just test the payer signing logic path with single signer
        message = MessageV0.try_compile(
            payer=keypair.pubkey(),
            instructions=[],
            address_lookup_table_accounts=[],
            recent_blockhash=Hash.default(),
        )
        tx = VersionedTransaction.populate(message, [Signature.default()])
        tx_b64 = base64.b64encode(bytes(tx)).decode()

        with (
            patch(
                "solana_agent.tools.dflow_prediction.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.dflow_prediction.replace_blockhash_in_transaction"
            ) as mock_replace,
            patch(
                "solana_agent.tools.dflow_prediction.send_raw_transaction_with_priority",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_blockhash.return_value = {
                "blockhash": "GHtXQBsoZHVnNFa9YevAzFr17DJjgHXk3ycTKD5xD3Zi",
                "lastValidBlockHeight": 12345678,
            }
            mock_replace.return_value = tx_b64
            mock_send.return_value = {"success": True, "signature": "testsig123"}

            signature = await configured_tool._sign_and_send(keypair, tx_b64)
            assert signature == "testsig123"

    @pytest.mark.asyncio
    async def test_sign_and_send_with_payer_found_in_signers(self, configured_tool):
        """Should sign when payer is found in transaction signers."""
        import base64
        from solders.transaction import VersionedTransaction
        from solders.message import MessageV0
        from solders.hash import Hash
        from solders.keypair import Keypair as SoldersKeypair
        from solders.signature import Signature

        # Create payer and use it as the transaction payer
        payer_keypair = SoldersKeypair()
        configured_tool._payer_private_key = str(payer_keypair)

        # Get main keypair
        main_keypair = configured_tool._get_keypair()

        # Create transaction with payer as the account
        # The payer is the first signer (index 0)
        message = MessageV0.try_compile(
            payer=payer_keypair.pubkey(),
            instructions=[],
            address_lookup_table_accounts=[],
            recent_blockhash=Hash.default(),
        )
        tx = VersionedTransaction.populate(message, [Signature.default()])
        tx_b64 = base64.b64encode(bytes(tx)).decode()

        with (
            patch(
                "solana_agent.tools.dflow_prediction.get_fresh_blockhash",
                new_callable=AsyncMock,
            ) as mock_blockhash,
            patch(
                "solana_agent.tools.dflow_prediction.replace_blockhash_in_transaction"
            ) as mock_replace,
        ):
            mock_blockhash.return_value = {
                "blockhash": "GHtXQBsoZHVnNFa9YevAzFr17DJjgHXk3ycTKD5xD3Zi",
                "lastValidBlockHeight": 12345678,
            }
            mock_replace.return_value = tx_b64

            # This should fail because main_keypair is not in the transaction
            # But payer signing logic should be executed first (lines 219-220)
            with pytest.raises(Exception) as exc_info:
                await configured_tool._sign_and_send(main_keypair, tx_b64)

            # Payer signing happens but main keypair not found
            assert "not found" in str(exc_info.value).lower()


# =============================================================================
# ADDITIONAL SELL ACTION TESTS
# =============================================================================


class TestSellActionExtra:
    """Additional sell action tests for coverage."""

    @pytest.mark.asyncio
    async def test_sell_requires_rpc_url(self, prediction_tool):
        """Sell should require rpc_url configuration."""
        prediction_tool._rpc_url = None
        result = await prediction_tool.execute(
            action="sell",
            market_id="TEST",
            side="YES",
            amount=10,
        )
        assert result["status"] == "error"
        assert "rpc_url" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_sell_requires_market_id_or_mint(self, prediction_tool):
        """Sell should require market_id or mint_address."""
        result = await prediction_tool.execute(
            action="sell",
            side="YES",
            amount=10,
        )
        assert result["status"] == "error"
        assert "market_id" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_sell_requires_amount(self, prediction_tool):
        """Sell should require amount."""
        result = await prediction_tool.execute(
            action="sell",
            market_id="TEST",
            side="YES",
        )
        assert result["status"] == "error"
        assert "amount" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_sell_no_outcome_mint_found(self, prediction_tool, sample_market):
        """Sell should handle missing outcome mint."""
        # Market without YES mint
        sample_market_no_mints = sample_market.copy()
        sample_market_no_mints["accounts"] = {}

        with (
            patch.object(
                DFlowPredictionClient,
                "get_market",
                new_callable=AsyncMock,
                return_value=sample_market_no_mints,
            ),
            patch("solana_agent.tools.dflow_prediction.Keypair") as MockKeypair,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            result = await prediction_tool.execute(
                action="sell",
                market_id="TEST",
                side="YES",
                amount=10,
            )
            assert result["status"] == "error"
            assert "outcome mint" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_sell_handles_order_failure(self, prediction_tool, sample_market):
        """Sell should handle order execution failure."""
        order_response = {
            "transaction": "base64tx...",
            "executionMode": "sync",
            "requestId": "req123",
        }

        order_result = DFlowPredictionOrderResult(
            success=False,
            signature="sig123",
            execution_mode="sync",
            in_amount=None,
            out_amount=None,
            min_out_amount=None,
            price_impact_pct=None,
            error="Insufficient balance",
        )

        with (
            patch.object(
                DFlowPredictionClient,
                "get_market",
                new_callable=AsyncMock,
                return_value=sample_market,
            ),
            patch.object(
                DFlowPredictionClient,
                "get_prediction_order",
                new_callable=AsyncMock,
                return_value=order_response,
            ),
            patch.object(
                DFlowPredictionClient,
                "execute_prediction_order_blocking",
                new_callable=AsyncMock,
                return_value=order_result,
            ),
            patch("solana_agent.tools.dflow_prediction.Keypair") as MockKeypair,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            result = await prediction_tool.execute(
                action="sell",
                market_id="PRES-2028-DEM-HARRIS",
                side="YES",
                amount=10,
            )

            assert result["status"] == "error"
            assert "Insufficient balance" in result["message"]


# =============================================================================
# ADDITIONAL BUY ACTION TESTS
# =============================================================================


class TestBuyActionExtra:
    """Additional buy action tests for coverage."""

    @pytest.mark.asyncio
    async def test_buy_no_outcome_mint_found(self, prediction_tool, sample_market):
        """Buy should handle missing outcome mint."""
        sample_market_no_mints = sample_market.copy()
        sample_market_no_mints["accounts"] = {}

        with (
            patch.object(
                DFlowPredictionClient,
                "get_market",
                new_callable=AsyncMock,
                return_value=sample_market_no_mints,
            ),
            patch("solana_agent.tools.dflow_prediction.Keypair") as MockKeypair,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            result = await prediction_tool.execute(
                action="buy",
                market_id="TEST",
                side="YES",
                amount=10,
            )
            assert result["status"] == "error"
            assert "outcome mint" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_buy_with_include_risky(self, prediction_tool, low_quality_market):
        """Buy should allow LOW safety markets with include_risky=True."""
        # Add the safety and mints directly
        low_quality_market["safety"] = {
            "score": "LOW",
            "warnings": ["Low volume"],
            "recommendation": "AVOID",
        }
        low_quality_market["yes_mint"] = "YesMint123"
        low_quality_market["no_mint"] = "NoMint456"

        order_response = {
            "transaction": "base64tx...",
            "executionMode": "sync",
            "requestId": "req123",
            "inAmount": "10000000",
            "outAmount": "28571428",
        }

        order_result = DFlowPredictionOrderResult(
            success=True,
            signature="sig123",
            execution_mode="sync",
            in_amount="10000000",
            out_amount="28571428",
            min_out_amount="28000000",
            price_impact_pct="0.1",
        )

        with (
            patch.object(
                DFlowPredictionTool,
                "_get_market_with_mints",
                new_callable=AsyncMock,
                return_value=low_quality_market,
            ),
            patch.object(
                DFlowPredictionClient,
                "get_prediction_order",
                new_callable=AsyncMock,
                return_value=order_response,
            ),
            patch.object(
                DFlowPredictionClient,
                "execute_prediction_order_blocking",
                new_callable=AsyncMock,
                return_value=order_result,
            ),
            patch("solana_agent.tools.dflow_prediction.Keypair") as MockKeypair,
        ):
            mock_keypair = MagicMock()
            mock_keypair.pubkey.return_value = "UserPubkey123"
            MockKeypair.from_base58_string.return_value = mock_keypair

            result = await prediction_tool.execute(
                action="buy",
                market_id="RANDOM-CAT-SLEEP",
                side="YES",
                amount=10,
                include_risky=True,
            )
            # Should succeed even with LOW safety when include_risky=True
            assert result["status"] == "success"


# =============================================================================
# ASYNC EXECUTION ADDITIONAL TESTS
# =============================================================================


class TestBlockingExecutionExtra:
    """Additional blocking execution tests."""

    @pytest.mark.asyncio
    async def test_async_execution_handles_expired_status(self):
        """Async execution should handle expired orders."""
        client = DFlowPredictionClient()

        order_response = {
            "transaction": "base64tx",
            "executionMode": "async",
            "requestId": "req123",
        }

        async def mock_sign_send(tx):
            return "sig123"

        async def mock_get_status(request_id):
            return {"status": "expired"}

        with patch.object(
            client, "get_prediction_order_status", side_effect=mock_get_status
        ):
            result = await client.execute_prediction_order_blocking(
                order_response,
                mock_sign_send,
                max_wait_seconds=5,
                poll_interval_seconds=0.1,
            )

        assert result.success is False
        assert "expired" in result.error

    @pytest.mark.asyncio
    async def test_async_execution_handles_next_transaction(self):
        """Async execution should handle next transaction requirement."""
        client = DFlowPredictionClient()

        order_response = {
            "transaction": "base64tx",
            "executionMode": "async",
            "requestId": "req123",
        }

        call_count = 0

        async def mock_sign_send(tx):
            return "sig123"

        async def mock_get_status(request_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"status": "pendingClose", "nextTransaction": "nextTx123"}
            else:
                return {
                    "status": "closed",
                    "inAmount": "1000",
                    "outAmount": "2000",
                    "fills": [],
                }

        with patch.object(
            client, "get_prediction_order_status", side_effect=mock_get_status
        ):
            result = await client.execute_prediction_order_blocking(
                order_response,
                mock_sign_send,
                max_wait_seconds=10,
                poll_interval_seconds=0.1,
            )

        assert result.success is True
        assert result.execution_mode == "async"

    @pytest.mark.asyncio
    async def test_async_execution_handles_status_check_exception(self):
        """Async execution should continue on status check exceptions."""
        client = DFlowPredictionClient()

        order_response = {
            "transaction": "base64tx",
            "executionMode": "async",
            "requestId": "req123",
        }

        call_count = 0

        async def mock_sign_send(tx):
            return "sig123"

        async def mock_get_status(request_id):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary network error")
            return {
                "status": "closed",
                "inAmount": "1000",
                "outAmount": "2000",
                "fills": [],
            }

        with patch.object(
            client, "get_prediction_order_status", side_effect=mock_get_status
        ):
            result = await client.execute_prediction_order_blocking(
                order_response,
                mock_sign_send,
                max_wait_seconds=10,
                poll_interval_seconds=0.1,
            )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execution_sign_and_send_exception(self):
        """Should handle exception during sign and send."""
        client = DFlowPredictionClient()

        order_response = {
            "transaction": "base64tx",
            "executionMode": "sync",
        }

        async def mock_sign_send(tx):
            raise Exception("Signing failed")

        result = await client.execute_prediction_order_blocking(
            order_response, mock_sign_send
        )

        assert result.success is False
        assert "Signing failed" in result.error


# =============================================================================
# FULL INTEGRATION TESTS
# =============================================================================


class TestFullIntegration:
    """Tests that exercise the full code path including inner closures."""

    @pytest.fixture
    def integration_tool(self):
        """Create a tool with valid keypair for integration tests."""
        from solders.keypair import Keypair as SoldersKeypair

        test_keypair = SoldersKeypair()

        tool = DFlowPredictionTool()
        tool.configure(
            {
                "tools": {
                    "dflow_prediction": {
                        "private_key": str(test_keypair),
                        "rpc_url": "https://mainnet.helius-rpc.com/?api-key=test-key",
                    }
                }
            }
        )
        return tool

    @pytest.fixture
    def integration_market(self):
        """Market with mints and safety for integration tests."""
        return {
            "ticker": "PRES-2028-DEM-HARRIS",
            "yes_mint": "YesMint123",
            "no_mint": "NoMint456",
            "safety": {"score": "HIGH", "warnings": [], "recommendation": "PROCEED"},
        }

    @pytest.mark.asyncio
    async def test_buy_full_path_with_inner_closure(
        self, integration_tool, integration_market
    ):
        """Buy should exercise the inner sign_and_send closure."""

        order_response = {
            "transaction": "base64tx",
            "executionMode": "sync",
            "requestId": "req123",
            "inAmount": "10000000",
            "outAmount": "28571428",
            "minOutAmount": "28000000",
            "priceImpactPct": "0.1",
        }

        with (
            patch.object(
                DFlowPredictionTool,
                "_get_market_with_mints",
                new_callable=AsyncMock,
                return_value=integration_market,
            ),
            patch.object(
                DFlowPredictionClient,
                "get_prediction_order",
                new_callable=AsyncMock,
                return_value=order_response,
            ),
            patch.object(
                DFlowPredictionTool,
                "_sign_and_send",
                new_callable=AsyncMock,
                return_value="sig123",
            ),
        ):
            result = await integration_tool.execute(
                action="buy",
                market_id="PRES-2028-DEM-HARRIS",
                side="YES",
                amount=10,
            )

            assert result["status"] == "success"
            assert result["signature"] == "sig123"

    @pytest.mark.asyncio
    async def test_sell_full_path_with_inner_closure(
        self, integration_tool, integration_market
    ):
        """Sell should exercise the inner sign_and_send closure."""

        order_response = {
            "transaction": "base64tx",
            "executionMode": "sync",
            "requestId": "req123",
            "inAmount": "28571428",
            "outAmount": "9500000",
            "minOutAmount": "9400000",
        }

        with (
            patch.object(
                DFlowPredictionTool,
                "_get_market_with_mints",
                new_callable=AsyncMock,
                return_value=integration_market,
            ),
            patch.object(
                DFlowPredictionClient,
                "get_prediction_order",
                new_callable=AsyncMock,
                return_value=order_response,
            ),
            patch.object(
                DFlowPredictionTool,
                "_sign_and_send",
                new_callable=AsyncMock,
                return_value="sig456",
            ),
        ):
            result = await integration_tool.execute(
                action="sell",
                market_id="PRES-2028-DEM-HARRIS",
                side="YES",
                amount=28.57,
            )

            assert result["status"] == "success"
            assert result["signature"] == "sig456"


# =============================================================================
# PREDICTION CLIENT HTTP TESTS
# =============================================================================


class TestDFlowPredictionClientHTTP:
    """Test the DFlowPredictionClient HTTP methods directly."""

    @pytest.fixture
    def prediction_client(self):
        return DFlowPredictionClient()

    @pytest.mark.asyncio
    async def test_search_http_success(self, prediction_client):
        """Search should make HTTP request and process response."""
        mock_response = {
            "events": [
                {
                    "ticker": "EVENT1",
                    "volume": 5000,
                    "liquidity": 2000,
                    "rulesPrimary": "x" * 100,
                }
            ]
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.search("test query")
            assert "events" in result
            assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_search_http_error(self, prediction_client):
        """Search should raise exception on HTTP error."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=500, text="Internal Error")
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            with pytest.raises(Exception) as exc_info:
                await prediction_client.search("test")
            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_events_http_success(self, prediction_client):
        """List events should make HTTP request and process response."""
        mock_response = {
            "events": [
                {
                    "ticker": "EVENT1",
                    "volume": 5000,
                    "liquidity": 2000,
                    "rulesPrimary": "x" * 100,
                }
            ],
            "cursor": "next_cursor_123",
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.list_events(
                limit=10, cursor="prev_cursor", series_tickers=["US-POLITICS"]
            )
            assert "events" in result
            assert result["cursor"] == "next_cursor_123"

    @pytest.mark.asyncio
    async def test_list_events_http_error(self, prediction_client):
        """List events should raise exception on HTTP error."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=404, text="Not Found")
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            with pytest.raises(Exception) as exc_info:
                await prediction_client.list_events()
            assert "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_event_http_success(self, prediction_client):
        """Get event should make HTTP request and add safety score."""
        mock_response = {
            "ticker": "EVENT1",
            "volume": 50000,
            "liquidity": 20000,
            "rulesPrimary": "x" * 100,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_event("EVENT1")
            assert result["ticker"] == "EVENT1"
            assert "safety" in result

    @pytest.mark.asyncio
    async def test_get_event_http_error(self, prediction_client):
        """Get event should raise exception on HTTP error."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=404, text="Not Found")
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            with pytest.raises(Exception) as exc_info:
                await prediction_client.get_event("NONEXISTENT")
            assert "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_markets_http_success(self, prediction_client):
        """List markets should make HTTP request and process response."""
        mock_response = {
            "markets": [
                {
                    "ticker": "MARKET1",
                    "volume": 5000,
                    "liquidity": 2000,
                    "rulesPrimary": "x" * 100,
                }
            ],
            "cursor": "next_cursor_123",
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.list_markets(
                limit=10, cursor="prev_cursor"
            )
            assert "markets" in result
            assert result["cursor"] == "next_cursor_123"

    @pytest.mark.asyncio
    async def test_list_markets_http_error(self, prediction_client):
        """List markets should raise exception on HTTP error."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=500, text="Server Error")
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            with pytest.raises(Exception) as exc_info:
                await prediction_client.list_markets()
            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_market_by_id_http_success(self, prediction_client):
        """Get market by ID should make HTTP request and add safety score."""
        mock_response = {
            "ticker": "MARKET1",
            "volume": 50000,
            "liquidity": 20000,
            "rulesPrimary": "x" * 100,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_market(market_id="MARKET1")
            assert result["ticker"] == "MARKET1"
            assert "safety" in result

    @pytest.mark.asyncio
    async def test_get_market_by_mint_http_success(self, prediction_client):
        """Get market by mint address should make HTTP request."""
        mock_response = {
            "ticker": "MARKET1",
            "volume": 50000,
            "liquidity": 20000,
            "rulesPrimary": "x" * 100,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_market(mint_address="MintAddress123")
            assert result["ticker"] == "MARKET1"
            # Verify the URL used by-mint path
            call_args = mock_instance.get.call_args
            url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
            assert "by-mint" in url

    @pytest.mark.asyncio
    async def test_get_market_no_params_error(self, prediction_client):
        """Get market should raise ValueError when no ID or mint provided."""
        with pytest.raises(ValueError, match="Either market_id or mint_address"):
            await prediction_client.get_market()

    @pytest.mark.asyncio
    async def test_get_market_http_error(self, prediction_client):
        """Get market should raise exception on HTTP error."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=404, text="Not Found")
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            with pytest.raises(Exception) as exc_info:
                await prediction_client.get_market(market_id="NONEXISTENT")
            assert "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_series_http_success(self, prediction_client):
        """List series should make HTTP request."""
        mock_response = {"series": [{"ticker": "US-POLITICS", "status": "active"}]}

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.list_series(category="politics")
            assert "series" in result

    @pytest.mark.asyncio
    async def test_list_series_http_error(self, prediction_client):
        """List series should raise exception on HTTP error."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=500, text="Server Error")
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            with pytest.raises(Exception) as exc_info:
                await prediction_client.list_series()
            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_categories_http_success(self, prediction_client):
        """Get categories should make HTTP request."""
        mock_response = {
            "politics": ["elections", "policy"],
            "sports": ["nfl", "nba"],
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_categories()
            assert "politics" in result

    @pytest.mark.asyncio
    async def test_get_categories_http_error(self, prediction_client):
        """Get categories should raise exception on HTTP error."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=500, text="Server Error")
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            with pytest.raises(Exception) as exc_info:
                await prediction_client.get_categories()
            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_trades_by_ticker_http_success(self, prediction_client):
        """Get trades by ticker should make HTTP request."""
        mock_response = {"trades": [{"amount": "1000", "price": "0.35"}]}

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_trades(ticker="MARKET1")
            assert "trades" in result

    @pytest.mark.asyncio
    async def test_get_trades_by_mint_http_success(self, prediction_client):
        """Get trades by mint should make HTTP request."""
        mock_response = {"trades": [{"amount": "1000", "price": "0.35"}]}

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_trades(mint_address="MintAddress123")
            assert "trades" in result
            # Verify the URL used by-mint path
            call_args = mock_instance.get.call_args
            url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
            assert "by-mint" in url

    @pytest.mark.asyncio
    async def test_get_trades_http_error(self, prediction_client):
        """Get trades should raise exception on HTTP error."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=500, text="Server Error")
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            with pytest.raises(Exception) as exc_info:
                await prediction_client.get_trades(ticker="MARKET1")
            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_outcome_mints_http_success(self, prediction_client):
        """Get outcome mints should make HTTP request."""
        mock_response = {
            "MintAddress1": {"market": "MARKET1", "side": "yes"},
            "MintAddress2": {"market": "MARKET1", "side": "no"},
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_outcome_mints()
            assert "MintAddress1" in result

    @pytest.mark.asyncio
    async def test_get_outcome_mints_http_error(self, prediction_client):
        """Get outcome mints should raise exception on HTTP error."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=500, text="Server Error")
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            with pytest.raises(Exception) as exc_info:
                await prediction_client.get_outcome_mints()
            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_prediction_order_http_success(self, prediction_client):
        """Get prediction order should make HTTP request."""
        mock_response = {
            "transaction": "base64tx",
            "executionMode": "sync",
            "inAmount": "1000000",
            "outAmount": "2857142",
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_prediction_order(
                input_mint="USDC",
                output_mint="YesMint",
                amount=1000000,
                user_public_key="UserPubkey",
                platform_fee_bps=50,
                platform_fee_scale=1000,
                fee_account="FeeAccount",
            )
            assert result["transaction"] == "base64tx"

    @pytest.mark.asyncio
    async def test_get_prediction_order_http_error(self, prediction_client):
        """Get prediction order should raise exception on HTTP error."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=400, text="Bad Request")
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            with pytest.raises(Exception) as exc_info:
                await prediction_client.get_prediction_order(
                    input_mint="USDC",
                    output_mint="YesMint",
                    amount=1000000,
                    user_public_key="UserPubkey",
                )
            assert "400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_prediction_order_status_http_success(self, prediction_client):
        """Get prediction order status should make HTTP request."""
        mock_response = {
            "status": "closed",
            "inAmount": "1000000",
            "outAmount": "2857142",
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=200, json=lambda: mock_response)
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_prediction_order_status("req123")
            assert result["status"] == "closed"

    @pytest.mark.asyncio
    async def test_get_prediction_order_status_http_error(self, prediction_client):
        """Get prediction order status should raise exception on HTTP error."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_response_obj = MagicMock(status_code=404, text="Not Found")
            mock_instance.get = AsyncMock(return_value=mock_response_obj)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            with pytest.raises(Exception) as exc_info:
                await prediction_client.get_prediction_order_status("req123")
            assert "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_positions_success_with_positions(self, prediction_client):
        """Get positions should return positions when outcome tokens found."""
        # Mock outcome mints response
        outcome_mints_response = {
            "OutcomeMint1": {"market": "PRES-2028-HARRIS", "side": "yes"},
            "OutcomeMint2": {"market": "PRES-2028-HARRIS", "side": "no"},
            "OutcomeMint3": {"market": "BTC-100K-2025", "side": "yes"},
        }

        # Mock RPC response with token accounts
        rpc_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "value": [
                    {
                        "account": {
                            "data": {
                                "parsed": {
                                    "info": {
                                        "mint": "OutcomeMint1",
                                        "tokenAmount": {
                                            "amount": "1500000",
                                            "uiAmount": 1.5,
                                            "decimals": 6,
                                        },
                                    }
                                }
                            }
                        }
                    },
                    {
                        "account": {
                            "data": {
                                "parsed": {
                                    "info": {
                                        "mint": "SomeOtherToken",
                                        "tokenAmount": {
                                            "amount": "1000000000",
                                            "uiAmount": 1.0,
                                            "decimals": 9,
                                        },
                                    }
                                }
                            }
                        }
                    },
                    {
                        "account": {
                            "data": {
                                "parsed": {
                                    "info": {
                                        "mint": "OutcomeMint3",
                                        "tokenAmount": {
                                            "amount": "5000000",
                                            "uiAmount": 5.0,
                                            "decimals": 6,
                                        },
                                    }
                                }
                            }
                        }
                    },
                ]
            },
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()

            # First call: get_outcome_mints
            mock_outcome_response = MagicMock(
                status_code=200, json=lambda: outcome_mints_response
            )
            # Second call: RPC getTokenAccountsByOwner
            mock_rpc_response = MagicMock(status_code=200, json=lambda: rpc_response)

            mock_instance.get = AsyncMock(return_value=mock_outcome_response)
            mock_instance.post = AsyncMock(return_value=mock_rpc_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_positions(
                wallet_address="TestWallet123",
                rpc_url="https://api.mainnet-beta.solana.com",
            )

            assert result["status"] == "success"
            assert result["wallet"] == "TestWallet123"
            assert result["position_count"] == 2
            assert len(result["positions"]) == 2

            # Check first position
            pos1 = result["positions"][0]
            assert pos1["mint"] == "OutcomeMint1"
            assert pos1["ticker"] == "PRES-2028-HARRIS"
            assert pos1["side"] == "YES"
            assert pos1["ui_amount"] == 1.5

            # Check second position
            pos2 = result["positions"][1]
            assert pos2["mint"] == "OutcomeMint3"
            assert pos2["ticker"] == "BTC-100K-2025"
            assert pos2["side"] == "YES"
            assert pos2["ui_amount"] == 5.0

    @pytest.mark.asyncio
    async def test_get_positions_no_positions(self, prediction_client):
        """Get positions should return empty when no outcome tokens held."""
        outcome_mints_response = {
            "OutcomeMint1": {"market": "PRES-2028", "side": "yes"},
        }

        rpc_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "value": [
                    {
                        "account": {
                            "data": {
                                "parsed": {
                                    "info": {
                                        "mint": "SomeOtherToken",
                                        "tokenAmount": {
                                            "amount": "1000000000",
                                            "uiAmount": 1.0,
                                            "decimals": 9,
                                        },
                                    }
                                }
                            }
                        }
                    }
                ]
            },
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_outcome_response = MagicMock(
                status_code=200, json=lambda: outcome_mints_response
            )
            mock_rpc_response = MagicMock(status_code=200, json=lambda: rpc_response)
            mock_instance.get = AsyncMock(return_value=mock_outcome_response)
            mock_instance.post = AsyncMock(return_value=mock_rpc_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_positions(
                wallet_address="TestWallet123",
                rpc_url="https://api.mainnet-beta.solana.com",
            )

            assert result["status"] == "success"
            assert result["position_count"] == 0
            assert result["positions"] == []
            assert "No prediction market positions found" in result["hint"]

    @pytest.mark.asyncio
    async def test_get_positions_zero_balance_excluded(self, prediction_client):
        """Positions with zero balance should be excluded."""
        outcome_mints_response = {
            "OutcomeMint1": {"market": "TEST", "side": "yes"},
        }

        rpc_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "value": [
                    {
                        "account": {
                            "data": {
                                "parsed": {
                                    "info": {
                                        "mint": "OutcomeMint1",
                                        "tokenAmount": {
                                            "amount": "0",
                                            "uiAmount": 0,
                                            "decimals": 6,
                                        },
                                    }
                                }
                            }
                        }
                    }
                ]
            },
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_outcome_response = MagicMock(
                status_code=200, json=lambda: outcome_mints_response
            )
            mock_rpc_response = MagicMock(status_code=200, json=lambda: rpc_response)
            mock_instance.get = AsyncMock(return_value=mock_outcome_response)
            mock_instance.post = AsyncMock(return_value=mock_rpc_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_positions(
                wallet_address="TestWallet123",
                rpc_url="https://api.mainnet-beta.solana.com",
            )

            assert result["status"] == "success"
            assert result["position_count"] == 0

    @pytest.mark.asyncio
    async def test_get_positions_rpc_error(self, prediction_client):
        """Get positions should return error when RPC fails."""
        outcome_mints_response = {
            "OutcomeMint1": {"market": "TEST", "side": "yes"},
        }

        rpc_error_response = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000}}

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_outcome_response = MagicMock(
                status_code=200, json=lambda: outcome_mints_response
            )
            mock_rpc_response = MagicMock(
                status_code=200, json=lambda: rpc_error_response
            )
            mock_instance.get = AsyncMock(return_value=mock_outcome_response)
            mock_instance.post = AsyncMock(return_value=mock_rpc_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_positions(
                wallet_address="TestWallet123",
                rpc_url="https://api.mainnet-beta.solana.com",
            )

            assert result["status"] == "error"
            assert "RPC error" in result["message"]

    @pytest.mark.asyncio
    async def test_get_positions_rpc_http_error(self, prediction_client):
        """Get positions should return error on HTTP failure."""
        outcome_mints_response = {
            "OutcomeMint1": {"market": "TEST", "side": "yes"},
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_outcome_response = MagicMock(
                status_code=200, json=lambda: outcome_mints_response
            )
            mock_rpc_response = MagicMock(status_code=503, text="Service Unavailable")
            mock_instance.get = AsyncMock(return_value=mock_outcome_response)
            mock_instance.post = AsyncMock(return_value=mock_rpc_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_positions(
                wallet_address="TestWallet123",
                rpc_url="https://api.mainnet-beta.solana.com",
            )

            assert result["status"] == "error"
            assert "503" in result["message"]

    @pytest.mark.asyncio
    async def test_get_positions_rpc_exception(self, prediction_client):
        """Get positions should return error when RPC throws exception."""
        outcome_mints_response = {
            "OutcomeMint1": {"market": "TEST", "side": "yes"},
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_outcome_response = MagicMock(
                status_code=200, json=lambda: outcome_mints_response
            )
            mock_instance.get = AsyncMock(return_value=mock_outcome_response)
            # RPC call throws exception
            mock_instance.post = AsyncMock(side_effect=Exception("Connection timeout"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_instance

            result = await prediction_client.get_positions(
                wallet_address="TestWallet123",
                rpc_url="https://api.mainnet-beta.solana.com",
            )

            assert result["status"] == "error"
            assert "Connection timeout" in result["message"]

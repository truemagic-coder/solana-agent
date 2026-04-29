"""Comprehensive tests for BirdeyeTool - all 50 actions."""

import pytest
import respx
from unittest.mock import AsyncMock, patch
from httpx import Response
from httpx import Request

from solana_agent.tools.birdeye import BirdeyeTool


@pytest.fixture
def tool():
    """Create a BirdeyeTool instance with mock API key."""
    t = BirdeyeTool()
    t.configure({"tools": {"birdeye": {"api_key": "test-api-key"}}})
    return t


@pytest.fixture
def tool_no_key():
    """Create a BirdeyeTool instance without API key."""
    t = BirdeyeTool()
    t.configure({})
    return t


@pytest.fixture
def tool_x402_private_key():
    """Create a BirdeyeTool instance with x402 private-key auth."""
    t = BirdeyeTool()
    t.configure(
        {
            "tools": {
                "birdeye": {
                    "auth_mode": "x402_private_key",
                    "private_key": "test-private-key",
                }
            }
        }
    )
    return t


# Test token addresses
SOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
BONK = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
TEST_WALLET = "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
TEST_PAIR = "8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj"


class TestNoApiKey:
    """Test behavior when API key is missing."""

    @pytest.mark.asyncio
    async def test_no_api_key(self, tool_no_key):
        """Should return error when API key is not set."""
        result = await tool_no_key.execute(action="price", address=SOL)
        assert result["success"] is False
        assert "not configured" in result["error"]


class TestX402PrivateKeyMode:
    """Tests for Birdeye x402 private-key mode."""

    def test_runtime_context_can_be_cleared(self):
        tool = BirdeyeTool()
        tool.set_runtime_context({"privy_wallet_id": " wallet-123 "})

        assert tool._get_runtime_privy_wallet_id() == "wallet-123"

        tool.clear_runtime_context()
        assert tool._get_runtime_privy_wallet_id() == ""

    @pytest.mark.asyncio
    async def test_x402_private_key_missing_key(self):
        """Should return error when x402 private key is not set."""
        tool = BirdeyeTool()
        tool.configure({"tools": {"birdeye": {"auth_mode": "x402_private_key"}}})

        result = await tool.execute(action="price", address=SOL)
        assert result["success"] is False
        assert "private key not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_x402_private_key_price(self, tool_x402_private_key):
        """Should route Birdeye price requests through the x402 helper."""
        response = Response(
            200,
            json={"data": {"value": 150.5}},
            request=Request("GET", "https://public-api.birdeye.so/x402/defi/price"),
        )

        with patch(
            "solana_agent.tools.birdeye.request_with_x402_private_key",
            AsyncMock(return_value=response),
        ) as mock_request:
            result = await tool_x402_private_key.execute(action="price", address=SOL)

        assert result["success"] is True
        assert result["data"]["value"] == 150.5
        assert result["payment_mode"] == "x402_private_key"
        assert (
            mock_request.await_args.kwargs["url"]
            == "https://public-api.birdeye.so/x402/defi/price"
        )

    @pytest.mark.asyncio
    async def test_x402_privy_returns_explicit_error(self):
        """Should require Privy runtime export config for x402 Privy mode."""
        tool = BirdeyeTool()
        tool.configure({"tools": {"birdeye": {"auth_mode": "x402_privy"}}})

        result = await tool.execute(action="price", address=SOL)

        assert result["success"] is False
        assert "runtime_context.privy_wallet_id" in result["error"]

    @pytest.mark.asyncio
    async def test_x402_privy_price_uses_runtime_export_config(self):
        """Should route Birdeye x402 Privy requests through runtime Privy export."""
        tool = BirdeyeTool()
        tool.configure(
            {
                "tools": {
                    "birdeye": {
                        "auth_mode": "x402_privy",
                        "privy_app_id": "app-123",
                        "privy_app_secret": "secret-123",
                    }
                }
            }
        )
        tool.set_runtime_context({"privy_wallet_id": "wallet-123"})
        response = Response(
            200,
            json={"data": {"value": 150.5}},
            request=Request("GET", "https://public-api.birdeye.so/x402/defi/price"),
        )

        with patch(
            "solana_agent.tools.birdeye.request_with_x402_privy",
            AsyncMock(return_value=response),
        ) as mock_request:
            result = await tool.execute(action="price", address=SOL)

        assert result["success"] is True
        assert result["payment_mode"] == "x402_privy"
        assert mock_request.await_args.kwargs["privy_wallet_id"] == "wallet-123"
        assert mock_request.await_args.kwargs["privy_app_id"] == "app-123"
        assert mock_request.await_args.kwargs["privy_app_secret"] == "secret-123"


class TestUnknownAction:
    """Test unknown action handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        """Should return error for unknown action."""
        result = await tool.execute(action="unknown_action")
        assert result["success"] is False
        assert "Unknown action" in result["error"]


class TestPriceActions:
    """Tests for price-related actions."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_price(self, tool):
        """Test price action."""
        respx.get("https://public-api.birdeye.so/defi/price").mock(
            return_value=Response(200, json={"data": {"value": 150.5}})
        )
        result = await tool.execute(action="price", address=SOL)
        assert result["success"] is True
        assert result["data"]["value"] == 150.5

    @pytest.mark.asyncio
    async def test_price_missing_address(self, tool):
        """Test price action without address."""
        result = await tool.execute(action="price")
        assert result["success"] is False
        assert "address is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_multi_price(self, tool):
        """Test multi_price action."""
        respx.get("https://public-api.birdeye.so/defi/multi_price").mock(
            return_value=Response(
                200, json={"data": {SOL: {"value": 150}, USDC: {"value": 1}}}
            )
        )
        result = await tool.execute(action="multi_price", list_address=f"{SOL},{USDC}")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_multi_price_missing_addresses(self, tool):
        """Test multi_price without addresses."""
        result = await tool.execute(action="multi_price")
        assert result["success"] is False
        assert "list_address is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_multi_price_post(self, tool):
        """Test multi_price_post action."""
        respx.post("https://public-api.birdeye.so/defi/multi_price").mock(
            return_value=Response(200, json={"data": {SOL: {"value": 150}}})
        )
        result = await tool.execute(action="multi_price_post", list_address=[SOL, USDC])
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_history_price(self, tool):
        """Test history_price action."""
        respx.get("https://public-api.birdeye.so/defi/history_price").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(action="history_price", address=SOL, type="1H")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_history_price_missing_address(self, tool):
        """Test history_price without address."""
        result = await tool.execute(action="history_price")
        assert result["success"] is False
        assert "address is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_historical_price_unix(self, tool):
        """Test historical_price_unix action."""
        respx.get("https://public-api.birdeye.so/defi/historical_price_unix").mock(
            return_value=Response(200, json={"data": {"value": 145.0}})
        )
        result = await tool.execute(
            action="historical_price_unix", address=SOL, unixtime=1700000000
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_historical_price_unix_missing_params(self, tool):
        """Test historical_price_unix without required params."""
        result = await tool.execute(action="historical_price_unix", address=SOL)
        assert result["success"] is False
        assert "unixtime are required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_price_volume_single(self, tool):
        """Test price_volume_single action."""
        respx.get("https://public-api.birdeye.so/defi/price_volume/single").mock(
            return_value=Response(200, json={"data": {"price": 150, "volume": 1000000}})
        )
        result = await tool.execute(action="price_volume_single", address=SOL)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_price_volume_multi(self, tool):
        """Test price_volume_multi action."""
        respx.post("https://public-api.birdeye.so/defi/price_volume/multi").mock(
            return_value=Response(200, json={"data": {}})
        )
        result = await tool.execute(action="price_volume_multi", list_address=[SOL])
        assert result["success"] is True


class TestOHLCVActions:
    """Tests for OHLCV-related actions."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_ohlcv(self, tool):
        """Test ohlcv action."""
        respx.get("https://public-api.birdeye.so/defi/ohlcv").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(action="ohlcv", address=SOL, type="1H")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_ohlcv_missing_address(self, tool):
        """Test ohlcv without address."""
        result = await tool.execute(action="ohlcv")
        assert result["success"] is False
        assert "address is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_ohlcv_pair(self, tool):
        """Test ohlcv_pair action."""
        respx.get("https://public-api.birdeye.so/defi/ohlcv/pair").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(action="ohlcv_pair", address=TEST_PAIR)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_ohlcv_base_quote(self, tool):
        """Test ohlcv_base_quote action."""
        respx.get("https://public-api.birdeye.so/defi/ohlcv/base_quote").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(
            action="ohlcv_base_quote", base_address=SOL, quote_address=USDC
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_ohlcv_base_quote_missing_params(self, tool):
        """Test ohlcv_base_quote without required params."""
        result = await tool.execute(action="ohlcv_base_quote", base_address=SOL)
        assert result["success"] is False
        assert "quote_address are required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_ohlcv_v3(self, tool):
        """Test ohlcv_v3 action."""
        respx.get("https://public-api.birdeye.so/defi/v3/ohlcv").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(action="ohlcv_v3", address=SOL)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_ohlcv_pair_v3(self, tool):
        """Test ohlcv_pair_v3 action."""
        respx.get("https://public-api.birdeye.so/defi/v3/ohlcv/pair").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(action="ohlcv_pair_v3", address=TEST_PAIR)
        assert result["success"] is True


class TestTradesActions:
    """Tests for trades-related actions."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_trades_token(self, tool):
        """Test trades_token action."""
        respx.get("https://public-api.birdeye.so/defi/txs/token").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(action="trades_token", address=SOL)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_trades_token_missing_address(self, tool):
        """Test trades_token without address."""
        result = await tool.execute(action="trades_token")
        assert result["success"] is False
        assert "address is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_trades_pair(self, tool):
        """Test trades_pair action."""
        respx.get("https://public-api.birdeye.so/defi/txs/pair").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(action="trades_pair", address=TEST_PAIR)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_trades_token_seek(self, tool):
        """Test trades_token_seek action."""
        respx.get("https://public-api.birdeye.so/defi/txs/token/seek_by_time").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(
            action="trades_token_seek", address=SOL, before_time=1700000000
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_trades_pair_seek(self, tool):
        """Test trades_pair_seek action."""
        respx.get("https://public-api.birdeye.so/defi/txs/pair/seek_by_time").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(
            action="trades_pair_seek", address=TEST_PAIR, after_time=1600000000
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_trades_v3(self, tool):
        """Test trades_v3 action."""
        respx.get("https://public-api.birdeye.so/defi/v3/txs").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(action="trades_v3", address=SOL, owner=TEST_WALLET)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_trades_token_v3(self, tool):
        """Test trades_token_v3 action."""
        respx.get("https://public-api.birdeye.so/defi/v3/token/txs").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(action="trades_token_v3", address=SOL)
        assert result["success"] is True


class TestTokenActions:
    """Tests for token-related actions."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_list(self, tool):
        """Test token_list action."""
        respx.get("https://public-api.birdeye.so/defi/tokenlist").mock(
            return_value=Response(200, json={"data": {"tokens": []}})
        )
        result = await tool.execute(action="token_list", limit=10)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_list_v3(self, tool):
        """Test token_list_v3 action."""
        respx.get("https://public-api.birdeye.so/defi/v3/token/list").mock(
            return_value=Response(200, json={"data": {"tokens": []}})
        )
        result = await tool.execute(
            action="token_list_v3", limit=10, min_liquidity=1000
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_list_scroll(self, tool):
        """Test token_list_scroll action."""
        respx.get("https://public-api.birdeye.so/defi/v3/token/list/scroll").mock(
            return_value=Response(
                200, json={"data": {"tokens": [], "scroll_id": "abc"}}
            )
        )
        result = await tool.execute(action="token_list_scroll", limit=50)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_overview(self, tool):
        """Test token_overview action."""
        respx.get("https://public-api.birdeye.so/defi/token_overview").mock(
            return_value=Response(
                200, json={"data": {"symbol": "SOL", "name": "Solana"}}
            )
        )
        result = await tool.execute(action="token_overview", address=SOL)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_token_overview_missing_address(self, tool):
        """Test token_overview without address."""
        result = await tool.execute(action="token_overview")
        assert result["success"] is False
        assert "address is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_metadata_single(self, tool):
        """Test token_metadata_single action."""
        respx.get("https://public-api.birdeye.so/defi/v3/token/meta-data/single").mock(
            return_value=Response(200, json={"data": {"symbol": "SOL"}})
        )
        result = await tool.execute(action="token_metadata_single", address=SOL)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_metadata_multiple(self, tool):
        """Test token_metadata_multiple action."""
        respx.get(
            "https://public-api.birdeye.so/defi/v3/token/meta-data/multiple"
        ).mock(return_value=Response(200, json={"data": {}}))
        result = await tool.execute(
            action="token_metadata_multiple", list_address=f"{SOL},{USDC}"
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_market_data(self, tool):
        """Test token_market_data action."""
        respx.get("https://public-api.birdeye.so/defi/v3/token/market-data").mock(
            return_value=Response(200, json={"data": {"price": 150}})
        )
        result = await tool.execute(action="token_market_data", address=SOL)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_market_data_multiple(self, tool):
        """Test token_market_data_multiple action."""
        respx.get(
            "https://public-api.birdeye.so/defi/v3/token/market-data/multiple"
        ).mock(return_value=Response(200, json={"data": {}}))
        result = await tool.execute(
            action="token_market_data_multiple", list_address=f"{SOL},{USDC}"
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_trade_data_single(self, tool):
        """Test token_trade_data_single action."""
        respx.get("https://public-api.birdeye.so/defi/v3/token/trade-data/single").mock(
            return_value=Response(200, json={"data": {"volume_24h": 1000000}})
        )
        result = await tool.execute(action="token_trade_data_single", address=SOL)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_trade_data_multiple(self, tool):
        """Test token_trade_data_multiple action."""
        respx.get(
            "https://public-api.birdeye.so/defi/v3/token/trade-data/multiple"
        ).mock(return_value=Response(200, json={"data": {}}))
        result = await tool.execute(
            action="token_trade_data_multiple", list_address=f"{SOL},{USDC}"
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_holder(self, tool):
        """Test token_holder action."""
        respx.get("https://public-api.birdeye.so/defi/v3/token/holder").mock(
            return_value=Response(200, json={"data": {"holders": []}})
        )
        result = await tool.execute(action="token_holder", address=SOL, limit=10)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_trending(self, tool):
        """Test token_trending action."""
        respx.get("https://public-api.birdeye.so/defi/token_trending").mock(
            return_value=Response(200, json={"data": {"tokens": []}})
        )
        result = await tool.execute(action="token_trending", limit=10)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_new_listing(self, tool):
        """Test token_new_listing action."""
        respx.get("https://public-api.birdeye.so/defi/v2/tokens/new_listing").mock(
            return_value=Response(200, json={"data": {"tokens": []}})
        )
        result = await tool.execute(action="token_new_listing", limit=10)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_top_traders(self, tool):
        """Test token_top_traders action."""
        respx.get("https://public-api.birdeye.so/defi/v2/tokens/top_traders").mock(
            return_value=Response(200, json={"data": {"traders": []}})
        )
        result = await tool.execute(
            action="token_top_traders", address=BONK, time_frame="24h"
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_token_top_traders_missing_address(self, tool):
        """Test token_top_traders without address."""
        result = await tool.execute(action="token_top_traders")
        assert result["success"] is False
        assert "address is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_markets(self, tool):
        """Test token_markets action."""
        respx.get("https://public-api.birdeye.so/defi/v2/markets").mock(
            return_value=Response(200, json={"data": {"markets": []}})
        )
        result = await tool.execute(action="token_markets", address=SOL)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_security(self, tool):
        """Test token_security action."""
        respx.get("https://public-api.birdeye.so/defi/token_security").mock(
            return_value=Response(200, json={"data": {"isScam": False}})
        )
        result = await tool.execute(action="token_security", address=SOL)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_creation_info(self, tool):
        """Test token_creation_info action."""
        respx.get("https://public-api.birdeye.so/defi/token_creation_info").mock(
            return_value=Response(200, json={"data": {"creator": TEST_WALLET}})
        )
        result = await tool.execute(action="token_creation_info", address=BONK)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_mint_burn(self, tool):
        """Test token_mint_burn action."""
        respx.get("https://public-api.birdeye.so/defi/v3/token/mint-burn-txs").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(
            action="token_mint_burn", address=BONK, tx_type="mint"
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_all_time_trades_single(self, tool):
        """Test token_all_time_trades_single action."""
        respx.get("https://public-api.birdeye.so/defi/v3/all-time-trades/single").mock(
            return_value=Response(200, json={"data": {"total_trades": 1000000}})
        )
        result = await tool.execute(action="token_all_time_trades_single", address=SOL)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_all_time_trades_multiple(self, tool):
        """Test token_all_time_trades_multiple action."""
        respx.get(
            "https://public-api.birdeye.so/defi/v3/all-time-trades/multiple"
        ).mock(return_value=Response(200, json={"data": {}}))
        result = await tool.execute(
            action="token_all_time_trades_multiple", list_address=f"{SOL},{USDC}"
        )
        assert result["success"] is True


class TestExitLiquidityActions:
    """Tests for exit liquidity actions."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_exit_liquidity(self, tool):
        """Test token_exit_liquidity action."""
        respx.get("https://public-api.birdeye.so/defi/v3/token/exit-liquidity").mock(
            return_value=Response(
                200, json={"data": {"liquidity": 500000, "price_impact": 0.5}}
            )
        )
        result = await tool.execute(action="token_exit_liquidity", address=SOL)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_token_exit_liquidity_missing_address(self, tool):
        """Test token_exit_liquidity without address."""
        result = await tool.execute(action="token_exit_liquidity")
        assert result["success"] is False
        assert "address is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_exit_liquidity_multiple(self, tool):
        """Test token_exit_liquidity_multiple action."""
        respx.get(
            "https://public-api.birdeye.so/defi/v3/token/exit-liquidity/multiple"
        ).mock(return_value=Response(200, json={"data": {}}))
        result = await tool.execute(
            action="token_exit_liquidity_multiple", list_address=f"{SOL},{BONK}"
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_token_exit_liquidity_multiple_missing_address(self, tool):
        """Test token_exit_liquidity_multiple without list_address."""
        result = await tool.execute(action="token_exit_liquidity_multiple")
        assert result["success"] is False
        assert "list_address is required" in result["error"]


class TestPairActions:
    """Tests for pair-related actions."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_pair_overview_single(self, tool):
        """Test pair_overview_single action."""
        respx.get("https://public-api.birdeye.so/defi/v3/pair/overview/single").mock(
            return_value=Response(200, json={"data": {"liquidity": 1000000}})
        )
        result = await tool.execute(action="pair_overview_single", address=TEST_PAIR)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_pair_overview_single_missing_address(self, tool):
        """Test pair_overview_single without address."""
        result = await tool.execute(action="pair_overview_single")
        assert result["success"] is False
        assert "address is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_pair_overview_multiple(self, tool):
        """Test pair_overview_multiple action."""
        respx.get("https://public-api.birdeye.so/defi/v3/pair/overview/multiple").mock(
            return_value=Response(200, json={"data": {}})
        )
        result = await tool.execute(
            action="pair_overview_multiple", list_address=TEST_PAIR
        )
        assert result["success"] is True


class TestTraderActions:
    """Tests for trader-related actions."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_trader_gainers_losers(self, tool):
        """Test trader_gainers_losers action."""
        respx.get("https://public-api.birdeye.so/trader/gainers-losers").mock(
            return_value=Response(200, json={"data": {"gainers": [], "losers": []}})
        )
        result = await tool.execute(action="trader_gainers_losers", type="1h", limit=10)
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_trader_txs_seek(self, tool):
        """Test trader_txs_seek action."""
        respx.get("https://public-api.birdeye.so/trader/txs/seek_by_time").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(
            action="trader_txs_seek", address=TEST_WALLET, before_time=1700000000
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_trader_txs_seek_missing_address(self, tool):
        """Test trader_txs_seek without address."""
        result = await tool.execute(action="trader_txs_seek")
        assert result["success"] is False
        assert "address is required" in result["error"]


class TestWalletActions:
    """Tests for wallet-related actions."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_wallet_token_list(self, tool):
        """Test wallet_token_list action."""
        respx.get("https://public-api.birdeye.so/v1/wallet/token_list").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(action="wallet_token_list", wallet=TEST_WALLET)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wallet_token_list_missing_wallet(self, tool):
        """Test wallet_token_list without wallet."""
        result = await tool.execute(action="wallet_token_list")
        assert result["success"] is False
        assert "wallet is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_wallet_token_balance(self, tool):
        """Test wallet_token_balance action."""
        respx.get("https://public-api.birdeye.so/v1/wallet/token_balance").mock(
            return_value=Response(200, json={"data": {"balance": 100}})
        )
        result = await tool.execute(
            action="wallet_token_balance", wallet=TEST_WALLET, token_address=SOL
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wallet_token_balance_missing_params(self, tool):
        """Test wallet_token_balance without required params."""
        result = await tool.execute(action="wallet_token_balance", wallet=TEST_WALLET)
        assert result["success"] is False
        assert "token_address are required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_wallet_tx_list(self, tool):
        """Test wallet_tx_list action."""
        respx.get("https://public-api.birdeye.so/v1/wallet/tx_list").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(
            action="wallet_tx_list", wallet=TEST_WALLET, limit=20
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_wallet_balance_change(self, tool):
        """Test wallet_balance_change action."""
        respx.get("https://public-api.birdeye.so/wallet/v2/balance-change").mock(
            return_value=Response(200, json={"data": {"changes": []}})
        )
        result = await tool.execute(
            action="wallet_balance_change", wallet=TEST_WALLET, token_address=SOL
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_wallet_pnl_summary(self, tool):
        """Test wallet_pnl_summary action."""
        respx.get("https://public-api.birdeye.so/wallet/v2/pnl/summary").mock(
            return_value=Response(
                200,
                json={
                    "data": {
                        "pnl": {"realized_profit_usd": 1000, "unrealized_usd": 500}
                    }
                },
            )
        )
        result = await tool.execute(action="wallet_pnl_summary", wallet=TEST_WALLET)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wallet_pnl_summary_missing_wallet(self, tool):
        """Test wallet_pnl_summary without wallet."""
        result = await tool.execute(action="wallet_pnl_summary")
        assert result["success"] is False
        assert "wallet is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_wallet_pnl_details(self, tool):
        """Test wallet_pnl_details action (POST)."""
        respx.post("https://public-api.birdeye.so/wallet/v2/pnl/details").mock(
            return_value=Response(200, json={"data": {"tokens": []}})
        )
        result = await tool.execute(
            action="wallet_pnl_details", wallet=TEST_WALLET, tokens=[SOL, "BonkMint123"]
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wallet_pnl_details_missing_wallet(self, tool):
        """Test wallet_pnl_details without wallet."""
        result = await tool.execute(action="wallet_pnl_details")
        assert result["success"] is False
        assert "wallet is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_wallet_pnl_multiple(self, tool):
        """Test wallet_pnl_multiple action."""
        respx.get("https://public-api.birdeye.so/wallet/v2/pnl/multiple").mock(
            return_value=Response(200, json={"data": {}})
        )
        result = await tool.execute(
            action="wallet_pnl_multiple", wallets=[TEST_WALLET, "AnotherWallet123"]
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wallet_pnl_multiple_missing_wallets(self, tool):
        """Test wallet_pnl_multiple without wallets."""
        result = await tool.execute(action="wallet_pnl_multiple")
        assert result["success"] is False
        assert "wallets list is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_wallet_current_net_worth(self, tool):
        """Test wallet_current_net_worth action."""
        respx.get("https://public-api.birdeye.so/wallet/v2/current-net-worth").mock(
            return_value=Response(
                200, json={"data": {"total_usd": 10000, "tokens": []}}
            )
        )
        result = await tool.execute(
            action="wallet_current_net_worth", wallet=TEST_WALLET
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wallet_current_net_worth_missing_wallet(self, tool):
        """Test wallet_current_net_worth without wallet."""
        result = await tool.execute(action="wallet_current_net_worth")
        assert result["success"] is False
        assert "wallet is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_wallet_net_worth(self, tool):
        """Test wallet_net_worth action."""
        respx.get("https://public-api.birdeye.so/wallet/v2/net-worth").mock(
            return_value=Response(200, json={"data": {"items": []}})
        )
        result = await tool.execute(
            action="wallet_net_worth",
            wallet=TEST_WALLET,
            time="2025-11-30 00:00:00",
            type="1d",
            count=7,
            direction="back",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wallet_net_worth_missing_wallet(self, tool):
        """Test wallet_net_worth without wallet."""
        result = await tool.execute(action="wallet_net_worth")
        assert result["success"] is False
        assert "wallet is required" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_wallet_net_worth_details(self, tool):
        """Test wallet_net_worth_details action."""
        respx.get("https://public-api.birdeye.so/wallet/v2/net-worth-details").mock(
            return_value=Response(200, json={"data": {"tokens": []}})
        )
        result = await tool.execute(
            action="wallet_net_worth_details",
            wallet=TEST_WALLET,
            time="2025-11-30 00:00:00",
            type="1d",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wallet_net_worth_details_missing_wallet(self, tool):
        """Test wallet_net_worth_details without wallet."""
        result = await tool.execute(action="wallet_net_worth_details")
        assert result["success"] is False
        assert "wallet is required" in result["error"]


class TestSearchAction:
    """Tests for search action."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_search(self, tool):
        """Test search action."""
        respx.get("https://public-api.birdeye.so/defi/v3/search").mock(
            return_value=Response(200, json={"data": {"tokens": [], "pairs": []}})
        )
        result = await tool.execute(action="search", keyword="bonk", limit=10)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_missing_keyword(self, tool):
        """Test search without keyword."""
        result = await tool.execute(action="search")
        assert result["success"] is False
        assert "keyword is required" in result["error"]


class TestUtilsActions:
    """Tests for utility actions."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_latest_block(self, tool):
        """Test latest_block action."""
        respx.get("https://public-api.birdeye.so/defi/v3/txs/latest_block").mock(
            return_value=Response(200, json={"data": {"slot": 250000000}})
        )
        result = await tool.execute(action="latest_block")
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_networks(self, tool):
        """Test networks action."""
        respx.get("https://public-api.birdeye.so/defi/networks").mock(
            return_value=Response(200, json={"data": ["solana", "ethereum"]})
        )
        result = await tool.execute(action="networks")
        assert result["success"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_supported_chains(self, tool):
        """Test supported_chains action."""
        respx.get("https://public-api.birdeye.so/v1/wallet/list_supported_chain").mock(
            return_value=Response(200, json={"data": ["solana"]})
        )
        result = await tool.execute(action="supported_chains")
        assert result["success"] is True


class TestAPIErrors:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_error_response(self, tool):
        """Test handling of API error response."""
        respx.get("https://public-api.birdeye.so/defi/price").mock(
            return_value=Response(500, text="Internal Server Error")
        )
        result = await tool.execute(action="price", address=SOL)
        assert result["success"] is False
        assert "API error: 500" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_unauthorized(self, tool):
        """Test handling of 401 unauthorized."""
        respx.get("https://public-api.birdeye.so/defi/price").mock(
            return_value=Response(401, text="Unauthorized")
        )
        result = await tool.execute(action="price", address=SOL)
        assert result["success"] is False
        assert "401" in result["error"]


class TestChainSupport:
    """Tests for multi-chain support."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_different_chain(self, tool):
        """Test specifying different chain."""
        route = respx.get("https://public-api.birdeye.so/defi/price").mock(
            return_value=Response(200, json={"data": {"value": 3000}})
        )
        result = await tool.execute(action="price", address="0xabc", chain="ethereum")
        assert result["success"] is True
        # Verify the chain header was set
        assert route.calls[0].request.headers["x-chain"] == "ethereum"

    @pytest.mark.asyncio
    @respx.mock
    async def test_chain_from_config(self):
        """Test chain configured via config locks all requests to that chain."""
        t = BirdeyeTool()
        t.configure(
            {"tools": {"birdeye": {"api_key": "test-key", "chain": "ethereum"}}}
        )

        route = respx.get("https://public-api.birdeye.so/defi/price").mock(
            return_value=Response(200, json={"data": {"value": 3000}})
        )
        # Don't pass chain - should use config chain
        result = await t.execute(action="price", address="0xabc")
        assert result["success"] is True
        assert route.calls[0].request.headers["x-chain"] == "ethereum"

    @pytest.mark.asyncio
    @respx.mock
    async def test_default_chain_is_solana(self, tool):
        """Test default chain is solana when not specified."""
        route = respx.get("https://public-api.birdeye.so/defi/price").mock(
            return_value=Response(200, json={"data": {"value": 150}})
        )
        result = await tool.execute(action="price", address=SOL)
        assert result["success"] is True
        assert route.calls[0].request.headers["x-chain"] == "solana"


class TestBirdeyePlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        from solana_agent.tools.birdeye import BirdeyePlugin

        plugin = BirdeyePlugin()
        assert plugin.name == "birdeye"

    def test_plugin_description(self):
        """Should have descriptive description."""
        from solana_agent.tools.birdeye import BirdeyePlugin

        plugin = BirdeyePlugin()
        assert "birdeye" in plugin.description.lower()

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        from solana_agent.tools.birdeye import BirdeyePlugin

        plugin = BirdeyePlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        from solana_agent.tools.birdeye import BirdeyePlugin
        from unittest.mock import MagicMock

        plugin = BirdeyePlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1


class TestBirdeyeSchema:
    """Test schema and tool structure."""

    def test_schema_has_action(self, tool):
        """Schema should include action parameter."""
        schema = tool.get_schema()
        assert "action" in schema["properties"]
        assert "action" in schema["required"]

    def test_schema_has_address(self, tool):
        """Schema should include address parameter."""
        schema = tool.get_schema()
        assert "address" in schema["properties"]

    def test_schema_has_wallet(self, tool):
        """Schema should include wallet parameter."""
        schema = tool.get_schema()
        assert "wallet" in schema["properties"]

    def test_schema_has_additional_properties_false(self, tool):
        """Schema should have additionalProperties: false for OpenAI compliance."""
        schema = tool.get_schema()
        assert schema.get("additionalProperties") is False

    def test_get_plugin_function(self):
        """Should have get_plugin function."""
        from solana_agent.tools.birdeye import get_plugin

        plugin = get_plugin()
        assert plugin is not None
        assert plugin.name == "birdeye"

"""Tests for the Vybe tool."""

import time

import pytest
import respx
from httpx import Response

from solana_agent.tools.vybe import VybeTool
import solana_agent.tools.vybe as vybe_module


@pytest.fixture
def tool():
    """Create a VybeTool instance with mock API key."""
    t = VybeTool()
    t.configure({"tools": {"vybe": {"api_key": "test-api-key"}}})
    return t


@pytest.fixture
def tool_no_key():
    """Create a VybeTool instance without API key."""
    t = VybeTool()
    t.configure({})
    return t


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the module-level cache before each test."""
    vybe_module._known_accounts_cache = {}
    vybe_module._cache_timestamp = 0
    yield
    vybe_module._known_accounts_cache = {}
    vybe_module._cache_timestamp = 0


# Sample known accounts response
KNOWN_ACCOUNTS_RESPONSE = {
    "data": [
        {
            "ownerAddress": "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9",
            "name": "Binance Hot Wallet 1",
            "labels": ["CEX", "Exchange"],
            "entityId": "binance",
            "entityName": "Binance",
            "type": "CEX",
        },
        {
            "ownerAddress": "2AQdpHJ2JpcEgPiATUXjQxA8QmafFegfQwSLWSprPicm",
            "name": "Coinbase Prime",
            "labels": ["CEX", "Exchange", "Institutional"],
            "entityId": "coinbase",
            "entityName": "Coinbase",
            "type": "CEX",
        },
        {
            "ownerAddress": "HWHvQhFmJB3NUcu1aihKmrKegfVxBEHzwVX6yZCKEsi1",
            "name": "Raydium AMM",
            "labels": ["AMM", "DeFi", "Liquidity"],
            "entityId": "raydium",
            "entityName": "Raydium",
            "type": "AMM",
        },
        {
            "ownerAddress": "MarketMaker111111111111111111111111111111111",
            "name": "Wintermute",
            "labels": ["Market Maker", "Trading"],
            "entityId": "wintermute",
            "entityName": "Wintermute Trading",
            "type": "Market Maker",
        },
    ]
}


class TestNoApiKey:
    """Tests for missing API key."""

    @pytest.mark.asyncio
    async def test_no_api_key(self, tool_no_key):
        """Test that missing API key returns error."""
        result = await tool_no_key.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )
        assert result["success"] is False
        assert "API key is required" in result["error"]


class TestAddressLookup:
    """Tests for address lookup functionality."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_single_known_address(self, tool):
        """Test looking up a single known address."""
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(200, json=KNOWN_ACCOUNTS_RESPONSE)
        )

        result = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )

        assert result["success"] is True
        assert result["known_count"] == 1
        assert result["unknown_count"] == 0
        assert len(result["results"]) == 1
        assert result["results"][0]["known"] is True
        assert result["results"][0]["name"] == "Binance Hot Wallet 1"
        assert "CEX" in result["results"][0]["labels"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_single_unknown_address(self, tool):
        """Test looking up a single unknown address."""
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(200, json=KNOWN_ACCOUNTS_RESPONSE)
        )

        result = await tool.execute(
            addresses="UnknownAddress111111111111111111111111111111"
        )

        assert result["success"] is True
        assert result["known_count"] == 0
        assert result["unknown_count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["known"] is False
        assert result["results"][0]["name"] is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_bulk_lookup_mixed(self, tool):
        """Test bulk lookup with mix of known and unknown addresses."""
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(200, json=KNOWN_ACCOUNTS_RESPONSE)
        )

        addresses = (
            "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9,"
            "2AQdpHJ2JpcEgPiATUXjQxA8QmafFegfQwSLWSprPicm,"
            "UnknownWallet11111111111111111111111111111111,"
            "HWHvQhFmJB3NUcu1aihKmrKegfVxBEHzwVX6yZCKEsi1"
        )

        result = await tool.execute(addresses=addresses)

        assert result["success"] is True
        assert result["known_count"] == 3
        assert result["unknown_count"] == 1
        assert len(result["results"]) == 4

        # Check Binance
        binance = next(
            r
            for r in result["results"]
            if r["address"] == "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )
        assert binance["known"] is True
        assert binance["entity_name"] == "Binance"

        # Check Coinbase
        coinbase = next(
            r
            for r in result["results"]
            if r["address"] == "2AQdpHJ2JpcEgPiATUXjQxA8QmafFegfQwSLWSprPicm"
        )
        assert coinbase["known"] is True
        assert coinbase["entity_name"] == "Coinbase"

        # Check unknown
        unknown = next(
            r
            for r in result["results"]
            if r["address"] == "UnknownWallet11111111111111111111111111111111"
        )
        assert unknown["known"] is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_all_known_addresses(self, tool):
        """Test bulk lookup where all addresses are known."""
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(200, json=KNOWN_ACCOUNTS_RESPONSE)
        )

        addresses = (
            "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9,"
            "MarketMaker111111111111111111111111111111111"
        )

        result = await tool.execute(addresses=addresses)

        assert result["success"] is True
        assert result["known_count"] == 2
        assert result["unknown_count"] == 0

    @pytest.mark.asyncio
    async def test_empty_addresses(self, tool):
        """Test with empty addresses string."""
        result = await tool.execute(addresses="")
        assert result["success"] is False
        assert "No valid addresses" in result["error"]

    @pytest.mark.asyncio
    async def test_whitespace_only_addresses(self, tool):
        """Test with whitespace-only addresses string."""
        result = await tool.execute(addresses="  ,  ,  ")
        assert result["success"] is False
        assert "No valid addresses" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_addresses_with_whitespace(self, tool):
        """Test addresses with extra whitespace are handled."""
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(200, json=KNOWN_ACCOUNTS_RESPONSE)
        )

        result = await tool.execute(
            addresses="  5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9  , 2AQdpHJ2JpcEgPiATUXjQxA8QmafFegfQwSLWSprPicm  "
        )

        assert result["success"] is True
        assert result["known_count"] == 2


class TestCaching:
    """Tests for cache behavior."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_cache_is_used(self, tool):
        """Test that cache is used on subsequent calls."""
        route = respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(200, json=KNOWN_ACCOUNTS_RESPONSE)
        )

        # First call - should hit API
        result1 = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )
        assert result1["success"] is True
        assert route.call_count == 1

        # Second call - should use cache
        result2 = await tool.execute(
            addresses="2AQdpHJ2JpcEgPiATUXjQxA8QmafFegfQwSLWSprPicm"
        )
        assert result2["success"] is True
        assert route.call_count == 1  # Still 1, cache was used

    @pytest.mark.asyncio
    @respx.mock
    async def test_refresh_cache_forces_api_call(self, tool):
        """Test that refresh_cache=True forces a new API call."""
        route = respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(200, json=KNOWN_ACCOUNTS_RESPONSE)
        )

        # First call
        result1 = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )
        assert result1["success"] is True
        assert route.call_count == 1

        # Second call with refresh
        result2 = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9",
            refresh_cache=True,
        )
        assert result2["success"] is True
        assert route.call_count == 2  # API was called again

    @pytest.mark.asyncio
    @respx.mock
    async def test_cache_expiry(self, tool):
        """Test that expired cache triggers new API call."""
        route = respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(200, json=KNOWN_ACCOUNTS_RESPONSE)
        )

        # First call
        result1 = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )
        assert result1["success"] is True
        assert route.call_count == 1

        # Simulate cache expiry by setting timestamp to past
        vybe_module._cache_timestamp = time.time() - 4000  # Older than 1 hour

        # Second call - should hit API due to expired cache
        result2 = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )
        assert result2["success"] is True
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_cache_age_in_response(self, tool):
        """Test that cache age is included in response."""
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(200, json=KNOWN_ACCOUNTS_RESPONSE)
        )

        result = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )

        assert result["success"] is True
        assert "cache_age_seconds" in result
        assert result["cache_age_seconds"] >= 0


class TestAPIErrors:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_error_response(self, tool):
        """Test handling of API error response."""
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(500, json={"error": "Internal server error"})
        )

        result = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )

        assert result["success"] is False
        assert "500" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_unauthorized(self, tool):
        """Test handling of unauthorized response."""
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(401, json={"error": "Unauthorized"})
        )

        result = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )

        assert result["success"] is False
        assert "401" in result["error"]


class TestResponseFormats:
    """Tests for different API response formats."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_response_as_list(self, tool):
        """Test handling response where data is directly a list."""
        # Some APIs return list directly instead of {"data": [...]}
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(
                200,
                json=[
                    {
                        "ownerAddress": "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9",
                        "name": "Test Wallet",
                        "labels": ["Test"],
                        "entityName": "Test Entity",
                    }
                ],
            )
        )

        result = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )

        assert result["success"] is True
        assert result["known_count"] == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_alternative_address_field(self, tool):
        """Test handling response with 'address' instead of 'ownerAddress'."""
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(
                200,
                json={
                    "data": [
                        {
                            "address": "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9",
                            "name": "Alt Format Wallet",
                            "labels": ["Alt"],
                        }
                    ]
                },
            )
        )

        result = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )

        assert result["success"] is True
        assert result["known_count"] == 1
        assert result["results"][0]["name"] == "Alt Format Wallet"


class TestSummaryOutput:
    """Tests for summary output formatting."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_summary_includes_counts(self, tool):
        """Test that summary includes address counts."""
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(200, json=KNOWN_ACCOUNTS_RESPONSE)
        )

        result = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9,Unknown111111111111111111111111111111111111"
        )

        assert result["success"] is True
        assert "2 addresses" in result["summary"]
        assert "1 known" in result["summary"]
        assert "1 unknown" in result["summary"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_summary_includes_labels(self, tool):
        """Test that summary includes labels for known addresses."""
        respx.get("https://api.vybenetwork.xyz/account/known-accounts").mock(
            return_value=Response(200, json=KNOWN_ACCOUNTS_RESPONSE)
        )

        result = await tool.execute(
            addresses="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9"
        )

        assert result["success"] is True
        assert "Binance Hot Wallet 1" in result["summary"]
        assert "CEX" in result["summary"]

"""
Tests for Token Math Tool.

Tests the TokenMathTool which provides reliable token amount calculations
for swaps and limit orders. LLMs are bad at math - this tool isn't.
"""

import pytest
from decimal import Decimal

from solana_agent.tools.token_math import (
    TokenMathTool,
    TokenMathPlugin,
    get_plugin,
    human_to_smallest_units,
    smallest_units_to_human,
    usd_to_token_amount,
    token_amount_to_usd,
    apply_percentage_change,
    calculate_swap_amount,
    calculate_limit_order_amounts,
    calculate_limit_order_info,
)


# =============================================================================
# Unit Tests for Pure Functions
# =============================================================================


class TestHumanToSmallestUnits:
    """Test human_to_smallest_units function."""

    def test_sol_conversion(self):
        """Should convert SOL amounts correctly (9 decimals)."""
        # 0.07 SOL = 70,000,000 lamports
        assert human_to_smallest_units("0.07", 9) == "70000000"
        # 1 SOL = 1,000,000,000 lamports
        assert human_to_smallest_units("1", 9) == "1000000000"
        # 0.001 SOL = 1,000,000 lamports
        assert human_to_smallest_units("0.001", 9) == "1000000"

    def test_usdc_conversion(self):
        """Should convert USDC amounts correctly (6 decimals)."""
        # 100 USDC = 100,000,000 smallest units
        assert human_to_smallest_units("100", 6) == "100000000"
        # 1 USDC = 1,000,000 smallest units
        assert human_to_smallest_units("1", 6) == "1000000"
        # 0.01 USDC = 10,000 smallest units
        assert human_to_smallest_units("0.01", 6) == "10000"

    def test_bonk_conversion(self):
        """Should convert BONK amounts correctly (5 decimals)."""
        # 1,000,000 BONK = 100,000,000,000 smallest units
        assert human_to_smallest_units("1000000", 5) == "100000000000"
        # 1 BONK = 100,000 smallest units
        assert human_to_smallest_units("1", 5) == "100000"

    def test_zero_amount(self):
        """Should handle zero amounts."""
        assert human_to_smallest_units("0", 9) == "0"
        assert human_to_smallest_units("0.0", 6) == "0"

    def test_large_amounts(self):
        """Should handle large amounts without overflow."""
        # 1 billion tokens with 9 decimals
        result = human_to_smallest_units("1000000000", 9)
        assert result == "1000000000000000000"

    def test_very_small_amounts(self):
        """Should handle very small amounts."""
        # 0.000000001 SOL (1 lamport)
        assert human_to_smallest_units("0.000000001", 9) == "1"

    def test_rounds_down_fractional_smallest_units(self):
        """Should round down when result would have fractional smallest units."""
        # 0.0000000015 SOL should round down to 1 lamport, not 2
        assert human_to_smallest_units("0.0000000015", 9) == "1"

    def test_invalid_amount_raises_error(self):
        """Should raise ValueError for invalid amounts."""
        with pytest.raises(ValueError):
            human_to_smallest_units("abc", 9)
        with pytest.raises(ValueError):
            human_to_smallest_units("", 9)

    def test_integer_input(self):
        """Should handle integer-like strings."""
        assert human_to_smallest_units("100", 6) == "100000000"

    def test_scientific_notation(self):
        """Should handle scientific notation."""
        assert human_to_smallest_units("1e-9", 9) == "1"
        assert human_to_smallest_units("1e6", 6) == "1000000000000"


class TestSmallestUnitsToHuman:
    """Test smallest_units_to_human function."""

    def test_sol_conversion(self):
        """Should convert lamports to SOL correctly."""
        assert smallest_units_to_human("70000000", 9) == "0.07"
        assert smallest_units_to_human("1000000000", 9) == "1"
        assert smallest_units_to_human("1", 9) == "1E-9"

    def test_usdc_conversion(self):
        """Should convert USDC smallest units correctly."""
        result = smallest_units_to_human("100000000", 6)
        assert Decimal(result) == Decimal("100")
        result2 = smallest_units_to_human("1000000", 6)
        assert Decimal(result2) == Decimal("1")

    def test_bonk_conversion(self):
        """Should convert BONK smallest units correctly."""
        result = smallest_units_to_human("100000000000", 5)
        assert Decimal(result) == Decimal("1000000")

    def test_zero(self):
        """Should handle zero."""
        assert smallest_units_to_human("0", 9) == "0"

    def test_invalid_units_raises_error(self):
        """Should raise ValueError for invalid input."""
        with pytest.raises(ValueError):
            smallest_units_to_human("abc", 9)


class TestUsdToTokenAmount:
    """Test usd_to_token_amount function."""

    def test_sol_calculation(self):
        """Should calculate SOL amount from USD correctly."""
        # $10 at $140/SOL = 0.0714... SOL
        result = usd_to_token_amount("10", "140")
        assert Decimal(result) == Decimal("10") / Decimal("140")

    def test_cheap_token_calculation(self):
        """Should handle cheap tokens correctly."""
        # $2 at $0.0000543/token = 36,832.41... tokens
        result = usd_to_token_amount("2", "0.0000543")
        expected = Decimal("2") / Decimal("0.0000543")
        assert abs(Decimal(result) - expected) < Decimal("0.000001")

    def test_exact_division(self):
        """Should handle exact divisions."""
        result = usd_to_token_amount("100", "10")
        assert result == "10"

    def test_zero_usd(self):
        """Should handle zero USD."""
        assert usd_to_token_amount("0", "100") == "0"

    def test_zero_price_raises_error(self):
        """Should raise error for zero price."""
        with pytest.raises(ValueError):
            usd_to_token_amount("10", "0")

    def test_negative_price_raises_error(self):
        """Should raise error for negative price."""
        with pytest.raises(ValueError):
            usd_to_token_amount("10", "-5")


class TestTokenAmountToUsd:
    """Test token_amount_to_usd function."""

    def test_sol_to_usd(self):
        """Should calculate USD from SOL correctly."""
        # 0.07 SOL at $140 = $9.80
        result = token_amount_to_usd("0.07", "140")
        assert result == "9.80"

    def test_large_token_amount(self):
        """Should handle large token amounts."""
        # 1,000,000 BONK at $0.00001 = $10
        result = token_amount_to_usd("1000000", "0.00001")
        assert result == "10.00000"


class TestApplyPercentageChange:
    """Test apply_percentage_change function."""

    def test_positive_percentage(self):
        """Should apply positive percentage correctly."""
        # 100 + 10% = 110
        result = apply_percentage_change("100", "10")
        assert Decimal(result) == Decimal("110")

    def test_negative_percentage(self):
        """Should apply negative percentage correctly."""
        # 100 - 0.5% = 99.5
        result = apply_percentage_change("100", "-0.5")
        assert Decimal(result) == Decimal("99.5")

    def test_zero_percentage(self):
        """Should handle zero percentage (no change)."""
        result = apply_percentage_change("100", "0")
        assert result == "100"

    def test_small_price_with_percentage(self):
        """Should handle small prices with percentage changes."""
        # $0.00001 - 0.5% = $0.00000995
        result = apply_percentage_change("0.00001", "-0.5")
        expected = Decimal("0.00001") * Decimal("0.995")
        assert Decimal(result) == expected


class TestCalculateSwapAmount:
    """Test calculate_swap_amount function."""

    def test_sol_swap(self):
        """Should calculate SOL swap amounts correctly."""
        result = calculate_swap_amount("10", "140", 9)
        # $10 / $140 = 0.0714... SOL
        human = Decimal(result["human_amount"])
        assert human > Decimal("0.071") and human < Decimal("0.072")
        # Smallest units should be ~71,428,571
        assert result["smallest_units"] == "71428571"

    def test_usdc_swap(self):
        """Should calculate USDC swap amounts correctly."""
        result = calculate_swap_amount("100", "1", 6)
        # $100 / $1 = 100 USDC
        assert result["human_amount"] == "100"
        assert result["smallest_units"] == "100000000"


class TestCalculateLimitOrderAmounts:
    """Test calculate_limit_order_amounts function."""

    def test_limit_order_at_current_price(self):
        """Should calculate limit order at current price."""
        result = calculate_limit_order_amounts(
            input_usd_amount="10",
            input_price_usd="140",  # SOL price
            input_decimals=9,
            output_price_usd="0.00001",  # BONK price
            output_decimals=5,
            price_change_percentage="0",  # Current price
        )
        # making_amount: $10 / $140 * 10^9 = 71,428,571 lamports
        assert result["making_amount"] == "71428571"
        # taking_amount: $10 / $0.00001 * 10^5 = 100,000,000,000
        assert result["taking_amount"] == "100000000000"

    def test_limit_order_buy_the_dip(self):
        """Should calculate limit order for buying the dip (get more output by waiting)."""
        result = calculate_limit_order_amounts(
            input_usd_amount="10",
            input_price_usd="140",  # SOL
            input_decimals=9,
            output_price_usd="0.00001",  # BONK
            output_decimals=5,
            price_change_percentage="0.5",  # Want 0.5% MORE output (buy the dip)
        )
        # making_amount unchanged
        assert result["making_amount"] == "71428571"
        # taking_amount: with +0.5%, we're asking for MORE output (1.005x)
        # $10 * 1.005 / $0.00001 * 10^5 = 100,500,000,000
        taking = int(result["taking_amount"])
        assert taking > 100000000000  # More than at current price
        assert taking < 101000000000  # But not too much more

    def test_limit_order_sell_high(self):
        """Should calculate limit order for selling at higher price (get more output)."""
        result = calculate_limit_order_amounts(
            input_usd_amount="10",
            input_price_usd="0.00001",  # BONK
            input_decimals=5,
            output_price_usd="140",  # SOL
            output_decimals=9,
            price_change_percentage="5",  # Want 5% MORE output (sell high)
        )
        # making: $10 / $0.00001 * 10^5 = 100,000,000,000
        making = int(result["making_amount"])
        assert making == 100000000000
        # taking: $10 * 1.05 / $140 * 10^9 = 75,000,000 (5% more SOL than current)
        taking = int(result["taking_amount"])
        # Current would be $10 / $140 * 10^9 = 71,428,571
        assert (
            taking > 71428571
        )  # More than at current price (we're asking for premium)
        assert taking < 80000000  # Reasonable upper bound


class TestCalculateLimitOrderInfo:
    """Test calculate_limit_order_info function."""

    def test_order_at_current_price(self):
        """Should calculate info for order at current market price."""
        # Order: sell 0.036001982 SOL for 521714.66282 BONK
        # Raw amounts: 36001982 lamports, 52171466282 smallest units
        result = calculate_limit_order_info(
            making_amount="36001982",  # 0.036001982 SOL in lamports
            taking_amount="52171466282",  # 521714.66282 BONK in smallest units (5 decimals)
            input_price_usd="139",  # Current SOL price
            output_price_usd="0.0000096",  # Current BONK price
            input_decimals=9,  # SOL decimals
            output_decimals=5,  # BONK decimals
        )
        # making_usd = 0.036001982 * 139 = ~$5.00
        assert float(result["making_usd"]) == pytest.approx(5.00, rel=0.01)
        # trigger_price = making_usd / taking_amount = $5 / 521714 = ~$0.00000958
        trigger_price = float(result["trigger_price_usd"])
        assert trigger_price == pytest.approx(0.00000958, rel=0.01)
        # Current price is $0.0000096, trigger is ~$0.00000958
        # Order wants to buy at LOWER price, so should NOT fill now
        assert result["should_fill_now"] is False

    def test_order_should_fill(self):
        """Should detect when order should fill (price dropped to trigger)."""
        # Raw: 36000000 lamports = 0.036 SOL, 52171400000 = 521714 BONK
        result = calculate_limit_order_info(
            making_amount="36000000",  # 0.036 SOL in lamports
            taking_amount="52171400000",  # 521714 BONK in smallest units
            input_price_usd="139",  # SOL price
            output_price_usd="0.0000090",  # BONK price dropped below trigger
            input_decimals=9,
            output_decimals=5,
        )
        # trigger_price = $5 / 521714 = ~$0.00000958
        # Current price = $0.0000090 which is BELOW trigger
        # Order should fill now!
        assert result["should_fill_now"] is True

    def test_price_difference_calculation(self):
        """Should calculate price difference correctly."""
        # Raw: 70000000 lamports = 0.07 SOL, 1000000 * 10^5 = 100000000000 smallest units
        result = calculate_limit_order_info(
            making_amount="70000000",  # 0.07 SOL in lamports
            taking_amount="100000000000",  # 1M BONK in smallest units (5 decimals)
            input_price_usd="140",
            output_price_usd="0.00001",  # Current BONK price
            input_decimals=9,
            output_decimals=5,
        )
        # trigger_price = (0.07 * 140) / 1000000 = $0.0000098 per BONK
        trigger_price = float(result["trigger_price_usd"])
        assert trigger_price == pytest.approx(0.0000098, rel=0.01)
        # price_diff = (trigger - current) / current * 100
        # = (0.0000098 - 0.00001) / 0.00001 * 100 = -2%
        price_diff = float(result["price_difference_percent"])
        assert price_diff == pytest.approx(-2.0, rel=0.1)

    def test_usd_values(self):
        """Should calculate USD values correctly."""
        # Raw: 1 SOL = 1000000000 lamports, 10000 USDC = 10000000000 (6 decimals)
        result = calculate_limit_order_info(
            making_amount="1000000000",  # 1 SOL in lamports
            taking_amount="10000000000",  # 10,000 USDC in smallest units (6 decimals)
            input_price_usd="140",
            output_price_usd="1",  # USDC
            input_decimals=9,
            output_decimals=6,
        )
        # making_usd = 1 * 140 = $140
        assert result["making_usd"] == "140"
        # taking_usd_at_current = 10000 * 1 = $10,000
        assert result["taking_usd_at_current"] == "10000"
        # trigger_price = $140 / 10000 = $0.014 per USDC
        assert float(result["trigger_price_usd"]) == pytest.approx(0.014, rel=0.001)

    def test_zero_taking_amount(self):
        """Should handle zero taking amount (edge case)."""
        result = calculate_limit_order_info(
            making_amount="1000000000",  # 1 SOL in lamports
            taking_amount="0",  # Zero output
            input_price_usd="140",
            output_price_usd="1",
            input_decimals=9,
            output_decimals=6,
        )
        # trigger_price should be 0 when taking is 0
        assert result["trigger_price_usd"] == "0"

    def test_zero_output_price(self):
        """Should handle zero output price (edge case)."""
        result = calculate_limit_order_info(
            making_amount="1000000000",  # 1 SOL in lamports
            taking_amount="1000000000",  # 1000 tokens (6 decimals)
            input_price_usd="140",
            output_price_usd="0",  # Zero price
            input_decimals=9,
            output_decimals=6,
        )
        # price_diff should be 0 when output_price is 0
        assert result["price_difference_percent"] == "0"

    def test_human_readable_amounts_no_decimals(self):
        """Should accept human-readable amounts when decimals are 0 (backwards compatibility)."""
        # When decimals=0, amounts are treated as already human-readable
        result = calculate_limit_order_info(
            making_amount="0.036",  # Already human-readable
            taking_amount="521714",  # Already human-readable
            input_price_usd="139",
            output_price_usd="0.0000096",
            input_decimals=0,  # No conversion
            output_decimals=0,  # No conversion
        )
        # making_usd = 0.036 * 139 = ~$5
        assert float(result["making_usd"]) == pytest.approx(5.00, rel=0.01)
        # Check trigger price is calculated correctly
        trigger_price = float(result["trigger_price_usd"])
        assert trigger_price == pytest.approx(0.00000958, rel=0.01)

    def test_invalid_amounts(self):
        """Should raise error for invalid amounts."""
        with pytest.raises(ValueError):
            calculate_limit_order_info(
                making_amount="invalid",
                taking_amount="100",
                input_price_usd="140",
                output_price_usd="1",
                input_decimals=9,
                output_decimals=6,
            )


# =============================================================================
# Integration Tests for Tool
# =============================================================================


@pytest.fixture
def math_tool():
    """Create a TokenMathTool instance."""
    tool = TokenMathTool()
    tool.configure({})
    return tool


class TestTokenMathToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, math_tool):
        """Should have correct tool name."""
        assert math_tool.name == "token_math"

    def test_schema_has_action(self, math_tool):
        """Should have action property."""
        schema = math_tool.get_schema()
        assert "action" in schema["properties"]
        assert "required" in schema
        assert "action" in schema["required"]

    def test_schema_action_enum(self, math_tool):
        """Should have correct action options."""
        schema = math_tool.get_schema()
        actions = schema["properties"]["action"]["enum"]
        assert "swap" in actions
        assert "transfer" in actions
        assert "limit_order" in actions
        assert "to_smallest_units" in actions
        assert "to_human" in actions
        assert "usd_to_tokens" in actions

    def test_schema_all_properties_required(self, math_tool):
        """Should have all properties in required list (OpenAI compatibility)."""
        schema = math_tool.get_schema()
        required = set(schema["required"])
        properties = set(schema["properties"].keys())
        assert required == properties, (
            "All properties must be required for OpenAI compatibility"
        )


class TestTokenMathToolExecuteSwap:
    """Test 'swap' action."""

    @pytest.mark.asyncio
    async def test_swap_success(self, math_tool):
        """Should calculate swap amounts correctly."""
        result = await math_tool.execute(
            action="swap",
            usd_amount="10",
            token_price_usd="140",
            decimals=9,
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "success"
        assert result["action"] == "swap"
        assert "smallest_units" in result
        assert result["smallest_units"] == "71428571"

    @pytest.mark.asyncio
    async def test_swap_missing_params(self, math_tool):
        """Should return error when params missing."""
        result = await math_tool.execute(
            action="swap",
            usd_amount="10",
            token_price_usd="",  # Missing
            decimals=0,  # Missing (0 is falsy)
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "error"
        assert "missing" in result["message"].lower()


class TestTokenMathToolExecuteTransfer:
    """Test 'transfer' action."""

    @pytest.mark.asyncio
    async def test_transfer_success(self, math_tool):
        """Should calculate transfer amount correctly (human-readable for privy_transfer)."""
        # $2 of AGENT at price $0.0000543
        result = await math_tool.execute(
            action="transfer",
            usd_amount="2",
            token_price_usd="0.0000543",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "success"
        assert result["action"] == "transfer"
        assert "amount" in result
        # $2 / $0.0000543 = 36,832.41... tokens
        amount = Decimal(result["amount"])
        assert amount > 36000 and amount < 37000

    @pytest.mark.asyncio
    async def test_transfer_sol(self, math_tool):
        """Should calculate SOL transfer amount correctly."""
        # $2 of SOL at price $142
        result = await math_tool.execute(
            action="transfer",
            usd_amount="2",
            token_price_usd="142",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "success"
        # $2 / $142 = 0.0140845... SOL
        amount = Decimal(result["amount"])
        assert amount > Decimal("0.014") and amount < Decimal("0.015")

    @pytest.mark.asyncio
    async def test_transfer_missing_params(self, math_tool):
        """Should return error when params missing."""
        result = await math_tool.execute(
            action="transfer",
            usd_amount="2",
            token_price_usd="",  # Missing
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "error"
        assert "missing" in result["message"].lower()


class TestTokenMathToolExecuteLimitOrder:
    """Test 'limit_order' action."""

    @pytest.mark.asyncio
    async def test_limit_order_success(self, math_tool):
        """Should calculate limit order amounts correctly."""
        result = await math_tool.execute(
            action="limit_order",
            usd_amount="10",
            token_price_usd="",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="140",
            input_decimals=9,
            output_price_usd="0.00001",
            output_decimals=5,
            price_change_percentage="-0.5",
        )
        assert result["status"] == "success"
        assert result["action"] == "limit_order"
        assert "making_amount" in result
        assert "taking_amount" in result
        # Verify the amounts are strings (as required by privy_trigger)
        assert isinstance(result["making_amount"], str)
        assert isinstance(result["taking_amount"], str)

    @pytest.mark.asyncio
    async def test_limit_order_missing_params(self, math_tool):
        """Should return error when params missing."""
        result = await math_tool.execute(
            action="limit_order",
            usd_amount="10",
            token_price_usd="",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="",  # Missing
            input_decimals=0,  # Missing
            output_price_usd="",  # Missing
            output_decimals=0,  # Missing
            price_change_percentage="0",
        )
        assert result["status"] == "error"


class TestTokenMathToolExecuteLimitOrderInfo:
    """Test 'limit_order_info' action."""

    @pytest.mark.asyncio
    async def test_limit_order_info_success(self, math_tool):
        """Should calculate order info correctly."""
        # Raw amounts: 36000000 lamports = 0.036 SOL, 52171400000 = 521714 BONK
        result = await math_tool.execute(
            action="limit_order_info",
            usd_amount="",
            token_price_usd="",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="139",  # SOL current price
            input_decimals=9,  # SOL decimals
            output_price_usd="0.0000096",  # BONK current price
            output_decimals=5,  # BONK decimals
            price_change_percentage="0",
            making_amount="36000000",  # 0.036 SOL in lamports
            taking_amount="52171400000",  # 521714 BONK in smallest units
        )
        assert result["status"] == "success"
        assert result["action"] == "limit_order_info"
        # Check USD values (0.036 * 139 = ~$5)
        assert float(result["making_usd"]) == pytest.approx(5.0, rel=0.01)
        # Check trigger price
        assert "trigger_price_usd" in result
        assert "should_fill_now" in result

    @pytest.mark.asyncio
    async def test_limit_order_info_should_fill(self, math_tool):
        """Should detect when order should fill."""
        result = await math_tool.execute(
            action="limit_order_info",
            usd_amount="",
            token_price_usd="",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="139",
            input_decimals=9,  # SOL decimals
            output_price_usd="0.0000090",  # Price dropped below trigger
            output_decimals=5,  # BONK decimals
            price_change_percentage="0",
            making_amount="36000000",  # 0.036 SOL in lamports
            taking_amount="52171400000",  # 521714 BONK in smallest units
        )
        assert result["status"] == "success"
        assert result["should_fill_now"] is True

    @pytest.mark.asyncio
    async def test_limit_order_info_missing_params(self, math_tool):
        """Should return error when params missing."""
        result = await math_tool.execute(
            action="limit_order_info",
            usd_amount="",
            token_price_usd="",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="",  # Missing
            input_decimals=0,
            output_price_usd="",  # Missing
            output_decimals=0,
            price_change_percentage="0",
            making_amount="",  # Missing
            taking_amount="",  # Missing
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_limit_order_info_missing_decimals(self, math_tool):
        """Should return error when decimals are 0."""
        result = await math_tool.execute(
            action="limit_order_info",
            usd_amount="",
            token_price_usd="",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="139",
            input_decimals=0,  # Missing! Should be 9 for SOL
            output_price_usd="0.0000096",
            output_decimals=0,  # Missing! Should be 5 for BONK
            price_change_percentage="0",
            making_amount="36000000",
            taking_amount="52171400000",
        )
        assert result["status"] == "error"
        assert (
            "input_decimals" in result["message"]
            or "output_decimals" in result["message"]
        )


class TestTokenMathToolExecuteToSmallestUnits:
    """Test 'to_smallest_units' action."""

    @pytest.mark.asyncio
    async def test_to_smallest_units_success(self, math_tool):
        """Should convert to smallest units correctly."""
        result = await math_tool.execute(
            action="to_smallest_units",
            usd_amount="",
            token_price_usd="",
            decimals=9,
            human_amount="0.07",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "success"
        assert result["smallest_units"] == "70000000"

    @pytest.mark.asyncio
    async def test_to_smallest_units_missing_params(self, math_tool):
        """Should return error when params missing."""
        result = await math_tool.execute(
            action="to_smallest_units",
            usd_amount="",
            token_price_usd="",
            decimals=0,  # Missing
            human_amount="0.07",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "error"


class TestTokenMathToolExecuteToHuman:
    """Test 'to_human' action."""

    @pytest.mark.asyncio
    async def test_to_human_success(self, math_tool):
        """Should convert to human readable correctly."""
        result = await math_tool.execute(
            action="to_human",
            usd_amount="",
            token_price_usd="",
            decimals=9,
            human_amount="",
            smallest_units="70000000",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "success"
        assert result["human_amount"] == "0.07"


class TestTokenMathToolExecuteUsdToTokens:
    """Test 'usd_to_tokens' action."""

    @pytest.mark.asyncio
    async def test_usd_to_tokens_success(self, math_tool):
        """Should calculate token amount from USD correctly."""
        result = await math_tool.execute(
            action="usd_to_tokens",
            usd_amount="10",
            token_price_usd="140",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "success"
        token_amount = Decimal(result["token_amount"])
        expected = Decimal("10") / Decimal("140")
        assert token_amount == expected


class TestTokenMathToolExecuteInvalidAction:
    """Test invalid action handling."""

    @pytest.mark.asyncio
    async def test_invalid_action(self, math_tool):
        """Should return error for invalid action."""
        result = await math_tool.execute(
            action="invalid_action",
            usd_amount="",
            token_price_usd="",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "error"
        assert "unknown action" in result["message"].lower()


class TestTokenMathToolExecuteErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_invalid_number_format(self, math_tool):
        """Should handle invalid number formats gracefully."""
        result = await math_tool.execute(
            action="swap",
            usd_amount="not_a_number",
            token_price_usd="140",
            decimals=9,
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_usd_to_tokens_missing_params(self, math_tool):
        """Should return error when usd_to_tokens params missing."""
        result = await math_tool.execute(
            action="usd_to_tokens",
            usd_amount="10",
            token_price_usd="",  # Missing
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "error"
        assert "missing" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_to_smallest_units_missing_params(self, math_tool):
        """Should return error when to_smallest_units params missing."""
        result = await math_tool.execute(
            action="to_smallest_units",
            usd_amount="",
            token_price_usd="",
            decimals=0,  # Missing (0 is falsy)
            human_amount="0.07",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "error"
        assert "missing" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_to_human_missing_params(self, math_tool):
        """Should return error when to_human params missing."""
        result = await math_tool.execute(
            action="to_human",
            usd_amount="",
            token_price_usd="",
            decimals=0,  # Missing (0 is falsy)
            human_amount="",
            smallest_units="70000000",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "error"
        assert "missing" in result["message"].lower()


class TestTokenAmountToUsdErrors:
    """Test token_amount_to_usd error handling."""

    def test_invalid_token_amount(self):
        """Should raise error for invalid token amount."""
        with pytest.raises(ValueError) as exc_info:
            token_amount_to_usd("not_a_number", "140")
        assert "invalid" in str(exc_info.value).lower()

    def test_invalid_price(self):
        """Should raise error for invalid price."""
        with pytest.raises(ValueError) as exc_info:
            token_amount_to_usd("0.07", "invalid_price")
        assert "invalid" in str(exc_info.value).lower()


class TestApplyPercentageChangeErrors:
    """Test apply_percentage_change error handling."""

    def test_invalid_amount(self):
        """Should raise error for invalid amount."""
        with pytest.raises(ValueError) as exc_info:
            apply_percentage_change("not_a_number", "10")
        assert "invalid" in str(exc_info.value).lower()

    def test_invalid_percentage(self):
        """Should raise error for invalid percentage."""
        with pytest.raises(ValueError) as exc_info:
            apply_percentage_change("100", "not_a_number")
        assert "invalid" in str(exc_info.value).lower()


class TestTokenMathToolExecuteGenericException:
    """Test generic exception handling in execute method."""

    @pytest.mark.asyncio
    async def test_generic_exception_handling(self, math_tool, monkeypatch):
        """Should catch and return generic exceptions."""
        import solana_agent.tools.token_math as token_math_module

        # Patch calculate_swap_amount to raise a generic exception
        def raise_runtime_error(*args, **kwargs):
            raise RuntimeError("Unexpected internal error")

        monkeypatch.setattr(
            token_math_module, "calculate_swap_amount", raise_runtime_error
        )

        result = await math_tool.execute(
            action="swap",
            usd_amount="10",
            token_price_usd="140",
            decimals=9,
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "error"
        assert "calculation error" in result["message"].lower()


# =============================================================================
# Plugin Tests
# =============================================================================


class TestTokenMathPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = TokenMathPlugin()
        assert plugin.name == "token_math"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = TokenMathPlugin()
        assert (
            "math" in plugin.description.lower()
            or "calculation" in plugin.description.lower()
        )

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = TokenMathPlugin()
        # _tool is None before initialize() is called
        assert plugin._tool is None
        # This should take the else branch in get_tools()
        result = plugin.get_tools()
        assert result == []
        assert isinstance(result, list)

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        from unittest.mock import MagicMock

        plugin = TokenMathPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

    def test_plugin_configure(self):
        """Should configure without error."""
        from unittest.mock import MagicMock

        plugin = TokenMathPlugin()
        mock_registry = MagicMock()
        plugin.initialize(mock_registry)

        # Should not raise
        plugin.configure({})
        assert plugin.config == {}

    def test_plugin_configure_before_init(self):
        """Should handle configure before initialize (no _tool yet)."""
        plugin = TokenMathPlugin()
        assert plugin._tool is None
        # This should take the else branch (if self._tool: is False)
        plugin.configure({"some": "config"})
        assert plugin.config == {"some": "config"}


class TestGetPlugin:
    """Test get_plugin function."""

    def test_get_plugin_returns_instance(self):
        """Should return a TokenMathPlugin instance."""
        plugin = get_plugin()
        assert isinstance(plugin, TokenMathPlugin)
        assert plugin.name == "token_math"


# =============================================================================
# Real-World Scenario Tests
# =============================================================================


class TestRealWorldScenarios:
    """Test real-world usage scenarios from the agent."""

    @pytest.mark.asyncio
    async def test_scenario_swap_2_dollars_of_sol(self, math_tool):
        """
        Scenario: User says "swap $2 of SOL to AGENT"
        Need to calculate SOL amount in lamports.
        """
        # From Birdeye: SOL price = $142, decimals = 9
        result = await math_tool.execute(
            action="swap",
            usd_amount="2",
            token_price_usd="142",
            decimals=9,
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "success"
        # $2 / $142 = 0.0140845... SOL
        # In lamports: 14,084,507
        smallest = int(result["smallest_units"])
        assert smallest > 14000000 and smallest < 15000000

    @pytest.mark.asyncio
    async def test_scenario_limit_buy_bonk_with_10_dollars_sol(self, math_tool):
        """
        Scenario: "limit buy BONK when price drops 0.5% with $10 of SOL"
        User wants 0.5% MORE BONK than current market (buy the dip).
        """
        # From Birdeye: SOL price=$140, decimals=9, BONK price=$0.00001, decimals=5
        result = await math_tool.execute(
            action="limit_order",
            usd_amount="10",
            token_price_usd="",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="140",  # SOL
            input_decimals=9,
            output_price_usd="0.00001",  # BONK
            output_decimals=5,
            price_change_percentage="0.5",  # Want 0.5% MORE output (buy the dip)
        )
        assert result["status"] == "success"

        # Verify making_amount (SOL to sell)
        # $10 / $140 = 0.0714... SOL = 71,428,571 lamports
        making = int(result["making_amount"])
        assert making == 71428571

        # Verify taking_amount (BONK to receive)
        # With +0.5%, we want 0.5% more BONK
        # Current: $10 / $0.00001 = 1,000,000 BONK
        # With +0.5%: 1,000,000 * 1.005 = 1,005,000 BONK
        # In smallest units: 1,005,000 * 100,000 = 100,500,000,000
        taking = int(result["taking_amount"])
        # Should be around 100.5 billion (12 digits)
        assert len(str(taking)) == 12  # Verify digit count
        assert taking > 100000000000  # More than at current price

    @pytest.mark.asyncio
    async def test_scenario_transfer_2_dollars_of_agent(self, math_tool):
        """
        Scenario: "send $2 of AGENT to ADDRESS"
        privy_transfer takes human amounts, so just need USD to tokens.
        """
        # From Birdeye: AGENT price = $0.0000543
        result = await math_tool.execute(
            action="usd_to_tokens",
            usd_amount="2",
            token_price_usd="0.0000543",
            decimals=0,
            human_amount="",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "success"
        # $2 / $0.0000543 = 36,832.41... AGENT
        token_amount = Decimal(result["token_amount"])
        assert token_amount > 36000 and token_amount < 37000

    @pytest.mark.asyncio
    async def test_bonk_digit_count_verification(self, math_tool):
        """
        From agent instructions:
        "For ~1M BONK (5 decimals): should have 11-12 digits"

        This test verifies our calculations match the expected digit counts.
        """
        result = await math_tool.execute(
            action="to_smallest_units",
            usd_amount="",
            token_price_usd="",
            decimals=5,
            human_amount="1000000",  # 1M BONK
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "success"
        # 1,000,000 * 100,000 = 100,000,000,000 (12 digits)
        assert result["smallest_units"] == "100000000000"
        assert len(result["smallest_units"]) == 12

    @pytest.mark.asyncio
    async def test_sol_digit_count_verification(self, math_tool):
        """
        From agent instructions:
        "0.07 SOL (9 decimals): should have 8 digits"
        """
        result = await math_tool.execute(
            action="to_smallest_units",
            usd_amount="",
            token_price_usd="",
            decimals=9,
            human_amount="0.07",
            smallest_units="",
            input_price_usd="",
            input_decimals=0,
            output_price_usd="",
            output_decimals=0,
            price_change_percentage="0",
        )
        assert result["status"] == "success"
        # 0.07 * 1,000,000,000 = 70,000,000 (8 digits)
        assert result["smallest_units"] == "70000000"
        assert len(result["smallest_units"]) == 8

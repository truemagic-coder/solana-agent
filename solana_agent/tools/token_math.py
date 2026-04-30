"""
Token Math tool for reliable token amount calculations.

LLMs are notoriously bad at math - they drop zeros, mess up decimal conversions,
and hallucinate calculations. This tool does the math reliably so the LLM just
passes inputs and gets outputs.
"""

import logging
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry

logger = logging.getLogger(__name__)


def human_to_smallest_units(human_amount: str, decimals: int) -> str:
    """
    Convert a human-readable token amount to smallest units (lamports, etc).

    Args:
        human_amount: Human readable amount as string (e.g., "0.07", "1000000")
        decimals: Number of decimal places for the token (e.g., 9 for SOL, 6 for USDC)

    Returns:
        String representation of amount in smallest units

    Examples:
        - human_to_smallest_units("0.07", 9) -> "70000000" (SOL)
        - human_to_smallest_units("1000000", 5) -> "100000000000" (BONK)
        - human_to_smallest_units("100", 6) -> "100000000" (USDC)
    """
    try:
        # Use Decimal for precise arithmetic
        amount = Decimal(str(human_amount))
        multiplier = Decimal(10) ** decimals
        smallest_units = amount * multiplier

        # Round down to avoid fractional smallest units
        smallest_units = smallest_units.to_integral_value(rounding=ROUND_DOWN)

        return str(int(smallest_units))
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid amount '{human_amount}': {e}")


def smallest_units_to_human(smallest_units: str, decimals: int) -> str:
    """
    Convert smallest units to human-readable amount.

    Args:
        smallest_units: Amount in smallest units as string (e.g., "70000000")
        decimals: Number of decimal places for the token

    Returns:
        String representation of human-readable amount

    Examples:
        - smallest_units_to_human("70000000", 9) -> "0.07"
        - smallest_units_to_human("100000000000", 5) -> "1000000"
    """
    try:
        units = Decimal(str(smallest_units))
        divisor = Decimal(10) ** decimals
        human_amount = units / divisor

        # Format without trailing zeros but preserve precision
        normalized = human_amount.normalize()
        return str(normalized)
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid smallest_units '{smallest_units}': {e}")


def usd_to_token_amount(usd_amount: str, token_price_usd: str) -> str:
    """
    Convert USD amount to token amount.

    Args:
        usd_amount: USD amount as string (e.g., "10", "2.50")
        token_price_usd: Token price in USD as string (e.g., "140.50", "0.00001")

    Returns:
        String representation of token amount (human-readable)

    Examples:
        - usd_to_token_amount("10", "140") -> "0.07142857142857142857..."
        - usd_to_token_amount("2", "0.0000543") -> "36832.41208791208791..."
    """
    try:
        usd = Decimal(str(usd_amount))
        price = Decimal(str(token_price_usd))

        if price <= 0:
            raise ValueError(f"Token price must be positive, got {token_price_usd}")

        token_amount = usd / price
        return str(token_amount)
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid USD calculation: {e}")


def token_amount_to_usd(token_amount: str, token_price_usd: str) -> str:
    """
    Convert token amount to USD value.

    Args:
        token_amount: Token amount as string (human-readable)
        token_price_usd: Token price in USD as string

    Returns:
        String representation of USD value
    """
    try:
        amount = Decimal(str(token_amount))
        price = Decimal(str(token_price_usd))
        usd_value = amount * price
        return str(usd_value)
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid token to USD calculation: {e}")


def apply_percentage_change(amount: str, percentage: str) -> str:
    """
    Apply a percentage change to an amount.

    Args:
        amount: Base amount as string
        percentage: Percentage change as string (e.g., "-0.5" for -0.5%, "10" for +10%)

    Returns:
        String representation of adjusted amount
    """
    try:
        base = Decimal(str(amount))
        pct = Decimal(str(percentage))
        multiplier = 1 + (pct / 100)
        result = base * multiplier
        return str(result)
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid percentage calculation: {e}")


def calculate_swap_amount(
    usd_amount: str,
    token_price_usd: str,
    decimals: int,
) -> Dict[str, str]:
    """
    Calculate the smallest units amount for a swap given USD value.

    This is a convenience function that combines usd_to_token_amount and
    human_to_smallest_units.

    Args:
        usd_amount: USD amount to swap (e.g., "10")
        token_price_usd: Current token price in USD
        decimals: Token decimals

    Returns:
        Dict with human_amount and smallest_units
    """
    human_amount = usd_to_token_amount(usd_amount, token_price_usd)
    smallest_units = human_to_smallest_units(human_amount, decimals)
    return {
        "human_amount": human_amount,
        "smallest_units": smallest_units,
    }


def calculate_limit_order_amounts(
    input_usd_amount: str,
    input_price_usd: str,
    input_decimals: int,
    output_price_usd: str,
    output_decimals: int,
    price_change_percentage: str = "0",
) -> Dict[str, str]:
    """
    Calculate both making_amount and taking_amount for a limit order.

    The price_change_percentage indicates how much MORE output you want than current market:
    - Positive = you want MORE output tokens (order waits for better price)
      Example: 0.5 = get 0.5% more output tokens (buy the dip)
      Example: 5 = get 5% more output tokens (sell high)
    - Zero = current market price (will fill immediately if liquidity available)
    - Negative = accept LESS output (will fill immediately, rare use case)

    For "buy the dip 0.5%" → use 0.5 (you want 0.5% more tokens)
    For "sell high 5%" → use 5 (you want 5% more of the output token)

    Args:
        input_usd_amount: USD value of input token to spend
        input_price_usd: Current price of input token in USD
        input_decimals: Decimals of input token
        output_price_usd: Current price of output token in USD
        output_decimals: Decimals of output token
        price_change_percentage: How much MORE output you want (positive = wait for better price)

    Returns:
        Dict with making_amount, taking_amount (both in smallest units)
    """
    # Calculate input token amount (making_amount)
    input_human = usd_to_token_amount(input_usd_amount, input_price_usd)
    making_amount = human_to_smallest_units(input_human, input_decimals)

    # Apply percentage change to get the adjusted USD value
    # Positive percentage = you want MORE output (input worth more, or output cheaper)
    # We ADD the percentage to the input USD value to get more output tokens
    pct = Decimal(str(price_change_percentage))
    adjusted_usd = Decimal(str(input_usd_amount)) * (
        Decimal("1") + pct / Decimal("100")
    )

    # Calculate output token amount at current price for the adjusted USD value
    output_human = usd_to_token_amount(str(adjusted_usd), output_price_usd)
    taking_amount = human_to_smallest_units(output_human, output_decimals)

    # Calculate what the target output price would be (for display purposes)
    # trigger_price = input_usd / output_tokens
    # If we're asking for more output, the effective output price we're paying is lower
    target_output_price = str(Decimal(str(input_usd_amount)) / Decimal(output_human))

    return {
        "making_amount": making_amount,
        "taking_amount": taking_amount,
        "input_human_amount": input_human,
        "output_human_amount": output_human,
        "target_output_price_usd": target_output_price,
    }


def calculate_limit_order_info(
    making_amount: str,
    taking_amount: str,
    input_price_usd: str,
    output_price_usd: str,
    input_decimals: int = 0,
    output_decimals: int = 0,
) -> Dict[str, str]:
    """
    Calculate display info for a limit order from its amounts.

    Use this when listing orders to show USD values and trigger prices.

    Args:
        making_amount: Raw amount of input token in smallest units (e.g., lamports)
        taking_amount: Raw amount of output token in smallest units
        input_price_usd: Current market price of input token in USD
        output_price_usd: Current market price of output token in USD
        input_decimals: Decimals for input token (e.g., 9 for SOL). If 0, assumes human-readable.
        output_decimals: Decimals for output token (e.g., 5 for BONK). If 0, assumes human-readable.

    Returns:
        Dict with USD values and trigger price info
    """
    try:
        raw_making = Decimal(str(making_amount))
        raw_taking = Decimal(str(taking_amount))
        input_price = Decimal(str(input_price_usd))
        output_price = Decimal(str(output_price_usd))

        # Convert from smallest units to human-readable if decimals provided
        if input_decimals > 0:
            making = raw_making / (Decimal(10) ** input_decimals)
        else:
            making = raw_making

        if output_decimals > 0:
            taking = raw_taking / (Decimal(10) ** output_decimals)
        else:
            taking = raw_taking

        # Calculate USD values at current prices
        making_usd = making * input_price
        taking_usd_at_current = taking * output_price

        # Calculate the trigger price (price per output token the order expects)
        # trigger_price = making_usd / taking_amount
        if taking > 0:
            trigger_price = making_usd / taking
        else:
            trigger_price = Decimal("0")

        # Calculate price difference from current market
        if output_price > 0:
            price_diff_pct = ((trigger_price - output_price) / output_price) * 100
        else:
            price_diff_pct = Decimal("0")

        # Determine if order should fill (for buy orders, trigger >= current means fill)
        # For a buy order: you want to buy when price drops TO trigger_price
        # Order fills when current_price <= trigger_price
        will_fill = output_price <= trigger_price

        return {
            "making_amount": str(making),
            "taking_amount": str(taking),
            "making_usd": str(making_usd),
            "taking_usd_at_current": str(taking_usd_at_current),
            "trigger_price_usd": str(trigger_price),
            "current_output_price_usd": str(output_price),
            "price_difference_percent": str(price_diff_pct),
            "should_fill_now": will_fill,
        }
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid limit order info calculation: {e}")


class TokenMathTool(AutoTool):
    """
    Reliable token math calculations for swaps, transfers, and limit orders.

    LLMs are bad at math - this tool does the calculations reliably.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="token_math",
            description=(
                "Calculate token amounts reliably for swaps, transfers, and limit orders. "
                "Use this BEFORE calling privy_ultra, privy_transfer, or privy_trigger to get the correct amounts. "
                "Actions: 'swap' (for privy_ultra - returns smallest_units), "
                "'transfer' (for privy_transfer - returns human-readable amount), "
                "'limit_order' (for privy_trigger - returns making_amount and taking_amount), "
                "'limit_order_info' (for displaying order list - calculates trigger price and USD values), "
                "'to_smallest_units', 'to_human', 'usd_to_tokens'. "
                "ALWAYS use this tool for any calculation - never do math yourself!"
            ),
            registry=registry,
        )

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "swap",
                        "transfer",
                        "limit_order",
                        "limit_order_info",
                        "to_smallest_units",
                        "to_human",
                        "usd_to_tokens",
                    ],
                    "description": (
                        "Action to perform: "
                        "'swap' - For privy_ultra: calculate smallest_units from USD amount, "
                        "'transfer' - For privy_transfer: calculate human-readable token amount from USD, "
                        "'limit_order' - For privy_trigger: calculate making_amount and taking_amount, "
                        "'limit_order_info' - For displaying orders: calculate trigger price and USD values from order amounts, "
                        "'to_smallest_units' - Convert human amount to smallest units, "
                        "'to_human' - Convert smallest units to human readable, "
                        "'usd_to_tokens' - Calculate token amount from USD value"
                    ),
                },
                "usd_amount": {
                    "type": "string",
                    "description": "USD amount (for 'swap', 'transfer', 'limit_order', 'usd_to_tokens'). E.g., '10' for $10. Pass empty string if not needed.",
                },
                "token_price_usd": {
                    "type": "string",
                    "description": "Token price in USD from Birdeye (for 'swap', 'transfer', 'usd_to_tokens'). E.g., '140.50'. Pass empty string if not needed.",
                },
                "decimals": {
                    "type": "integer",
                    "description": "Token decimals from Birdeye (for 'swap', 'to_smallest_units', 'to_human'). E.g., 9 for SOL, 6 for USDC, 5 for BONK. Pass 0 if not needed.",
                },
                "human_amount": {
                    "type": "string",
                    "description": "Human-readable token amount (for 'to_smallest_units'). E.g., '0.07' for SOL. Pass empty string if not needed.",
                },
                "smallest_units": {
                    "type": "string",
                    "description": "Amount in smallest units (for 'to_human'). E.g., '70000000' for 0.07 SOL. Pass empty string if not needed.",
                },
                "input_price_usd": {
                    "type": "string",
                    "description": "Input token price in USD (for 'limit_order'). Get from Birdeye. Pass empty string if not needed.",
                },
                "input_decimals": {
                    "type": "integer",
                    "description": "Input token decimals (for 'limit_order', 'limit_order_info'). Get from Birdeye. E.g., 9 for SOL, 5 for BONK. REQUIRED for limit_order_info.",
                },
                "output_price_usd": {
                    "type": "string",
                    "description": "Output token price in USD (for 'limit_order', 'limit_order_info'). Get from Birdeye. Pass empty string if not needed.",
                },
                "output_decimals": {
                    "type": "integer",
                    "description": "Output token decimals (for 'limit_order', 'limit_order_info'). Get from Birdeye. E.g., 9 for SOL, 5 for BONK. REQUIRED for limit_order_info.",
                },
                "price_change_percentage": {
                    "type": "string",
                    "description": (
                        "How much MORE output you want compared to current market (for 'limit_order'). "
                        "ALWAYS POSITIVE for limit orders that wait for better prices: "
                        "e.g., '0.5' = get 0.5%% more output (buy the dip), "
                        "'5' = get 5%% more output (sell high). "
                        "Pass '0' for current market price (will fill immediately). "
                        "Negative values mean accepting LESS than market (rare, fills immediately)."
                    ),
                },
                "making_amount": {
                    "type": "string",
                    "description": "Amount of input token being sold (for 'limit_order_info'). From order's rawMakingAmount field (smallest units). Pass empty string if not needed.",
                },
                "taking_amount": {
                    "type": "string",
                    "description": "Amount of output token to receive (for 'limit_order_info'). From order's rawTakingAmount field (smallest units). Pass empty string if not needed.",
                },
            },
            "required": [
                "action",
                "usd_amount",
                "token_price_usd",
                "decimals",
                "human_amount",
                "smallest_units",
                "input_price_usd",
                "input_decimals",
                "output_price_usd",
                "output_decimals",
                "price_change_percentage",
                "making_amount",
                "taking_amount",
            ],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        """No configuration needed for math operations."""
        pass

    async def execute(
        self,
        action: str,
        usd_amount: str = "",
        token_price_usd: str = "",
        decimals: int = 0,
        human_amount: str = "",
        smallest_units: str = "",
        input_price_usd: str = "",
        input_decimals: int = 0,
        output_price_usd: str = "",
        output_decimals: int = 0,
        price_change_percentage: str = "0",
        making_amount: str = "",
        taking_amount: str = "",
    ) -> Dict[str, Any]:
        """Execute the math calculation."""
        action = action.lower().strip()

        try:
            if action == "swap":
                # Calculate amount for a swap given USD value
                if not all([usd_amount, token_price_usd, decimals]):
                    return {
                        "status": "error",
                        "message": "Missing required params for 'swap': usd_amount, token_price_usd, decimals",
                    }
                result = calculate_swap_amount(usd_amount, token_price_usd, decimals)
                return {
                    "status": "success",
                    "action": "swap",
                    "human_amount": result["human_amount"],
                    "smallest_units": result["smallest_units"],
                    "message": f"For ${usd_amount}, you get {result['human_amount']} tokens ({result['smallest_units']} smallest units). Use smallest_units for privy_ultra amount.",
                }

            elif action == "transfer":
                # Calculate human-readable amount for a transfer given USD value
                # privy_transfer takes human amounts, NOT smallest units
                if not usd_amount or not token_price_usd:
                    return {
                        "status": "error",
                        "message": "Missing required params for 'transfer': usd_amount, token_price_usd",
                    }
                token_amount = usd_to_token_amount(usd_amount, token_price_usd)
                return {
                    "status": "success",
                    "action": "transfer",
                    "amount": token_amount,
                    "message": f"For ${usd_amount} at price ${token_price_usd}, transfer {token_amount} tokens. Use this amount directly for privy_transfer.",
                }

            elif action == "limit_order":
                # Calculate both amounts for a limit order
                if not all(
                    [
                        usd_amount,
                        input_price_usd,
                        input_decimals,
                        output_price_usd,
                        output_decimals,
                    ]
                ):
                    return {
                        "status": "error",
                        "message": "Missing required params for 'limit_order': usd_amount, input_price_usd, input_decimals, output_price_usd, output_decimals",
                    }
                result = calculate_limit_order_amounts(
                    input_usd_amount=usd_amount,
                    input_price_usd=input_price_usd,
                    input_decimals=input_decimals,
                    output_price_usd=output_price_usd,
                    output_decimals=output_decimals,
                    price_change_percentage=price_change_percentage,
                )
                return {
                    "status": "success",
                    "action": "limit_order",
                    "making_amount": result["making_amount"],
                    "taking_amount": result["taking_amount"],
                    "input_human_amount": result["input_human_amount"],
                    "output_human_amount": result["output_human_amount"],
                    "target_output_price_usd": result["target_output_price_usd"],
                    "message": f"Limit order: sell {result['input_human_amount']} input tokens (making_amount={result['making_amount']}) for {result['output_human_amount']} output tokens (taking_amount={result['taking_amount']}) at target price ${result['target_output_price_usd']}",
                }

            elif action == "to_smallest_units":
                # Convert human amount to smallest units
                if not human_amount or not decimals:
                    return {
                        "status": "error",
                        "message": "Missing required params for 'to_smallest_units': human_amount, decimals",
                    }
                result = human_to_smallest_units(human_amount, decimals)
                return {
                    "status": "success",
                    "action": "to_smallest_units",
                    "smallest_units": result,
                    "message": f"{human_amount} tokens with {decimals} decimals = {result} smallest units",
                }

            elif action == "to_human":
                # Convert smallest units to human readable
                if not smallest_units or not decimals:
                    return {
                        "status": "error",
                        "message": "Missing required params for 'to_human': smallest_units, decimals",
                    }
                result = smallest_units_to_human(smallest_units, decimals)
                return {
                    "status": "success",
                    "action": "to_human",
                    "human_amount": result,
                    "message": f"{smallest_units} smallest units with {decimals} decimals = {result} tokens",
                }

            elif action == "usd_to_tokens":
                # Calculate token amount from USD value
                if not usd_amount or not token_price_usd:
                    return {
                        "status": "error",
                        "message": "Missing required params for 'usd_to_tokens': usd_amount, token_price_usd",
                    }
                result = usd_to_token_amount(usd_amount, token_price_usd)
                return {
                    "status": "success",
                    "action": "usd_to_tokens",
                    "token_amount": result,
                    "message": f"${usd_amount} at price ${token_price_usd} = {result} tokens",
                }

            elif action == "limit_order_info":
                # Calculate display info for a limit order from its amounts
                if not all(
                    [making_amount, taking_amount, input_price_usd, output_price_usd]
                ):
                    return {
                        "status": "error",
                        "message": "Missing required params for 'limit_order_info': making_amount, taking_amount, input_price_usd, output_price_usd",
                    }
                # Validate that decimals are provided (should be > 0)
                if input_decimals == 0 or output_decimals == 0:
                    return {
                        "status": "error",
                        "message": "Missing required params for 'limit_order_info': input_decimals and output_decimals must be > 0. Get these from Birdeye. SOL=9, USDC=6, BONK=5.",
                    }
                result = calculate_limit_order_info(
                    making_amount=making_amount,
                    taking_amount=taking_amount,
                    input_price_usd=input_price_usd,
                    output_price_usd=output_price_usd,
                    input_decimals=input_decimals,
                    output_decimals=output_decimals,
                )
                return {
                    "status": "success",
                    "action": "limit_order_info",
                    "making_amount": result["making_amount"],
                    "taking_amount": result["taking_amount"],
                    "making_usd": result["making_usd"],
                    "taking_usd_at_current": result["taking_usd_at_current"],
                    "trigger_price_usd": result["trigger_price_usd"],
                    "current_output_price_usd": result["current_output_price_usd"],
                    "price_difference_percent": result["price_difference_percent"],
                    "should_fill_now": result["should_fill_now"],
                    "message": f"Order: sell {result['making_amount']} (${result['making_usd']}) for {result['taking_amount']} tokens. "
                    f"Trigger price: ${result['trigger_price_usd']}, Current price: ${result['current_output_price_usd']} "
                    f"({result['price_difference_percent']}% diff). Will fill now: {result['should_fill_now']}",
                }

            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}. Valid: swap, transfer, limit_order, limit_order_info, to_smallest_units, to_human, usd_to_tokens",
                }

        except ValueError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.exception(f"Token math error: {e}")
            return {"status": "error", "message": f"Calculation error: {e}"}


class TokenMathPlugin:
    """Plugin for reliable token math calculations."""

    def __init__(self):
        self.name = "token_math"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for reliable token amount calculations. Use before swaps and limit orders."

    def initialize(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._tool = TokenMathTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:
        return [self._tool] if self._tool else []


def get_plugin():
    return TokenMathPlugin()

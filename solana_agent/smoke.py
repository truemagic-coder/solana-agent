"""Live smoke helpers for the public Solana Agent SDK."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import inspect
from statistics import median
import time
from typing import Any, Mapping
from uuid import uuid4

import tiktoken

from solana_agent.factories.agent_factory import (
    DEFAULT_AGI_MEMORY_MODEL,
    DEFAULT_AGI_STATELESS_MODEL,
)


SMOKE_MESSAGE = "Reply with exactly SDK_SMOKE_OK"
PRIORITY_SMOKE_MESSAGE = "Reply with exactly SDK_SMOKE_PRIORITY_OK"
SEARCH_SMOKE_MESSAGE = (
    "Search if needed, then reply with the current year followed by SDK_SMOKE_SEARCH_OK"
)
MEMORY_STORE_SMOKE_SENTINEL = "SDK_SMOKE_MEMORY_STORED"
MEMORY_WORK_SMOKE_SENTINEL = "SDK_SMOKE_MEMORY_WORK_OK"
MEMORY_PROJECT_SMOKE_SENTINEL = "SDK_SMOKE_MEMORY_PROJECT_OK"
MEMORY_PRIORITY_WORK_SMOKE_SENTINEL = "SDK_SMOKE_MEMORY_PRIORITY_WORK_OK"
MEMORY_PRIORITY_PROJECT_SMOKE_SENTINEL = "SDK_SMOKE_MEMORY_PRIORITY_PROJECT_OK"
JUPITER_SMOKE_SENTINEL = "SDK_SMOKE_JUPITER_OK"
BIRDEYE_SMOKE_SENTINEL = "SDK_SMOKE_BIRDEYE_OK"
SWAP_SMOKE_SENTINEL = "SDK_SMOKE_SWAP_OK"
TRIGGER_SMOKE_SENTINEL = "SDK_SMOKE_TRIGGER_OK"
EARN_SMOKE_SENTINEL = "SDK_SMOKE_EARN_OK"
TECHNICAL_ANALYSIS_SMOKE_SENTINEL = "SDK_SMOKE_TECHNICAL_ANALYSIS_OK"
TOKEN_MATH_SMOKE_SENTINEL = "SDK_SMOKE_TOKEN_MATH_OK"
TRANSFER_SMOKE_SENTINEL = "SDK_SMOKE_TRANSFER_OK"
WRITE_SMOKE_FAILURE_TERMS = ("failed", "could not", "unable", "error")
DEFAULT_TOKENIZER_MODEL = "gpt-oss-120b"
SMOKE_OUTPUT_TOKEN_BUDGET = 48
SEARCH_OUTPUT_TOKEN_BUDGET = 96
TOOL_OUTPUT_TOKEN_BUDGET = 128
MINIMUM_SUGGESTED_FUNDING_USDC = Decimal("1.00")
DEFAULT_SMOKE_FUNDING_BUFFER_USDC = Decimal("0.50")
DESTRUCTIVE_STEP_BUFFER_USDC = Decimal("0.25")
READ_ONLY_TOOL_BUFFER_USDC = Decimal("0.25")
DEFAULT_TRANSFER_AMOUNT_USDC = Decimal("0.10")
DEFAULT_SWAP_AMOUNT_USDC = Decimal("0.05")
DEFAULT_TRIGGER_AMOUNT_USDC = Decimal("5.00")
DEFAULT_EARN_AMOUNT_USDC = Decimal("0.05")
WORK_MEMORY_TTL_DAYS = 7
PROJECT_MEMORY_TTL_DAYS = 30
MEMORY_WORK_SMOKE_TOKEN = "sdk-memory-work-7d"
MEMORY_PROJECT_SMOKE_TOKEN = "sdk-memory-project-30d"
MEMORY_PRIORITY_WORK_SMOKE_TOKEN = "sdk-memory-work-7d-priority"
MEMORY_PRIORITY_PROJECT_SMOKE_TOKEN = "sdk-memory-project-30d-priority"
SOLANA_USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
WRAPPED_SOL_MINT = "So11111111111111111111111111111111111111112"
DEFAULT_JUPITER_QUOTE_AMOUNT = 1_000_000


def _round_ms(value: float) -> float:
    return round(float(value), 2)


def _latency_summary(samples_ms: list[float]) -> dict[str, Any]:
    normalized = [_round_ms(sample) for sample in samples_ms if sample >= 0]
    if not normalized:
        return {
            "count": 0,
            "min_ms": None,
            "max_ms": None,
            "avg_ms": None,
            "median_ms": None,
            "samples_ms": [],
        }

    return {
        "count": len(normalized),
        "min_ms": _round_ms(min(normalized)),
        "max_ms": _round_ms(max(normalized)),
        "avg_ms": _round_ms(sum(normalized) / len(normalized)),
        "median_ms": _round_ms(median(normalized)),
        "samples_ms": normalized,
    }


def _money_string(value: Decimal) -> str:
    quantized = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return format(quantized, "f")


def _decimal_value(raw_value: Any) -> Decimal:
    try:
        return Decimal(str(raw_value or "0"))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _positive_decimal_value(raw_value: Any, *, field_name: str) -> Decimal:
    normalized = _decimal_value(raw_value)
    if normalized <= Decimal("0"):
        raise ValueError(f"{field_name} must be greater than 0")
    return normalized


def _path(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _encoding_for_model(model_name: str | None):
    try:
        return tiktoken.encoding_for_model(model_name or DEFAULT_TOKENIZER_MODEL)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _estimate_text_tokens(text: str, model_name: str | None = None) -> int:
    raw_text = str(text or "")
    if not raw_text:
        return 0
    return len(_encoding_for_model(model_name).encode(raw_text))


def _estimate_model_request_cost_usd(
    pricing_payload: Mapping[str, Any],
    *,
    prompt_text: str,
    expected_output_tokens: int,
    model_id: str = DEFAULT_AGI_STATELESS_MODEL,
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
) -> Decimal:
    base_rates = _path(pricing_payload, "base_rates", model_id)
    if not isinstance(base_rates, Mapping):
        return Decimal("0")

    input_rate = _decimal_value(base_rates.get("input_per_million"))
    output_rate = _decimal_value(base_rates.get("output_per_million"))
    prompt_tokens = Decimal(_estimate_text_tokens(prompt_text, tokenizer_model))
    output_tokens = Decimal(max(0, int(expected_output_tokens)))
    return (prompt_tokens * input_rate / Decimal("1000000")) + (
        output_tokens * output_rate / Decimal("1000000")
    )


def _jupiter_smoke_message() -> str:
    return (
        "Use the privy_swap_quote tool to preview a Jupiter swap from USDC to wrapped SOL. "
        f"Set input_mint to {SOLANA_USDC_MINT}, output_mint to {WRAPPED_SOL_MINT}, "
        f"and amount to {DEFAULT_JUPITER_QUOTE_AMOUNT}. Do not execute a swap. "
        f"After the preview succeeds, reply with {JUPITER_SMOKE_SENTINEL} and the quoted out_amount."
    )


def _memory_store_smoke_message(
    *, remember_token: str, retention_days: int, memory_ttl_tier: str
) -> str:
    return (
        f"This is the {retention_days}-day {memory_ttl_tier} memory smoke check. "
        f"Remember the exact token {remember_token} for this conversation. "
        f"Reply with exactly {MEMORY_STORE_SMOKE_SENTINEL} {remember_token}."
    )


def _memory_recall_smoke_message(
    *,
    remember_token: str,
    expected_sentinel: str,
    retention_days: int,
    memory_ttl_tier: str,
) -> str:
    return (
        f"In this same {retention_days}-day {memory_ttl_tier} memory smoke conversation, "
        f"what exact token did I ask you to remember? Reply with exactly {expected_sentinel} {remember_token}."
    )


def _birdeye_smoke_message() -> str:
    return (
        "Use the birdeye tool to fetch the current Solana USDC price. "
        f"Set action to price and address to {SOLANA_USDC_MINT}. Use empty strings or 0 for the other fields. "
        f"After the tool succeeds, reply with {BIRDEYE_SMOKE_SENTINEL} and the observed price."
    )


def _token_math_smoke_message() -> str:
    return (
        "Use the token_math tool to convert 0.10 USDC into smallest units and then back into a human-readable amount. "
        "First call token_math with action to_smallest_units, human_amount 0.10, and decimals 6. "
        "Then call token_math with action to_human using the smallest_units result and decimals 6. "
        f"After both succeed, reply with {TOKEN_MATH_SMOKE_SENTINEL} 100000 0.1 exactly once in the response."
    )


def _technical_analysis_smoke_message() -> str:
    return (
        "Use the technical_analysis tool to analyze wrapped SOL on Solana. "
        f"Set address to {WRAPPED_SOL_MINT} and timeframe to 4h. "
        f"After the tool succeeds, reply with {TECHNICAL_ANALYSIS_SMOKE_SENTINEL} and include the latest RSI value."
    )


def _swap_smoke_message(*, amount_usdc: Decimal) -> str:
    return (
        "Use token_math and privy_swap to execute a tiny live USDC to wrapped SOL swap. "
        f"First convert {_money_string(amount_usdc)} USDC to smallest units with token_math action to_smallest_units and decimals 6. "
        f"Then call privy_swap with input_mint {SOLANA_USDC_MINT}, output_mint {WRAPPED_SOL_MINT}, and the smallest-units amount. "
        f"After the swap succeeds, reply with {SWAP_SMOKE_SENTINEL} and the transaction signature."
    )


def _trigger_smoke_message(*, amount_usdc: Decimal) -> str:
    return (
        "Create and then cancel one Jupiter Trigger limit order using the current Privy wallet. "
        f"First use birdeye action price for address {WRAPPED_SOL_MINT} to fetch the current wrapped SOL price in USD. "
        "Then use token_math action limit_order with usd_amount "
        f"{_money_string(amount_usdc)}, input_price_usd 1, input_decimals 6, output_price_usd equal to the fetched wrapped SOL price, output_decimals 9, and price_change_percentage 100 so the order is far from market and should stay open for cancellation. "
        f"Next use privy_trigger action create with input_mint {SOLANA_USDC_MINT}, output_mint {WRAPPED_SOL_MINT}, and the making_amount and taking_amount from token_math. "
        "After creation, use privy_trigger action list to find the order and then use privy_trigger action cancel for that exact order. "
        f"After the cancel succeeds, reply with {TRIGGER_SMOKE_SENTINEL} and the canceled order public key."
    )


def _earn_smoke_message(*, amount_usdc: Decimal) -> str:
    return (
        "Use Jupiter Earn with a tiny reversible USDC position. "
        f"First use token_math action to_smallest_units for {_money_string(amount_usdc)} USDC with decimals 6. "
        f"Then call privy_earn action deposit exactly once with asset {SOLANA_USDC_MINT} and the smallest-units amount, "
        "then call privy_earn action positions exactly once for the current wallet, and finally call privy_earn action withdraw exactly once with the same asset and amount. "
        "Do not repeat any privy_earn action and do not use any other Jupiter tools. "
        f"After the withdraw succeeds, reply with {EARN_SMOKE_SENTINEL} and any transaction signatures returned."
    )


def _transfer_smoke_message(*, recipient: str, amount_usdc: Decimal) -> str:
    return (
        "Use the privy_transfer tool to send a live USDC transfer from the current Privy wallet. "
        f"Transfer {_money_string(amount_usdc)} USDC using mint {SOLANA_USDC_MINT} to {recipient}. "
        "Set memo to 'sdk smoke transfer'. "
        f"After the transfer succeeds, reply with {TRANSFER_SMOKE_SENTINEL} and the transaction signature."
    )


def _resolve_transfer_plan(
    *,
    include_transfer: bool,
    transfer_recipient: str | None,
    transfer_amount_usdc: Decimal | str | None,
) -> tuple[str, Decimal | None]:
    recipient = str(transfer_recipient or "").strip()
    if not include_transfer:
        return recipient, None
    if not recipient:
        raise ValueError(
            "transfer_recipient is required when include_transfer is enabled"
        )
    raw_amount = (
        transfer_amount_usdc
        if transfer_amount_usdc not in (None, "")
        else DEFAULT_TRANSFER_AMOUNT_USDC
    )
    return recipient, _positive_decimal_value(
        raw_amount,
        field_name="transfer_amount_usdc",
    )


def _transfer_payload(
    *, recipient: str, amount_usdc: Decimal | None
) -> dict[str, Any] | None:
    if not recipient or amount_usdc is None:
        return None
    return {
        "recipient": recipient,
        "amount_usdc": _money_string(amount_usdc),
        "mint": SOLANA_USDC_MINT,
    }


def _memory_project_surcharge_usd(pricing_payload: Mapping[str, Any]) -> Decimal:
    return _decimal_value(
        _path(
            pricing_payload,
            "base_rates",
            DEFAULT_AGI_MEMORY_MODEL,
            "project_memory_surcharge_usd",
        )
    )


def _estimate_memory_smoke_cost_usd(
    pricing_payload: Mapping[str, Any],
    *,
    memory_ttl_tier: str,
    retention_days: int,
    remember_token: str,
    expected_sentinel: str,
) -> Decimal:
    total_cost = _estimate_model_request_cost_usd(
        pricing_payload,
        prompt_text=_memory_store_smoke_message(
            remember_token=remember_token,
            retention_days=retention_days,
            memory_ttl_tier=memory_ttl_tier,
        ),
        expected_output_tokens=SMOKE_OUTPUT_TOKEN_BUDGET,
        model_id=DEFAULT_AGI_MEMORY_MODEL,
    )
    total_cost += _estimate_model_request_cost_usd(
        pricing_payload,
        prompt_text=_memory_recall_smoke_message(
            remember_token=remember_token,
            expected_sentinel=expected_sentinel,
            retention_days=retention_days,
            memory_ttl_tier=memory_ttl_tier,
        ),
        expected_output_tokens=SMOKE_OUTPUT_TOKEN_BUDGET,
        model_id=DEFAULT_AGI_MEMORY_MODEL,
    )
    if memory_ttl_tier == "project":
        total_cost += _memory_project_surcharge_usd(pricing_payload) * Decimal("2")
    return total_cost


def build_public_sdk_smoke_estimate(
    pricing_payload: Mapping[str, Any],
    forecast_payload: Mapping[str, Any] | None,
    *,
    include_search: bool,
    include_rotate: bool,
    include_export: bool,
    include_priority: bool = False,
    include_memory: bool = False,
    include_jupiter: bool = False,
    include_birdeye: bool = False,
    include_swap: bool = False,
    include_trigger: bool = False,
    include_earn: bool = False,
    include_technical_analysis: bool = False,
    include_token_math: bool = False,
    include_transfer: bool = False,
    transfer_amount_usdc: Decimal | str | None = None,
) -> dict[str, Any]:
    _, normalized_transfer_amount = _resolve_transfer_plan(
        include_transfer=include_transfer,
        transfer_recipient="sdk-smoke-estimate",
        transfer_amount_usdc=transfer_amount_usdc,
    )
    base_chat_cost = _estimate_model_request_cost_usd(
        pricing_payload,
        prompt_text=SMOKE_MESSAGE,
        expected_output_tokens=SMOKE_OUTPUT_TOKEN_BUDGET,
    )
    memory_work_cost = Decimal("0")
    memory_project_cost = Decimal("0")
    priority_chat_cost = Decimal("0")
    priority_memory_work_cost = Decimal("0")
    priority_memory_project_cost = Decimal("0")
    search_chat_cost = Decimal("0")
    tooling_chat_cost = Decimal("0")
    transfer_chat_cost = Decimal("0")
    search_surcharge = Decimal("0")
    search_provider_cost_ceiling = Decimal("0")
    tooling_buffer = Decimal("0")

    if include_priority:
        priority_chat_cost = _estimate_model_request_cost_usd(
            pricing_payload,
            prompt_text=PRIORITY_SMOKE_MESSAGE,
            expected_output_tokens=SMOKE_OUTPUT_TOKEN_BUDGET,
        )

    if include_memory:
        memory_work_cost = _estimate_memory_smoke_cost_usd(
            pricing_payload,
            memory_ttl_tier="work",
            retention_days=WORK_MEMORY_TTL_DAYS,
            remember_token=MEMORY_WORK_SMOKE_TOKEN,
            expected_sentinel=MEMORY_WORK_SMOKE_SENTINEL,
        )
        memory_project_cost = _estimate_memory_smoke_cost_usd(
            pricing_payload,
            memory_ttl_tier="project",
            retention_days=PROJECT_MEMORY_TTL_DAYS,
            remember_token=MEMORY_PROJECT_SMOKE_TOKEN,
            expected_sentinel=MEMORY_PROJECT_SMOKE_SENTINEL,
        )
        if include_priority:
            priority_memory_work_cost = _estimate_memory_smoke_cost_usd(
                pricing_payload,
                memory_ttl_tier="work",
                retention_days=WORK_MEMORY_TTL_DAYS,
                remember_token=MEMORY_PRIORITY_WORK_SMOKE_TOKEN,
                expected_sentinel=MEMORY_PRIORITY_WORK_SMOKE_SENTINEL,
            )
            priority_memory_project_cost = _estimate_memory_smoke_cost_usd(
                pricing_payload,
                memory_ttl_tier="project",
                retention_days=PROJECT_MEMORY_TTL_DAYS,
                remember_token=MEMORY_PRIORITY_PROJECT_SMOKE_TOKEN,
                expected_sentinel=MEMORY_PRIORITY_PROJECT_SMOKE_SENTINEL,
            )

    search_add_on = _path(pricing_payload, "search_add_on")
    if include_search and isinstance(search_add_on, Mapping):
        search_chat_cost = _estimate_model_request_cost_usd(
            pricing_payload,
            prompt_text=SEARCH_SMOKE_MESSAGE,
            expected_output_tokens=SEARCH_OUTPUT_TOKEN_BUDGET,
        )
        search_surcharge = _decimal_value(
            search_add_on.get("surcharge_per_request_usd")
        )
        search_provider_cost_ceiling = _decimal_value(
            search_add_on.get("search_max_provider_cost_usd")
        )

    tool_prompts: list[str] = []
    if include_jupiter:
        tool_prompts.append(_jupiter_smoke_message())
    if include_birdeye:
        tool_prompts.append(_birdeye_smoke_message())
    if include_technical_analysis:
        tool_prompts.append(_technical_analysis_smoke_message())
    if include_token_math:
        tool_prompts.append(_token_math_smoke_message())

    destructive_tool_prompts: list[str] = []
    if include_swap:
        destructive_tool_prompts.append(
            _swap_smoke_message(amount_usdc=DEFAULT_SWAP_AMOUNT_USDC)
        )
    if include_trigger:
        destructive_tool_prompts.append(
            _trigger_smoke_message(amount_usdc=DEFAULT_TRIGGER_AMOUNT_USDC)
        )
    if include_earn:
        destructive_tool_prompts.append(
            _earn_smoke_message(amount_usdc=DEFAULT_EARN_AMOUNT_USDC)
        )

    for prompt_text in tool_prompts:
        tooling_chat_cost += _estimate_model_request_cost_usd(
            pricing_payload,
            prompt_text=prompt_text,
            expected_output_tokens=TOOL_OUTPUT_TOKEN_BUDGET,
        )
        tooling_buffer += READ_ONLY_TOOL_BUFFER_USDC

    for prompt_text in destructive_tool_prompts:
        tooling_chat_cost += _estimate_model_request_cost_usd(
            pricing_payload,
            prompt_text=prompt_text,
            expected_output_tokens=TOOL_OUTPUT_TOKEN_BUDGET,
        )
        tooling_buffer += DESTRUCTIVE_STEP_BUFFER_USDC

    if include_transfer and normalized_transfer_amount is not None:
        transfer_chat_cost = _estimate_model_request_cost_usd(
            pricing_payload,
            prompt_text=_transfer_smoke_message(
                recipient="sdk-smoke-transfer-recipient",
                amount_usdc=normalized_transfer_amount,
            ),
            expected_output_tokens=TOOL_OUTPUT_TOKEN_BUDGET,
        )

    estimated_smoke_spend_ceiling = (
        base_chat_cost
        + memory_work_cost
        + memory_project_cost
        + priority_chat_cost
        + priority_memory_work_cost
        + priority_memory_project_cost
        + search_chat_cost
        + tooling_chat_cost
        + transfer_chat_cost
        + search_surcharge
        + search_provider_cost_ceiling
    )
    funding_buffer = DEFAULT_SMOKE_FUNDING_BUFFER_USDC
    funding_buffer += tooling_buffer
    if include_rotate:
        funding_buffer += DESTRUCTIVE_STEP_BUFFER_USDC
    if include_export:
        funding_buffer += DESTRUCTIVE_STEP_BUFFER_USDC
    if include_transfer:
        funding_buffer += DESTRUCTIVE_STEP_BUFFER_USDC

    live_transfer_amount = normalized_transfer_amount or Decimal("0")

    suggested_wallet_funding = max(
        MINIMUM_SUGGESTED_FUNDING_USDC,
        estimated_smoke_spend_ceiling + funding_buffer + live_transfer_amount,
    )

    forecast_projection = None
    current_month_spend = None
    if isinstance(forecast_payload, Mapping):
        projected_month_end = _path(forecast_payload, "projected_month_end", "spend")
        if projected_month_end is not None:
            forecast_projection = str(projected_month_end)
        current_month = _path(forecast_payload, "current_month", "spend")
        if current_month is not None:
            current_month_spend = str(current_month)

    assumptions = [
        "Estimate covers the standard hosted chat check, any enabled hosted memory recall checks, and all non-chat account/wallet helper calls.",
        "If search is enabled, estimate includes the fixed search surcharge and the configured provider cost ceiling.",
        "Wallet create/fetch, address lookup, summary, usage, forecast, and pricing endpoints are treated as unpriced control-plane checks.",
    ]
    if include_priority:
        assumptions.append(
            "Priority-tier validation is estimated using the base hosted model rates for both chat and memory because the pricing surface does not expose a separate tier surcharge."
        )
    if include_memory:
        assumptions.append(
            "Hosted memory validation uses two-turn recall checks for the 7-day work tier and the 30-day project tier; the project tier estimate includes the exposed project-memory surcharge on both turns."
        )
    if tool_prompts:
        assumptions.append(
            "Read-only Jupiter, Birdeye, technical-analysis, and token-math checks are budgeted as extra hosted turns plus a safety buffer because external provider and x402 fees are not fully exposed by the pricing endpoint."
        )
    if destructive_tool_prompts:
        assumptions.append(
            "Swap, trigger, and earn checks are budgeted with additional safety buffer only; the estimate does not fully model live routing fees, slippage, or temporary working capital."
        )
    if include_rotate or include_export:
        assumptions.append(
            "Rotate and export are included as live checks, but the hosted pricing surface does not expose a direct per-call fee for them, so the recommendation adds only a safety buffer."
        )
    if include_transfer:
        assumptions.append(
            "Optional USDC transfer adds the requested transfer amount to suggested funding plus a safety buffer for live execution overhead."
        )

    return {
        "estimated_smoke_spend_ceiling_usd": _money_string(
            estimated_smoke_spend_ceiling
        ),
        "suggested_wallet_funding_usdc": _money_string(suggested_wallet_funding),
        "components": {
            "standard_chat_request_usd": _money_string(base_chat_cost),
            "memory_work_request_usd": _money_string(memory_work_cost),
            "memory_project_request_usd": _money_string(memory_project_cost),
            "priority_chat_request_usd": _money_string(priority_chat_cost),
            "priority_memory_work_request_usd": _money_string(
                priority_memory_work_cost
            ),
            "priority_memory_project_request_usd": _money_string(
                priority_memory_project_cost
            ),
            "search_chat_request_usd": _money_string(search_chat_cost),
            "tooling_chat_requests_usd": _money_string(tooling_chat_cost),
            "transfer_chat_request_usd": _money_string(transfer_chat_cost),
            "search_surcharge_usd": _money_string(search_surcharge),
            "search_provider_cost_ceiling_usd": _money_string(
                search_provider_cost_ceiling
            ),
            "protocol_tooling_buffer_usdc": _money_string(tooling_buffer),
            "funding_buffer_usdc": _money_string(funding_buffer),
            "transfer_amount_usdc": _money_string(live_transfer_amount),
        },
        "account_forecast_context": {
            "current_month_spend_usd": current_month_spend,
            "projected_month_end_spend_usd": forecast_projection,
        },
        "coverage": {
            "includes_search": include_search,
            "includes_rotate": include_rotate,
            "includes_export": include_export,
            "includes_priority": include_priority,
            "includes_priority_chat": include_priority,
            "includes_memory_work": include_memory,
            "includes_memory_project": include_memory,
            "includes_priority_memory_work": include_memory and include_priority,
            "includes_priority_memory_project": include_memory and include_priority,
            "includes_jupiter_quote": include_jupiter,
            "includes_birdeye_read": include_birdeye,
            "includes_swap": include_swap,
            "includes_trigger": include_trigger,
            "includes_earn": include_earn,
            "includes_technical_analysis": include_technical_analysis,
            "includes_token_math": include_token_math,
            "includes_transfer": include_transfer,
            "excludes": [
                "Wallet creation through LLM tool-calling is intentionally excluded from hosted smoke coverage.",
            ],
        },
        "assumptions": assumptions,
    }


def _excerpt(value: Any, *, max_length: int = 160) -> str:
    rendered = str(value or "").strip().replace("\n", " ")
    if len(rendered) <= max_length:
        return rendered
    return rendered[: max_length - 3] + "..."


def _wallet_id_from_payload(payload: Mapping[str, Any]) -> str:
    return str(payload.get("wallet_id") or payload.get("id") or "").strip()


def _wallet_address_from_payload(payload: Mapping[str, Any]) -> str:
    return str(
        payload.get("address")
        or payload.get("public_address")
        or payload.get("wallet_address")
        or payload.get("public_key")
        or ""
    ).strip()


def _should_bootstrap_local_hosted_x402_signer(agent: Any) -> bool:
    query_service = getattr(agent, "query_service", None)
    agent_service = getattr(query_service, "agent_service", None)
    llm_provider = getattr(agent_service, "llm_provider", None)
    if llm_provider is None:
        return False

    auth_mode = str(getattr(llm_provider, "auth_mode", "") or "").strip()
    base_url = str(getattr(llm_provider, "base_url", "") or "").strip().lower()
    is_openai_endpoint = bool(getattr(llm_provider, "_is_openai_endpoint", False))

    return bool(
        auth_mode == "hosted_managed"
        and base_url
        and not is_openai_endpoint
        and ("127.0.0.1" in base_url or "localhost" in base_url)
    )


def _is_wallet_export_ownership_mismatch(exc: Exception) -> bool:
    error_text = str(exc or "")
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    if response_text:
        error_text = f"{error_text} {response_text}"
    return "wallet_id does not belong to privy_user_id" in error_text.lower()


async def _export_wallet_private_key_without_saved_wallet_id(
    agent: Any,
    *,
    privy_user_id: str,
    chain_type: str,
) -> str:
    resolver = getattr(agent, "_get_provider_method", None)
    if callable(resolver):
        provider_export = resolver(
            "export_wallet_private_key",
            "Hosted wallet management",
        )
        if callable(provider_export):
            provider_result = provider_export(
                privy_user_id=privy_user_id,
                chain_type=chain_type,
            )
            if inspect.isawaitable(provider_result):
                provider_payload = await provider_result
                if isinstance(provider_payload, Mapping):
                    private_key = str(provider_payload.get("private_key") or "").strip()
                else:
                    private_key = str(provider_payload or "").strip()
                if private_key:
                    return private_key

    return str(
        await agent.export_wallet_private_key(
            privy_user_id=privy_user_id,
            chain_type=chain_type,
        )
        or ""
    ).strip()


async def _bootstrap_local_hosted_x402_signer(
    agent: Any,
    *,
    wallet_id: str,
    privy_user_id: str,
    chain_type: str,
) -> dict[str, Any] | None:
    if not _should_bootstrap_local_hosted_x402_signer(agent):
        return None

    query_service = getattr(agent, "query_service", None)
    agent_service = getattr(query_service, "agent_service", None)
    llm_provider = getattr(agent_service, "llm_provider", None)
    if llm_provider is None:
        return None

    export_kwargs = {
        "wallet_id": wallet_id or None,
        "privy_user_id": privy_user_id,
        "chain_type": chain_type,
    }
    try:
        exported_private_key = str(
            await agent.export_wallet_private_key(**export_kwargs) or ""
        ).strip()
    except Exception as exc:
        if not wallet_id or not _is_wallet_export_ownership_mismatch(exc):
            raise
        exported_private_key = await _export_wallet_private_key_without_saved_wallet_id(
            agent,
            privy_user_id=privy_user_id,
            chain_type=chain_type,
        )
    if not exported_private_key:
        raise ValueError(
            "Local hosted x402 signer bootstrap returned an empty private key"
        )

    llm_provider.private_key = exported_private_key
    llm_provider.client = llm_provider._create_client(
        api_key=str(getattr(llm_provider, "api_key", "") or "x402").strip() or "x402",
        base_url=getattr(llm_provider, "base_url", None),
    )
    return {
        "name": "bootstrap_local_x402_signer",
        "status": "passed",
        "wallet_id": wallet_id,
        "private_key_redacted": True,
        "private_key_length": len(exported_private_key),
    }


async def _run_message_smoke_step(
    agent: Any,
    *,
    step_name: str,
    prompt_text: str,
    expected_sentinel: str,
    error_message: str,
    conversation_prefix: str,
    chain_type: str,
    model: str = "chat",
    memory_ttl_tier: str | None = "work",
    service_tier: str = "standard",
    search_enabled: bool = False,
    step_payload: Mapping[str, Any] | None = None,
    required_response_terms: tuple[str, ...] = (),
    failure_response_terms: tuple[str, ...] = (),
    max_attempts: int = 2,
    max_tool_iterations: int = 8,
    request_timeout_seconds: int = 120,
) -> dict[str, Any]:
    attempts = max(1, int(max_attempts))
    last_response: Any = None
    missing_term_error: ValueError | None = None
    for attempt in range(attempts):
        context_kwargs: dict[str, Any] = {
            "conversation_id": f"{conversation_prefix}-{uuid4().hex[:12]}",
            "model": model,
            "service_tier": service_tier,
            "search_enabled": search_enabled,
            "chain_type": chain_type,
        }
        if memory_ttl_tier is not None:
            context_kwargs["memory_ttl_tier"] = memory_ttl_tier
        context = await agent.context(**context_kwargs)
        context["_raise_stream_errors"] = True
        context["max_tool_iterations"] = max_tool_iterations
        context["request_timeout_seconds"] = request_timeout_seconds

        response_text = await _send_smoke_message(
            agent,
            step_name=step_name,
            prompt_text=prompt_text,
            context=context,
        )
        last_response = response_text

        if expected_sentinel not in response_text:
            if attempt + 1 < attempts:
                continue
            raise ValueError(error_message)

        response_text_casefold = response_text.casefold()
        for term in failure_response_terms:
            if term.casefold() in response_text_casefold:
                raise ValueError(
                    f"Smoke step '{step_name}' response included failure text: {term}"
                )

        missing_term_error = None
        for term in required_response_terms:
            if term.casefold() not in response_text_casefold:
                missing_term_error = ValueError(
                    f"Smoke step '{step_name}' response did not include required text: {term}"
                )
                break
        if missing_term_error is None:
            break
        if attempt + 1 >= attempts:
            raise missing_term_error

    response = last_response

    step = {
        "name": step_name,
        "status": "passed",
        "service_tier": service_tier,
        "search_enabled": search_enabled,
        "response_excerpt": _excerpt(response),
    }
    if isinstance(step_payload, Mapping):
        step.update(step_payload)
    return step


async def _send_smoke_message(
    agent: Any,
    *,
    step_name: str,
    prompt_text: str,
    context: Mapping[str, Any],
) -> str:
    try:
        response = await agent.message(prompt_text, **context)
    except Exception as exc:
        error_text = str(exc or "").strip() or type(exc).__name__
        if "Error code: 402" in error_text:
            raise ValueError(
                f"Smoke step '{step_name}' hit hosted payment error 402. Fund the hosted wallet/account using the preview estimate and retry. Original error: {error_text}"
            ) from exc
        raise ValueError(
            f"Smoke step '{step_name}' failed before returning a response: {error_text}"
        ) from exc
    return str(response or "")


async def _run_memory_smoke_step(
    agent: Any,
    *,
    step_name: str,
    expected_sentinel: str,
    error_message: str,
    conversation_prefix: str,
    chain_type: str,
    memory_ttl_tier: str,
    retention_days: int,
    remember_token: str,
    service_tier: str = "standard",
    max_attempts: int = 2,
    max_tool_iterations: int = 8,
    request_timeout_seconds: int = 120,
    warm_recall_count: int = 0,
) -> dict[str, Any]:
    attempts = max(1, int(max_attempts))
    warm_recall_iterations = max(0, int(warm_recall_count))
    for attempt in range(attempts):
        context_started_at = time.perf_counter()
        context = await agent.context(
            conversation_id=f"{conversation_prefix}-{uuid4().hex[:12]}",
            model="memory",
            memory_ttl_tier=memory_ttl_tier,
            service_tier=service_tier,
            search_enabled=False,
            chain_type=chain_type,
        )
        context_build_ms = _round_ms((time.perf_counter() - context_started_at) * 1000)
        context["_raise_stream_errors"] = True
        context["max_tool_iterations"] = max_tool_iterations
        context["request_timeout_seconds"] = request_timeout_seconds

        store_started_at = time.perf_counter()
        store_response = await _send_smoke_message(
            agent,
            step_name=step_name,
            prompt_text=_memory_store_smoke_message(
                remember_token=remember_token,
                retention_days=retention_days,
                memory_ttl_tier=memory_ttl_tier,
            ),
            context=context,
        )
        store_ms = _round_ms((time.perf_counter() - store_started_at) * 1000)
        if (
            MEMORY_STORE_SMOKE_SENTINEL not in store_response
            or remember_token not in store_response
        ):
            if attempt + 1 < attempts:
                continue
            raise ValueError(
                f"Smoke step '{step_name}' response did not include {MEMORY_STORE_SMOKE_SENTINEL} {remember_token}"
            )

        recall_prompt = _memory_recall_smoke_message(
            remember_token=remember_token,
            expected_sentinel=expected_sentinel,
            retention_days=retention_days,
            memory_ttl_tier=memory_ttl_tier,
        )
        cold_recall_started_at = time.perf_counter()
        recall_response = await _send_smoke_message(
            agent,
            step_name=step_name,
            prompt_text=recall_prompt,
            context=context,
        )
        cold_recall_ms = _round_ms(
            (time.perf_counter() - cold_recall_started_at) * 1000
        )
        if (
            expected_sentinel not in recall_response
            or remember_token not in recall_response
        ):
            if attempt + 1 < attempts:
                continue
            raise ValueError(error_message)

        warm_recall_samples_ms: list[float] = []
        warm_recall_response = ""
        warm_recall_failed = False
        for warm_index in range(warm_recall_iterations):
            warm_started_at = time.perf_counter()
            warm_recall_response = await _send_smoke_message(
                agent,
                step_name=f"{step_name}_warm_{warm_index + 1}",
                prompt_text=recall_prompt,
                context=context,
            )
            warm_recall_samples_ms.append(
                _round_ms((time.perf_counter() - warm_started_at) * 1000)
            )
            if (
                expected_sentinel not in warm_recall_response
                or remember_token not in warm_recall_response
            ):
                warm_recall_failed = True
                break

        if warm_recall_failed:
            if attempt + 1 < attempts:
                continue
            raise ValueError(error_message)

        warm_recall_summary = _latency_summary(warm_recall_samples_ms)

        return {
            "name": step_name,
            "status": "passed",
            "model": DEFAULT_AGI_MEMORY_MODEL,
            "memory_ttl_tier": memory_ttl_tier,
            "retention_days": retention_days,
            "remember_token": remember_token,
            "service_tier": service_tier,
            "response_excerpt": _excerpt(recall_response),
            "warm_response_excerpt": _excerpt(warm_recall_response or recall_response),
            "latency_ms": {
                "context_build_ms": context_build_ms,
                "store_ms": store_ms,
                "cold_recall_ms": cold_recall_ms,
                "warm_recall": warm_recall_summary,
            },
        }

    raise ValueError(error_message)


def _resolve_memory_smoke_profile(
    *,
    memory_ttl_tier: str,
    service_tier: str,
) -> tuple[int, str, str]:
    normalized_memory_tier = str(memory_ttl_tier or "").strip().lower()
    normalized_service_tier = str(service_tier or "").strip().lower()
    if normalized_memory_tier not in {"work", "project"}:
        raise ValueError("memory_ttl_tier must be one of: work, project")
    if normalized_service_tier not in {"standard", "priority"}:
        raise ValueError("service_tier must be one of: standard, priority")

    if normalized_memory_tier == "work":
        if normalized_service_tier == "priority":
            return (
                WORK_MEMORY_TTL_DAYS,
                MEMORY_PRIORITY_WORK_SMOKE_SENTINEL,
                MEMORY_PRIORITY_WORK_SMOKE_TOKEN,
            )
        return WORK_MEMORY_TTL_DAYS, MEMORY_WORK_SMOKE_SENTINEL, MEMORY_WORK_SMOKE_TOKEN

    if normalized_service_tier == "priority":
        return (
            PROJECT_MEMORY_TTL_DAYS,
            MEMORY_PRIORITY_PROJECT_SMOKE_SENTINEL,
            MEMORY_PRIORITY_PROJECT_SMOKE_TOKEN,
        )
    return (
        PROJECT_MEMORY_TTL_DAYS,
        MEMORY_PROJECT_SMOKE_SENTINEL,
        MEMORY_PROJECT_SMOKE_TOKEN,
    )


async def run_public_sdk_memory_benchmark(
    agent: Any,
    *,
    chain_type: str = "solana",
    memory_ttl_tier: str = "project",
    service_tier: str = "standard",
    warm_recall_count: int = 3,
    preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retention_days, expected_sentinel, remember_token = _resolve_memory_smoke_profile(
        memory_ttl_tier=memory_ttl_tier,
        service_tier=service_tier,
    )
    smoke_preview = preview or await build_public_sdk_smoke_preview(
        agent,
        chain_type=chain_type,
        include_search=False,
        include_priority=service_tier == "priority",
        include_memory=True,
    )
    privy_user_id = str(smoke_preview.get("privy_user_id") or "").strip()
    wallet_payload = dict(smoke_preview.get("wallet") or {})
    wallet_id = str(wallet_payload.get("wallet_id") or "").strip()
    steps = list(smoke_preview.get("steps") or [])

    bootstrap_step = await _bootstrap_local_hosted_x402_signer(
        agent,
        wallet_id=wallet_id,
        privy_user_id=privy_user_id,
        chain_type=chain_type,
    )
    if bootstrap_step is not None:
        steps.append(bootstrap_step)

    step = await _run_memory_smoke_step(
        agent,
        step_name=f"memory_benchmark_{memory_ttl_tier}_{service_tier}",
        expected_sentinel=expected_sentinel,
        error_message=(
            f"Memory benchmark response did not include {expected_sentinel} {remember_token}"
        ),
        conversation_prefix=f"sdk-benchmark-memory-{memory_ttl_tier}-{service_tier}",
        chain_type=chain_type,
        memory_ttl_tier=memory_ttl_tier,
        retention_days=retention_days,
        remember_token=remember_token,
        service_tier=service_tier,
        warm_recall_count=warm_recall_count,
    )
    steps.append(step)
    return {
        "ok": True,
        "preview_only": False,
        "memory_ttl_tier": memory_ttl_tier,
        "service_tier": service_tier,
        "retention_days": retention_days,
        "warm_recall_count": max(0, int(warm_recall_count)),
        "privy_user_id": privy_user_id,
        "wallet": wallet_payload,
        "coverage": dict(smoke_preview.get("coverage") or {}),
        "latency_ms": dict(step.get("latency_ms") or {}),
        "step": step,
        "steps": steps,
        "account": dict(smoke_preview.get("account") or {}),
    }


async def build_public_sdk_smoke_preview(
    agent: Any,
    *,
    chain_type: str = "solana",
    forecast_window_days: int = 30,
    include_search: bool = True,
    include_rotate: bool = False,
    include_export: bool = False,
    include_priority: bool = False,
    include_memory: bool = False,
    include_jupiter: bool = False,
    include_birdeye: bool = False,
    include_swap: bool = False,
    include_trigger: bool = False,
    include_earn: bool = False,
    include_technical_analysis: bool = False,
    include_token_math: bool = False,
    include_transfer: bool = False,
    transfer_recipient: str | None = None,
    transfer_amount_usdc: Decimal | str | None = None,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    resolved_transfer_recipient, resolved_transfer_amount = _resolve_transfer_plan(
        include_transfer=include_transfer,
        transfer_recipient=transfer_recipient,
        transfer_amount_usdc=transfer_amount_usdc,
    )

    try:
        privy_user_id = agent._configured_privy_user_id()
        user_created = False
    except Exception:
        user_payload = await agent.create_privy_user()
        privy_user_id = str(
            user_payload.get("privy_user_id") or user_payload.get("id") or ""
        ).strip()
        if not privy_user_id:
            raise ValueError("Privy user creation did not return a privy_user_id")
        user_created = bool(user_payload.get("created", True))

    steps.append(
        {
            "name": "resolve_privy_user",
            "status": "passed",
            "privy_user_id": privy_user_id,
            "created": user_created,
        }
    )

    wallet_payload = await agent.create_wallet(
        privy_user_id=privy_user_id,
        chain_type=chain_type,
    )
    wallet_id = _wallet_id_from_payload(wallet_payload)
    wallet_address = _wallet_address_from_payload(wallet_payload)
    if not wallet_id:
        raise ValueError("Smoke preview wallet check did not return a wallet_id")
    steps.append(
        {
            "name": "create_or_fetch_wallet",
            "status": "passed",
            "wallet_id": wallet_id,
            "address": wallet_address,
            "created": bool(wallet_payload.get("created", False)),
        }
    )

    resolved_wallet_address = await agent.get_wallet_address(wallet_id=wallet_id)
    resolved_wallet_address = str(resolved_wallet_address or "").strip()
    if not resolved_wallet_address:
        raise ValueError("Smoke preview wallet address check returned no address")
    steps.append(
        {
            "name": "get_wallet_address",
            "status": "passed",
            "wallet_id": wallet_id,
            "address": resolved_wallet_address,
            "matches_wallet_payload": resolved_wallet_address == wallet_address,
        }
    )

    summary = await agent.get_account_summary()
    steps.append(
        {
            "name": "account_summary",
            "status": "passed",
            "requests_lifetime": _path(summary, "requests", "lifetime") or 0,
            "month_spend_usd": _path(summary, "spend", "month"),
        }
    )

    usage = await agent.get_usage_report("day", group_by="conversation")
    buckets = usage.get("buckets") if isinstance(usage, Mapping) else []
    steps.append(
        {
            "name": "account_usage",
            "status": "passed",
            "bucket_count": len(list(buckets or [])),
        }
    )

    forecast = await agent.get_usage_forecast(window_days=forecast_window_days)
    steps.append(
        {
            "name": "account_forecast",
            "status": "passed",
            "forecast_window_days": forecast_window_days,
            "projected_month_end_spend_usd": _path(
                forecast,
                "projected_month_end",
                "spend",
            ),
        }
    )

    pricing = await agent.get_pricing_info()
    steps.append(
        {
            "name": "account_pricing",
            "status": "passed",
            "search_surcharge_usd": _path(
                pricing,
                "search_add_on",
                "surcharge_per_request_usd",
            ),
            "search_provider_cost_ceiling_usd": _path(
                pricing,
                "search_add_on",
                "search_max_provider_cost_usd",
            ),
        }
    )

    estimate = build_public_sdk_smoke_estimate(
        pricing,
        forecast,
        include_search=include_search,
        include_rotate=include_rotate,
        include_export=include_export,
        include_priority=include_priority,
        include_memory=include_memory,
        include_jupiter=include_jupiter,
        include_birdeye=include_birdeye,
        include_swap=include_swap,
        include_trigger=include_trigger,
        include_earn=include_earn,
        include_technical_analysis=include_technical_analysis,
        include_token_math=include_token_math,
        include_transfer=include_transfer,
        transfer_amount_usdc=resolved_transfer_amount,
    )
    coverage = dict(estimate.get("coverage") or {})

    return {
        "ok": True,
        "preview_only": True,
        "privy_user_id": privy_user_id,
        "wallet": {
            "wallet_id": wallet_id,
            "address": resolved_wallet_address,
            "chain_type": chain_type,
        },
        "coverage": coverage,
        "transfer": _transfer_payload(
            recipient=resolved_transfer_recipient,
            amount_usdc=resolved_transfer_amount,
        ),
        "estimate": estimate,
        "steps": steps,
        "account": {
            "summary": summary,
            "usage": usage,
            "forecast": forecast,
            "pricing": pricing,
        },
    }


async def run_public_sdk_smoke(
    agent: Any,
    *,
    chain_type: str = "solana",
    forecast_window_days: int = 30,
    include_search: bool = True,
    include_rotate: bool = False,
    include_export: bool = False,
    include_priority: bool = False,
    include_memory: bool = False,
    include_jupiter: bool = False,
    include_birdeye: bool = False,
    include_swap: bool = False,
    include_trigger: bool = False,
    include_earn: bool = False,
    include_technical_analysis: bool = False,
    include_token_math: bool = False,
    include_transfer: bool = False,
    transfer_recipient: str | None = None,
    transfer_amount_usdc: Decimal | str | None = None,
    preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_transfer_recipient, resolved_transfer_amount = _resolve_transfer_plan(
        include_transfer=include_transfer,
        transfer_recipient=transfer_recipient,
        transfer_amount_usdc=transfer_amount_usdc,
    )
    smoke_preview = preview or await build_public_sdk_smoke_preview(
        agent,
        chain_type=chain_type,
        forecast_window_days=forecast_window_days,
        include_search=include_search,
        include_rotate=include_rotate,
        include_export=include_export,
        include_priority=include_priority,
        include_memory=include_memory,
        include_jupiter=include_jupiter,
        include_birdeye=include_birdeye,
        include_swap=include_swap,
        include_trigger=include_trigger,
        include_earn=include_earn,
        include_technical_analysis=include_technical_analysis,
        include_token_math=include_token_math,
        include_transfer=include_transfer,
        transfer_recipient=resolved_transfer_recipient,
        transfer_amount_usdc=resolved_transfer_amount,
    )
    steps = list(smoke_preview.get("steps") or [])
    privy_user_id = str(smoke_preview.get("privy_user_id") or "").strip()
    wallet_payload = dict(smoke_preview.get("wallet") or {})
    wallet_id = str(wallet_payload.get("wallet_id") or "").strip()
    wallet_address = str(wallet_payload.get("address") or "").strip()
    transfer_payload = _transfer_payload(
        recipient=resolved_transfer_recipient,
        amount_usdc=resolved_transfer_amount,
    )

    bootstrap_step = await _bootstrap_local_hosted_x402_signer(
        agent,
        wallet_id=wallet_id,
        privy_user_id=privy_user_id,
        chain_type=chain_type,
    )
    if bootstrap_step is not None:
        steps.append(bootstrap_step)

    steps.append(
        await _run_message_smoke_step(
            agent,
            step_name="chat_message",
            prompt_text=SMOKE_MESSAGE,
            expected_sentinel="SDK_SMOKE_OK",
            error_message="Smoke chat response did not include SDK_SMOKE_OK",
            conversation_prefix="sdk-smoke",
            chain_type=chain_type,
        )
    )

    if include_priority:
        steps.append(
            await _run_message_smoke_step(
                agent,
                step_name="chat_message_priority_tier",
                prompt_text=PRIORITY_SMOKE_MESSAGE,
                expected_sentinel="SDK_SMOKE_PRIORITY_OK",
                error_message=(
                    "Priority-tier smoke response did not include SDK_SMOKE_PRIORITY_OK"
                ),
                conversation_prefix="sdk-smoke-priority",
                chain_type=chain_type,
                service_tier="priority",
            )
        )

    if include_memory:
        steps.append(
            await _run_memory_smoke_step(
                agent,
                step_name="memory_recall_work_7d",
                expected_sentinel=MEMORY_WORK_SMOKE_SENTINEL,
                error_message=(
                    f"Memory smoke response did not include {MEMORY_WORK_SMOKE_SENTINEL} {MEMORY_WORK_SMOKE_TOKEN}"
                ),
                conversation_prefix="sdk-smoke-memory-work",
                chain_type=chain_type,
                memory_ttl_tier="work",
                retention_days=WORK_MEMORY_TTL_DAYS,
                remember_token=MEMORY_WORK_SMOKE_TOKEN,
            )
        )
        if include_priority:
            steps.append(
                await _run_memory_smoke_step(
                    agent,
                    step_name="memory_recall_work_7d_priority_tier",
                    expected_sentinel=MEMORY_PRIORITY_WORK_SMOKE_SENTINEL,
                    error_message=(
                        f"Priority memory smoke response did not include {MEMORY_PRIORITY_WORK_SMOKE_SENTINEL} {MEMORY_PRIORITY_WORK_SMOKE_TOKEN}"
                    ),
                    conversation_prefix="sdk-smoke-memory-work-priority",
                    chain_type=chain_type,
                    memory_ttl_tier="work",
                    retention_days=WORK_MEMORY_TTL_DAYS,
                    remember_token=MEMORY_PRIORITY_WORK_SMOKE_TOKEN,
                    service_tier="priority",
                )
            )
        steps.append(
            await _run_memory_smoke_step(
                agent,
                step_name="memory_recall_project_30d",
                expected_sentinel=MEMORY_PROJECT_SMOKE_SENTINEL,
                error_message=(
                    f"Memory smoke response did not include {MEMORY_PROJECT_SMOKE_SENTINEL} {MEMORY_PROJECT_SMOKE_TOKEN}"
                ),
                conversation_prefix="sdk-smoke-memory-project",
                chain_type=chain_type,
                memory_ttl_tier="project",
                retention_days=PROJECT_MEMORY_TTL_DAYS,
                remember_token=MEMORY_PROJECT_SMOKE_TOKEN,
            )
        )
        if include_priority:
            steps.append(
                await _run_memory_smoke_step(
                    agent,
                    step_name="memory_recall_project_30d_priority_tier",
                    expected_sentinel=MEMORY_PRIORITY_PROJECT_SMOKE_SENTINEL,
                    error_message=(
                        f"Priority memory smoke response did not include {MEMORY_PRIORITY_PROJECT_SMOKE_SENTINEL} {MEMORY_PRIORITY_PROJECT_SMOKE_TOKEN}"
                    ),
                    conversation_prefix="sdk-smoke-memory-project-priority",
                    chain_type=chain_type,
                    memory_ttl_tier="project",
                    retention_days=PROJECT_MEMORY_TTL_DAYS,
                    remember_token=MEMORY_PRIORITY_PROJECT_SMOKE_TOKEN,
                    service_tier="priority",
                )
            )

    if include_search:
        steps.append(
            await _run_message_smoke_step(
                agent,
                step_name="chat_message_search_enabled",
                prompt_text=SEARCH_SMOKE_MESSAGE,
                expected_sentinel="SDK_SMOKE_SEARCH_OK",
                error_message=(
                    "Smoke search-enabled response did not include SDK_SMOKE_SEARCH_OK"
                ),
                conversation_prefix="sdk-smoke-search",
                chain_type=chain_type,
                search_enabled=True,
            )
        )

    if include_jupiter:
        steps.append(
            await _run_message_smoke_step(
                agent,
                step_name="jupiter_swap_quote",
                prompt_text=_jupiter_smoke_message(),
                expected_sentinel=JUPITER_SMOKE_SENTINEL,
                error_message=(
                    f"Jupiter smoke response did not include {JUPITER_SMOKE_SENTINEL}"
                ),
                conversation_prefix="sdk-smoke-jupiter",
                chain_type=chain_type,
            )
        )

    if include_birdeye:
        steps.append(
            await _run_message_smoke_step(
                agent,
                step_name="birdeye_read",
                prompt_text=_birdeye_smoke_message(),
                expected_sentinel=BIRDEYE_SMOKE_SENTINEL,
                error_message=(
                    f"Birdeye smoke response did not include {BIRDEYE_SMOKE_SENTINEL}"
                ),
                conversation_prefix="sdk-smoke-birdeye",
                chain_type=chain_type,
            )
        )

    if include_token_math:
        steps.append(
            await _run_message_smoke_step(
                agent,
                step_name="token_math",
                prompt_text=_token_math_smoke_message(),
                expected_sentinel=TOKEN_MATH_SMOKE_SENTINEL,
                error_message=(
                    f"Token-math smoke response did not include {TOKEN_MATH_SMOKE_SENTINEL}"
                ),
                conversation_prefix="sdk-smoke-token-math",
                chain_type=chain_type,
                required_response_terms=("100000", "0.1"),
            )
        )

    if include_technical_analysis:
        steps.append(
            await _run_message_smoke_step(
                agent,
                step_name="technical_analysis",
                prompt_text=_technical_analysis_smoke_message(),
                expected_sentinel=TECHNICAL_ANALYSIS_SMOKE_SENTINEL,
                error_message=(
                    f"Technical-analysis smoke response did not include {TECHNICAL_ANALYSIS_SMOKE_SENTINEL}"
                ),
                conversation_prefix="sdk-smoke-technical-analysis",
                chain_type=chain_type,
            )
        )

    if include_swap:
        steps.append(
            await _run_message_smoke_step(
                agent,
                step_name="swap_live",
                prompt_text=_swap_smoke_message(amount_usdc=DEFAULT_SWAP_AMOUNT_USDC),
                expected_sentinel=SWAP_SMOKE_SENTINEL,
                error_message=(
                    f"Swap smoke response did not include {SWAP_SMOKE_SENTINEL}"
                ),
                conversation_prefix="sdk-smoke-swap",
                chain_type=chain_type,
                step_payload={
                    "amount_usdc": _money_string(DEFAULT_SWAP_AMOUNT_USDC),
                    "input_mint": SOLANA_USDC_MINT,
                    "output_mint": WRAPPED_SOL_MINT,
                },
                failure_response_terms=WRITE_SMOKE_FAILURE_TERMS,
            )
        )

    if include_trigger:
        steps.append(
            await _run_message_smoke_step(
                agent,
                step_name="trigger_limit_order",
                prompt_text=_trigger_smoke_message(
                    amount_usdc=DEFAULT_TRIGGER_AMOUNT_USDC
                ),
                expected_sentinel=TRIGGER_SMOKE_SENTINEL,
                error_message=(
                    f"Trigger smoke response did not include {TRIGGER_SMOKE_SENTINEL}"
                ),
                conversation_prefix="sdk-smoke-trigger",
                chain_type=chain_type,
                step_payload={
                    "amount_usdc": _money_string(DEFAULT_TRIGGER_AMOUNT_USDC),
                    "input_mint": SOLANA_USDC_MINT,
                    "output_mint": WRAPPED_SOL_MINT,
                },
                failure_response_terms=WRITE_SMOKE_FAILURE_TERMS,
            )
        )

    if include_earn:
        steps.append(
            await _run_message_smoke_step(
                agent,
                step_name="earn_reversible",
                prompt_text=_earn_smoke_message(amount_usdc=DEFAULT_EARN_AMOUNT_USDC),
                expected_sentinel=EARN_SMOKE_SENTINEL,
                error_message=(
                    f"Earn smoke response did not include {EARN_SMOKE_SENTINEL}"
                ),
                conversation_prefix="sdk-smoke-earn",
                chain_type=chain_type,
                step_payload={
                    "amount_usdc": _money_string(DEFAULT_EARN_AMOUNT_USDC),
                    "asset": SOLANA_USDC_MINT,
                },
                required_response_terms=("deposit", "withdraw"),
                failure_response_terms=WRITE_SMOKE_FAILURE_TERMS,
            )
        )

    if include_transfer and resolved_transfer_amount is not None:
        steps.append(
            await _run_message_smoke_step(
                agent,
                step_name="transfer_usdc",
                prompt_text=_transfer_smoke_message(
                    recipient=resolved_transfer_recipient,
                    amount_usdc=resolved_transfer_amount,
                ),
                expected_sentinel=TRANSFER_SMOKE_SENTINEL,
                error_message=(
                    f"Transfer smoke response did not include {TRANSFER_SMOKE_SENTINEL}"
                ),
                conversation_prefix="sdk-smoke-transfer",
                chain_type=chain_type,
                step_payload={
                    "recipient": resolved_transfer_recipient,
                    "amount_usdc": _money_string(resolved_transfer_amount),
                    "mint": SOLANA_USDC_MINT,
                },
            )
        )

    if include_rotate:
        rotated_wallet = await agent.rotate_wallet(
            privy_user_id=privy_user_id,
            chain_type=chain_type,
        )
        rotated_wallet_id = _wallet_id_from_payload(rotated_wallet)
        rotated_wallet_address = _wallet_address_from_payload(rotated_wallet)
        if not rotated_wallet_id or not rotated_wallet_address:
            raise ValueError(
                "Wallet rotation smoke check returned an incomplete wallet"
            )
        wallet_id = rotated_wallet_id
        wallet_address = rotated_wallet_address
        steps.append(
            {
                "name": "rotate_wallet",
                "status": "passed",
                "wallet_id": wallet_id,
                "address": wallet_address,
            }
        )

    if include_export:
        private_key = await agent.export_wallet_private_key(
            wallet_id=wallet_id or None,
            privy_user_id=privy_user_id,
            chain_type=chain_type,
        )
        normalized_private_key = str(private_key or "").strip()
        if not normalized_private_key:
            raise ValueError("Wallet export smoke check returned an empty private key")
        steps.append(
            {
                "name": "export_wallet_private_key",
                "status": "passed",
                "wallet_id": wallet_id,
                "private_key_redacted": True,
                "private_key_length": len(normalized_private_key),
            }
        )

    return {
        "ok": True,
        "preview_only": False,
        "privy_user_id": privy_user_id,
        "wallet": {
            "wallet_id": wallet_id,
            "address": wallet_address,
            "chain_type": chain_type,
        },
        "transfer": transfer_payload,
        "estimate": smoke_preview.get("estimate") or {},
        "coverage": {
            "includes_search": include_search,
            "includes_rotate": include_rotate,
            "includes_export": include_export,
            "includes_priority": include_priority,
            "includes_priority_chat": include_priority,
            "includes_memory_work": include_memory,
            "includes_memory_project": include_memory,
            "includes_priority_memory_work": include_memory and include_priority,
            "includes_priority_memory_project": include_memory and include_priority,
            "includes_jupiter_quote": include_jupiter,
            "includes_birdeye_read": include_birdeye,
            "includes_swap": include_swap,
            "includes_trigger": include_trigger,
            "includes_earn": include_earn,
            "includes_technical_analysis": include_technical_analysis,
            "includes_token_math": include_token_math,
            "includes_transfer": include_transfer,
        },
        "steps": steps,
    }

"""Live smoke helpers for the public Solana Agent SDK."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Mapping
from uuid import uuid4

import tiktoken

from solana_agent.factories.agent_factory import DEFAULT_AGI_STATELESS_MODEL


SMOKE_MESSAGE = "Reply with exactly SDK_SMOKE_OK"
PRIORITY_SMOKE_MESSAGE = "Reply with exactly SDK_SMOKE_PRIORITY_OK"
SEARCH_SMOKE_MESSAGE = (
    "Search if needed, then reply with the current year followed by SDK_SMOKE_SEARCH_OK"
)
JUPITER_SMOKE_SENTINEL = "SDK_SMOKE_JUPITER_OK"
KAMINO_SMOKE_SENTINEL = "SDK_SMOKE_KAMINO_OK"
BIRDEYE_SMOKE_SENTINEL = "SDK_SMOKE_BIRDEYE_OK"
TRANSFER_SMOKE_SENTINEL = "SDK_SMOKE_TRANSFER_OK"
DEFAULT_TOKENIZER_MODEL = "gpt-oss-120b"
SMOKE_OUTPUT_TOKEN_BUDGET = 48
SEARCH_OUTPUT_TOKEN_BUDGET = 96
TOOL_OUTPUT_TOKEN_BUDGET = 128
MINIMUM_SUGGESTED_FUNDING_USDC = Decimal("1.00")
DEFAULT_SMOKE_FUNDING_BUFFER_USDC = Decimal("0.50")
DESTRUCTIVE_STEP_BUFFER_USDC = Decimal("0.25")
READ_ONLY_TOOL_BUFFER_USDC = Decimal("0.25")
DEFAULT_TRANSFER_AMOUNT_USDC = Decimal("0.10")
SOLANA_USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
WRAPPED_SOL_MINT = "So11111111111111111111111111111111111111112"
DEFAULT_JUPITER_QUOTE_AMOUNT = 1_000_000


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


def _kamino_smoke_message() -> str:
    return (
        "Use the privy_kamino tool with action list_vaults to verify Kamino read access. "
        "Set kvault, market, reserve, amount, user_pubkey, path, params_json, body_json, "
        "referrer, and referral_code to empty strings. "
        f"After the tool succeeds, reply with {KAMINO_SMOKE_SENTINEL} and the number of vaults returned."
    )


def _birdeye_smoke_message() -> str:
    return (
        "Use the birdeye tool to fetch the current Solana USDC price. "
        f"Set action to price and address to {SOLANA_USDC_MINT}. Use empty strings or 0 for the other fields. "
        f"After the tool succeeds, reply with {BIRDEYE_SMOKE_SENTINEL} and the observed price."
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


def build_public_sdk_smoke_estimate(
    pricing_payload: Mapping[str, Any],
    forecast_payload: Mapping[str, Any] | None,
    *,
    include_search: bool,
    include_rotate: bool,
    include_export: bool,
    include_priority: bool = False,
    include_jupiter: bool = False,
    include_kamino: bool = False,
    include_birdeye: bool = False,
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
    priority_chat_cost = Decimal("0")
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
    if include_kamino:
        tool_prompts.append(_kamino_smoke_message())
    if include_birdeye:
        tool_prompts.append(_birdeye_smoke_message())

    for prompt_text in tool_prompts:
        tooling_chat_cost += _estimate_model_request_cost_usd(
            pricing_payload,
            prompt_text=prompt_text,
            expected_output_tokens=TOOL_OUTPUT_TOKEN_BUDGET,
        )
        tooling_buffer += READ_ONLY_TOOL_BUFFER_USDC

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
        + priority_chat_cost
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
        "Estimate covers one standard hosted chat check and all non-chat account/wallet helper calls.",
        "If search is enabled, estimate includes the fixed search surcharge and the configured provider cost ceiling.",
        "Wallet create/fetch, address lookup, summary, usage, forecast, and pricing endpoints are treated as unpriced control-plane checks.",
    ]
    if include_priority:
        assumptions.append(
            "Priority-tier validation is estimated using the base hosted model rates because the pricing surface does not expose a separate tier surcharge."
        )
    if tool_prompts:
        assumptions.append(
            "Read-only Jupiter, Kamino, and Birdeye checks are budgeted as extra hosted turns plus a safety buffer because external provider and x402 fees are not exposed by the pricing endpoint."
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
            "priority_chat_request_usd": _money_string(priority_chat_cost),
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
            "includes_jupiter_quote": include_jupiter,
            "includes_kamino_read": include_kamino,
            "includes_birdeye_read": include_birdeye,
            "includes_transfer": include_transfer,
            "excludes": [
                "Live Jupiter swap execution, Kamino write actions, and other destructive protocol actions remain outside the built-in smoke run.",
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


async def _run_message_smoke_step(
    agent: Any,
    *,
    step_name: str,
    prompt_text: str,
    expected_sentinel: str,
    error_message: str,
    conversation_prefix: str,
    chain_type: str,
    service_tier: str = "standard",
    search_enabled: bool = False,
    step_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = await agent.context(
        conversation_id=f"{conversation_prefix}-{uuid4().hex[:12]}",
        model="chat",
        memory_ttl_tier="work",
        service_tier=service_tier,
        search_enabled=search_enabled,
        chain_type=chain_type,
    )
    response = await agent.message(prompt_text, **context)
    if expected_sentinel not in str(response or ""):
        raise ValueError(error_message)

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


async def build_public_sdk_smoke_preview(
    agent: Any,
    *,
    chain_type: str = "solana",
    forecast_window_days: int = 30,
    include_search: bool = True,
    include_rotate: bool = False,
    include_export: bool = False,
    include_priority: bool = False,
    include_jupiter: bool = False,
    include_kamino: bool = False,
    include_birdeye: bool = False,
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
        include_jupiter=include_jupiter,
        include_kamino=include_kamino,
        include_birdeye=include_birdeye,
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
    include_jupiter: bool = False,
    include_kamino: bool = False,
    include_birdeye: bool = False,
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
        include_jupiter=include_jupiter,
        include_kamino=include_kamino,
        include_birdeye=include_birdeye,
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

    if include_kamino:
        steps.append(
            await _run_message_smoke_step(
                agent,
                step_name="kamino_read",
                prompt_text=_kamino_smoke_message(),
                expected_sentinel=KAMINO_SMOKE_SENTINEL,
                error_message=(
                    f"Kamino smoke response did not include {KAMINO_SMOKE_SENTINEL}"
                ),
                conversation_prefix="sdk-smoke-kamino",
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
            "includes_jupiter_quote": include_jupiter,
            "includes_kamino_read": include_kamino,
            "includes_birdeye_read": include_birdeye,
            "includes_transfer": include_transfer,
        },
        "steps": steps,
    }

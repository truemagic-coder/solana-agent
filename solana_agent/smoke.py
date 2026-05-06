"""Live smoke helpers for the public Solana Agent SDK."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Mapping
from uuid import uuid4

import tiktoken

from solana_agent.factories.agent_factory import DEFAULT_AGI_STATELESS_MODEL


SMOKE_MESSAGE = "Reply with exactly SDK_SMOKE_OK"
SEARCH_SMOKE_MESSAGE = (
    "Search if needed, then reply with the current year followed by SDK_SMOKE_SEARCH_OK"
)
DEFAULT_TOKENIZER_MODEL = "gpt-oss-120b"
SMOKE_OUTPUT_TOKEN_BUDGET = 48
SEARCH_OUTPUT_TOKEN_BUDGET = 96
MINIMUM_SUGGESTED_FUNDING_USDC = Decimal("1.00")
DEFAULT_SMOKE_FUNDING_BUFFER_USDC = Decimal("0.50")
DESTRUCTIVE_STEP_BUFFER_USDC = Decimal("0.25")


def _money_string(value: Decimal) -> str:
    quantized = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return format(quantized, "f")


def _decimal_value(raw_value: Any) -> Decimal:
    try:
        return Decimal(str(raw_value or "0"))
    except (InvalidOperation, ValueError):
        return Decimal("0")


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


def build_public_sdk_smoke_estimate(
    pricing_payload: Mapping[str, Any],
    forecast_payload: Mapping[str, Any] | None,
    *,
    include_search: bool,
    include_rotate: bool,
    include_export: bool,
) -> dict[str, Any]:
    base_chat_cost = _estimate_model_request_cost_usd(
        pricing_payload,
        prompt_text=SMOKE_MESSAGE,
        expected_output_tokens=SMOKE_OUTPUT_TOKEN_BUDGET,
    )
    search_chat_cost = Decimal("0")
    search_surcharge = Decimal("0")
    search_provider_cost_ceiling = Decimal("0")

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

    estimated_smoke_spend_ceiling = (
        base_chat_cost
        + search_chat_cost
        + search_surcharge
        + search_provider_cost_ceiling
    )
    funding_buffer = DEFAULT_SMOKE_FUNDING_BUFFER_USDC
    if include_rotate:
        funding_buffer += DESTRUCTIVE_STEP_BUFFER_USDC
    if include_export:
        funding_buffer += DESTRUCTIVE_STEP_BUFFER_USDC

    suggested_wallet_funding = max(
        MINIMUM_SUGGESTED_FUNDING_USDC,
        estimated_smoke_spend_ceiling + funding_buffer,
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
    if include_rotate or include_export:
        assumptions.append(
            "Rotate and export are included as live checks, but the hosted pricing surface does not expose a direct per-call fee for them, so the recommendation adds only a safety buffer."
        )

    return {
        "estimated_smoke_spend_ceiling_usd": _money_string(
            estimated_smoke_spend_ceiling
        ),
        "suggested_wallet_funding_usdc": _money_string(suggested_wallet_funding),
        "components": {
            "standard_chat_request_usd": _money_string(base_chat_cost),
            "search_chat_request_usd": _money_string(search_chat_cost),
            "search_surcharge_usd": _money_string(search_surcharge),
            "search_provider_cost_ceiling_usd": _money_string(
                search_provider_cost_ceiling
            ),
            "funding_buffer_usdc": _money_string(funding_buffer),
        },
        "account_forecast_context": {
            "current_month_spend_usd": current_month_spend,
            "projected_month_end_spend_usd": forecast_projection,
        },
        "coverage": {
            "includes_search": include_search,
            "includes_rotate": include_rotate,
            "includes_export": include_export,
            "excludes": [
                "Protocol tool execution is not part of the built-in smoke run because tool choice is model-driven and may trigger live market or on-chain actions.",
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


async def build_public_sdk_smoke_preview(
    agent: Any,
    *,
    chain_type: str = "solana",
    forecast_window_days: int = 30,
    include_search: bool = True,
    include_rotate: bool = False,
    include_export: bool = False,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []

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

    return {
        "ok": True,
        "preview_only": True,
        "privy_user_id": privy_user_id,
        "wallet": {
            "wallet_id": wallet_id,
            "address": resolved_wallet_address,
            "chain_type": chain_type,
        },
        "estimate": build_public_sdk_smoke_estimate(
            pricing,
            forecast,
            include_search=include_search,
            include_rotate=include_rotate,
            include_export=include_export,
        ),
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
    preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    smoke_preview = preview or await build_public_sdk_smoke_preview(
        agent,
        chain_type=chain_type,
        forecast_window_days=forecast_window_days,
        include_search=include_search,
        include_rotate=include_rotate,
        include_export=include_export,
    )
    steps = list(smoke_preview.get("steps") or [])
    privy_user_id = str(smoke_preview.get("privy_user_id") or "").strip()
    wallet_payload = dict(smoke_preview.get("wallet") or {})
    wallet_id = str(wallet_payload.get("wallet_id") or "").strip()
    wallet_address = str(wallet_payload.get("address") or "").strip()

    context = await agent.context(
        conversation_id=f"sdk-smoke-{uuid4().hex[:12]}",
        model="chat",
        memory_ttl_tier="work",
        service_tier="standard",
        search_enabled=False,
        chain_type=chain_type,
    )
    message_response = await agent.message(SMOKE_MESSAGE, **context)
    if "SDK_SMOKE_OK" not in str(message_response or ""):
        raise ValueError("Smoke chat response did not include SDK_SMOKE_OK")
    steps.append(
        {
            "name": "chat_message",
            "status": "passed",
            "response_excerpt": _excerpt(message_response),
        }
    )

    if include_search:
        search_context = await agent.context(
            conversation_id=f"sdk-smoke-search-{uuid4().hex[:12]}",
            model="chat",
            memory_ttl_tier="work",
            service_tier="standard",
            search_enabled=True,
            chain_type=chain_type,
        )
        search_response = await agent.message(SEARCH_SMOKE_MESSAGE, **search_context)
        if "SDK_SMOKE_SEARCH_OK" not in str(search_response or ""):
            raise ValueError(
                "Smoke search-enabled response did not include SDK_SMOKE_SEARCH_OK"
            )
        steps.append(
            {
                "name": "chat_message_search_enabled",
                "status": "passed",
                "response_excerpt": _excerpt(search_response),
            }
        )

    if include_rotate:
        rotated_wallet = await agent.rotate_wallet(
            privy_user_id=privy_user_id,
            chain_type=chain_type,
        )
        rotated_wallet_id = _wallet_id_from_payload(rotated_wallet)
        rotated_wallet_address = _wallet_address_from_payload(rotated_wallet)
        if not rotated_wallet_id or not rotated_wallet_address:
            raise ValueError("Wallet rotation smoke check returned an incomplete wallet")
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
        "estimate": smoke_preview.get("estimate") or {},
        "coverage": {
            "includes_search": include_search,
            "includes_rotate": include_rotate,
            "includes_export": include_export,
        },
        "steps": steps,
    }
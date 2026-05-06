from unittest.mock import AsyncMock, MagicMock

import pytest

from solana_agent.smoke import (
    build_public_sdk_smoke_estimate,
    build_public_sdk_smoke_preview,
    run_public_sdk_smoke,
)


def test_build_public_sdk_smoke_estimate_includes_search_cost_ceiling() -> None:
    estimate = build_public_sdk_smoke_estimate(
        {
            "base_rates": {
                "solana-agent-chat": {
                    "input_per_million": "5",
                    "output_per_million": "15",
                }
            },
            "search_add_on": {
                "surcharge_per_request_usd": "0.25",
                "search_max_provider_cost_usd": "0.10",
            },
        },
        {
            "current_month": {"spend": "0.03"},
            "projected_month_end": {"spend": "0.30"},
        },
        include_search=True,
        include_rotate=False,
        include_export=False,
    )

    assert estimate["components"]["search_surcharge_usd"] == "0.25"
    assert estimate["components"]["search_provider_cost_ceiling_usd"] == "0.10"
    assert estimate["suggested_wallet_funding_usdc"] == "1.00"
    assert estimate["account_forecast_context"]["projected_month_end_spend_usd"] == (
        "0.30"
    )


@pytest.mark.asyncio
async def test_build_public_sdk_smoke_preview_creates_user_when_missing() -> None:
    agent = MagicMock()
    agent._configured_privy_user_id.side_effect = ValueError("missing")
    agent.create_privy_user = AsyncMock(
        return_value={"privy_user_id": "did:privy:new-user", "created": True}
    )
    agent.create_wallet = AsyncMock(
        return_value={
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
            "created": True,
        }
    )
    agent.get_wallet_address = AsyncMock(return_value="WalletPubkey123")
    agent.get_account_summary = AsyncMock(
        return_value={"requests": {"lifetime": 1}, "spend": {"month": "0.01"}}
    )
    agent.get_usage_report = AsyncMock(return_value={"buckets": []})
    agent.get_usage_forecast = AsyncMock(
        return_value={
            "current_month": {"spend": "0.01"},
            "projected_month_end": {"spend": "0.02"},
        }
    )
    agent.get_pricing_info = AsyncMock(
        return_value={
            "base_rates": {
                "solana-agent-chat": {
                    "input_per_million": "5",
                    "output_per_million": "15",
                }
            },
            "search_add_on": {
                "surcharge_per_request_usd": "0.25",
                "search_max_provider_cost_usd": "0.10",
            },
        }
    )

    preview = await build_public_sdk_smoke_preview(agent, include_export=True)

    assert preview["privy_user_id"] == "did:privy:new-user"
    assert preview["wallet"]["wallet_id"] == "wallet-123"
    assert preview["estimate"]["coverage"]["includes_export"] is True
    assert [step["name"] for step in preview["steps"]] == [
        "resolve_privy_user",
        "create_or_fetch_wallet",
        "get_wallet_address",
        "account_summary",
        "account_usage",
        "account_forecast",
        "account_pricing",
    ]


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_executes_chat_search_rotate_and_export() -> None:
    agent = MagicMock()
    agent.context = AsyncMock(
        side_effect=[
            {"conversation_id": "conv-1"},
            {"conversation_id": "conv-2", "search_enabled": True},
        ]
    )
    agent.message = AsyncMock(
        side_effect=[
            "SDK_SMOKE_OK",
            "2026 SDK_SMOKE_SEARCH_OK",
        ]
    )
    agent.rotate_wallet = AsyncMock(
        return_value={
            "wallet_id": "wallet-456",
            "address": "WalletPubkey456",
        }
    )
    agent.export_wallet_private_key = AsyncMock(return_value="base58-private-key")

    result = await run_public_sdk_smoke(
        agent,
        include_search=True,
        include_rotate=True,
        include_export=True,
        preview={
            "privy_user_id": "did:privy:user123",
            "wallet": {
                "wallet_id": "wallet-123",
                "address": "WalletPubkey123",
                "chain_type": "solana",
            },
            "estimate": {"suggested_wallet_funding_usdc": "1.00"},
            "steps": [{"name": "account_pricing", "status": "passed"}],
        },
    )

    assert result["ok"] is True
    assert result["wallet"]["wallet_id"] == "wallet-456"
    assert [step["name"] for step in result["steps"]][-4:] == [
        "chat_message",
        "chat_message_search_enabled",
        "rotate_wallet",
        "export_wallet_private_key",
    ]
    assert result["steps"][-1]["private_key_redacted"] is True
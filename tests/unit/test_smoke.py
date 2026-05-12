from unittest.mock import AsyncMock, MagicMock

import pytest

from solana_agent.factories.agent_factory import DEFAULT_AGI_MEMORY_MODEL
from solana_agent.smoke import (
    DEFAULT_TRIGGER_AMOUNT_USDC,
    build_public_sdk_smoke_estimate,
    build_public_sdk_smoke_preview,
    run_public_sdk_memory_benchmark,
    run_public_sdk_smoke,
    _trigger_smoke_message,
)


def _preview_agent() -> MagicMock:
    agent = MagicMock()
    agent._configured_privy_user_id.return_value = "did:privy:existing-user"
    agent.create_wallet = AsyncMock(
        return_value={
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
            "created": False,
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
                },
                "solana-agent-memory": {
                    "input_per_million": "10",
                    "output_per_million": "60",
                    "project_memory_surcharge_usd": "0.50",
                },
            },
            "search_add_on": {
                "surcharge_per_request_usd": "0.25",
                "search_max_provider_cost_usd": "0.10",
            },
        }
    )
    return agent


def _smoke_preview_payload() -> dict[str, object]:
    return {
        "privy_user_id": "did:privy:user123",
        "wallet": {
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
            "chain_type": "solana",
        },
        "coverage": {
            "includes_search": True,
            "includes_rotate": False,
            "includes_export": False,
        },
        "estimate": {"suggested_wallet_funding_usdc": "1.00"},
        "steps": [],
    }


def test_trigger_smoke_uses_far_from_market_cancelable_order() -> None:
    message = _trigger_smoke_message(amount_usdc=DEFAULT_TRIGGER_AMOUNT_USDC)

    assert "price_change_percentage 100" in message
    assert "should stay open for cancellation" in message


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


def test_build_public_sdk_smoke_estimate_adds_rotate_buffer() -> None:
    estimate = build_public_sdk_smoke_estimate(
        {
            "base_rates": {
                "solana-agent-chat": {
                    "input_per_million": "5",
                    "output_per_million": "15",
                }
            }
        },
        None,
        include_search=False,
        include_rotate=True,
        include_export=False,
    )

    assert estimate["components"]["funding_buffer_usdc"] == "0.75"
    assert estimate["account_forecast_context"]["current_month_spend_usd"] is None


def test_build_public_sdk_smoke_estimate_includes_big_smoke_components() -> None:
    estimate = build_public_sdk_smoke_estimate(
        {
            "base_rates": {
                "solana-agent-chat": {
                    "input_per_million": "5000",
                    "output_per_million": "15000",
                },
                "solana-agent-memory": {
                    "input_per_million": "10000",
                    "output_per_million": "60000",
                    "project_memory_surcharge_usd": "0.50",
                },
            }
        },
        None,
        include_search=False,
        include_rotate=False,
        include_export=False,
        include_priority=True,
        include_memory=True,
        include_jupiter=True,
        include_birdeye=True,
        include_swap=True,
        include_trigger=True,
        include_earn=True,
        include_technical_analysis=True,
        include_token_math=True,
        include_transfer=True,
        transfer_amount_usdc="0.10",
    )

    assert estimate["components"]["priority_chat_request_usd"] != "0.00"
    assert estimate["components"]["memory_work_request_usd"] != "0.00"
    assert estimate["components"]["memory_project_request_usd"] != "0.00"
    assert estimate["components"]["priority_memory_work_request_usd"] != "0.00"
    assert estimate["components"]["priority_memory_project_request_usd"] != "0.00"
    assert estimate["components"]["tooling_chat_requests_usd"] != "0.00"
    assert estimate["components"]["protocol_tooling_buffer_usdc"] == "1.75"
    assert estimate["components"]["transfer_amount_usdc"] == "0.10"
    assert estimate["coverage"]["includes_priority"] is True
    assert estimate["coverage"]["includes_memory_work"] is True
    assert estimate["coverage"]["includes_memory_project"] is True
    assert estimate["coverage"]["includes_priority_memory_work"] is True
    assert estimate["coverage"]["includes_priority_memory_project"] is True
    assert estimate["coverage"]["includes_swap"] is True
    assert estimate["coverage"]["includes_trigger"] is True
    assert estimate["coverage"]["includes_earn"] is True
    assert estimate["coverage"]["includes_technical_analysis"] is True
    assert estimate["coverage"]["includes_token_math"] is True
    assert estimate["coverage"]["includes_transfer"] is True


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
                },
                "solana-agent-memory": {
                    "input_per_million": "10",
                    "output_per_million": "60",
                    "project_memory_surcharge_usd": "0.50",
                },
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
    assert preview["coverage"]["includes_export"] is True
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
async def test_build_public_sdk_smoke_preview_uses_existing_user() -> None:
    agent = _preview_agent()

    preview = await build_public_sdk_smoke_preview(agent)

    assert preview["privy_user_id"] == "did:privy:existing-user"
    agent.create_privy_user.assert_not_called()
    assert preview["steps"][0]["created"] is False


@pytest.mark.asyncio
async def test_build_public_sdk_smoke_preview_requires_created_user_id() -> None:
    agent = _preview_agent()
    agent._configured_privy_user_id.side_effect = ValueError("missing")
    agent.create_privy_user = AsyncMock(return_value={"created": True})

    with pytest.raises(ValueError, match="privy_user_id"):
        await build_public_sdk_smoke_preview(agent)


@pytest.mark.asyncio
async def test_build_public_sdk_smoke_preview_requires_wallet_id() -> None:
    agent = _preview_agent()
    agent.create_wallet = AsyncMock(return_value={"address": "WalletPubkey123"})

    with pytest.raises(ValueError, match="wallet_id"):
        await build_public_sdk_smoke_preview(agent)


@pytest.mark.asyncio
async def test_build_public_sdk_smoke_preview_requires_wallet_address() -> None:
    agent = _preview_agent()
    agent.get_wallet_address = AsyncMock(return_value=" ")

    with pytest.raises(ValueError, match="returned no address"):
        await build_public_sdk_smoke_preview(agent)


@pytest.mark.asyncio
async def test_build_public_sdk_smoke_preview_requires_transfer_recipient() -> None:
    agent = _preview_agent()

    with pytest.raises(ValueError, match="transfer_recipient"):
        await build_public_sdk_smoke_preview(agent, include_transfer=True)


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


@pytest.mark.asyncio
async def test_run_public_sdk_memory_benchmark_reports_cold_and_warm_timings() -> None:
    rebuilt_client = object()
    llm_provider = MagicMock(
        auth_mode="hosted_managed",
        base_url="http://127.0.0.1:8000/v1",
        private_key="merchant-private-key",
        api_key="x402",
        _is_openai_endpoint=False,
    )
    llm_provider._create_client.return_value = rebuilt_client

    agent = MagicMock()
    agent.query_service = MagicMock(agent_service=MagicMock(llm_provider=llm_provider))
    agent.context = AsyncMock(return_value={"conversation_id": "conv-memory"})
    agent.message = AsyncMock(
        side_effect=[
            "SDK_SMOKE_MEMORY_STORED sdk-memory-project-30d",
            "SDK_SMOKE_MEMORY_PROJECT_OK sdk-memory-project-30d",
            "SDK_SMOKE_MEMORY_PROJECT_OK sdk-memory-project-30d",
            "SDK_SMOKE_MEMORY_PROJECT_OK sdk-memory-project-30d",
        ]
    )
    agent.export_wallet_private_key = AsyncMock(return_value="base58-private-key")

    result = await run_public_sdk_memory_benchmark(
        agent,
        memory_ttl_tier="project",
        warm_recall_count=2,
        preview=_smoke_preview_payload(),
    )

    assert result["ok"] is True
    assert result["memory_ttl_tier"] == "project"
    assert result["service_tier"] == "standard"
    assert result["steps"][0]["name"] == "bootstrap_local_x402_signer"
    latency = result["latency_ms"]
    assert latency["context_build_ms"] >= 0
    assert latency["store_ms"] >= 0
    assert latency["cold_recall_ms"] >= 0
    assert latency["warm_recall"]["count"] == 2
    assert len(latency["warm_recall"]["samples_ms"]) == 2
    assert result["step"]["remember_token"] == "sdk-memory-project-30d"
    agent.export_wallet_private_key.assert_awaited_once_with(
        wallet_id="wallet-123",
        privy_user_id="did:privy:user123",
        chain_type="solana",
    )


@pytest.mark.asyncio
async def test_run_public_sdk_memory_benchmark_validates_memory_tier() -> None:
    with pytest.raises(ValueError, match="memory_ttl_tier"):
        await run_public_sdk_memory_benchmark(
            MagicMock(),
            memory_ttl_tier="archive",
            preview=_smoke_preview_payload(),
        )


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_executes_priority_tooling_and_transfer() -> None:
    agent = MagicMock()
    agent.context = AsyncMock(
        side_effect=[
            {"conversation_id": "conv-1"},
            {"conversation_id": "conv-2", "service_tier": "priority"},
            {"conversation_id": "conv-3", "model": DEFAULT_AGI_MEMORY_MODEL},
            {
                "conversation_id": "conv-4",
                "model": DEFAULT_AGI_MEMORY_MODEL,
                "service_tier": "priority",
            },
            {"conversation_id": "conv-5", "model": DEFAULT_AGI_MEMORY_MODEL},
            {
                "conversation_id": "conv-6",
                "model": DEFAULT_AGI_MEMORY_MODEL,
                "service_tier": "priority",
            },
            {"conversation_id": "conv-7", "search_enabled": True},
            {"conversation_id": "conv-8"},
            {"conversation_id": "conv-9"},
            {"conversation_id": "conv-10"},
            {"conversation_id": "conv-11"},
            {"conversation_id": "conv-12"},
            {"conversation_id": "conv-13"},
            {"conversation_id": "conv-14"},
            {"conversation_id": "conv-15"},
        ]
    )
    agent.message = AsyncMock(
        side_effect=[
            "SDK_SMOKE_OK",
            "SDK_SMOKE_PRIORITY_OK",
            "SDK_SMOKE_MEMORY_STORED sdk-memory-work-7d",
            "SDK_SMOKE_MEMORY_WORK_OK sdk-memory-work-7d",
            "SDK_SMOKE_MEMORY_STORED sdk-memory-work-7d-priority",
            "SDK_SMOKE_MEMORY_PRIORITY_WORK_OK sdk-memory-work-7d-priority",
            "SDK_SMOKE_MEMORY_STORED sdk-memory-project-30d",
            "SDK_SMOKE_MEMORY_PROJECT_OK sdk-memory-project-30d",
            "SDK_SMOKE_MEMORY_STORED sdk-memory-project-30d-priority",
            "SDK_SMOKE_MEMORY_PRIORITY_PROJECT_OK sdk-memory-project-30d-priority",
            "2026 SDK_SMOKE_SEARCH_OK",
            "SDK_SMOKE_JUPITER_OK out_amount=123",
            "SDK_SMOKE_BIRDEYE_OK price=1.00",
            "SDK_SMOKE_TOKEN_MATH_OK 100000 0.1",
            "SDK_SMOKE_TECHNICAL_ANALYSIS_OK rsi=52.1",
            "SDK_SMOKE_SWAP_OK sig=swap123",
            "SDK_SMOKE_TRIGGER_OK order=trigger123",
            "SDK_SMOKE_EARN_OK deposit=earn123 withdraw=earn124",
            "SDK_SMOKE_TRANSFER_OK sig=abc123",
        ]
    )

    result = await run_public_sdk_smoke(
        agent,
        include_search=True,
        include_priority=True,
        include_memory=True,
        include_jupiter=True,
        include_birdeye=True,
        include_swap=True,
        include_trigger=True,
        include_earn=True,
        include_technical_analysis=True,
        include_token_math=True,
        include_transfer=True,
        transfer_recipient="RecipientPubkey123",
        transfer_amount_usdc="0.10",
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
    assert result["coverage"]["includes_priority"] is True
    assert result["coverage"]["includes_memory_work"] is True
    assert result["coverage"]["includes_memory_project"] is True
    assert result["coverage"]["includes_priority_memory_work"] is True
    assert result["coverage"]["includes_priority_memory_project"] is True
    assert result["coverage"]["includes_jupiter_quote"] is True
    assert result["coverage"]["includes_birdeye_read"] is True
    assert result["coverage"]["includes_swap"] is True
    assert result["coverage"]["includes_trigger"] is True
    assert result["coverage"]["includes_earn"] is True
    assert result["coverage"]["includes_technical_analysis"] is True
    assert result["coverage"]["includes_token_math"] is True
    assert result["coverage"]["includes_transfer"] is True
    assert result["transfer"] == {
        "recipient": "RecipientPubkey123",
        "amount_usdc": "0.10",
        "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    }
    assert [step["name"] for step in result["steps"]][-15:] == [
        "chat_message",
        "chat_message_priority_tier",
        "memory_recall_work_7d",
        "memory_recall_work_7d_priority_tier",
        "memory_recall_project_30d",
        "memory_recall_project_30d_priority_tier",
        "chat_message_search_enabled",
        "jupiter_swap_quote",
        "birdeye_read",
        "token_math",
        "technical_analysis",
        "swap_live",
        "trigger_limit_order",
        "earn_reversible",
        "transfer_usdc",
    ]
    assert agent.context.await_args_list[1].kwargs["service_tier"] == "priority"
    assert agent.context.await_args_list[2].kwargs["model"] == "memory"
    assert agent.context.await_args_list[2].kwargs["memory_ttl_tier"] == "work"
    assert agent.context.await_args_list[3].kwargs["service_tier"] == "priority"
    assert agent.context.await_args_list[4].kwargs["memory_ttl_tier"] == "project"
    assert agent.context.await_args_list[5].kwargs["service_tier"] == "priority"


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_requires_memory_recall_token() -> None:
    agent = MagicMock()
    agent.context = AsyncMock(
        side_effect=[
            {"conversation_id": "conv-1"},
            {"conversation_id": "conv-2", "model": DEFAULT_AGI_MEMORY_MODEL},
            {"conversation_id": "conv-3", "model": DEFAULT_AGI_MEMORY_MODEL},
        ]
    )
    agent.message = AsyncMock(
        side_effect=[
            "SDK_SMOKE_OK",
            "SDK_SMOKE_MEMORY_STORED sdk-memory-work-7d",
            "SDK_SMOKE_MEMORY_WORK_OK wrong-token",
            "SDK_SMOKE_MEMORY_STORED sdk-memory-work-7d",
            "SDK_SMOKE_MEMORY_WORK_OK still-wrong-token",
        ]
    )

    with pytest.raises(ValueError, match="SDK_SMOKE_MEMORY_WORK_OK sdk-memory-work-7d"):
        await run_public_sdk_smoke(
            agent,
            include_search=False,
            include_memory=True,
            preview=_smoke_preview_payload(),
        )


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_requires_token_math_required_terms() -> None:
    agent = MagicMock()
    agent.context = AsyncMock(
        side_effect=[
            {"conversation_id": "conv-1"},
            {"conversation_id": "conv-2"},
            {"conversation_id": "conv-3"},
        ]
    )
    agent.message = AsyncMock(
        side_effect=[
            "SDK_SMOKE_OK",
            "SDK_SMOKE_TOKEN_MATH_OK but missing the numeric round-trip",
            "SDK_SMOKE_TOKEN_MATH_OK but still missing the numeric round-trip",
        ]
    )

    with pytest.raises(ValueError, match="100000"):
        await run_public_sdk_smoke(
            agent,
            include_search=False,
            include_token_math=True,
            preview=_smoke_preview_payload(),
        )


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_rejects_write_sentinel_with_failure_text() -> None:
    agent = MagicMock()
    agent.context = AsyncMock(
        side_effect=[
            {"conversation_id": "conv-1"},
            {"conversation_id": "conv-2"},
        ]
    )
    agent.message = AsyncMock(
        side_effect=[
            "SDK_SMOKE_OK",
            "SDK_SMOKE_EARN_OK Deposit failed, so I could not complete the full reversible write path.",
        ]
    )

    with pytest.raises(ValueError, match="failure text"):
        await run_public_sdk_smoke(
            agent,
            include_search=False,
            include_earn=True,
            preview=_smoke_preview_payload(),
        )


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_requires_transfer_recipient() -> None:
    agent = MagicMock()

    with pytest.raises(ValueError, match="transfer_recipient"):
        await run_public_sdk_smoke(
            agent,
            include_search=False,
            include_transfer=True,
            preview=_smoke_preview_payload(),
        )


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_requires_chat_sentinel() -> None:
    agent = MagicMock()
    agent.context = AsyncMock(
        side_effect=[
            {"conversation_id": "conv-1"},
            {"conversation_id": "conv-2"},
        ]
    )
    agent.message = AsyncMock(side_effect=["wrong", "still wrong"])

    with pytest.raises(ValueError, match="SDK_SMOKE_OK"):
        await run_public_sdk_smoke(
            agent,
            include_search=False,
            preview=_smoke_preview_payload(),
        )


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_surfaces_hosted_payment_error() -> None:
    agent = MagicMock()
    agent.context = AsyncMock(return_value={"conversation_id": "conv-1"})
    agent.message = AsyncMock(side_effect=RuntimeError("Error code: 402 - {}"))

    with pytest.raises(ValueError, match="hosted payment error 402"):
        await run_public_sdk_smoke(
            agent,
            include_search=False,
            preview=_smoke_preview_payload(),
        )

    agent.message.assert_awaited_once_with(
        "Reply with exactly SDK_SMOKE_OK",
        conversation_id="conv-1",
        _raise_stream_errors=True,
        max_tool_iterations=8,
        request_timeout_seconds=120,
    )


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_bootstraps_local_hosted_x402_signer() -> None:
    rebuilt_client = object()
    llm_provider = MagicMock(
        auth_mode="hosted_managed",
        base_url="http://127.0.0.1:8000/v1",
        private_key="merchant-private-key",
        api_key="x402",
        _is_openai_endpoint=False,
    )
    llm_provider._create_client.return_value = rebuilt_client

    agent = MagicMock()
    agent.query_service = MagicMock(agent_service=MagicMock(llm_provider=llm_provider))
    agent.context = AsyncMock(return_value={"conversation_id": "conv-1"})
    agent.message = AsyncMock(return_value="SDK_SMOKE_OK")
    agent.export_wallet_private_key = AsyncMock(return_value="base58-private-key")

    result = await run_public_sdk_smoke(
        agent,
        include_search=False,
        preview=_smoke_preview_payload(),
    )

    assert result["ok"] is True
    assert [step["name"] for step in result["steps"]][-2:] == [
        "bootstrap_local_x402_signer",
        "chat_message",
    ]
    agent.export_wallet_private_key.assert_awaited_once_with(
        wallet_id="wallet-123",
        privy_user_id="did:privy:user123",
        chain_type="solana",
    )
    assert llm_provider.private_key == "base58-private-key"
    llm_provider._create_client.assert_called_once_with(
        api_key="x402",
        base_url="http://127.0.0.1:8000/v1",
    )
    assert llm_provider.client is rebuilt_client


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_retries_local_hosted_x402_bootstrap_without_wallet_id() -> (
    None
):
    rebuilt_client = object()
    llm_provider = MagicMock(
        auth_mode="hosted_managed",
        base_url="http://127.0.0.1:8000/v1",
        private_key="merchant-private-key",
        api_key="x402",
        _is_openai_endpoint=False,
    )
    llm_provider._create_client.return_value = rebuilt_client

    agent = MagicMock()
    agent.query_service = MagicMock(agent_service=MagicMock(llm_provider=llm_provider))
    agent.context = AsyncMock(return_value={"conversation_id": "conv-1"})
    agent.message = AsyncMock(return_value="SDK_SMOKE_OK")
    agent.export_wallet_private_key = AsyncMock(
        side_effect=[
            RuntimeError("wallet_id does not belong to privy_user_id"),
            "base58-private-key",
        ]
    )

    result = await run_public_sdk_smoke(
        agent,
        include_search=False,
        preview=_smoke_preview_payload(),
    )

    assert result["ok"] is True
    assert agent.export_wallet_private_key.await_args_list[0].kwargs == {
        "wallet_id": "wallet-123",
        "privy_user_id": "did:privy:user123",
        "chain_type": "solana",
    }
    assert agent.export_wallet_private_key.await_args_list[1].kwargs == {
        "privy_user_id": "did:privy:user123",
        "chain_type": "solana",
    }


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_retry_uses_provider_export_without_saved_wallet_id() -> (
    None
):
    rebuilt_client = object()
    llm_provider = MagicMock(
        auth_mode="hosted_managed",
        base_url="http://127.0.0.1:8000/v1",
        private_key="merchant-private-key",
        api_key="x402",
        _is_openai_endpoint=False,
    )
    llm_provider._create_client.return_value = rebuilt_client

    provider_export = AsyncMock(
        return_value={
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
            "private_key": "base58-private-key",
        }
    )

    agent = MagicMock()
    agent.query_service = MagicMock(agent_service=MagicMock(llm_provider=llm_provider))
    agent.context = AsyncMock(return_value={"conversation_id": "conv-1"})
    agent.message = AsyncMock(return_value="SDK_SMOKE_OK")
    agent.export_wallet_private_key = AsyncMock(
        side_effect=RuntimeError("wallet_id does not belong to privy_user_id")
    )
    agent._get_provider_method.return_value = provider_export

    result = await run_public_sdk_smoke(
        agent,
        include_search=False,
        preview=_smoke_preview_payload(),
    )

    assert result["ok"] is True
    agent.export_wallet_private_key.assert_awaited_once_with(
        wallet_id="wallet-123",
        privy_user_id="did:privy:user123",
        chain_type="solana",
    )
    provider_export.assert_awaited_once_with(
        privy_user_id="did:privy:user123",
        chain_type="solana",
    )
    assert llm_provider.private_key == "base58-private-key"
    assert llm_provider.client is rebuilt_client


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_requires_search_sentinel() -> None:
    agent = MagicMock()
    agent.context = AsyncMock(
        side_effect=[
            {"conversation_id": "conv-1"},
            {"conversation_id": "conv-2", "search_enabled": True},
            {"conversation_id": "conv-3", "search_enabled": True},
        ]
    )
    agent.message = AsyncMock(side_effect=["SDK_SMOKE_OK", "wrong", "still wrong"])

    with pytest.raises(ValueError, match="SDK_SMOKE_SEARCH_OK"):
        await run_public_sdk_smoke(
            agent,
            include_search=True,
            preview=_smoke_preview_payload(),
        )


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_retries_format_miss_once() -> None:
    agent = MagicMock()
    agent.context = AsyncMock(
        side_effect=[
            {"conversation_id": "conv-1"},
            {"conversation_id": "conv-2"},
        ]
    )
    agent.message = AsyncMock(side_effect=["almost there", "SDK_SMOKE_OK"])

    result = await run_public_sdk_smoke(
        agent,
        include_search=False,
        preview=_smoke_preview_payload(),
    )

    assert result["ok"] is True
    assert agent.message.await_count == 2


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_requires_complete_rotated_wallet() -> None:
    agent = MagicMock()
    agent.context = AsyncMock(return_value={"conversation_id": "conv-1"})
    agent.message = AsyncMock(return_value="SDK_SMOKE_OK")
    agent.rotate_wallet = AsyncMock(return_value={"wallet_id": "wallet-456"})

    with pytest.raises(ValueError, match="incomplete wallet"):
        await run_public_sdk_smoke(
            agent,
            include_search=False,
            include_rotate=True,
            preview=_smoke_preview_payload(),
        )


@pytest.mark.asyncio
async def test_run_public_sdk_smoke_requires_exported_private_key() -> None:
    agent = MagicMock()
    agent.context = AsyncMock(return_value={"conversation_id": "conv-1"})
    agent.message = AsyncMock(return_value="SDK_SMOKE_OK")
    agent.export_wallet_private_key = AsyncMock(return_value=" ")

    with pytest.raises(ValueError, match="empty private key"):
        await run_public_sdk_smoke(
            agent,
            include_search=False,
            include_export=True,
            preview=_smoke_preview_payload(),
        )

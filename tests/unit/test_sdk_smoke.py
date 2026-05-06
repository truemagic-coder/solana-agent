from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from solana_agent.client.solana_agent import SolanaAgent
from solana_agent.factories.agent_factory import (
    DEFAULT_AGI_BASE_URL,
    DEFAULT_AGI_STATELESS_MODEL,
    SolanaAgentFactory,
    UNSUPPORTED_PUBLIC_TOOL_ERROR,
)
from solana_agent.plugins.registry import ToolRegistry
from solana_agent.services.query import QueryService
from solana_agent.tools.mcp import get_plugin as get_mcp_plugin


pytestmark = pytest.mark.smoke


def test_public_factory_smoke_builds_hosted_managed_runtime_and_mcp_surface() -> None:
    config = {
        "ai": {
            "instructions": "Use hosted Solana Agent APIs.",
            "privy_user_id": "did:privy:smoke-user",
            "model": "chat",
            "stateless_model": DEFAULT_AGI_STATELESS_MODEL,
            "base_url": DEFAULT_AGI_BASE_URL,
            "api_key": "x402",
            "tools": ["mcp"],
            "x402_preferred_asset": "USDC",
            "max_output_tokens": 4096,
            "context_window_tokens": 128000,
        },
        "tools": {
            "mcp": {
                "url": "https://mcp.example.test/mcp",
                "headers": {"Authorization": "Bearer test-token"},
                "llm_provider": "openai",
                "api_key": "test-openai-key",
                "llm_model": "gpt-4.1-mini",
            }
        },
    }

    with (
        patch("solana_agent.factories.agent_factory.OpenAIAdapter") as adapter_class,
        patch("solana_agent.factories.agent_factory.PluginManager") as manager_class,
    ):
        adapter_class.return_value = MagicMock()
        manager_class.return_value = MagicMock(
            load_plugins=MagicMock(return_value=["mcp"])
        )

        service = SolanaAgentFactory.create_from_config(config)

    assert isinstance(service, QueryService)
    adapter_class.assert_called_once_with(
        api_key="x402",
        model=DEFAULT_AGI_STATELESS_MODEL,
        base_url=DEFAULT_AGI_BASE_URL,
        context_window_tokens=128000,
        max_output_tokens=4096,
        auth_mode="hosted_managed",
        x402_preferred_asset="USDC",
        privy_user_id="did:privy:smoke-user",
    )
    manager_class.return_value.load_plugins.assert_called_once_with()


def test_public_factory_smoke_rejects_removed_x402_request_plugin() -> None:
    with pytest.raises(ValueError, match=UNSUPPORTED_PUBLIC_TOOL_ERROR):
        SolanaAgentFactory.create_from_config(
            {
                "ai": {
                    "instructions": "Use hosted Solana Agent APIs.",
                    "privy_user_id": "did:privy:smoke-user",
                    "tools": ["x402_request"],
                }
            }
        )


@pytest.mark.asyncio
async def test_public_client_smoke_covers_message_context_wallet_and_account_helpers() -> (
    None
):
    async def process_chunks(*args, **kwargs):
        del args, kwargs
        yield "hello "
        yield "world"

    provider = AsyncMock()
    provider.create_privy_user = AsyncMock(
        return_value={"privy_user_id": "did:privy:new-user", "created": True}
    )
    provider.create_wallet = AsyncMock(
        return_value={
            "privy_user_id": "did:privy:smoke-user",
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
        }
    )
    provider.rotate_wallet = AsyncMock(
        return_value={
            "privy_user_id": "did:privy:smoke-user",
            "wallet_id": "wallet-456",
            "address": "WalletPubkey456",
            "old_wallets": [{"wallet_id": "wallet-123"}],
        }
    )
    provider.export_wallet_private_key = AsyncMock(
        return_value={"private_key": "base58-private-key"}
    )
    provider.get_wallet_address = AsyncMock(
        return_value={"wallet_id": "wallet-456", "address": "WalletPubkey456"}
    )
    provider.get_account_summary = AsyncMock(return_value={"spend": {"month": 1}})
    provider.get_usage_report = AsyncMock(return_value={"buckets": []})
    provider.get_usage_forecast = AsyncMock(return_value={"window_days": 30})
    provider.get_pricing_info = AsyncMock(return_value={"models": []})

    query_service = MagicMock()
    query_service.process = MagicMock(side_effect=process_chunks)
    query_service.agent_service = SimpleNamespace(llm_provider=provider)

    with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as factory:
        factory.create_from_config.return_value = query_service
        agent = SolanaAgent(
            instructions="Use hosted Solana Agent APIs.",
            privy_user_id="did:privy:smoke-user",
            model="memory",
            stateless_model=DEFAULT_AGI_STATELESS_MODEL,
            base_url=DEFAULT_AGI_BASE_URL,
            api_key="x402",
            tools=["mcp"],
        )

    runtime_context = await agent.context(
        conversation_id="conv-123",
        model="chat",
        memory_ttl_tier="project",
        service_tier="priority",
        search_enabled=True,
    )
    result = await agent.message("hello", **runtime_context)

    assert result == "hello world"
    assert runtime_context == {
        "conversation_id": "conv-123",
        "model": DEFAULT_AGI_STATELESS_MODEL,
        "memory_ttl_tier": "project",
        "service_tier": "priority",
        "search_enabled": True,
        "privy_wallet_id": "wallet-123",
        "hosted_privy_wallet_id": "wallet-123",
        "privy_wallet_address": "WalletPubkey123",
        "privy_wallet_public_key": "WalletPubkey123",
    }
    assert query_service.process.call_args.kwargs == {
        "privy_user_id": "did:privy:smoke-user",
        "query": "hello",
        "runtime_context": runtime_context,
        "images": None,
        "output_format": "text",
        "audio_voice": "nova",
        "audio_output_format": "aac",
        "audio_input_format": "mp4",
        "prompt": None,
        "output_model": None,
        "capture_schema": None,
        "capture_name": None,
    }

    assert await agent.create_privy_user() == {
        "privy_user_id": "did:privy:new-user",
        "created": True,
    }
    assert await agent.rotate_wallet() == {
        "privy_user_id": "did:privy:smoke-user",
        "wallet_id": "wallet-456",
        "address": "WalletPubkey456",
        "old_wallets": [{"wallet_id": "wallet-123"}],
    }
    assert await agent.get_wallet_address("wallet-456") == "WalletPubkey456"
    assert await agent.export_wallet_private_key("wallet-456") == "base58-private-key"
    assert await agent.get_account_summary(privy_wallet_id="wallet-123") == {
        "spend": {"month": 1}
    }
    assert await agent.get_usage_report("day", group_by="model") == {"buckets": []}
    assert await agent.get_usage_forecast(window_days=30) == {"window_days": 30}
    assert await agent.get_pricing_info() == {"models": []}


def test_public_mcp_plugin_smoke_configures_single_server_and_schema() -> None:
    config = {
        "ai": {"api_key": "x402", "model": "memory"},
        "tools": {
            "mcp": {
                "url": "https://mcp.example.test/mcp",
                "headers": {"Authorization": "Bearer test-token"},
                "llm_provider": "openai",
                "api_key": "test-openai-key",
                "llm_model": "gpt-4.1-mini",
            }
        },
    }
    registry = ToolRegistry(config=config)
    plugin = get_mcp_plugin()

    plugin.initialize(registry)
    plugin.configure(config)
    tool = registry.get_tool("mcp")

    assert tool is plugin.get_tools()[0]
    assert tool.get_schema()["required"] == ["query"]
    assert tool._servers == [
        {
            "url": "https://mcp.example.test/mcp",
            "headers": {"Authorization": "Bearer test-token"},
        }
    ]
    assert tool._llm_provider == "openai"
    assert tool._llm_model == "gpt-4.1-mini"
    assert tool._llm_api_key == "x402"

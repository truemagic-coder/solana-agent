import re
from unittest.mock import MagicMock, patch

import pytest

from solana_agent.factories.agent_factory import (
    DEFAULT_AGI_BASE_URL,
    DEFAULT_AGI_MEMORY_MODEL,
    DEFAULT_AGI_STATELESS_MODEL,
    GUARDRAILS_CONFIG_ERROR,
    LEGACY_AGENTS_CONFIG_ERROR,
    LOCAL_MEMORY_CONFIG_ERROR,
    SolanaAgentFactory,
    UNSUPPORTED_PUBLIC_TOOL_ERROR,
)
from solana_agent.services.query import QueryService


@pytest.fixture
def hosted_config():
    return {
        "ai": {
            "name": "assistant",
            "instructions": "You are a concise hosted Solana assistant.",
            "specialization": "general",
            "tools": ["mcp"],
            "privy_user_id": "did:privy:user123",
        }
    }


def test_create_from_config_rejects_local_memory(hosted_config):
    config = dict(hosted_config)
    config["mongo"] = {"uri": "mongodb://localhost:27017"}

    with pytest.raises(ValueError, match=LOCAL_MEMORY_CONFIG_ERROR):
        SolanaAgentFactory.create_from_config(config)


def test_create_from_config_rejects_guardrails(hosted_config):
    config = dict(hosted_config)
    config["guardrails"] = {"input": [{"class": "example.Guardrail"}]}

    with pytest.raises(ValueError, match=GUARDRAILS_CONFIG_ERROR):
        SolanaAgentFactory.create_from_config(config)


def test_create_from_config_rejects_multiple_agents(hosted_config):
    config = dict(hosted_config)
    config["agents"] = [
        {
            "name": "assistant",
            "instructions": "You are a concise hosted Solana assistant.",
            "specialization": "general",
        },
        {
            "name": "second",
            "instructions": "You are another assistant.",
            "specialization": "general",
        },
    ]

    with pytest.raises(ValueError, match=re.escape(LEGACY_AGENTS_CONFIG_ERROR)):
        SolanaAgentFactory.create_from_config(config)


@patch("solana_agent.factories.agent_factory.PluginManager")
@patch("solana_agent.factories.agent_factory.OpenAIAdapter")
def test_create_from_config_builds_single_agent_runtime(
    mock_adapter_class,
    mock_plugin_manager_class,
    hosted_config,
):
    mock_adapter = MagicMock()
    mock_adapter_class.return_value = mock_adapter
    mock_plugin_manager = MagicMock()
    mock_plugin_manager.load_plugins.return_value = ["mcp"]
    mock_plugin_manager_class.return_value = mock_plugin_manager

    service = SolanaAgentFactory.create_from_config(hosted_config)

    assert isinstance(service, QueryService)
    assert set(service.agent_service.get_all_ai_agents()) == {"assistant"}
    mock_adapter_class.assert_called_once_with(
        api_key="x402",
        model=DEFAULT_AGI_MEMORY_MODEL,
        base_url=DEFAULT_AGI_BASE_URL,
        auth_mode="hosted_managed",
        privy_user_id="did:privy:user123",
    )
    mock_plugin_manager.load_plugins.assert_called_once_with()


@patch("solana_agent.factories.agent_factory.PluginManager")
@patch("solana_agent.factories.agent_factory.OpenAIAdapter")
def test_create_from_config_creates_default_agent_when_omitted(
    mock_adapter_class,
    mock_plugin_manager_class,
):
    mock_adapter_class.return_value = MagicMock()
    mock_plugin_manager_class.return_value = MagicMock(load_plugins=MagicMock())

    service = SolanaAgentFactory.create_from_config(
        {
            "ai": {
                "instructions": "You are a helpful default hosted assistant.",
            }
        }
    )

    assert set(service.agent_service.get_all_ai_agents()) == {"default"}


@patch("solana_agent.factories.agent_factory.PluginManager")
@patch("solana_agent.factories.agent_factory.OpenAIAdapter")
def test_create_from_config_resolves_chat_model_alias(
    mock_adapter_class,
    mock_plugin_manager_class,
    hosted_config,
):
    mock_adapter_class.return_value = MagicMock()
    mock_plugin_manager_class.return_value = MagicMock(load_plugins=MagicMock())
    config = dict(hosted_config)
    config["ai"] = dict(hosted_config["ai"], model="chat")

    SolanaAgentFactory.create_from_config(config)

    assert mock_adapter_class.call_args.kwargs["model"] == DEFAULT_AGI_STATELESS_MODEL


def test_create_from_config_rejects_agent_tools_for_unknown_agent(hosted_config):
    config = dict(hosted_config)
    config["agent_tools"] = {"other-agent": ["mcp"]}

    with pytest.raises(ValueError, match=re.escape(LEGACY_AGENTS_CONFIG_ERROR)):
        SolanaAgentFactory.create_from_config(config)


def test_create_from_config_rejects_x402_request_tool(hosted_config):
    config = {"ai": dict(hosted_config["ai"], tools=["x402_request"])}

    with pytest.raises(ValueError, match=re.escape(UNSUPPORTED_PUBLIC_TOOL_ERROR)):
        SolanaAgentFactory.create_from_config(config)


def test_create_from_config_rejects_x402_request_tool_config(hosted_config):
    config = dict(hosted_config)
    config["tools"] = {"x402_request": {"allowed_hosts": ["api.example.com"]}}

    with pytest.raises(ValueError, match=re.escape(UNSUPPORTED_PUBLIC_TOOL_ERROR)):
        SolanaAgentFactory.create_from_config(config)

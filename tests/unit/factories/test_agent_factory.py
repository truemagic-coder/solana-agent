from unittest.mock import MagicMock, patch

import pytest

from solana_agent.factories.agent_factory import (
    DEFAULT_AGI_BASE_URL,
    DEFAULT_AGI_MEMORY_MODEL,
    GUARDRAILS_CONFIG_ERROR,
    LOCAL_MEMORY_CONFIG_ERROR,
    MULTI_AGENT_CONFIG_ERROR,
    SolanaAgentFactory,
)
from solana_agent.services.query import QueryService


@pytest.fixture
def hosted_config():
    return {
        "ai": {
            "auth_mode": "x402_private_key",
            "private_key": "test-private-key",
        },
        "agents": [
            {
                "name": "assistant",
                "instructions": "You are a concise hosted Solana assistant.",
                "specialization": "general",
                "tools": ["x402_request"],
            }
        ],
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
        hosted_config["agents"][0],
        {
            "name": "second",
            "instructions": "You are another assistant.",
            "specialization": "general",
        },
    ]

    with pytest.raises(ValueError, match=MULTI_AGENT_CONFIG_ERROR):
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
    mock_plugin_manager.load_plugins.return_value = ["x402_request"]
    mock_plugin_manager_class.return_value = mock_plugin_manager

    service = SolanaAgentFactory.create_from_config(hosted_config)

    assert isinstance(service, QueryService)
    assert set(service.agent_service.get_all_ai_agents()) == {"assistant"}
    mock_adapter_class.assert_called_once_with(
        api_key="x402",
        model=DEFAULT_AGI_MEMORY_MODEL,
        base_url=DEFAULT_AGI_BASE_URL,
        auth_mode="x402_private_key",
        private_key="test-private-key",
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
                "auth_mode": "x402_private_key",
                "private_key": "test-private-key",
            }
        }
    )

    assert set(service.agent_service.get_all_ai_agents()) == {"default"}


def test_create_from_config_rejects_agent_tools_for_unknown_agent(hosted_config):
    config = dict(hosted_config)
    config["agent_tools"] = {"other-agent": ["x402_request"]}

    with pytest.raises(ValueError, match=MULTI_AGENT_CONFIG_ERROR):
        SolanaAgentFactory.create_from_config(config)

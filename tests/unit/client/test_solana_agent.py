"""
Tests for the SolanaAgent client interface.

This module provides comprehensive test coverage for the SolanaAgent client
including initialization, message processing, and tool registration.
"""

import pytest
from pydantic import BaseModel
from unittest.mock import MagicMock, patch, AsyncMock

from solana_agent.client.solana_agent import SolanaAgent
from solana_agent.interfaces.plugins.plugins import Tool
from solana_agent.local_state import load_saved_privy_user_id, save_privy_user_id


@pytest.fixture
def config_dict():
    """Fixture providing test configuration."""
    return {
        "ai": {
            "name": "test_agent",
            "instructions": "Test agent instructions",
            "specialization": "Testing",
            "privy_user_id": "did:privy:test-user",
        }
    }


@pytest.fixture
def mock_query_service():
    """Create a mock query service with all required methods."""
    mock = AsyncMock()

    async def mock_process(*args, **kwargs):
        yield "Test response"

    mock.process = MagicMock(side_effect=mock_process)

    # Configure agent service
    mock.agent_service = MagicMock()
    mock.agent_service.llm_provider = AsyncMock()
    mock.agent_service.tool_registry = MagicMock()
    mock.agent_service.get_all_ai_agents = MagicMock(return_value=["test_agent"])
    mock.agent_service.assign_tool_for_agent = MagicMock()

    return mock


class TestSolanaAgent:
    """Test suite for SolanaAgent client."""

    @patch("solana_agent.client.solana_agent.SolanaAgentFactory")
    def test_init_with_config(self, mock_factory, config_dict, mock_query_service):
        """Test initialization with configuration dictionary."""
        mock_factory.create_from_config.return_value = mock_query_service
        agent = SolanaAgent(config=config_dict)
        mock_factory.create_from_config.assert_called_once_with(config_dict)
        assert agent.query_service == mock_query_service

    @patch("solana_agent.client.solana_agent.SolanaAgentFactory")
    def test_init_without_config_uses_hosted_defaults(
        self, mock_factory, mock_query_service
    ):
        """The public SDK can initialize from hosted defaults."""
        mock_factory.create_from_config.return_value = mock_query_service

        agent = SolanaAgent()

        mock_factory.create_from_config.assert_called_once_with({"ai": {}})
        assert agent.query_service == mock_query_service

    @patch("solana_agent.client.solana_agent.SolanaAgentFactory")
    def test_init_uses_saved_privy_user_id(
        self, mock_factory, mock_query_service
    ):
        """The public SDK should load the last saved Privy DID from local state."""
        save_privy_user_id("did:privy:saved-user")
        mock_factory.create_from_config.return_value = mock_query_service

        agent = SolanaAgent()

        mock_factory.create_from_config.assert_called_once_with(
            {"ai": {"privy_user_id": "did:privy:saved-user"}}
        )
        assert agent.config["ai"]["privy_user_id"] == "did:privy:saved-user"

    @patch("solana_agent.client.solana_agent.SolanaAgentFactory")
    def test_init_with_public_kwargs_builds_ai_config(
        self, mock_factory, mock_query_service
    ):
        """README-style constructor kwargs should become config.ai fields."""
        mock_factory.create_from_config.return_value = mock_query_service

        agent = SolanaAgent(
            instructions="You are a Solana trading bot.",
            privy_user_id="did:privy:user123",
            model="chat",
        )

        mock_factory.create_from_config.assert_called_once_with(
            {
                "ai": {
                    "instructions": "You are a Solana trading bot.",
                    "privy_user_id": "did:privy:user123",
                    "model": "chat",
                }
            }
        )
        assert agent.config["ai"]["privy_user_id"] == "did:privy:user123"

    @pytest.mark.asyncio
    async def test_process_requires_configured_privy_user_id(self, mock_query_service):
        """Process should require config.ai.privy_user_id."""
        config = {
            "ai": {
                "name": "test_agent",
                "instructions": "Test agent instructions",
                "specialization": "Testing",
            }
        }

        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config)

            with pytest.raises(
                ValueError,
                match="config.ai.privy_user_id",
            ):
                async for _chunk in agent.process(message="hello"):
                    pass

    @pytest.mark.asyncio
    async def test_process_rejects_nested_runtime_context(
        self, config_dict, mock_query_service
    ):
        """Process should require flat runtime keyword arguments."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            with pytest.raises(
                ValueError,
                match="flat keyword arguments",
            ):
                async for _chunk in agent.process(
                    message="hello",
                    runtime_context={"conversation_id": "conv-123"},
                ):
                    pass

    @pytest.mark.asyncio
    async def test_prepare_x402_runtime_context_rejects_runtime_privy_user_id(
        self, config_dict, mock_query_service
    ):
        """Runtime metadata should not redeclare the configured Privy DID."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            with pytest.raises(
                ValueError,
                match="config.ai.privy_user_id",
            ):
                await agent.prepare_x402_runtime_context(
                    privy_user_id="did:privy:user123",
                )

    @pytest.mark.asyncio
    async def test_process_passes_runtime_context(
        self, config_dict, mock_query_service
    ):
        """Process should pass runtime context through to QueryService."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            chunks = []
            async for chunk in agent.process(
                message="hello",
                privy_wallet_id="wallet-123",
            ):
                chunks.append(chunk)

            assert chunks == ["Test response"]
            assert mock_query_service.process.call_args.kwargs["runtime_context"] == {
                "privy_wallet_id": "wallet-123",
            }

    @pytest.mark.asyncio
    async def test_process_message_collects_non_streaming_text_response(
        self, config_dict, mock_query_service
    ):
        """process_message should collect the final hosted response automatically."""

        async def mock_process(*args, **kwargs):
            del args, kwargs
            yield "Test "
            yield "response"

        mock_query_service.process = MagicMock(side_effect=mock_process)

        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            result = await agent.process_message(
                message="hello",
                conversation_id="conv-123",
            )

            assert result == "Test response"
            assert mock_query_service.process.call_args.kwargs["runtime_context"] == {
                "conversation_id": "conv-123",
            }

    @pytest.mark.asyncio
    async def test_message_alias_collects_non_streaming_text_response(
        self, config_dict, mock_query_service
    ):
        """message should be the README-friendly process_message alias."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            result = await agent.message(
                message="hello",
                conversation_id="conv-123",
            )

            assert result == "Test response"
            assert mock_query_service.process.call_args.kwargs["runtime_context"] == {
                "conversation_id": "conv-123",
            }

    @pytest.mark.asyncio
    async def test_context_builds_wallet_model_and_tier_context(
        self, config_dict, mock_query_service
    ):
        """context should create flat runtime metadata for message()."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            mock_query_service.agent_service.llm_provider.create_wallet = AsyncMock(
                return_value={
                    "privy_user_id": "did:privy:test-user",
                    "wallet_id": "wallet-123",
                    "address": "WalletPubkey123",
                }
            )

            result = await agent.context(
                conversation_id="conv-123",
                model="chat",
                memory_ttl_tier="project",
                service_tier="priority",
            )

            assert result == {
                "conversation_id": "conv-123",
                "model": "solana-agent-chat",
                "memory_ttl_tier": "project",
                "service_tier": "priority",
                "privy_wallet_id": "wallet-123",
                "hosted_privy_wallet_id": "wallet-123",
                "privy_wallet_address": "WalletPubkey123",
                "privy_wallet_public_key": "WalletPubkey123",
            }

    @pytest.mark.asyncio
    async def test_context_rejects_invalid_service_tier(
        self, config_dict, mock_query_service
    ):
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            with pytest.raises(
                ValueError,
                match="service_tier must be one of: standard, priority",
            ):
                await agent.context(service_tier="express")

    @pytest.mark.asyncio
    async def test_process_message_returns_structured_output(
        self, config_dict, mock_query_service
    ):
        """process_message should return a captured Pydantic model unchanged."""

        class CapturedOutput(BaseModel):
            status: str

        captured = CapturedOutput(status="ok")

        async def mock_process(*args, **kwargs):
            del args, kwargs
            yield captured

        mock_query_service.process = MagicMock(side_effect=mock_process)

        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            result = await agent.process_message(message="hello")

            assert result == captured

    @pytest.mark.asyncio
    async def test_process_message_returns_audio_bytes(
        self, config_dict, mock_query_service
    ):
        """process_message should concatenate binary audio chunks."""

        async def mock_process(*args, **kwargs):
            del args, kwargs
            yield b"abc"
            yield b"123"

        mock_query_service.process = MagicMock(side_effect=mock_process)

        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            result = await agent.process_message(message="hello")

            assert result == b"abc123"

    @pytest.mark.asyncio
    async def test_process_message_returns_none_when_provider_yields_nothing(
        self, config_dict, mock_query_service
    ):
        """process_message should surface None when no chunks are emitted."""

        async def mock_process(*args, **kwargs):
            del args, kwargs
            if False:
                yield None

        mock_query_service.process = MagicMock(side_effect=mock_process)

        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            result = await agent.process_message(message="hello")

            assert result is None

    @pytest.mark.asyncio
    async def test_process_merges_search_enabled_into_runtime_context(
        self, config_dict, mock_query_service
    ):
        """Process should expose hosted search without forcing callers into runtime_context."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            chunks = []
            async for chunk in agent.process(
                message="hello",
                conversation_id="conv-123",
                search_enabled=True,
            ):
                chunks.append(chunk)

            assert chunks == ["Test response"]
            assert mock_query_service.process.call_args.kwargs["runtime_context"] == {
                "conversation_id": "conv-123",
                "search_enabled": True,
            }

    @pytest.mark.asyncio
    async def test_process_does_not_prepare_wallet_context_for_mcp_tools(
        self, mock_query_service
    ):
        """MCP is a local plugin and should not trigger hidden wallet setup."""
        config = {
            "ai": {
                "instructions": "Use MCP tools when useful.",
                "tools": ["mcp"],
                "privy_user_id": "did:privy:user123",
            }
        }
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config)

            mock_query_service.agent_service.llm_provider.create_wallet = AsyncMock()

            chunks = []
            async for chunk in agent.process(
                message="hello",
                conversation_id="conv-123",
            ):
                chunks.append(chunk)

            assert chunks == ["Test response"]
            mock_query_service.agent_service.llm_provider.create_wallet.assert_not_awaited()
            assert mock_query_service.process.call_args.kwargs["runtime_context"] == {
                "conversation_id": "conv-123",
            }

    def test_register_tool_success(self, config_dict, mock_query_service):
        """Test successful tool registration."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            # Setup mock tool and registry
            mock_tool = MagicMock(spec=Tool)
            mock_tool.name = "test_tool"
            mock_query_service.agent_service.tool_registry.register_tool.return_value = True

            # Test registration
            result = agent.register_tool("test_agent", mock_tool)

            # Verify results
            assert result is True
            mock_query_service.agent_service.tool_registry.register_tool.assert_called_once_with(
                mock_tool
            )
            mock_query_service.agent_service.assign_tool_for_agent.assert_called_once_with(
                "test_agent", "test_tool"
            )

    def test_register_tool_failure(self, config_dict, mock_query_service):
        """Test failed tool registration."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            # Setup mock tool and registry
            mock_tool = MagicMock(spec=Tool)
            mock_tool.name = "test_tool"
            mock_query_service.agent_service.tool_registry.register_tool.return_value = False

            # Test registration
            result = agent.register_tool("test_agent", mock_tool)

            # Verify results
            assert result is False
            mock_query_service.agent_service.tool_registry.register_tool.assert_called_once_with(
                mock_tool
            )
            # Verify assign_tool_for_agent was not called
            mock_query_service.agent_service.assign_tool_for_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_account_summary(self, config_dict, mock_query_service):
        """Client account summary should delegate to the hosted provider."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {
                "spend": {"month": 42.0},
                "protocols": {"lifetime": {"jupiter": {"executed_requests": 3}}},
                "tooling": {"lifetime": {"totals": {"requests": 5}}},
            }
            mock_query_service.agent_service.llm_provider.get_account_summary = (
                AsyncMock(return_value=expected)
            )

            result = await agent.get_account_summary(privy_wallet_id="wallet-123")

            assert result == expected
            assert result["tooling"]["lifetime"]["totals"]["requests"] == 5
            mock_query_service.agent_service.llm_provider.get_account_summary.assert_awaited_once_with(
                runtime_context={"privy_wallet_id": "wallet-123"}
            )

    @pytest.mark.asyncio
    async def test_create_wallet(self, config_dict, mock_query_service):
        """Client wallet creation should delegate to the hosted provider."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {
                "privy_user_id": "did:privy:user123",
                "wallet_id": "wallet-123",
                "address": "WalletPubkey123",
                "chain_type": "solana",
                "created": True,
                "old_wallets": [],
            }
            mock_query_service.agent_service.llm_provider.create_wallet = AsyncMock(
                return_value=expected
            )

            result = await agent.create_wallet("did:privy:user123")

            assert result == expected
            mock_query_service.agent_service.llm_provider.create_wallet.assert_awaited_once_with(
                privy_user_id="did:privy:user123",
                chain_type="solana",
            )

    @pytest.mark.asyncio
    async def test_create_wallet_uses_configured_privy_user_id(
        self, config_dict, mock_query_service
    ):
        """Client wallet creation can use the configured Privy DID."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {
                "privy_user_id": "did:privy:test-user",
                "wallet_id": "wallet-123",
                "address": "WalletPubkey123",
            }
            mock_query_service.agent_service.llm_provider.create_wallet = AsyncMock(
                return_value=expected
            )

            result = await agent.create_wallet()

            assert result == expected
            mock_query_service.agent_service.llm_provider.create_wallet.assert_awaited_once_with(
                privy_user_id="did:privy:test-user",
                chain_type="solana",
            )

    @pytest.mark.asyncio
    async def test_create_privy_user(self, config_dict, mock_query_service):
        """Client user creation should delegate to the hosted provider."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {
                "privy_user_id": "did:privy:user123",
                "created": True,
            }
            mock_query_service.agent_service.llm_provider.create_privy_user = AsyncMock(
                return_value=expected
            )

            result = await agent.create_privy_user()

            assert result == expected
            assert agent.config["ai"]["privy_user_id"] == "did:privy:user123"
            assert (
                mock_query_service.agent_service.llm_provider.privy_user_id
                == "did:privy:user123"
            )
            assert load_saved_privy_user_id() == "did:privy:user123"
            mock_query_service.agent_service.llm_provider.create_privy_user.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_rotate_wallet(self, config_dict, mock_query_service):
        """Client wallet rotation should delegate to the hosted provider."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {
                "privy_user_id": "did:privy:user123",
                "wallet_id": "wallet-new",
                "address": "WalletPubkeyNew",
                "created": True,
                "old_wallets": [{"wallet_id": "wallet-old", "address": "WalletOld"}],
            }
            mock_query_service.agent_service.llm_provider.rotate_wallet = AsyncMock(
                return_value=expected
            )

            result = await agent.rotate_wallet("did:privy:user123")

            assert result == expected
            mock_query_service.agent_service.llm_provider.rotate_wallet.assert_awaited_once_with(
                privy_user_id="did:privy:user123",
                chain_type="solana",
            )

    @pytest.mark.asyncio
    async def test_rotate_wallet_uses_configured_privy_user_id(
        self, config_dict, mock_query_service
    ):
        """Client wallet rotation can use the configured Privy DID."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {
                "privy_user_id": "did:privy:test-user",
                "wallet_id": "wallet-new",
                "address": "WalletPubkeyNew",
                "old_wallets": [],
            }
            mock_query_service.agent_service.llm_provider.rotate_wallet = AsyncMock(
                return_value=expected
            )

            result = await agent.rotate_wallet()

            assert result == expected
            mock_query_service.agent_service.llm_provider.rotate_wallet.assert_awaited_once_with(
                privy_user_id="did:privy:test-user",
                chain_type="solana",
            )

    @pytest.mark.asyncio
    async def test_export_wallet_private_key_uses_configured_privy_user_id(
        self, config_dict, mock_query_service
    ):
        """Client wallet export can use the configured Privy DID."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            mock_query_service.agent_service.llm_provider.export_wallet_private_key = (
                AsyncMock(return_value={"private_key": "base58-private-key"})
            )

            result = await agent.export_wallet_private_key()

            assert result == "base58-private-key"
            mock_query_service.agent_service.llm_provider.export_wallet_private_key.assert_awaited_once_with(
                privy_user_id="did:privy:test-user",
                wallet_id=None,
                chain_type="solana",
            )

    @pytest.mark.asyncio
    async def test_export_wallet_private_key_accepts_wallet_id_and_privy_user_id(
        self, config_dict, mock_query_service
    ):
        """Client wallet export should support historical wallet IDs."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            mock_query_service.agent_service.llm_provider.export_wallet_private_key = (
                AsyncMock(return_value={"private_key": "old-base58-private-key"})
            )

            result = await agent.export_wallet_private_key(
                wallet_id="wallet-old",
                privy_user_id="did:privy:user123",
            )

            assert result == "old-base58-private-key"
            mock_query_service.agent_service.llm_provider.export_wallet_private_key.assert_awaited_once_with(
                privy_user_id="did:privy:user123",
                wallet_id="wallet-old",
                chain_type="solana",
            )

    @pytest.mark.asyncio
    async def test_export_wallet_private_key_requires_private_key_in_response(
        self, config_dict, mock_query_service
    ):
        """Client wallet export should fail if the hosted response omits key material."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            mock_query_service.agent_service.llm_provider.export_wallet_private_key = (
                AsyncMock(return_value={"wallet_id": "wallet-123"})
            )

            with pytest.raises(
                ValueError,
                match="missing a private_key",
            ):
                await agent.export_wallet_private_key()

    @pytest.mark.asyncio
    async def test_get_wallet_address(self, config_dict, mock_query_service):
        """Client wallet address lookup should return the hosted public address."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            mock_query_service.agent_service.llm_provider.get_wallet_address = (
                AsyncMock(
                    return_value={
                        "wallet_id": "wallet-123",
                        "address": "WalletPubkey123",
                    }
                )
            )

            result = await agent.get_wallet_address("wallet-123")

            assert result == "WalletPubkey123"
            mock_query_service.agent_service.llm_provider.get_wallet_address.assert_awaited_once_with(
                wallet_id="wallet-123"
            )

    @pytest.mark.asyncio
    async def test_get_wallet_address_without_wallet_id_uses_configured_privy_user(
        self, config_dict, mock_query_service
    ):
        """No-arg wallet address should fetch the active wallet for config.ai.privy_user_id."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            mock_query_service.agent_service.llm_provider.create_wallet = AsyncMock(
                return_value={
                    "privy_user_id": "did:privy:test-user",
                    "wallet_id": "wallet-123",
                    "address": "WalletPubkey123",
                }
            )
            mock_query_service.agent_service.llm_provider.get_wallet_address = (
                AsyncMock()
            )

            result = await agent.get_wallet_address()

            assert result == "WalletPubkey123"
            mock_query_service.agent_service.llm_provider.create_wallet.assert_awaited_once_with(
                privy_user_id="did:privy:test-user",
                chain_type="solana",
            )
            mock_query_service.agent_service.llm_provider.get_wallet_address.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_wallet_address_without_wallet_id_falls_back_to_wallet_lookup(
        self, config_dict, mock_query_service
    ):
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            mock_query_service.agent_service.llm_provider.create_wallet = AsyncMock(
                return_value={
                    "privy_user_id": "did:privy:test-user",
                    "wallet_id": "wallet-123",
                }
            )
            mock_query_service.agent_service.llm_provider.get_wallet_address = (
                AsyncMock(return_value={"address": "WalletPubkey123"})
            )

            result = await agent.get_wallet_address()

            assert result == "WalletPubkey123"
            mock_query_service.agent_service.llm_provider.get_wallet_address.assert_awaited_once_with(
                wallet_id="wallet-123"
            )

    @pytest.mark.asyncio
    async def test_get_wallet_address_requires_address_in_provider_response(
        self, config_dict, mock_query_service
    ):
        """Client wallet address lookup should fail when the hosted response omits an address."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            mock_query_service.agent_service.llm_provider.get_wallet_address = (
                AsyncMock(return_value={"wallet_id": "wallet-123"})
            )

            with pytest.raises(
                ValueError,
                match="Hosted wallet response is missing an address",
            ):
                await agent.get_wallet_address("wallet-123")

    @pytest.mark.asyncio
    async def test_prepare_x402_runtime_context(self, config_dict, mock_query_service):
        """Client helper should return hosted Privy wallet runtime context."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            mock_query_service.agent_service.llm_provider.create_wallet = AsyncMock(
                return_value={
                    "privy_user_id": "did:privy:user123",
                    "wallet_id": "wallet-123",
                    "address": "WalletPubkey123",
                    "old_wallets": [],
                }
            )

            result = await agent.prepare_x402_runtime_context(
                conversation_id="conv-123",
            )

            assert result == {
                "conversation_id": "conv-123",
                "privy_wallet_id": "wallet-123",
                "hosted_privy_wallet_id": "wallet-123",
                "privy_wallet_address": "WalletPubkey123",
                "privy_wallet_public_key": "WalletPubkey123",
            }

    @pytest.mark.asyncio
    async def test_prepare_x402_runtime_context_reuses_hosted_wallet_alias(
        self, config_dict, mock_query_service
    ):
        """Client helper should normalize an existing hosted Privy wallet alias."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            mock_query_service.agent_service.llm_provider.create_wallet = AsyncMock()

            result = await agent.prepare_x402_runtime_context(
                conversation_id="conv-123",
                hosted_privy_wallet_id="wallet-123",
            )

            assert result == {
                "conversation_id": "conv-123",
                "hosted_privy_wallet_id": "wallet-123",
                "privy_wallet_id": "wallet-123",
            }
            mock_query_service.agent_service.llm_provider.create_wallet.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_usage_report(self, config_dict, mock_query_service):
        """Client usage reports should delegate query parameters to the hosted provider."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {
                "buckets": [
                    {
                        "totals": {
                            "tooling": {
                                "totals": {
                                    "requests": 7,
                                    "signature_count": 2,
                                }
                            }
                        }
                    }
                ]
            }
            mock_query_service.agent_service.llm_provider.get_usage_report = AsyncMock(
                return_value=expected
            )

            result = await agent.get_usage_report(
                "month",
                from_date="2026-05-01",
                to_date="2026-05-31",
                group_by="conversation",
                privy_wallet_id="wallet-123",
            )

            assert result == expected
            assert result["buckets"][0]["totals"]["tooling"]["totals"]["requests"] == 7
            mock_query_service.agent_service.llm_provider.get_usage_report.assert_awaited_once_with(
                "month",
                from_date="2026-05-01",
                to_date="2026-05-31",
                group_by="conversation",
                runtime_context={"privy_wallet_id": "wallet-123"},
            )

    @pytest.mark.asyncio
    async def test_get_usage_forecast(self, config_dict, mock_query_service):
        """Client usage forecasts should delegate to the hosted provider."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {
                "forecast": {"projected_spend": 12.5},
                "tooling": {
                    "current_month": {"totals": {"requests_to_date": 2}},
                    "projected_month_end": {"totals": {"projected_requests": 4}},
                },
            }
            mock_query_service.agent_service.llm_provider.get_usage_forecast = (
                AsyncMock(return_value=expected)
            )

            result = await agent.get_usage_forecast(
                window_days=14,
                privy_wallet_id="wallet-123",
            )

            assert result == expected
            assert (
                result["tooling"]["projected_month_end"]["totals"]["projected_requests"]
                == 4
            )
            mock_query_service.agent_service.llm_provider.get_usage_forecast.assert_awaited_once_with(
                window_days=14,
                runtime_context={"privy_wallet_id": "wallet-123"},
            )

    @pytest.mark.asyncio
    async def test_get_pricing_info(self, config_dict, mock_query_service):
        """Client pricing info should delegate to the hosted provider."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {"pricing": {"explorer": {"included_requests": 25}}}
            mock_query_service.agent_service.llm_provider.get_pricing_info = AsyncMock(
                return_value=expected
            )

            result = await agent.get_pricing_info(privy_wallet_id="wallet-123")

            assert result == expected
            mock_query_service.agent_service.llm_provider.get_pricing_info.assert_awaited_once_with(
                runtime_context={"privy_wallet_id": "wallet-123"}
            )

    @pytest.mark.asyncio
    async def test_account_reporting_raises_when_provider_lacks_support(
        self, config_dict, mock_query_service
    ):
        """Client should fail clearly when the configured provider has no account surface."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)
            mock_query_service.agent_service.llm_provider = MagicMock(spec=[])

            with pytest.raises(
                NotImplementedError,
                match="Account reporting is not available for the configured provider",
            ):
                await agent.get_account_summary()

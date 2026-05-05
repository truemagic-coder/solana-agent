"""
Tests for the SolanaAgent client interface.

This module provides comprehensive test coverage for the SolanaAgent client
including initialization, message processing, history management, and tool registration.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from solana_agent.client.solana_agent import SolanaAgent
from solana_agent.interfaces.plugins.plugins import Tool


@pytest.fixture
def config_dict():
    """Fixture providing test configuration."""
    return {
        "ai": {"api_key": "test_key"},
        "agents": [
            {
                "name": "test_agent",
                "instructions": "Test agent instructions",
                "specialization": "Testing",
            }
        ],
    }


@pytest.fixture
def mock_query_service():
    """Create a mock query service with all required methods."""
    mock = AsyncMock()

    async def mock_process(*args, **kwargs):
        yield "Test response"

    mock.process = MagicMock(side_effect=mock_process)
    mock.delete_user_history = AsyncMock()
    mock.get_user_history = AsyncMock(return_value={"messages": [], "total": 0})

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

    def test_init_without_config(self):
        """Test initialization fails without configuration."""
        with pytest.raises(
            ValueError, match="Either config or config_path must be provided"
        ):
            SolanaAgent()

    @pytest.mark.asyncio
    async def test_get_user_history(self, config_dict, mock_query_service):
        """Test retrieving user message history."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {"messages": [], "total": 0}
            mock_query_service.get_user_history.return_value = expected

            result = await agent.get_user_history(
                user_id="test_user", page_num=1, page_size=20, sort_order="desc"
            )

            assert result == expected
            mock_query_service.get_user_history.assert_called_once_with(
                "test_user", 1, 20, "desc"
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
                user_id="test_user",
                message="hello",
                runtime_context={"privy_wallet_id": "wallet-123"},
            ):
                chunks.append(chunk)

            assert chunks == ["Test response"]
            assert mock_query_service.process.call_args.kwargs["runtime_context"] == {
                "privy_wallet_id": "wallet-123",
                "user_id": "test_user",
            }

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
                user_id="test_user",
                message="hello",
                runtime_context={"conversation_id": "conv-123"},
                search_enabled=True,
            ):
                chunks.append(chunk)

            assert chunks == ["Test response"]
            assert mock_query_service.process.call_args.kwargs["runtime_context"] == {
                "conversation_id": "conv-123",
                "search_enabled": True,
                "user_id": "test_user",
            }

    @pytest.mark.asyncio
    async def test_process_prepares_hosted_privy_wallet_for_x402_privy(
        self, mock_query_service
    ):
        """Process should attach hosted wallet context for x402_privy calls."""
        config = {
            "ai": {
                "auth_mode": "x402_privy",
                "privy_app_id": "app-123",
                "privy_app_secret": "secret-123",
            }
        }
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config)

            mock_query_service.agent_service.llm_provider.create_wallet = AsyncMock(
                return_value={
                    "wallet_id": "wallet-123",
                    "address": "WalletPubkey123",
                }
            )

            chunks = []
            async for chunk in agent.process(
                user_id="did:privy:user123",
                message="hello",
                runtime_context={"conversation_id": "conv-123"},
            ):
                chunks.append(chunk)

            assert chunks == ["Test response"]
            mock_query_service.agent_service.llm_provider.create_wallet.assert_awaited_once_with(
                user_id="did:privy:user123",
                chain_type="solana",
            )
            assert mock_query_service.process.call_args.kwargs["runtime_context"] == {
                "conversation_id": "conv-123",
                "user_id": "did:privy:user123",
                "privy_wallet_id": "wallet-123",
                "hosted_privy_wallet_id": "wallet-123",
                "privy_wallet_address": "WalletPubkey123",
                "privy_wallet_public_key": "WalletPubkey123",
            }

    @pytest.mark.asyncio
    async def test_process_reuses_hosted_privy_wallet_alias_for_x402_privy(
        self, mock_query_service
    ):
        """Process should not recreate a hosted Privy wallet when the alias is already present."""
        config = {
            "ai": {
                "auth_mode": "x402_privy",
                "privy_app_id": "app-123",
                "privy_app_secret": "secret-123",
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
                user_id="did:privy:user123",
                message="hello",
                runtime_context={
                    "conversation_id": "conv-123",
                    "hosted_privy_wallet_id": "wallet-123",
                },
            ):
                chunks.append(chunk)

            assert chunks == ["Test response"]
            mock_query_service.agent_service.llm_provider.create_wallet.assert_not_awaited()
            assert mock_query_service.process.call_args.kwargs["runtime_context"] == {
                "conversation_id": "conv-123",
                "hosted_privy_wallet_id": "wallet-123",
                "user_id": "did:privy:user123",
            }

    @pytest.mark.asyncio
    async def test_delete_user_history(self, config_dict, mock_query_service):
        """Test deleting user message history."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            await agent.delete_user_history("test_user")
            mock_query_service.delete_user_history.assert_called_once_with("test_user")

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

            expected = {"spend": {"month": 42.0}}
            mock_query_service.agent_service.llm_provider.get_account_summary = (
                AsyncMock(return_value=expected)
            )

            result = await agent.get_account_summary(
                runtime_context={"privy_wallet_id": "wallet-123"}
            )

            assert result == expected
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
                "user_id": "did:privy:user123",
                "wallet_id": "wallet-123",
                "address": "WalletPubkey123",
                "chain_type": "solana",
                "created": True,
            }
            mock_query_service.agent_service.llm_provider.create_wallet = AsyncMock(
                return_value=expected
            )

            result = await agent.create_wallet("did:privy:user123")

            assert result == expected
            mock_query_service.agent_service.llm_provider.create_wallet.assert_awaited_once_with(
                user_id="did:privy:user123",
                chain_type="solana",
            )

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
                        "user_id": "did:privy:user123",
                        "address": "WalletPubkey123",
                    }
                )
            )

            result = await agent.get_wallet_address("did:privy:user123")

            assert result == "WalletPubkey123"
            mock_query_service.agent_service.llm_provider.get_wallet_address.assert_awaited_once_with(
                user_id="did:privy:user123"
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
                AsyncMock(return_value={"user_id": "did:privy:user123"})
            )

            with pytest.raises(
                ValueError,
                match="Hosted wallet response is missing an address",
            ):
                await agent.get_wallet_address("did:privy:user123")

    @pytest.mark.asyncio
    async def test_export_wallet_private_key(self, config_dict, mock_query_service):
        """Client wallet export should reuse the local Privy export helper."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            llm_provider = mock_query_service.agent_service.llm_provider
            llm_provider.privy_app_id = "app-123"
            llm_provider.privy_app_secret = "secret-123"
            llm_provider.privy_authorization_signature = None
            llm_provider.privy_request_expiry = None
            llm_provider.privy_api_url = None
            llm_provider.x402_rpc_url = None

            with patch(
                "solana_agent.client.solana_agent.export_privy_wallet_private_key",
                new_callable=AsyncMock,
                return_value="base58-private-key",
            ) as mock_export:
                result = await agent.export_wallet_private_key(wallet_id="wallet-123")

            assert result == "base58-private-key"
            assert mock_export.await_args.args[0].wallet_id == "wallet-123"

    @pytest.mark.asyncio
    async def test_export_wallet_private_key_requires_wallet_context(
        self, config_dict, mock_query_service
    ):
        """Client wallet export should fail fast when no wallet context is available."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            with pytest.raises(ValueError, match="wallet_id is required"):
                await agent.export_wallet_private_key()

    @pytest.mark.asyncio
    async def test_export_wallet_private_key_accepts_hosted_wallet_alias(
        self, config_dict, mock_query_service
    ):
        """Client wallet export should accept the hosted wallet alias in runtime context."""
        with patch(
            "solana_agent.client.solana_agent.SolanaAgentFactory"
        ) as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            llm_provider = mock_query_service.agent_service.llm_provider
            llm_provider.privy_app_id = "app-123"
            llm_provider.privy_app_secret = "secret-123"
            llm_provider.privy_authorization_signature = None
            llm_provider.privy_request_expiry = None
            llm_provider.privy_api_url = None
            llm_provider.x402_rpc_url = None

            with patch(
                "solana_agent.client.solana_agent.export_privy_wallet_private_key",
                new_callable=AsyncMock,
                return_value="base58-private-key",
            ) as mock_export:
                result = await agent.export_wallet_private_key(
                    runtime_context={"hosted_privy_wallet_id": "wallet-123"}
                )

            assert result == "base58-private-key"
            assert mock_export.await_args.args[0].wallet_id == "wallet-123"

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
                    "wallet_id": "wallet-123",
                    "address": "WalletPubkey123",
                }
            )

            result = await agent.prepare_x402_runtime_context(
                "did:privy:user123",
                runtime_context={"conversation_id": "conv-123"},
            )

            assert result == {
                "conversation_id": "conv-123",
                "user_id": "did:privy:user123",
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
                "did:privy:user123",
                runtime_context={
                    "conversation_id": "conv-123",
                    "hosted_privy_wallet_id": "wallet-123",
                },
            )

            assert result == {
                "conversation_id": "conv-123",
                "user_id": "did:privy:user123",
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

            expected = {"buckets": []}
            mock_query_service.agent_service.llm_provider.get_usage_report = AsyncMock(
                return_value=expected
            )

            result = await agent.get_usage_report(
                "month",
                from_date="2026-05-01",
                to_date="2026-05-31",
                group_by="conversation",
                runtime_context={"privy_wallet_id": "wallet-123"},
            )

            assert result == expected
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

            expected = {"forecast": {"projected_spend": 12.5}}
            mock_query_service.agent_service.llm_provider.get_usage_forecast = (
                AsyncMock(return_value=expected)
            )

            result = await agent.get_usage_forecast(
                window_days=14,
                runtime_context={"privy_wallet_id": "wallet-123"},
            )

            assert result == expected
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

            result = await agent.get_pricing_info(
                runtime_context={"privy_wallet_id": "wallet-123"}
            )

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

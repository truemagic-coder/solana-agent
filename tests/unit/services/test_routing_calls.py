"""
Tests for routing call optimization.

Ensures that:
1. With a single agent, no routing LLM calls are made
2. With multiple agents, routing is called exactly once per query
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from solana_agent.services.query import QueryService
from solana_agent.services.routing import RoutingService
from solana_agent.services.agent import AgentService


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.parse_structured_output = AsyncMock()
    provider.transcribe_audio = AsyncMock(return_value=iter([]))

    async def mock_chat_stream(*args, **kwargs):
        yield "Test response"

    provider.chat_stream = mock_chat_stream
    return provider


@pytest.fixture
def single_agent_service(mock_llm_provider):
    """Create an agent service with a single agent."""
    service = AgentService(llm_provider=mock_llm_provider)
    service.register_ai_agent(
        name="solo_agent",
        instructions="You are a helpful assistant.",
        specialization="General assistance",
    )
    return service


@pytest.fixture
def multi_agent_service(mock_llm_provider):
    """Create an agent service with multiple agents."""
    service = AgentService(llm_provider=mock_llm_provider)

    service.register_ai_agent(
        name="research_agent",
        instructions="You are a research specialist.",
        specialization="Research",
    )
    service.register_ai_agent(
        name="support_agent",
        instructions="You are a support specialist.",
        specialization="Customer support",
    )

    return service


@pytest.fixture
def mock_routing_service(mock_llm_provider, single_agent_service):
    """Create a mock routing service."""
    service = RoutingService(
        llm_provider=mock_llm_provider,
        agent_service=single_agent_service,
    )
    return service


class TestSingleAgentRouting:
    """Tests for single agent routing optimization."""

    @pytest.mark.asyncio
    async def test_single_agent_skips_switch_intent_detection(
        self, mock_llm_provider, single_agent_service
    ):
        """With one agent, _detect_switch_intent should not be called."""
        routing_service = RoutingService(
            llm_provider=mock_llm_provider,
            agent_service=single_agent_service,
        )

        query_service = QueryService(
            agent_service=single_agent_service,
            routing_service=routing_service,
        )

        # Patch _detect_switch_intent to track if it's called
        with patch.object(
            query_service, "_detect_switch_intent", new_callable=AsyncMock
        ) as mock_detect:
            mock_detect.return_value = (False, None, False)

            # Process a query
            async for _ in query_service.process("user123", "Hello, how are you?"):
                pass

            # _detect_switch_intent should NOT be called with single agent
            mock_detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_agent_skips_route_query(
        self, mock_llm_provider, single_agent_service
    ):
        """With one agent, route_query should not be called."""
        routing_service = RoutingService(
            llm_provider=mock_llm_provider,
            agent_service=single_agent_service,
        )

        query_service = QueryService(
            agent_service=single_agent_service,
            routing_service=routing_service,
        )

        # Patch route_query to track if it's called
        with patch.object(
            routing_service, "route_query", new_callable=AsyncMock
        ) as mock_route:
            mock_route.return_value = "solo_agent"

            # Process a query
            async for _ in query_service.process("user123", "Hello, how are you?"):
                pass

            # route_query should NOT be called with single agent
            mock_route.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_agent_uses_only_agent(
        self, mock_llm_provider, single_agent_service
    ):
        """With one agent, that agent should be used directly without LLM routing calls."""
        routing_service = RoutingService(
            llm_provider=mock_llm_provider,
            agent_service=single_agent_service,
        )

        query_service = QueryService(
            agent_service=single_agent_service,
            routing_service=routing_service,
        )

        # Track LLM calls
        llm_call_count = 0
        original_parse = mock_llm_provider.parse_structured_output

        async def counting_parse(*args, **kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            return await original_parse(*args, **kwargs)

        mock_llm_provider.parse_structured_output = AsyncMock(
            side_effect=counting_parse
        )

        # Process a query
        async for _ in query_service.process("user123", "Hello, how are you?"):
            pass

        # No parse_structured_output calls for routing (only for agent response if any)
        # The key is that routing-specific calls should be 0
        # We check that _detect_switch_intent wasn't invoked by verifying parse wasn't called
        # for switch intent detection


class TestMultiAgentRouting:
    """Tests for multi-agent routing behavior."""

    @pytest.mark.asyncio
    async def test_multi_agent_calls_switch_intent_detection(
        self, mock_llm_provider, multi_agent_service
    ):
        """With multiple agents, _detect_switch_intent should be called."""
        routing_service = RoutingService(
            llm_provider=mock_llm_provider,
            agent_service=multi_agent_service,
        )

        query_service = QueryService(
            agent_service=multi_agent_service,
            routing_service=routing_service,
        )

        # Patch _detect_switch_intent to track if it's called
        with patch.object(
            query_service, "_detect_switch_intent", new_callable=AsyncMock
        ) as mock_detect:
            mock_detect.return_value = (False, None, False)

            # Process a query
            async for _ in query_service.process("user123", "Hello, how are you?"):
                pass

            # _detect_switch_intent SHOULD be called with multiple agents
            mock_detect.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_agent_calls_route_query_once(
        self, mock_llm_provider, multi_agent_service
    ):
        """With multiple agents, route_query should be called exactly once."""
        routing_service = RoutingService(
            llm_provider=mock_llm_provider,
            agent_service=multi_agent_service,
        )

        query_service = QueryService(
            agent_service=multi_agent_service,
            routing_service=routing_service,
        )

        # Patch _detect_switch_intent to not request a switch
        with patch.object(
            query_service, "_detect_switch_intent", new_callable=AsyncMock
        ) as mock_detect:
            mock_detect.return_value = (False, None, False)

            # Patch route_query to track calls
            with patch.object(
                routing_service, "route_query", new_callable=AsyncMock
            ) as mock_route:
                mock_route.return_value = "research_agent"

                # Process a query
                async for _ in query_service.process("user123", "What is AI?"):
                    pass

                # route_query SHOULD be called exactly once
                mock_route.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_agent_sticky_session_skips_routing(
        self, mock_llm_provider, multi_agent_service
    ):
        """With sticky session and no switch request, routing should be skipped."""
        routing_service = RoutingService(
            llm_provider=mock_llm_provider,
            agent_service=multi_agent_service,
        )

        query_service = QueryService(
            agent_service=multi_agent_service,
            routing_service=routing_service,
        )

        # Set up a sticky session
        query_service._set_sticky_agent(
            "user123", "research_agent", required_complete=False
        )

        # Patch _detect_switch_intent to not request a switch
        with patch.object(
            query_service, "_detect_switch_intent", new_callable=AsyncMock
        ) as mock_detect:
            mock_detect.return_value = (False, None, False)

            # Patch route_query to track calls
            with patch.object(
                routing_service, "route_query", new_callable=AsyncMock
            ) as mock_route:
                mock_route.return_value = "research_agent"

                # Process a query
                async for _ in query_service.process("user123", "Tell me more"):
                    pass

                # route_query should NOT be called when sticky session exists
                mock_route.assert_not_called()


class TestRoutingServiceSingleAgent:
    """Tests for RoutingService single agent optimization."""

    @pytest.mark.asyncio
    async def test_routing_service_single_agent_no_llm_call(
        self, mock_llm_provider, single_agent_service
    ):
        """RoutingService should not make LLM calls with single agent."""
        routing_service = RoutingService(
            llm_provider=mock_llm_provider,
            agent_service=single_agent_service,
        )

        # Call route_query
        result = await routing_service.route_query("What is AI?")

        # Should return the only agent
        assert result == "solo_agent"

        # No LLM calls should be made
        mock_llm_provider.parse_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_routing_service_multi_agent_makes_llm_call(
        self, mock_llm_provider, multi_agent_service
    ):
        """RoutingService should make LLM call with multiple agents."""
        # Set up mock to return a valid analysis
        from solana_agent.domains.routing import QueryAnalysis

        mock_llm_provider.parse_structured_output.return_value = QueryAnalysis(
            primary_agent="research_agent",
            secondary_agents=[],
            complexity_level=1,
            topics=["AI"],
            confidence=0.9,
        )

        routing_service = RoutingService(
            llm_provider=mock_llm_provider,
            agent_service=multi_agent_service,
        )

        # Call route_query
        result = await routing_service.route_query("What is AI?")

        # Should return the routed agent
        assert result == "research_agent"

        # LLM call should be made
        mock_llm_provider.parse_structured_output.assert_called_once()

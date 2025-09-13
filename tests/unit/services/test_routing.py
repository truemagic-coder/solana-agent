"""
Tests for the QueryService implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.interfaces.providers.memory import MemoryProvider


@pytest.fixture
def mock_agent_service():
    """Create a mock agent service."""
    service = AsyncMock(spec=AgentService)
    service.last_text_response = "Test response"
    service.llm_provider = AsyncMock()

    async def mock_tts():
        yield b"audio_data"

    async def mock_transcribe():
        yield "transcribed text"

    async def mock_generate():
        yield "generated text"

    service.llm_provider.tts.side_effect = mock_tts
    service.llm_provider.transcribe_audio.side_effect = mock_transcribe
    service.generate_response.side_effect = mock_generate
    return service


@pytest.fixture
def mock_routing_service():
    """Create a mock routing service."""
    service = AsyncMock(spec=RoutingService)
    service.route_query.return_value = "test_agent"
    return service


@pytest.fixture
def mock_memory_provider():
    """Create a mock memory provider."""
    provider = AsyncMock(spec=MemoryProvider)
    provider.retrieve.return_value = "memory context"
    provider.find.return_value = [
        {
            "_id": "123",
            "user_message": "hello",
            "assistant_message": "hi",
            "timestamp": MagicMock(),
        }
    ]
    provider.count_documents.return_value = 1
    return provider


class TestQueryService:
    """Test suite for QueryService."""

    @pytest.mark.asyncio
    async def test_process_greeting(
        self, mock_agent_service, mock_routing_service, mock_memory_provider
    ):
        """Test processing simple greetings."""
        service = QueryService(
            mock_agent_service, mock_routing_service, mock_memory_provider
        )

        greetings = ["hello", "hi", "hey", "test", "ping"]
        for greeting in greetings:
            async for response in service.process(user_id="user123", query=greeting):
                assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_process_error_handling(
        self,
        mock_agent_service,
        mock_routing_service,
        mock_memory_provider,  # Add mock_memory_provider if needed for init
    ):
        """Test error handling during processing."""
        # Ensure generate_response is an AsyncMock for async iteration
        mock_agent_service.generate_response = AsyncMock(
            side_effect=Exception("Test error")
        )

        # Instantiate QueryService with all required args, including input_guardrails
        service = QueryService(
            agent_service=mock_agent_service,
            routing_service=mock_routing_service,
            memory_provider=mock_memory_provider,  # Pass memory provider
            input_guardrails=None,  # Add input_guardrails
        )

        response_chunks = []
        async for response in service.process(user_id="user123", query="test query"):
            response_chunks.append(response)

        # Assert that only the generic error message was yielded
        assert len(response_chunks) == 1
        assert "I apologize for the technical difficulty" in response_chunks[0]
        # Do NOT assert the internal error message ("Test error") is present
        # assert "Test error" not in response_chunks[0] # Optional: Explicitly check it's not there

    @pytest.mark.asyncio
    async def test_delete_user_history(
        self, mock_agent_service, mock_routing_service, mock_memory_provider
    ):
        """Test deleting user history."""
        service = QueryService(
            mock_agent_service, mock_routing_service, mock_memory_provider
        )
        await service.delete_user_history("user123")
        mock_memory_provider.delete.assert_called_once_with("user123")

    @pytest.mark.asyncio
    async def test_delete_user_history_error(
        self, mock_agent_service, mock_routing_service, mock_memory_provider
    ):
        """Test error handling in delete user history."""
        mock_memory_provider.delete.side_effect = Exception("Delete error")
        service = QueryService(
            mock_agent_service, mock_routing_service, mock_memory_provider
        )
        # Should not raise exception
        await service.delete_user_history("user123")
        mock_memory_provider.delete.assert_called_once_with("user123")

    @pytest.mark.asyncio
    async def test_get_user_history_no_memory_provider(
        self, mock_agent_service, mock_routing_service
    ):
        """Test getting user history without memory provider."""
        service = QueryService(mock_agent_service, mock_routing_service)
        result = await service.get_user_history("user123")
        assert result["error"] == "Memory provider not available"
        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_get_user_history_success(
        self, mock_agent_service, mock_routing_service, mock_memory_provider
    ):
        """Test successful retrieval of user history."""
        service = QueryService(
            mock_agent_service, mock_routing_service, mock_memory_provider
        )
        result = await service.get_user_history("user123")

        assert result["total"] == 1
        assert len(result["data"]) == 1
        assert result["error"] is None
        mock_memory_provider.find.assert_called_once()
        mock_memory_provider.count_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_history_error(
        self, mock_agent_service, mock_routing_service, mock_memory_provider
    ):
        """Test error handling in get user history."""
        mock_memory_provider.find.side_effect = Exception("Find error")
        service = QueryService(
            mock_agent_service, mock_routing_service, mock_memory_provider
        )

        result = await service.get_user_history("user123")
        assert result["error"] == "Error retrieving history: Find error"
        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_store_conversation_error(
        self, mock_agent_service, mock_routing_service, mock_memory_provider
    ):
        """Test error handling in store conversation."""
        mock_memory_provider.store.side_effect = Exception("Store error")
        service = QueryService(
            mock_agent_service, mock_routing_service, mock_memory_provider
        )

        await service._store_conversation(
            "user123", "test message", "test response"
        )  # Should not raise exception
        mock_memory_provider.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_pagination(
        self, mock_agent_service, mock_routing_service, mock_memory_provider
    ):
        """Test pagination in get user history."""
        service = QueryService(
            mock_agent_service, mock_routing_service, mock_memory_provider
        )
        result = await service.get_user_history(
            user_id="user123", page_num=2, page_size=10, sort_order="asc"
        )

        assert result["page"] == 2
        assert result["page_size"] == 10
        mock_memory_provider.find.assert_called_once_with(
            collection="conversations",
            query={"user_id": "user123"},
            sort=[("timestamp", 1)],
            skip=10,
            limit=10,
        )

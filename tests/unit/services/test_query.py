import pytest
from unittest.mock import Mock, AsyncMock  # Import Mock as well
from datetime import datetime, timezone
import asyncio

from solana_agent.services.query import QueryService
from solana_agent.services.agent import (
    AgentService,
)  # Import concrete types if needed for isinstance checks or type hints
from solana_agent.services.routing import RoutingService
from solana_agent.interfaces.providers.memory import MemoryProvider

# Test Data
TEST_USER_ID = "test_user"
TEST_QUERY = "What is Solana?"
TEST_RESPONSE = "Solana is a blockchain."
HARDCODED_GREETING = None  # Greeting shortcut removed


# Helper async generator function for mocking
async def mock_async_generator(*args):
    for item in args:
        yield item
        await asyncio.sleep(0)  # Yield control briefly


@pytest.fixture
def mock_agent_service():
    # Use AsyncMock for the service object itself
    service = AsyncMock(spec=AgentService)  # Use spec for better mocking

    # Provide a generate_response that returns an async generator honoring kwargs
    async def mock_generate(**kwargs):
        # yield a single chunk of text as normal LLM output
        async for item in mock_async_generator(TEST_RESPONSE):
            yield item

    service.generate_response.side_effect = lambda **kwargs: mock_generate(**kwargs)

    # Mock the attribute accessed after generate_response finishes
    service.last_text_response = TEST_RESPONSE

    # Mock the llm_provider attribute if it's accessed directly (e.g., for TTS in error paths)
    service.llm_provider = AsyncMock()
    # Mock the tts method on the llm_provider
    tts_generator_instance = mock_async_generator(b"audio_chunk_1", b"audio_chunk_2")
    service.llm_provider.tts = AsyncMock(return_value=tts_generator_instance)
    # Mock transcribe_audio if needed
    service.llm_provider.transcribe_audio = AsyncMock(
        return_value=mock_async_generator("transcribed text")
    )

    return service


@pytest.fixture
def mock_routing_service():
    # Use AsyncMock for the service object
    service = AsyncMock(spec=RoutingService)
    service.route_query = AsyncMock(return_value="test_agent")
    return service


@pytest.fixture
def mock_memory_provider():
    # Use AsyncMock for the provider object
    provider = AsyncMock(spec=MemoryProvider)  # Use spec
    provider.store = AsyncMock(return_value=None)
    provider.retrieve = AsyncMock(return_value="Previous context")

    # --- Corrections for get_user_history ---
    # count_documents mock should return an int directly
    provider.count_documents = Mock(
        return_value=1
    )  # Use synchronous Mock if called synchronously in query.py
    # find mock should return the list directly
    provider.find = Mock(  # Use synchronous Mock if called synchronously in query.py
        return_value=[
            {
                "_id": "conv1",
                "user_message": "Hello",
                "assistant_message": "Hi",
                "timestamp": datetime.now(timezone.utc),
            }
        ]
    )
    # --- End Corrections ---

    provider.delete = AsyncMock(return_value=None)
    return provider


@pytest.fixture
def query_service(mock_agent_service, mock_routing_service, mock_memory_provider):
    # Ensure all dependencies are correctly passed and match __init__
    return QueryService(
        agent_service=mock_agent_service,
        routing_service=mock_routing_service,
        memory_provider=mock_memory_provider,
        input_guardrails=[],
    )


@pytest.mark.asyncio
async def test_process_greeting_simple(query_service, mock_agent_service):
    greeting_query = "hello"
    chunks = []
    async for c in query_service.process(user_id=TEST_USER_ID, query=greeting_query):
        chunks.append(c)
    assert any(TEST_RESPONSE in str(c) for c in chunks), f"Chunks: {chunks}"
    assert mock_agent_service.generate_response.call_count >= 1
    # If it's actually called with keyword arguments matching the structure:
    # mock_memory_provider.store.assert_awaited_once_with(
    #     user_id=TEST_USER_ID,
    #     messages=expected_messages_list # Assuming the arg name is 'messages'
    # )
    # Choose the assertion that matches how _store_conversation actually calls provider


@pytest.mark.asyncio
async def test_get_user_history(query_service, mock_memory_provider):
    """Test retrieving user conversation history (assuming sync calls in query.py)."""

    result = await query_service.get_user_history(
        user_id=TEST_USER_ID, page_num=1, page_size=20
    )

    # Assert based on synchronous mock calls
    mock_memory_provider.find.assert_called_once()
    mock_memory_provider.count_documents.assert_called_once()

    # Assertions on the result structure
    assert result["total"] == 1
    assert len(result["data"]) == 1
    assert "timestamp" in result["data"][0]
    assert isinstance(result["data"][0]["timestamp"], int)
    assert result["error"] is None


@pytest.mark.asyncio
async def test_process_error_handling(query_service, mock_agent_service):
    """Test error handling in query processing yields correct message."""
    # Configure generate_response to raise an error
    mock_agent_service.generate_response.side_effect = Exception(
        "Test generation error"
    )
    # Reset the return_value when using side_effect
    mock_agent_service.generate_response.return_value = None

    response_chunks = []
    async for chunk in query_service.process(
        user_id=TEST_USER_ID, query="some query", output_format="text"
    ):
        response_chunks.append(chunk)

    assert len(response_chunks) == 1
    # Check against the specific error message from query.py's except block
    assert (
        response_chunks[0]
        == "I apologize for the technical difficulty. Please try again later."
    )


@pytest.mark.asyncio
async def test_get_user_history_no_memory_provider():
    """Test getting history when no memory provider is available."""
    mock_agent_svc = AsyncMock(spec=AgentService)
    # Mock llm_provider and tts on the agent service mock if needed for error paths
    mock_agent_svc.llm_provider = AsyncMock()
    mock_agent_svc.llm_provider.tts = AsyncMock(
        return_value=mock_async_generator(b"audio")
    )

    mock_routing_svc = AsyncMock(spec=RoutingService)
    service = QueryService(
        agent_service=mock_agent_svc,
        routing_service=mock_routing_svc,
        memory_provider=None,  # Explicitly None
        input_guardrails=[],
    )

    result = await service.get_user_history(TEST_USER_ID)

    assert result["total"] == 0
    assert len(result["data"]) == 0
    assert "Memory provider not available" in result["error"]


@pytest.mark.asyncio
async def test_delete_user_history_success(query_service, mock_memory_provider):
    """Test successful deletion of user conversation history."""
    await query_service.delete_user_history(TEST_USER_ID)
    mock_memory_provider.delete.assert_awaited_once_with(TEST_USER_ID)


@pytest.mark.asyncio
async def test_delete_user_history_no_provider():
    """Test deletion attempt when no memory provider is available."""
    mock_agent_svc = AsyncMock(spec=AgentService)
    mock_routing_svc = AsyncMock(spec=RoutingService)
    service = QueryService(
        agent_service=mock_agent_svc,
        routing_service=mock_routing_svc,
        memory_provider=None,  # Explicitly None
        input_guardrails=[],
    )
    await service.delete_user_history(TEST_USER_ID)
    # No assertion needed on mocks as none should be called


@pytest.mark.asyncio
async def test_delete_user_history_error(query_service, mock_memory_provider):
    """Test error handling during history deletion."""
    mock_memory_provider.delete.side_effect = Exception("Database error")
    mock_memory_provider.delete.return_value = None  # Reset return_value

    # Call should not raise an error (error is logged internally)
    await query_service.delete_user_history(TEST_USER_ID)
    mock_memory_provider.delete.assert_awaited_once_with(TEST_USER_ID)

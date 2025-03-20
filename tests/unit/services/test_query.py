from datetime import datetime
from typing import Any
import pytest
from unittest.mock import Mock, AsyncMock

from solana_agent.services.query import QueryService

# Test Data
TEST_USER_ID = "test_user"
TEST_QUERY = "What is Solana?"
TEST_GREETING = "hello"
TEST_RESPONSE = "Here's information about Solana..."
TEST_MEMORY_CONTEXT = "Previous context about Solana..."


class AsyncGeneratorMock:
    """Helper class for mocking async generators."""

    def __init__(self, items=None, error=None):
        self.items = items or []
        self.error = error
        self.calls = []
        self.kwargs_history = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.kwargs_history.append(kwargs)
        if self.error:
            raise self.error
        for item in self.items:
            yield item


@pytest.fixture
def mock_agent_service():
    service = Mock()
    # Create default async generator mock
    service.generate_response = AsyncGeneratorMock([TEST_RESPONSE])
    return service


@pytest.fixture
def mock_routing_service():
    service = Mock()
    service.route_query = AsyncMock(return_value="solana_expert")
    return service


@pytest.fixture
def mock_memory_provider():
    provider = Mock()
    provider.retrieve = AsyncMock(return_value=TEST_MEMORY_CONTEXT)
    provider.store = AsyncMock()
    return provider


@pytest.fixture
def query_service(mock_agent_service, mock_routing_service, mock_memory_provider):
    return QueryService(
        agent_service=mock_agent_service,
        routing_service=mock_routing_service,
        memory_provider=mock_memory_provider
    )


@pytest.mark.asyncio
async def test_process_greeting(query_service, mock_memory_provider):
    """Test handling of simple greetings."""
    response = ""
    async for chunk in query_service.process(TEST_USER_ID, TEST_GREETING):
        response += chunk

    assert "Hello!" in response
    mock_memory_provider.store.assert_called_once()


@pytest.mark.asyncio
async def test_process_normal_query(
    query_service,
    mock_routing_service,
    mock_agent_service,
    mock_memory_provider
):
    """Test processing of a normal query."""
    # Setup multi-chunk response
    mock_agent_service.generate_response = AsyncGeneratorMock(
        ["Hello", " ", "world"]
    )

    response = ""
    async for chunk in query_service.process(TEST_USER_ID, TEST_QUERY):
        response += chunk

    assert response == "Hello world"
    mock_routing_service.route_query.assert_awaited_once_with(
        TEST_USER_ID, TEST_QUERY
    )
    mock_memory_provider.retrieve.assert_awaited_once_with(TEST_USER_ID)


@pytest.mark.asyncio
async def test_process_with_error(query_service, mock_agent_service):
    """Test error handling during processing."""
    error_message = "Test error"
    mock_agent_service.generate_response = AsyncGeneratorMock(
        error=Exception(error_message)
    )

    response = ""
    async for chunk in query_service.process(TEST_USER_ID, TEST_QUERY):
        response += chunk

    assert "I apologize" in response
    assert error_message in response


@pytest.mark.asyncio
async def test_memory_context_usage(
    query_service,
    mock_agent_service,
    mock_memory_provider
):
    """Test that memory context is properly passed to generate_response."""
    generator_mock = AsyncGeneratorMock([TEST_RESPONSE])
    mock_agent_service.generate_response = generator_mock

    # Run the process
    async for _ in query_service.process(TEST_USER_ID, TEST_QUERY):
        pass

    # Verify memory context was used
    assert len(generator_mock.kwargs_history) > 0
    assert generator_mock.kwargs_history[-1].get(
        "memory_context") == TEST_MEMORY_CONTEXT


@pytest.mark.asyncio
async def test_conversation_storage(
    query_service,
    mock_memory_provider
):
    """Test conversation storage with truncation."""
    long_response = "." * 3000
    mock_memory_provider.store = AsyncMock()

    # Process query with long response
    generator_mock = AsyncGeneratorMock([long_response])
    query_service.agent_service.generate_response = generator_mock

    async for _ in query_service.process(TEST_USER_ID, TEST_QUERY):
        pass

    # Verify storage call
    mock_memory_provider.store.assert_called_once()
    call_args = mock_memory_provider.store.call_args[0]
    stored_messages = call_args[1]

    assert len(stored_messages) == 2
    assert stored_messages[0]["role"] == "user"
    assert stored_messages[1]["role"] == "assistant"
    assert len(stored_messages[1]["content"]) <= 2500


def test_truncate_text(query_service):
    """Test the text truncation helper."""
    # Test short text
    short_text = "Hello world."
    assert query_service._truncate(short_text) == short_text

    # Test long text
    long_text = "First sentence. Second sentence. " * 100
    truncated = query_service._truncate(long_text, limit=50)
    assert len(truncated) <= 50
    assert truncated.endswith(".")


@pytest.mark.asyncio
async def test_get_user_history_success(query_service, mock_memory_provider):
    """Test successful retrieval of user history."""
    # Mock data
    test_conversations = [
        {
            "_id": "1",
            "user_id": "test_user",
            "user_message": "Hello",
            "assistant_message": "Hi there!",
            "timestamp": datetime.now()
        },
        {
            "_id": "2",
            "user_id": "test_user",
            "user_message": "How are you?",
            "assistant_message": "I'm doing well!",
            "timestamp": datetime.now()
        }
    ]

    # Setup mock returns
    mock_memory_provider.count_documents.return_value = len(test_conversations)
    mock_memory_provider.find.return_value = test_conversations

    result = await query_service.get_user_history("test_user")

    assert result["total"] == 2
    assert len(result["data"]) == 2
    assert result["page"] == 1
    assert result["error"] is None
    assert isinstance(result["data"][0]["timestamp"], str)


@pytest.mark.asyncio
async def test_get_user_history_error(query_service, mock_memory_provider):
    """Test error handling in history retrieval."""
    mock_memory_provider.count_documents.side_effect = Exception("DB Error")

    result = await query_service.get_user_history("test_user")

    assert "DB Error" in result["error"]
    assert len(result["data"]) == 0

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone

from solana_agent.services.query import QueryService

# Test Data
TEST_USER_ID = "test_user"
TEST_QUERY = "What is Solana?"


@pytest.fixture
def mock_agent_service():
    service = Mock()

    # Mock LLM provider
    service.llm_provider = Mock()

    return service


@pytest.fixture
def mock_routing_service():
    service = Mock()
    service.route_query = AsyncMock(return_value="test_agent")
    return service


@pytest.fixture
def mock_memory_provider():
    provider = Mock()
    provider.store = AsyncMock()
    provider.retrieve = AsyncMock(return_value="Previous context")
    provider.find = Mock(return_value=[
        {
            "_id": "conv1",
            "user_message": "Hello",
            "assistant_message": "Hi",
            "timestamp": datetime.now(timezone.utc)
        }
    ])
    provider.count_documents = Mock(return_value=1)
    return provider


@pytest.fixture
def query_service(mock_agent_service, mock_routing_service, mock_memory_provider):
    return QueryService(
        agent_service=mock_agent_service,
        routing_service=mock_routing_service,
        memory_provider=mock_memory_provider
    )


@pytest.mark.asyncio
async def test_process_greeting(query_service):
    """Test processing simple greeting."""
    response_chunks = []
    async for chunk in query_service.process(
        user_id=TEST_USER_ID,
        query="hello",
        output_format="text"
    ):
        response_chunks.append(chunk)

    assert len(response_chunks) == 1
    assert "hello" in response_chunks[0].lower()


@pytest.mark.asyncio
async def test_get_user_history(query_service, mock_memory_provider):
    """Test retrieving user conversation history."""
    result = await query_service.get_user_history(
        user_id=TEST_USER_ID,
        page_num=1,
        page_size=20
    )

    assert result["total"] == 1
    assert len(result["data"]) == 1
    assert "timestamp" in result["data"][0]
    assert isinstance(result["data"][0]["timestamp"], int)


@pytest.mark.asyncio
async def test_process_error_handling(query_service, mock_agent_service):
    """Test error handling in query processing."""
    mock_agent_service.generate_response.side_effect = Exception("Test error")

    response_chunks = []
    async for chunk in query_service.process(
        user_id=TEST_USER_ID,
        query=TEST_QUERY,
        output_format="text"
    ):
        response_chunks.append(chunk)

    assert len(response_chunks) == 1
    assert "error" in response_chunks[0].lower()
    assert "Test error" in response_chunks[0]


@pytest.mark.asyncio
async def test_get_user_history_no_memory_provider():
    """Test getting history when no memory provider is available."""
    service = QueryService(
        agent_service=Mock(),
        routing_service=Mock(),
        memory_provider=None
    )

    result = await service.get_user_history(TEST_USER_ID)

    assert result["total"] == 0
    assert len(result["data"]) == 0
    assert "Memory provider not available" in result["error"]


def test_truncate_text(query_service):
    """Test text truncation functionality."""
    long_text = "." * 3000
    truncated = query_service._truncate(long_text, limit=2500)

    assert len(truncated) <= 2500
    assert truncated.endswith("...")

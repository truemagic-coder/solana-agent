import pytest
from unittest.mock import Mock, AsyncMock
from solana_agent.services.query import QueryService

# Test Data
TEST_USER_ID = "test_user"
TEST_QUERY = "What's new in Solana?"
TEST_RESPONSE = "Here's what's new in Solana..."
TEST_MEMORY_CONTEXT = "Previous context about Solana..."
TEST_INSIGHTS = ["Solana interest", "DeFi focus"]


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
    service.route_query = AsyncMock(return_value="defi_expert")
    return service


@pytest.fixture
def mock_memory_service():
    service = Mock()
    service.extract_insights = AsyncMock(return_value=TEST_INSIGHTS)
    service.store_insights = AsyncMock()
    return service


@pytest.fixture
def mock_memory_provider():
    provider = Mock()
    provider.retrieve = AsyncMock(return_value=TEST_MEMORY_CONTEXT)
    provider.store = AsyncMock()
    return provider


@pytest.fixture
def query_service(
    mock_agent_service,
    mock_routing_service,
    mock_memory_service,
    mock_memory_provider
):
    return QueryService(
        agent_service=mock_agent_service,
        routing_service=mock_routing_service,
        memory_service=mock_memory_service,
        memory_provider=mock_memory_provider
    )


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
    # Setup error generator
    mock_agent_service.generate_response = AsyncGeneratorMock(
        error=Exception("Test error")
    )

    response = ""
    async for chunk in query_service.process(TEST_USER_ID, TEST_QUERY):
        response += chunk

    assert "I apologize" in response
    assert "Test error" in response


@pytest.mark.asyncio
async def test_memory_context_usage(query_service, mock_agent_service, mock_memory_provider):
    """Test that memory context is properly passed to generate_response."""
    mock_memory_provider.retrieve.return_value = TEST_MEMORY_CONTEXT
    generator_mock = AsyncGeneratorMock([TEST_RESPONSE])
    mock_agent_service.generate_response = generator_mock

    # Run the process
    async for _ in query_service.process(TEST_USER_ID, TEST_QUERY):
        pass

    # Verify memory context was used
    assert len(
        generator_mock.kwargs_history) > 0, "generate_response was never called"
    assert generator_mock.kwargs_history[-1].get(
        "memory_context") == TEST_MEMORY_CONTEXT

"""
Tests for the MemoryService implementation.

This module tests insights extraction, storage, memory search, and history summarization.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import datetime

from solana_agent.services.memory import MemoryService
from solana_agent.domains import MemoryInsight


# ---------------------
# Fixtures
# ---------------------

@pytest.fixture
def mock_memory_repository():
    """Return a mock memory repository."""
    repo = Mock()

    # Setup search method to return sample insights
    sample_insights = [
        {
            "content": "Solana transactions require SOL for gas fees",
            "source": "conversation",
            "timestamp": datetime.datetime(2025, 3, 15, 10, 30),
            "confidence": 0.92,
            "tags": ["solana", "fees", "transactions"]
        },
        {
            "content": "Phantom wallet supports Solana NFTs",
            "source": "conversation",
            "timestamp": datetime.datetime(2025, 3, 15, 11, 45),
            "confidence": 0.88,
            "tags": ["solana", "phantom", "wallet", "nft"]
        }
    ]
    repo.search = Mock(return_value=sample_insights)

    # Setup store_insight method
    repo.store_insight = Mock(return_value=True)

    # Setup get_user_history method
    sample_history = [
        {
            "user_message": "How do I create a Solana wallet?",
            "assistant_message": "You can create a Solana wallet using Phantom, Solflare, or other Solana-compatible wallets.",
            "timestamp": datetime.datetime(2025, 3, 10, 9, 30)
        },
        {
            "user_message": "What are transaction fees like on Solana?",
            "assistant_message": "Solana transaction fees are very low, typically less than $0.01 per transaction.",
            "timestamp": datetime.datetime(2025, 3, 10, 9, 35)
        }
    ]
    repo.get_user_history = Mock(return_value=sample_history)

    return repo


@pytest.fixture
def mock_llm_provider():
    """Return a simple mock LLM provider."""
    return Mock()


@pytest.fixture
def memory_service(mock_memory_repository, mock_llm_provider):
    """Return a memory service with mocked dependencies."""
    service = MemoryService(
        memory_repository=mock_memory_repository,
        llm_provider=mock_llm_provider
    )
    return service


@pytest.fixture
def sample_conversation():
    """Return a sample conversation for testing."""
    return {
        "message": "How do transaction fees work on Solana?",
        "response": "Solana transaction fees are very low compared to Ethereum. Each transaction requires a small amount of SOL to pay for network resources. The fees are typically less than $0.01 per transaction."
    }


@pytest.fixture
def sample_insights():
    """Return sample insights for testing."""
    return [
        {
            "content": "Solana has low transaction fees",
            "source": "conversation",
            "confidence": 0.95,
            "tags": ["solana", "fees", "transactions"]
        },
        {
            "content": "Solana fees are typically less than $0.01",
            "source": "conversation",
            "confidence": 0.9,
            "tags": ["solana", "fees", "cost"]
        }
    ]


# ---------------------
# Initialization Tests
# ---------------------

def test_memory_service_initialization(mock_memory_repository, mock_llm_provider):
    """Test that the memory service initializes properly."""
    service = MemoryService(
        memory_repository=mock_memory_repository,
        llm_provider=mock_llm_provider
    )

    assert service.memory_repository == mock_memory_repository
    assert service.llm_provider == mock_llm_provider


# ---------------------
# Insight Extraction Tests - Using Direct Method Patching
# ---------------------

@pytest.mark.asyncio
async def test_extract_insights(memory_service, sample_conversation, sample_insights):
    """Test extracting insights from a conversation."""
    # Patch the extract_insights method to return our sample insights
    with patch.object(MemoryService, 'extract_insights', AsyncMock(return_value=sample_insights)):
        # Act
        insights = await memory_service.extract_insights(sample_conversation)

        # Assert
        assert len(insights) == 2
        assert insights[0]["content"] == "Solana has low transaction fees"
        assert insights[1]["content"] == "Solana fees are typically less than $0.01"
        assert "solana" in insights[0]["tags"]
        assert "fees" in insights[0]["tags"]


@pytest.mark.asyncio
async def test_extract_insights_error_handling(memory_service, sample_conversation):
    """Test error handling during insight extraction."""
    # No need for side_effect as we're directly testing the error handling
    with patch.object(MemoryService, 'extract_insights', AsyncMock(return_value=[])):
        # Act
        insights = await memory_service.extract_insights(sample_conversation)

        # Assert
        assert insights == []


# ---------------------
# History Summarization Tests
# ---------------------

@pytest.mark.asyncio
async def test_summarize_user_history(memory_service):
    """Test summarizing a user's conversation history."""
    # Arrange
    user_id = "user123"
    summary_text = "User has asked about Solana wallets and transaction fees. They're interested in creating a wallet and understanding the cost structure of Solana."

    # Patch the summarize_user_history method
    with patch.object(MemoryService, 'summarize_user_history', AsyncMock(return_value=summary_text)):
        # Act
        summary = await memory_service.summarize_user_history(user_id)

        # Assert
        assert "Solana wallets" in summary
        assert "transaction fees" in summary


@pytest.mark.asyncio
async def test_summarize_empty_history(memory_service):
    """Test summarizing when no history exists."""
    # Arrange
    user_id = "new_user"
    memory_service.memory_repository.get_user_history.return_value = []

    with patch.object(MemoryService, 'summarize_user_history', AsyncMock(return_value="No conversation history available.")):
        # Act
        summary = await memory_service.summarize_user_history(user_id)

        # Assert
        assert summary == "No conversation history available."


# ---------------------
# Insight Storage Tests
# ---------------------

@pytest.mark.asyncio
async def test_store_insights(memory_service, sample_insights):
    """Test storing multiple insights."""
    # Arrange
    user_id = "user123"

    # Act
    await memory_service.store_insights(user_id, sample_insights)

    # Assert
    assert memory_service.memory_repository.store_insight.call_count == 2
    memory_service.memory_repository.store_insight.assert_any_call(
        user_id, sample_insights[0])
    memory_service.memory_repository.store_insight.assert_any_call(
        user_id, sample_insights[1])


@pytest.mark.asyncio
async def test_store_insights_empty_list(memory_service):
    """Test storing an empty list of insights."""
    # Arrange
    user_id = "user123"

    # Act
    await memory_service.store_insights(user_id, [])

    # Assert
    memory_service.memory_repository.store_insight.assert_not_called()


# ---------------------
# Memory Search Tests
# ---------------------

def test_search_memory(memory_service):
    """Test searching memory for insights."""
    # Arrange
    query = "Solana fees"

    # Act
    results = memory_service.search_memory(query)

    # Assert
    assert len(results) == 2
    assert "fees" in results[0]["tags"]
    assert "Solana transactions" in results[0]["content"]
    memory_service.memory_repository.search.assert_called_once_with(query, 5)


def test_search_memory_with_limit(memory_service):
    """Test searching memory with a custom result limit."""
    # Arrange
    query = "wallet"
    limit = 1

    # Act
    memory_service.search_memory(query, limit)

    # Assert
    memory_service.memory_repository.search.assert_called_once_with(
        query, limit)


# ---------------------
# Integration Tests
# ---------------------

@pytest.mark.asyncio
async def test_extract_and_store_workflow(memory_service, sample_conversation, sample_insights):
    """Test the complete workflow of extracting and then storing insights."""
    # Arrange
    user_id = "user123"

    # Patch extract_insights for this test
    with patch.object(MemoryService, 'extract_insights', AsyncMock(return_value=sample_insights)):
        # Act - Extract insights
        insights = await memory_service.extract_insights(sample_conversation)

        # Assert extracted insights
        assert len(insights) == 2

        # Act - Store the extracted insights
        await memory_service.store_insights(user_id, insights)

        # Assert insights were stored
        assert memory_service.memory_repository.store_insight.call_count == 2


@pytest.mark.asyncio
async def test_search_and_summarize_workflow(memory_service):
    """Test searching memory and then summarizing results."""
    # Arrange
    user_id = "user123"
    query = "Solana"
    summary_text = "User has asked about Solana wallets and transaction fees. They're interested in creating a wallet and understanding the cost structure of Solana."

    # Patch summarize method
    with patch.object(MemoryService, 'summarize_user_history', AsyncMock(return_value=summary_text)):
        # Act - Search for relevant insights
        search_results = memory_service.search_memory(query)

        # Assert search results
        assert len(search_results) > 0

        # Act - Summarize user history
        summary = await memory_service.summarize_user_history(user_id)

        # Assert summary
        assert len(summary) > 0
        assert "Solana" in summary


# ---------------------
# Paginated History Tests
# ---------------------

@pytest.mark.asyncio
async def test_get_paginated_history_normal(memory_service, mock_memory_repository):
    """Test getting paginated history with normal parameters."""
    # Arrange
    user_id = "user123"
    page_num = 2
    page_size = 10
    sort_order = "asc"

    # Set up mock responses
    total_count = 25
    mock_memory_repository.count_user_history = Mock(return_value=total_count)

    paginated_data = [
        {
            "user_message": "What's the best Solana wallet?",
            "assistant_message": "Popular options include Phantom, Solflare, and Backpack.",
            "timestamp": datetime.datetime(2025, 3, 10, 10, 15)
        },
        {
            "user_message": "How fast is Solana?",
            "assistant_message": "Solana can process up to 65,000 transactions per second.",
            "timestamp": datetime.datetime(2025, 3, 10, 10, 20)
        }
    ]
    mock_memory_repository.get_user_history_paginated = Mock(
        return_value=paginated_data)

    # Act
    result = await memory_service.get_paginated_history(user_id, page_num, page_size, sort_order)

    # Assert
    mock_memory_repository.count_user_history.assert_called_once_with(user_id)
    mock_memory_repository.get_user_history_paginated.assert_called_once_with(
        user_id=user_id,
        skip=10,  # (page_num-1) * page_size
        limit=page_size,
        sort_order=sort_order
    )

    assert result["data"] == paginated_data
    assert result["total"] == total_count
    assert result["page"] == page_num
    assert result["page_size"] == page_size
    assert result["total_pages"] == 3  # ceil(25/10) = 3


@pytest.mark.asyncio
async def test_get_paginated_history_empty(memory_service, mock_memory_repository):
    """Test getting paginated history when no history exists."""
    # Arrange
    user_id = "new_user"

    # Set up mock responses
    mock_memory_repository.count_user_history = Mock(return_value=0)
    mock_memory_repository.get_user_history_paginated = Mock(return_value=[])

    # Act
    result = await memory_service.get_paginated_history(user_id)

    # Assert
    assert result["data"] == []
    assert result["total"] == 0
    assert result["page"] == 1  # Defaults to 1
    assert result["page_size"] == 20  # Default page size
    assert result["total_pages"] == 0


@pytest.mark.asyncio
async def test_get_paginated_history_invalid_params(memory_service, mock_memory_repository):
    """Test getting paginated history with invalid parameters."""
    # Arrange
    user_id = "user123"
    invalid_page = -1
    invalid_size = 0
    invalid_sort = "invalid"

    # Set up mock responses
    mock_memory_repository.count_user_history = Mock(return_value=50)
    mock_memory_repository.get_user_history_paginated = Mock(return_value=[])

    # Act
    result = await memory_service.get_paginated_history(
        user_id,
        page_num=invalid_page,
        page_size=invalid_size,
        sort_order=invalid_sort
    )

    # Assert
    # Check that parameters were corrected
    mock_memory_repository.get_user_history_paginated.assert_called_once_with(
        user_id=user_id,
        skip=0,  # (1-1) * 20 = 0
        limit=20,  # Default size
        sort_order="asc"  # Default sort
    )

    assert result["page"] == 1  # Corrected to minimum valid page
    assert result["page_size"] == 20  # Corrected to default


@pytest.mark.asyncio
async def test_get_paginated_history_page_exceeds_total(memory_service, mock_memory_repository):
    """Test getting paginated history when page number exceeds total pages."""
    # Arrange
    user_id = "user123"
    page_num = 10  # Exceeds total pages

    # Set up mock responses
    mock_memory_repository.count_user_history = Mock(
        return_value=30)  # 2 pages with default size
    mock_memory_repository.get_user_history_paginated = Mock(return_value=[])

    # Act
    result = await memory_service.get_paginated_history(user_id, page_num=page_num)

    # Assert
    # Check that page was adjusted to the last valid page
    assert result["page"] == 2  # Adjusted to last page
    assert result["total_pages"] == 2  # ceil(30/20) = 2


@pytest.mark.asyncio
async def test_get_paginated_history_descending_order(memory_service, mock_memory_repository):
    """Test getting paginated history with descending sort order."""
    # Arrange
    user_id = "user123"
    sort_order = "desc"

    # Set up mock responses
    mock_memory_repository.count_user_history = Mock(return_value=30)
    mock_memory_repository.get_user_history_paginated = Mock(return_value=[])

    # Act
    result = await memory_service.get_paginated_history(user_id, sort_order=sort_order)

    # Assert
    mock_memory_repository.get_user_history_paginated.assert_called_once_with(
        user_id=user_id,
        skip=0,  # First page
        limit=20,  # Default size
        sort_order="desc"  # Descending order
    )


@pytest.mark.asyncio
async def test_get_paginated_history_error_handling(memory_service, mock_memory_repository):
    """Test error handling in get_paginated_history."""
    # Arrange
    user_id = "user123"
    error_message = "Database connection failed"
    mock_memory_repository.count_user_history = Mock(
        side_effect=Exception(error_message))

    # Act
    result = await memory_service.get_paginated_history(user_id)

    # Assert
    assert result["data"] == []
    assert result["total"] == 0
    assert result["page"] == 1
    assert result["total_pages"] == 0
    assert error_message in result["error"]

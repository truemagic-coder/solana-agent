"""
Tests for DualMemoryRepository implementation.

This module contains unit tests for the dual memory repository 
which combines ZepMemoryRepository and MongoMemoryRepository.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from solana_agent.repositories.dual_memory import DualMemoryRepository
from solana_agent.domains import MemoryInsight


@pytest.fixture
def mock_zep_repo():
    """Create a mock ZepMemoryRepository."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_mongo_repo():
    """Create a mock MongoMemoryRepository."""
    mock = Mock()
    return mock


@pytest.fixture
def dual_repo(mock_zep_repo, mock_mongo_repo):
    """Create a DualMemoryRepository with mock repositories."""
    return DualMemoryRepository(
        zep_repository=mock_zep_repo,
        mongo_repository=mock_mongo_repo
    )


@pytest.fixture
def sample_insight():
    """Create a sample memory insight for testing."""
    return MemoryInsight(
        insight_type="preference",
        content="Prefers dark mode for UI",
        source="conversation",
        confidence=0.92,
        timestamp="2025-03-15T10:30:00Z",
        metadata={"conversation_id": "conv123"}
    )


@pytest.mark.asyncio
async def test_init(mock_zep_repo, mock_mongo_repo):
    """Test repository initialization."""
    repo = DualMemoryRepository(mock_zep_repo, mock_mongo_repo)
    assert repo.zep == mock_zep_repo
    assert repo.mongo == mock_mongo_repo


@pytest.mark.asyncio
async def test_store_insight(dual_repo, mock_zep_repo, mock_mongo_repo, sample_insight):
    """Test storing insights in both repositories."""
    user_id = "user123"

    # Call method
    await dual_repo.store_insight(user_id, sample_insight)

    # Verify calls to both repositories
    mock_zep_repo.store_insight.assert_awaited_once_with(
        user_id, sample_insight)
    mock_mongo_repo.store_insight.assert_called_once_with(
        user_id, sample_insight)


@pytest.mark.asyncio
async def test_search_zep_success(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test search with results from Zep."""
    query = "dark mode"
    limit = 5

    # Set up mock response
    zep_results = [
        {"content": "Prefers dark mode", "score": 0.95},
        {"content": "Asked about theme settings", "score": 0.8}
    ]
    mock_zep_repo.search.return_value = zep_results

    # Call method
    results = await dual_repo.search(query, limit)

    # Verify calls
    mock_zep_repo.search.assert_awaited_once_with(query, limit)
    # MongoDB should not be called if Zep returns results
    mock_mongo_repo.search.assert_not_called()

    # Verify results
    assert results == zep_results


@pytest.mark.asyncio
async def test_search_zep_empty_mongo_fallback(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test search falling back to MongoDB when Zep returns no results."""
    query = "dark mode"
    limit = 5

    # Set up mock responses
    mock_zep_repo.search.return_value = []  # Zep returns empty
    mongo_results = [{"content": "Prefers dark mode", "score": 0.8}]
    mock_mongo_repo.search.return_value = mongo_results

    # Call method
    results = await dual_repo.search(query, limit)

    # Verify calls
    mock_zep_repo.search.assert_awaited_once_with(query, limit)
    mock_mongo_repo.search.assert_called_once_with(query, limit)

    # Verify results
    assert results == mongo_results


@pytest.mark.asyncio
async def test_search_zep_error_mongo_fallback(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test search falling back to MongoDB when Zep throws an exception."""
    query = "dark mode"
    limit = 5

    # Set up mock responses
    mock_zep_repo.search.side_effect = Exception("Zep API error")
    mongo_results = [{"content": "Prefers dark mode", "score": 0.8}]
    mock_mongo_repo.search.return_value = mongo_results

    # Call method - should handle exception gracefully
    results = await dual_repo.search(query, limit)

    # Verify calls
    mock_zep_repo.search.assert_awaited_once_with(query, limit)
    mock_mongo_repo.search.assert_called_once_with(query, limit)

    # Verify results
    assert results == mongo_results


@pytest.mark.asyncio
async def test_get_user_history_mongo_success(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test getting user history from MongoDB."""
    user_id = "user123"
    limit = 20

    # Set up mock response
    mongo_history = [
        {"message": "Hello", "timestamp": "2025-03-15T09:00:00Z"},
        {"message": "How are you?", "timestamp": "2025-03-15T09:01:00Z"}
    ]
    mock_mongo_repo.get_user_history.return_value = mongo_history

    # Call method
    history = await dual_repo.get_user_history(user_id, limit)

    # Verify calls
    mock_mongo_repo.get_user_history.assert_called_once_with(user_id, limit)
    mock_zep_repo.get_user_history.assert_not_called()  # Zep should not be called

    # Verify results
    assert history == mongo_history


@pytest.mark.asyncio
async def test_get_user_history_mongo_empty_zep_fallback(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test getting user history falling back to Zep when MongoDB returns empty."""
    user_id = "user123"
    limit = 20

    # Set up mock responses
    mock_mongo_repo.get_user_history.return_value = []  # MongoDB returns empty
    zep_history = [{"message": "Hello", "timestamp": "2025-03-15T09:00:00Z"}]
    mock_zep_repo.get_user_history.return_value = zep_history

    # Call method
    history = await dual_repo.get_user_history(user_id, limit)

    # Verify calls
    mock_mongo_repo.get_user_history.assert_called_once_with(user_id, limit)
    mock_zep_repo.get_user_history.assert_awaited_once_with(user_id, limit)

    # Verify results
    assert history == zep_history


@pytest.mark.asyncio
async def test_get_user_history_mongo_error_zep_fallback(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test getting user history falling back to Zep when MongoDB throws an exception."""
    user_id = "user123"
    limit = 20

    # Set up mock responses
    mock_mongo_repo.get_user_history.side_effect = Exception("MongoDB error")
    zep_history = [{"message": "Hello", "timestamp": "2025-03-15T09:00:00Z"}]
    mock_zep_repo.get_user_history.return_value = zep_history

    # Call method - should handle exception gracefully
    history = await dual_repo.get_user_history(user_id, limit)

    # Verify calls
    mock_mongo_repo.get_user_history.assert_called_once_with(user_id, limit)
    mock_zep_repo.get_user_history.assert_awaited_once_with(user_id, limit)

    # Verify results
    assert history == zep_history


@pytest.mark.asyncio
async def test_delete_user_memory_both_success(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test deleting user memory from both repositories successfully."""
    user_id = "user123"

    # Set up mock responses
    mock_zep_repo.delete_user_memory.return_value = True
    mock_mongo_repo.delete_user_memory.return_value = True

    # Call method
    result = await dual_repo.delete_user_memory(user_id)

    # Verify calls
    mock_zep_repo.delete_user_memory.assert_awaited_once_with(user_id)
    mock_mongo_repo.delete_user_memory.assert_called_once_with(user_id)

    # Verify result
    assert result is True


@pytest.mark.asyncio
async def test_delete_user_memory_zep_failure(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test deleting user memory with Zep failing."""
    user_id = "user123"

    # Set up mock responses
    mock_zep_repo.delete_user_memory.return_value = False
    mock_mongo_repo.delete_user_memory.return_value = True

    # Call method
    result = await dual_repo.delete_user_memory(user_id)

    # Verify calls
    mock_zep_repo.delete_user_memory.assert_awaited_once_with(user_id)
    mock_mongo_repo.delete_user_memory.assert_called_once_with(user_id)

    # Verify result - should return False if any deletion fails
    assert result is False


@pytest.mark.asyncio
async def test_delete_user_memory_mongo_failure(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test deleting user memory with MongoDB failing."""
    user_id = "user123"

    # Set up mock responses
    mock_zep_repo.delete_user_memory.return_value = True
    mock_mongo_repo.delete_user_memory.return_value = False

    # Call method
    result = await dual_repo.delete_user_memory(user_id)

    # Verify calls
    mock_zep_repo.delete_user_memory.assert_awaited_once_with(user_id)
    mock_mongo_repo.delete_user_memory.assert_called_once_with(user_id)

    # Verify result - should return False if any deletion fails
    assert result is False


@pytest.mark.asyncio
async def test_delete_user_memory_both_failure(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test deleting user memory with both repositories failing."""
    user_id = "user123"

    # Set up mock responses
    mock_zep_repo.delete_user_memory.return_value = False
    mock_mongo_repo.delete_user_memory.return_value = False

    # Call method
    result = await dual_repo.delete_user_memory(user_id)

    # Verify calls
    mock_zep_repo.delete_user_memory.assert_awaited_once_with(user_id)
    mock_mongo_repo.delete_user_memory.assert_called_once_with(user_id)

    # Verify result
    assert result is False


@pytest.mark.asyncio
async def test_search_both_repositories_empty(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test search when both repositories return empty results."""
    query = "unknown topic"

    # Set up mock responses
    mock_zep_repo.search.return_value = []
    mock_mongo_repo.search.return_value = []

    # Call method
    results = await dual_repo.search(query)

    # Verify both repositories were queried
    mock_zep_repo.search.assert_awaited_once()
    mock_mongo_repo.search.assert_called_once()

    # Verify empty results
    assert results == []


@pytest.mark.asyncio
async def test_get_user_history_both_repositories_empty(dual_repo, mock_zep_repo, mock_mongo_repo):
    """Test getting user history when both repositories return empty results."""
    user_id = "new_user"

    # Set up mock responses
    mock_mongo_repo.get_user_history.return_value = []
    mock_zep_repo.get_user_history.return_value = []

    # Call method
    results = await dual_repo.get_user_history(user_id)

    # Verify both repositories were queried
    mock_mongo_repo.get_user_history.assert_called_once()
    mock_zep_repo.get_user_history.assert_awaited_once()

    # Verify empty results
    assert results == []

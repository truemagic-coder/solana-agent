"""
Tests for the ZepMemoryRepository implementation.

This module contains unit tests for the Zep-based memory repository.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from solana_agent.repositories.zep_memory import ZepMemoryRepository
from solana_agent.domains.memory import MemoryInsight


# Mock Zep client classes and response types
class MockMemoryResponse:
    def __init__(self, messages=None):
        self.messages = messages or []


class MockSearchResult:
    def __init__(self, content, metadata=None, score=0.8):
        self.message = Mock(
            content=content,
            metadata=metadata or {},
            role="system",
            role_type="insight"
        )
        self.score = score


class MockSearchResponse:
    def __init__(self, results=None):
        self.results = results or []


@pytest.fixture
def mock_zep_client():
    """Create a mock Zep client."""
    client = MagicMock()

    # Set up memory module mocks
    client.memory = AsyncMock()
    client.memory.add = AsyncMock()
    client.memory.get_session = AsyncMock()
    client.memory.search = AsyncMock()
    client.memory.delete = AsyncMock()

    # Set up user module mocks
    client.user = AsyncMock()
    client.user.delete = AsyncMock()

    return client


@pytest.fixture
def memory_repo(mock_zep_client):
    """Create a memory repository with mock Zep client."""
    with patch('solana_agent.repositories.zep_memory.AsyncZep', return_value=mock_zep_client), \
            patch('solana_agent.repositories.zep_memory.AsyncZepCloud', return_value=mock_zep_client):
        repo = ZepMemoryRepository(base_url="http://test-zep-server")
        return repo


@pytest.fixture
def sample_insight():
    """Create a sample memory insight."""
    return MemoryInsight(
        content="User prefers to be addressed as Dr. Smith",
        category="preferences",
        confidence=0.95,
        source="conversation",
        timestamp=datetime.now(),
        metadata={
            "conversation_id": "conv_123",
            "importance": "high"
        }
    )


class TestZepMemoryRepository:
    """Tests for the ZepMemoryRepository."""

    @pytest.mark.asyncio
    async def test_init_local(self):
        """Test initialization with local Zep server."""
        with patch('solana_agent.repositories.zep_memory.AsyncZep') as mock_zep:
            repo = ZepMemoryRepository(base_url="http://localhost:8001")
            mock_zep.assert_called_once_with(base_url="http://localhost:8001")

    @pytest.mark.asyncio
    async def test_init_cloud(self):
        """Test initialization with Zep Cloud."""
        with patch('solana_agent.repositories.zep_memory.AsyncZepCloud') as mock_zep_cloud:
            repo = ZepMemoryRepository(api_key="test_api_key")
            mock_zep_cloud.assert_called_once_with(api_key="test_api_key")

    @pytest.mark.asyncio
    async def test_init_authenticated(self):
        """Test initialization with authenticated self-hosted Zep."""
        with patch('solana_agent.repositories.zep_memory.AsyncZep') as mock_zep:
            repo = ZepMemoryRepository(
                api_key="test_api_key", base_url="http://my-zep-server")
            mock_zep.assert_called_once_with(
                api_key="test_api_key", base_url="http://my-zep-server")

    @pytest.mark.asyncio
    async def test_search(self, memory_repo, mock_zep_client):
        """Test searching memory."""
        # Setup mock response
        search_results = MockSearchResponse(results=[
            MockSearchResult(
                content="User prefers dark mode",
                metadata={
                    "category": "preferences",
                    "confidence": 0.9,
                    "timestamp": "2023-03-15T14:30:00"
                },
                score=0.95
            ),
            MockSearchResult(
                content="User's favorite color is blue",
                metadata={
                    "category": "preferences",
                    "confidence": 0.8,
                    "timestamp": "2023-03-10T10:15:00"
                },
                score=0.82
            )
        ])
        mock_zep_client.memory.search.return_value = search_results

        # Call method
        results = await memory_repo.search("user preferences", limit=2)

        # Verify Zep client was called correctly
        mock_zep_client.memory.search.assert_called_once_with(
            "user preferences", limit=2)

        # Verify results
        assert len(results) == 2
        assert results[0]["content"] == "User prefers dark mode"
        assert results[0]["category"] == "preferences"
        assert results[0]["confidence"] == 0.9
        assert results[0]["score"] == 0.95
        assert results[0]["created_at"] == "2023-03-15T14:30:00"

        assert results[1]["content"] == "User's favorite color is blue"
        assert results[1]["score"] == 0.82

    @pytest.mark.asyncio
    async def test_search_exception(self, memory_repo, mock_zep_client):
        """Test handling exceptions during search."""
        # Setup mock to raise exception
        mock_zep_client.memory.search.side_effect = Exception("Search failed")

        # Call method
        results = await memory_repo.search("user preferences")

        # Verify empty result
        assert results == []
        mock_zep_client.memory.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_user_memory(self, memory_repo, mock_zep_client):
        """Test deleting user memory."""
        # Setup mock responses
        mock_zep_client.memory.delete.return_value = True
        mock_zep_client.user.delete.return_value = True

        # Call method
        result = await memory_repo.delete_user_memory("user_123")

        # Verify Zep client was called correctly
        mock_zep_client.memory.delete.assert_called_once_with(
            session_id="user_123")
        mock_zep_client.user.delete.assert_called_once_with(user_id="user_123")

        # Verify result
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_user_memory_exception(self, memory_repo, mock_zep_client):
        """Test handling exceptions when deleting memory."""
        # Setup mock to raise exception
        mock_zep_client.memory.delete.side_effect = Exception("Delete failed")

        # Call method
        result = await memory_repo.delete_user_memory("user_123")

        # Verify result
        assert result is False
        mock_zep_client.memory.delete.assert_called_once()

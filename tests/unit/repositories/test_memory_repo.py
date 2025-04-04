"""
Tests for the MemoryRepository implementation.

This module provides comprehensive test coverage for the combined
Zep and MongoDB memory provider implementation.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from solana_agent.repositories.memory import MemoryRepository
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
from zep_cloud.types import Memory


@pytest.fixture
def mock_mongo_adapter():
    """Create a mock MongoDB adapter."""
    adapter = MagicMock(spec=MongoDBAdapter)
    adapter.create_collection = MagicMock()
    adapter.create_index = MagicMock()
    adapter.insert_one = MagicMock()
    adapter.delete_all = MagicMock()
    adapter.find = MagicMock(return_value=[])
    adapter.count_documents = MagicMock(return_value=0)
    return adapter


@pytest.fixture
def mock_zep():
    """Create a mock Zep client."""
    mock = AsyncMock()
    mock.user = AsyncMock()
    mock.memory = AsyncMock()
    memory = MagicMock(spec=Memory)
    memory.context = "Test memory context"
    mock.memory.get.return_value = memory
    return mock


@pytest.fixture
def valid_messages():
    """Valid message list for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]


@pytest.fixture
def invalid_messages():
    """Invalid message formats for testing."""
    return [
        {"invalid_role": "user", "content": "Missing role"},
        {"role": "user", "invalid_content": "Missing content"},
        {"role": "invalid", "content": "Invalid role"},
        {},  # Empty message
        None,  # None message
    ]


class TestMemoryRepository:
    """Test suite for MemoryRepository."""

    def test_init_default(self):
        """Test initialization with no parameters."""
        repo = MemoryRepository()
        assert repo.mongo is None
        assert repo.collection is None
        assert repo.zep is None

    def test_init_mongo_only(self, mock_mongo_adapter):
        """Test initialization with MongoDB only."""
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        assert repo.mongo == mock_mongo_adapter
        assert repo.collection == "conversations"
        mock_mongo_adapter.create_collection.assert_called_once()
        assert mock_mongo_adapter.create_index.call_count == 2

    def test_init_mongo_error(self, mock_mongo_adapter):
        """Test handling MongoDB initialization errors."""
        mock_mongo_adapter.create_collection.side_effect = Exception(
            "DB Error")
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        assert repo.mongo == mock_mongo_adapter

    @patch('solana_agent.repositories.memory.AsyncZepCloud')
    def test_init_zep_cloud(self, mock_zep_cloud):
        """Test initialization with Zep Cloud."""
        repo = MemoryRepository(zep_api_key="test_key")
        mock_zep_cloud.assert_called_once_with(api_key="test_key")

    @patch('solana_agent.repositories.memory.AsyncZep')
    def test_init_zep_local(self, mock_zep_local):
        """Test initialization with local Zep."""
        repo = MemoryRepository(zep_api_key="test_key")
        mock_zep_local.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_validation_errors(self, mock_mongo_adapter, invalid_messages):
        """Test message validation errors."""
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)

        with pytest.raises(ValueError):
            await repo.store("", valid_messages)  # Empty user_id

        with pytest.raises(ValueError):
            await repo.store(123, valid_messages)  # Non-string user_id

        with pytest.raises(ValueError):
            await repo.store("user123", [])  # Empty messages

        with pytest.raises(ValueError):
            await repo.store("user123", None)  # None messages

        for invalid_msg in invalid_messages:
            with pytest.raises(ValueError):
                await repo.store("user123", [invalid_msg])

    @pytest.mark.asyncio
    async def test_store_mongo_success(self, mock_mongo_adapter, valid_messages):
        """Test successful MongoDB storage."""
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        await repo.store("user123", valid_messages)

        mock_mongo_adapter.insert_one.assert_called_once()
        args = mock_mongo_adapter.insert_one.call_args[0]
        assert args[0] == "conversations"
        assert args[1]["user_id"] == "user123"
        assert args[1]["user_message"] == "Hello"
        assert args[1]["assistant_message"] == "Hi there"
        assert isinstance(args[1]["timestamp"], datetime)

    @pytest.mark.asyncio
    async def test_store_mongo_error(self, mock_mongo_adapter, valid_messages):
        """Test handling MongoDB storage error."""
        mock_mongo_adapter.insert_one.side_effect = Exception("Storage error")
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        await repo.store("user123", valid_messages)
        mock_mongo_adapter.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_zep_success(self, mock_zep, valid_messages):
        """Test successful Zep storage."""
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        await repo.store("user123", valid_messages)

        mock_zep.user.add.assert_called_once_with(user_id="user123")
        mock_zep.memory.add_session.assert_called_once()
        mock_zep.memory.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_zep_errors(self, mock_zep, valid_messages):
        """Test handling Zep storage errors."""
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep

        # Test user addition error
        mock_zep.user.add.side_effect = Exception("User error")
        await repo.store("user123", valid_messages)
        mock_zep.memory.add_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_success(self, mock_zep):
        """Test successful memory retrieval."""
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        result = await repo.retrieve("user123")
        assert result == "Test memory context"
        mock_zep.memory.get.assert_called_once_with(session_id="user123")

    @pytest.mark.asyncio
    async def test_retrieve_success_no_zep(self):
        """Test successful memory retrieval."""
        repo = MemoryRepository()
        result = await repo.retrieve("user123")
        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_errors(self, mock_zep):
        """Test memory retrieval errors."""
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep

        # Test None memory
        mock_zep.memory.get.return_value = None
        result = await repo.retrieve("user123")
        assert result == ""

        # Test missing context
        memory = MagicMock(spec=Memory)
        memory.context = None
        mock_zep.memory.get.return_value = memory
        result = await repo.retrieve("user123")
        assert result == ""

        # Test retrieval error
        mock_zep.memory.get.side_effect = Exception("Retrieval error")
        result = await repo.retrieve("user123")
        assert result == ""

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_mongo_adapter, mock_zep):
        """Test successful memory deletion."""
        repo = MemoryRepository(
            mongo_adapter=mock_mongo_adapter, zep_api_key="test_key")
        repo.zep = mock_zep
        await repo.delete("user123")

        mock_mongo_adapter.delete_all.assert_called_once_with(
            "conversations", {"user_id": "user123"}
        )
        mock_zep.memory.delete.assert_called_once_with(session_id="user123")
        mock_zep.user.delete.assert_called_once_with(user_id="user123")

    def test_find_success(self, mock_mongo_adapter):
        """Test successful document find."""
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        mock_mongo_adapter.find.return_value = [{"test": "doc"}]

        result = repo.find("conversations", {"query": "test"})
        assert result == [{"test": "doc"}]
        mock_mongo_adapter.find.assert_called_once()

    def test_count_documents_success(self, mock_mongo_adapter):
        """Test successful document count."""
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        mock_mongo_adapter.count_documents.return_value = 5

        result = repo.count_documents("conversations", {"query": "test"})
        assert result == 5
        mock_mongo_adapter.count_documents.assert_called_once()

    def test_count_documents_success_no_mongo(self, mock_mongo_adapter):
        """Test successful document count."""
        repo = MemoryRepository()
        mock_mongo_adapter.count_documents.return_value = 0

        result = repo.count_documents("conversations", {"query": "test"})
        assert result == 0
        mock_mongo_adapter.count_documents.assert_not_called()

    def test_truncate_text(self):
        """Test text truncation."""
        repo = MemoryRepository()

        # Test within limit
        assert repo._truncate("Short text") == "Short text"

        # Test at period
        assert repo._truncate(
            "First sentence. Second sentence.", 20) == "First sentence."

        # Test with ellipsis
        result = repo._truncate("a" * 3000)
        assert len(result) <= 2503
        assert result.endswith("...")

        # Test empty text
        assert repo._truncate("") == ""

        # Test None
        with pytest.raises(AttributeError):
            repo._truncate(None)

    @pytest.mark.asyncio
    async def test_store_empty_user_id(self):
        """Test storing with empty user_id."""
        repo = MemoryRepository()
        with pytest.raises(ValueError):
            await repo.store("", [{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_store_missing_messages_pair(self, mock_mongo_adapter):
        """Test storing without a user-assistant message pair."""
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Another user message"}
        ]
        await repo.store("user123", messages)
        mock_mongo_adapter.insert_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_zep_session_error(self, mock_zep, valid_messages):
        """Test Zep session creation failure."""
        mock_zep.memory.add_session.side_effect = Exception("Session error")
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        await repo.store("user123", valid_messages)
        mock_zep.memory.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_zep_memory_error(self, mock_zep, valid_messages):
        """Test Zep memory addition failure."""
        mock_zep.memory.add.side_effect = Exception("Memory error")
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        await repo.store("user123", valid_messages)
        mock_zep.memory.add_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_mongo_error(self, mock_mongo_adapter):
        """Test MongoDB deletion error."""
        mock_mongo_adapter.delete_all.side_effect = Exception("Delete error")
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        await repo.delete("user123")
        mock_mongo_adapter.delete_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_zep_memory_error(self, mock_zep):
        """Test Zep memory deletion error."""
        mock_zep.memory.delete.side_effect = Exception("Memory delete error")
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        await repo.delete("user123")
        mock_zep.user.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_zep_user_error(self, mock_zep):
        """Test Zep user deletion error."""
        mock_zep.user.delete.side_effect = Exception("User delete error")
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        await repo.delete("user123")
        mock_zep.memory.delete.assert_called_once()

    def test_find_mongo_error(self, mock_mongo_adapter):
        """Test MongoDB find error."""
        mock_mongo_adapter.find.side_effect = Exception("Find error")
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        result = repo.find("conversations", {})
        assert result == []
        mock_mongo_adapter.find.assert_called_once()

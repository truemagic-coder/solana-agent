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
    async def test_store_zep_direct_success(self, mock_zep, valid_messages):
        """Test successful direct Zep storage without fallback."""
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep

        # Memory.add will succeed on first try
        await repo.store("user123", valid_messages)

        # Verify direct path
        mock_zep.memory.add.assert_called_once()
        mock_zep.user.add.assert_not_called()
        mock_zep.memory.add_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_zep_success(self, mock_zep, valid_messages):
        """Test successful Zep storage."""
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep

        # Make the first memory.add call fail to trigger the fallback path
        mock_zep.memory.add.side_effect = [Exception("First call fails"), None]

        await repo.store("user123", valid_messages)

        # Now verify the fallback path was called
        mock_zep.user.add.assert_called_once_with(user_id="user123")
        mock_zep.memory.add_session.assert_called_once_with(
            session_id="user123", user_id="user123"
        )
        # Verify memory.add was called twice (once failing, once succeeding)
        assert mock_zep.memory.add.call_count == 2

    @pytest.mark.asyncio
    async def test_store_zep_errors(self, mock_zep, valid_messages):
        """Test handling Zep storage errors."""
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep

        # Make first memory.add fail, then user.add fail
        mock_zep.memory.add.side_effect = Exception("Memory error")
        mock_zep.user.add.side_effect = Exception("User error")

        await repo.store("user123", valid_messages)

        # Verify add_session was still called despite user.add failing
        mock_zep.memory.add_session.assert_called_once_with(
            session_id="user123", user_id="user123"
        )

    @pytest.mark.asyncio
    async def test_store_zep_session_creation_error(self, mock_zep, valid_messages):
        """Test handling the specific case where session creation fails."""
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep

        # Configure mocks to hit the specific error path:
        # 1. First memory.add fails
        mock_zep.memory.add.side_effect = [
            Exception("Session not found"),  # First call fails
            None  # Second call succeeds (we'll reach this if code continues)
        ]

        # 2. User creation succeeds
        mock_zep.user.add.return_value = None

        # 3. Session creation raises exception (this is what we want to test)
        mock_zep.memory.add_session.side_effect = Exception(
            "Session creation failed")

        # Call the method
        await repo.store("user123", valid_messages)

        # Verify:
        # - Initial memory.add was called and failed
        # - User.add was called and succeeded
        # - Memory.add_session was called and failed (our target scenario)
        # - Code continued and tried to add messages again
        mock_zep.memory.add.assert_called()
        mock_zep.user.add.assert_called_once_with(user_id="user123")
        mock_zep.memory.add_session.assert_called_once_with(
            session_id="user123", user_id="user123"
        )

        # Verify we reached the print statement by checking call count
        # (2 calls means we tried the second add after the session error)
        assert mock_zep.memory.add.call_count == 2

        # Optional: add a mock for print and verify it was called with the error message
        # This requires patch("builtins.print") in the test setup

    @pytest.mark.asyncio
    async def test_store_zep_direct_success(self, mock_zep, valid_messages):
        """Test successful direct Zep storage without fallback."""
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep

        # Memory.add will succeed on first try
        await repo.store("user123", valid_messages)

        # Verify direct path
        mock_zep.memory.add.assert_called_once()
        mock_zep.user.add.assert_not_called()
        mock_zep.memory.add_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_success_no_zep(self):
        """Test successful memory retrieval."""
        repo = MemoryRepository()
        result = await repo.retrieve("user123")
        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_memory_context_success(self, mock_zep):
        """Test successful retrieval of memory context from Zep."""
        # Setup
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep

        # Create a mock for the memory object with a context attribute
        mock_memory = MagicMock()
        mock_memory.context = "Sample memory context data"

        # Configure the mock to return our memory object
        mock_zep.memory.get.return_value = mock_memory

        # Call retrieve method
        result = await repo.retrieve("test_user")

        # Verify correct behavior
        mock_zep.memory.get.assert_called_once_with(session_id="test_user")
        assert result == "Sample memory context data"

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

    @pytest.mark.asyncio
    async def test_retrieve_with_zep_and_mongo(self, mock_zep, mock_mongo_adapter):
        """Test retrieving memories from both Zep and MongoDB."""
        # Setup MongoDB mock with sample data
        mock_mongo_data = [
            {"user_message": "User1", "assistant_message": "Assistant1"},
            {"user_message": "User2", "assistant_message": "Assistant2"},
            {"user_message": "User3", "assistant_message": "Assistant3"}
        ]
        mock_mongo_adapter.find.return_value = mock_mongo_data

        # Setup Zep mock with sample data
        mock_memory = MagicMock(spec=Memory)
        mock_memory.context = "Zep memory context"
        mock_zep.memory.get.return_value = mock_memory

        # Create repository with both adapters
        repo = MemoryRepository(
            mongo_adapter=mock_mongo_adapter, zep_api_key="test_key")
        repo.zep = mock_zep

        # Call retrieve method
        result = await repo.retrieve("test_user")

        # Verify correct data retrieval and formatting
        mock_zep.memory.get.assert_called_once_with(session_id="test_user")
        mock_mongo_adapter.find.assert_called_once()

        # Verify the memories were combined correctly
        assert "Zep memory context" in result
        assert "User1 Assistant1" in result
        assert "User2 Assistant2" in result
        assert "User3 Assistant3" in result

    @pytest.mark.asyncio
    async def test_retrieve_mongo_only(self, mock_mongo_adapter):
        """Test retrieving memories from MongoDB only (no Zep)."""
        # Setup MongoDB mock with sample data
        mock_mongo_data = [
            {"user_message": "MongoUser1", "assistant_message": "MongoAssistant1"},
            {"user_message": "MongoUser2", "assistant_message": "MongoAssistant2"}
        ]
        mock_mongo_adapter.find.return_value = mock_mongo_data

        # Create repository with MongoDB only
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)

        # Call retrieve method
        result = await repo.retrieve("test_user")

        # Verify correct behavior
        mock_mongo_adapter.find.assert_called_once_with(
            "conversations",
            {"user_id": "test_user"},
            sort=[("timestamp", -1)],
            limit=3
        )
        assert "MongoUser1 MongoAssistant1" in result
        assert "MongoUser2 MongoAssistant2" in result

    @pytest.mark.asyncio
    async def test_retrieve_mongo_empty_results(self, mock_mongo_adapter):
        """Test retrieving when MongoDB returns empty results."""
        # Setup MongoDB mock with empty data
        mock_mongo_adapter.find.return_value = []

        # Create repository with MongoDB only
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)

        # Call retrieve method
        result = await repo.retrieve("test_user")

        # Verify correct behavior
        mock_mongo_adapter.find.assert_called_once()
        assert result == ""  # Should be empty string

    @pytest.mark.asyncio
    async def test_retrieve_mongo_error(self, mock_mongo_adapter):
        """Test handling MongoDB retrieval error."""
        # Setup MongoDB mock to throw an error
        mock_mongo_adapter.find.side_effect = Exception(
            "MongoDB retrieval error")

        # Create repository with MongoDB only
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)

        # Call retrieve method - should handle the error gracefully
        result = await repo.retrieve("test_user")

        # Verify behavior
        mock_mongo_adapter.find.assert_called_once()
        assert result == ""  # Should return empty string on error

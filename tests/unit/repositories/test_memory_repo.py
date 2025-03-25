from typing import Dict, List, Any
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from solana_agent.repositories.memory import MemoryRepository

# Test constants
TEST_USER_ID = "test_user123"
TEST_USER_MESSAGE = "What is Solana?"
TEST_ASSISTANT_MESSAGE = "Solana is a high-performance blockchain..."
TEST_SUMMARY = "Discussion about Solana blockchain."
TEST_MESSAGES = [
    {"role": "user", "content": TEST_USER_MESSAGE},
    {"role": "assistant", "content": TEST_ASSISTANT_MESSAGE}
]


@pytest.fixture
def mock_mongo_adapter():
    adapter = Mock()
    adapter.create_collection = AsyncMock()
    adapter.create_index = AsyncMock()
    adapter.insert_one = AsyncMock()
    adapter.delete_all = AsyncMock()
    adapter.find = Mock(return_value=[])
    adapter.count_documents = Mock(return_value=0)
    return adapter


@pytest.fixture
def mock_zep_client():
    client = Mock()
    # Mock memory methods
    memory = AsyncMock()
    memory.add = AsyncMock()
    memory.get = AsyncMock(return_value=Mock(context=TEST_SUMMARY))
    memory.delete = AsyncMock()
    memory.add_session = AsyncMock()
    client.memory = memory
    # Mock user methods
    user = AsyncMock()
    user.add = AsyncMock()
    user.delete = AsyncMock()
    client.user = user
    return client


@pytest.fixture
def memory_repository_no_zep(mock_mongo_adapter):
    """Create a memory repository without Zep configuration."""
    return MemoryRepository(
        mongo_adapter=mock_mongo_adapter,
        zep_api_key=None,
        zep_base_url=None
    )


@pytest.fixture
def memory_repository_with_zep(mock_mongo_adapter):
    """Create a memory repository with Zep configuration."""
    with patch('solana_agent.repositories.memory.AsyncZep', autospec=True) as mock_zep_class:
        mock_zep = Mock()
        # Setup memory methods
        mock_memory = AsyncMock()
        mock_memory.add = AsyncMock()
        mock_memory.get = AsyncMock(return_value=Mock(context=TEST_SUMMARY))
        mock_memory.delete = AsyncMock()
        mock_memory.add_session = AsyncMock()
        mock_zep.memory = mock_memory
        # Setup user methods
        mock_user = AsyncMock()
        mock_user.add = AsyncMock()
        mock_user.delete = AsyncMock()
        mock_zep.user = mock_user

        mock_zep_class.return_value = mock_zep

        repo = MemoryRepository(
            mongo_adapter=mock_mongo_adapter,
            zep_api_key="test-key",
            zep_base_url="http://test-url"
        )
        repo.zep = mock_zep  # Ensure mock is properly set
        return repo

# Test cases without Zep


@pytest.mark.asyncio
async def test_store_no_zep(memory_repository_no_zep, mock_mongo_adapter):
    """Test message storage with only MongoDB."""
    await memory_repository_no_zep.store(TEST_USER_ID, TEST_MESSAGES)

    mock_mongo_adapter.insert_one.assert_called_once()
    call_args = mock_mongo_adapter.insert_one.call_args[0]
    assert call_args[0] == "conversations"
    doc = call_args[1]
    assert doc["user_id"] == TEST_USER_ID
    assert doc["user_message"] == TEST_USER_MESSAGE
    assert doc["assistant_message"] == TEST_ASSISTANT_MESSAGE
    assert isinstance(doc["timestamp"], datetime)


@pytest.mark.asyncio
async def test_retrieve_no_zep(memory_repository_no_zep):
    """Test memory retrieval without Zep."""
    result = await memory_repository_no_zep.retrieve(TEST_USER_ID)
    assert result == ""


@pytest.mark.asyncio
async def test_delete_no_zep(memory_repository_no_zep, mock_mongo_adapter):
    """Test deletion with only MongoDB."""
    await memory_repository_no_zep.delete(TEST_USER_ID)
    mock_mongo_adapter.delete_all.assert_called_once_with(
        "conversations",
        {"user_id": TEST_USER_ID}
    )

# Test cases with Zep


@pytest.mark.asyncio
async def test_store_with_zep(memory_repository_with_zep, mock_mongo_adapter):
    """Test message storage in both systems."""
    await memory_repository_with_zep.store(TEST_USER_ID, TEST_MESSAGES)

    # Verify MongoDB storage
    mock_mongo_adapter.insert_one.assert_called_once()
    call_args = mock_mongo_adapter.insert_one.call_args[0]
    assert call_args[0] == "conversations"
    doc = call_args[1]
    assert doc["user_id"] == TEST_USER_ID
    assert doc["user_message"] == TEST_USER_MESSAGE
    assert doc["assistant_message"] == TEST_ASSISTANT_MESSAGE
    assert isinstance(doc["timestamp"], datetime)

    # Verify Zep storage
    memory_repository_with_zep.zep.memory.add.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_with_zep(memory_repository_with_zep):
    """Test memory retrieval from Zep."""
    result = await memory_repository_with_zep.retrieve(TEST_USER_ID)
    assert result == TEST_SUMMARY
    memory_repository_with_zep.zep.memory.get.assert_called_once_with(
        session_id=TEST_USER_ID
    )


@pytest.mark.asyncio
async def test_delete_with_zep(memory_repository_with_zep, mock_mongo_adapter):
    """Test deletion from both systems."""
    await memory_repository_with_zep.delete(TEST_USER_ID)

    mock_mongo_adapter.delete_all.assert_called_once_with(
        "conversations",
        {"user_id": TEST_USER_ID}
    )
    memory_repository_with_zep.zep.memory.delete.assert_called_once_with(
        session_id=TEST_USER_ID
    )
    memory_repository_with_zep.zep.user.delete.assert_called_once_with(
        user_id=TEST_USER_ID
    )

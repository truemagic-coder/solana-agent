from typing import Dict, List, Any
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from solana_agent.repositories.memory import MemoryRepository

# Test Data
TEST_USER_ID = "test_user123"
TEST_USER_MESSAGE = "What is Solana?"
TEST_ASSISTANT_MESSAGE = "Solana is a high-performance blockchain..."
TEST_SUMMARY = "Discussion about Solana blockchain."
TEST_FACTS = [
    {"fact": "Solana is a blockchain platform"},
    {"fact": "Solana uses Proof of Stake"}
]

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
    adapter.delete_many = AsyncMock()
    return adapter


@pytest.fixture
def mock_zep_client():
    client = Mock()

    # Mock memory object
    memory = AsyncMock()
    memory.add = AsyncMock()
    memory.get_session = AsyncMock()
    memory.summarize = AsyncMock()
    memory.delete = AsyncMock()

    # Mock user object
    user = AsyncMock()
    user.delete = AsyncMock()

    # Attach to client
    client.memory = memory
    client.user = user
    return client


@pytest.fixture
def memory_repository(mock_mongo_adapter, mock_zep_client):
    with patch('solana_agent.repositories.memory.AsyncZep') as mock_zep:
        mock_zep.return_value = mock_zep_client
        repo = MemoryRepository(
            mongo_adapter=mock_mongo_adapter,
            zep_base_url="http://localhost:8000"
        )
        return repo


@pytest.mark.asyncio
async def test_store_success(memory_repository, mock_mongo_adapter, mock_zep_client):
    """Test successful message storage in both systems."""
    await memory_repository.store(TEST_USER_ID, TEST_MESSAGES)

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
    mock_zep_client.memory.add.assert_called_once()
    zep_args = mock_zep_client.memory.add.call_args[1]
    assert zep_args["session_id"] == TEST_USER_ID
    assert len(zep_args["messages"]) == 2


@pytest.mark.asyncio
async def test_retrieve_success(memory_repository, mock_zep_client):
    """Test successful memory retrieval from Zep."""
    # Setup mock returns
    mock_zep_client.memory.get_session.return_value = Mock(
        metadata={"facts": TEST_FACTS}
    )
    mock_zep_client.memory.get.return_value = Mock(
        context=TEST_SUMMARY
    )

    result = await memory_repository.retrieve(TEST_USER_ID)

    assert TEST_SUMMARY == result
    mock_zep_client.memory.get_session.assert_called_once_with(TEST_USER_ID)


@pytest.mark.asyncio
async def test_delete_success(memory_repository, mock_mongo_adapter, mock_zep_client):
    """Test successful deletion from both systems."""
    await memory_repository.delete(TEST_USER_ID)

    mock_mongo_adapter.delete_many.assert_called_once_with(
        "conversations",
        {"user_id": TEST_USER_ID}
    )
    mock_zep_client.memory.delete.assert_called_once_with(
        session_id=TEST_USER_ID
    )
    mock_zep_client.user.delete.assert_called_once_with(
        user_id=TEST_USER_ID
    )


@pytest.mark.asyncio
async def test_store_mongo_error(memory_repository, mock_mongo_adapter, mock_zep_client):
    """Test handling of MongoDB storage error."""
    mock_mongo_adapter.insert_one.side_effect = Exception("DB Error")

    # Should not raise exception
    await memory_repository.store(TEST_USER_ID, TEST_MESSAGES)

    # Zep storage should still be attempted
    mock_zep_client.memory.add.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_error(memory_repository, mock_zep_client):
    """Test handling of Zep retrieval error."""
    mock_zep_client.memory.get_session.side_effect = Exception("Zep Error")

    result = await memory_repository.retrieve(TEST_USER_ID)
    assert result == ""


def test_truncate_text(memory_repository):
    """Test text truncation functionality."""
    # Test short text
    short_text = "Hello world."
    assert memory_repository._truncate(short_text) == short_text

    # Test long text
    long_text = "." * 3000
    truncated = memory_repository._truncate(long_text, limit=2500)
    assert len(truncated) <= 2500
    assert truncated.endswith("...")

    # Test truncation at sentence boundary
    text_with_sentences = "First sentence. Second sentence. Third sentence."
    truncated = memory_repository._truncate(text_with_sentences, limit=20)
    assert truncated.endswith(".")

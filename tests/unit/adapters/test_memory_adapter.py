"""
Tests for memory adapter implementations.

This module contains unit tests for MongoMemoryProvider and DualMemoryProvider.
"""
import pytest
import datetime
from unittest.mock import Mock, AsyncMock, MagicMock

from solana_agent.adapters.memory_adapter import (
    MongoMemoryProvider,
    DualMemoryProvider
)


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, how can you help me with Solana?"},
        {"role": "assistant",
            "content": "I can help with Solana development, blockchain concepts, and more."}
    ]


@pytest.fixture
def mock_db_adapter():
    """Create a mock database adapter."""
    adapter = Mock()
    adapter.create_collection = Mock()
    adapter.create_index = Mock()
    adapter.insert_one = Mock()
    adapter.find = Mock(return_value=[
        {"role": "user", "content": "How do I create a Solana token?"},
        {"role": "assistant", "content": "You'll need to use the SPL Token program."}
    ])
    adapter.delete_one = Mock()
    return adapter


@pytest.fixture
def mock_memory_provider():
    """Create a mock memory provider for testing dual provider."""
    provider = MagicMock()
    provider.store = AsyncMock()
    provider.retrieve = AsyncMock(return_value="Mocked memory context")
    provider.delete = AsyncMock()
    return provider


# MongoMemoryProvider Tests
@pytest.mark.asyncio
async def test_mongo_memory_init(mock_db_adapter):
    """Test initialization of MongoMemoryProvider."""
    provider = MongoMemoryProvider(mock_db_adapter)

    # Verify collection and indexes were created
    mock_db_adapter.create_collection.assert_called_once_with("messages")
    assert mock_db_adapter.create_index.call_count == 2
    mock_db_adapter.create_index.assert_any_call("messages", [("user_id", 1)])
    mock_db_adapter.create_index.assert_any_call(
        "messages", [("timestamp", 1)])


@pytest.mark.asyncio
async def test_mongo_memory_store(mock_db_adapter, sample_messages):
    """Test storing messages in MongoMemoryProvider."""
    provider = MongoMemoryProvider(mock_db_adapter)
    await provider.store("user123", sample_messages)

    # Verify insert_one was called for each message
    assert mock_db_adapter.insert_one.call_count == 2

    # Check fields of the first inserted message
    args, _ = mock_db_adapter.insert_one.call_args_list[0]
    assert args[0] == "messages"
    assert args[1]["user_id"] == "user123"
    assert args[1]["role"] == sample_messages[0]["role"]
    assert args[1]["content"] == sample_messages[0]["content"]
    assert isinstance(args[1]["timestamp"], datetime.datetime)


@pytest.mark.asyncio
async def test_mongo_memory_retrieve(mock_db_adapter):
    """Test retrieving memory from MongoMemoryProvider."""
    provider = MongoMemoryProvider(mock_db_adapter)
    context = await provider.retrieve("user123")

    # Verify find was called with correct parameters
    mock_db_adapter.find.assert_called_once()
    args, kwargs = mock_db_adapter.find.call_args
    assert args[0] == "messages"
    assert args[1]["user_id"] == "user123"
    assert kwargs["sort"] == [("timestamp", 1)]

    # Check content of returned context
    assert "USER: How do I create a Solana token?" in context
    assert "ASSISTANT: You'll need to use the SPL Token program." in context


@pytest.mark.asyncio
async def test_mongo_memory_delete(mock_db_adapter):
    """Test deleting memory in MongoMemoryProvider."""
    provider = MongoMemoryProvider(mock_db_adapter)
    await provider.delete("user123")

    # Verify delete_one was called with correct parameters
    mock_db_adapter.delete_one.assert_called_once_with(
        "messages", {"user_id": "user123"}
    )


# DualMemoryProvider Tests
@pytest.mark.asyncio
async def test_dual_memory_store_both(mock_memory_provider, sample_messages):
    """Test storing messages in DualMemoryProvider with both providers."""
    # Create a second mock provider
    second_provider = MagicMock()
    second_provider.store = AsyncMock()

    # Create dual provider
    dual_provider = DualMemoryProvider(mock_memory_provider, second_provider)

    # Store messages
    await dual_provider.store("user123", sample_messages)

    # Verify both storage methods were called
    mock_memory_provider.store.assert_called_once_with(
        "user123", sample_messages)
    second_provider.store.assert_called_once_with("user123", sample_messages)


@pytest.mark.asyncio
async def test_dual_memory_store_mongo_only(mock_memory_provider, sample_messages):
    """Test storing messages in DualMemoryProvider with only MongoDB."""
    # Create dual provider with only one provider
    dual_provider = DualMemoryProvider(mock_memory_provider)

    # Store messages
    await dual_provider.store("user123", sample_messages)

    # Verify only first provider was used
    mock_memory_provider.store.assert_called_once_with(
        "user123", sample_messages)


@pytest.mark.asyncio
async def test_dual_memory_retrieve_primary(mock_memory_provider):
    """Test retrieving memory from primary provider in DualMemoryProvider."""
    # Create a secondary provider with different return value
    secondary_provider = MagicMock()
    secondary_provider.retrieve = AsyncMock(
        return_value="Secondary provider content")

    # Create dual provider
    dual_provider = DualMemoryProvider(
        mock_memory_provider, secondary_provider)

    # Retrieve memory
    context = await dual_provider.retrieve("user123")

    # Should use secondary provider first
    secondary_provider.retrieve.assert_called_once_with("user123")
    mock_memory_provider.retrieve.assert_not_called()
    assert context == "Secondary provider content"


@pytest.mark.asyncio
async def test_dual_memory_retrieve_fallback(mock_memory_provider):
    """Test retrieving memory from fallback in DualMemoryProvider."""
    # Create dual provider with only one provider
    dual_provider = DualMemoryProvider(mock_memory_provider)

    # Retrieve memory
    context = await dual_provider.retrieve("user123")

    # Should use primary provider
    mock_memory_provider.retrieve.assert_called_once_with("user123")
    assert context == "Mocked memory context"


@pytest.mark.asyncio
async def test_dual_memory_delete(mock_memory_provider):
    """Test deleting memory from DualMemoryProvider with both providers."""
    # Create a second mock provider
    second_provider = MagicMock()
    second_provider.delete = AsyncMock()

    # Create dual provider
    dual_provider = DualMemoryProvider(mock_memory_provider, second_provider)

    # Delete memory
    await dual_provider.delete("user123")

    # Verify both delete methods were called
    mock_memory_provider.delete.assert_called_once_with("user123")
    second_provider.delete.assert_called_once_with("user123")

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from solana_agent.repositories.memory import MemoryRepository
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter


@pytest.fixture
def mock_mongo_adapter():
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
    mock = AsyncMock()
    mock.user = AsyncMock()
    mock.thread = AsyncMock()
    return mock


@pytest.fixture
def valid_messages():
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


@pytest.fixture
def invalid_messages():
    return [
        {"invalid_role": "user", "content": "Missing role"},
        {"role": "user", "invalid_content": "Missing content"},
        {"role": "invalid", "content": "Invalid role"},
        {},  # Empty message
        None,  # None message
    ]


class TestMemoryRepository:
    def test_init_default(self):
        repo = MemoryRepository()
        assert repo.mongo is None
        assert repo.collection is None
        assert repo.zep is None

    def test_init_mongo_only(self, mock_mongo_adapter):
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        assert repo.mongo == mock_mongo_adapter
        assert repo.collection == "conversations"
        # Two collections should be created: conversations and captures
        assert mock_mongo_adapter.create_collection.call_count == 2
        created_names = [
            c.args[0] for c in mock_mongo_adapter.create_collection.call_args_list
        ]
        assert "conversations" in created_names
        assert "captures" in created_names

    def test_init_creates_expected_indexes(self, mock_mongo_adapter):
        MemoryRepository(mongo_adapter=mock_mongo_adapter)
        # Indexes: 2 for conversations (user_id, timestamp) and 5 for captures
        # (user_id, capture_name, agent_name, timestamp, and unique partial compound index)
        assert mock_mongo_adapter.create_index.call_count == 7

    def test_init_mongo_error(self, mock_mongo_adapter):
        mock_mongo_adapter.create_collection.side_effect = Exception("DB Error")
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        assert repo.mongo == mock_mongo_adapter

    @patch("solana_agent.repositories.memory.AsyncZepCloud")
    def test_init_zep_cloud(self, mock_zep_cloud):
        MemoryRepository(zep_api_key="test_key")
        mock_zep_cloud.assert_called_once_with(api_key="test_key")

    @pytest.mark.asyncio
    async def test_store_validation_errors(
        self, mock_mongo_adapter, invalid_messages, valid_messages
    ):
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        with pytest.raises(ValueError):
            await repo.store("", valid_messages)
        with pytest.raises(ValueError):
            await repo.store(123, valid_messages)
        with pytest.raises(ValueError):
            await repo.store("user123", [])
        with pytest.raises(ValueError):
            await repo.store("user123", None)
        for invalid_msg in invalid_messages:
            with pytest.raises(ValueError):
                await repo.store("user123", [invalid_msg])

    @pytest.mark.asyncio
    async def test_store_mongo_success(self, mock_mongo_adapter, valid_messages):
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
        mock_mongo_adapter.insert_one.side_effect = Exception("Storage error")
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        await repo.store("user123", valid_messages)
        mock_mongo_adapter.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_zep_thread_success(self, mock_zep, valid_messages):
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        await repo.store("user123", valid_messages)
        mock_zep.thread.add_messages.assert_called_once()
        mock_zep.user.add.assert_not_called()
        mock_zep.thread.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_zep_thread_fallback(self, mock_zep, valid_messages):
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        # Simulate thread.add_messages failing once, then succeeding
        mock_zep.thread.add_messages.side_effect = [Exception("fail"), None]
        await repo.store("user123", valid_messages)
        mock_zep.user.add.assert_called_once_with(user_id="user123")
        mock_zep.thread.create.assert_called_once_with(
            thread_id="user123", user_id="user123"
        )
        assert mock_zep.thread.add_messages.call_count == 2

    @pytest.mark.asyncio
    async def test_store_zep_thread_fallback_error(self, mock_zep, valid_messages):
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        # Simulate all Zep calls failing
        mock_zep.thread.add_messages.side_effect = Exception("fail")
        mock_zep.user.add.side_effect = Exception("fail")
        mock_zep.thread.create.side_effect = Exception("fail")
        await repo.store("user123", valid_messages)
        assert mock_zep.thread.add_messages.call_count == 2
        mock_zep.user.add.assert_called_once_with(user_id="user123")
        mock_zep.thread.create.assert_called_once_with(
            thread_id="user123", user_id="user123"
        )

    @pytest.mark.asyncio
    async def test_retrieve_success_no_zep(self):
        repo = MemoryRepository()
        result = await repo.retrieve("user123")
        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_memory_context_success(self, mock_zep):
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        mock_memory = MagicMock()
        mock_memory.context = "Sample memory context data"
        mock_zep.thread.get_user_context.return_value = mock_memory
        result = await repo.retrieve("test_user")
        mock_zep.thread.get_user_context.assert_called_once_with(thread_id="test_user")
        assert result == "Sample memory context data"

    @pytest.mark.asyncio
    async def test_retrieve_errors(self, mock_zep):
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        # None memory
        mock_zep.thread.get_user_context.return_value = None
        result = await repo.retrieve("user123")
        assert result == ""
        # Missing context
        memory = MagicMock()
        memory.context = None
        mock_zep.thread.get_user_context.return_value = memory
        result = await repo.retrieve("user123")
        assert result == ""
        # Retrieval error
        mock_zep.thread.get_user_context.side_effect = Exception("Retrieval error")
        result = await repo.retrieve("user123")
        assert result == ""

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_mongo_adapter, mock_zep):
        repo = MemoryRepository(
            mongo_adapter=mock_mongo_adapter, zep_api_key="test_key"
        )
        repo.zep = mock_zep
        await repo.delete("user123")
        mock_mongo_adapter.delete_all.assert_called_once_with(
            "conversations", {"user_id": "user123"}
        )
        mock_zep.thread.delete.assert_called_once_with(thread_id="user123")
        mock_zep.user.delete.assert_called_once_with(user_id="user123")

    def test_find_success(self, mock_mongo_adapter):
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        mock_mongo_adapter.find.return_value = [{"test": "doc"}]
        result = repo.find("conversations", {"query": "test"})
        assert result == [{"test": "doc"}]
        mock_mongo_adapter.find.assert_called_once()

    def test_count_documents_success(self, mock_mongo_adapter):
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        mock_mongo_adapter.count_documents.return_value = 5
        result = repo.count_documents("conversations", {"query": "test"})
        assert result == 5
        mock_mongo_adapter.count_documents.assert_called_once()

    def test_count_documents_success_no_mongo(self):
        repo = MemoryRepository()
        result = repo.count_documents("conversations", {"query": "test"})
        assert result == 0

    def test_truncate_text(self):
        repo = MemoryRepository()
        assert repo._truncate("Short text") == "Short text"
        assert (
            repo._truncate("First sentence. Second sentence.", 20) == "First sentence."
        )
        result = repo._truncate("a" * 3000)
        assert len(result) <= 2503
        assert result.endswith("...")
        assert repo._truncate("") == ""
        with pytest.raises(AttributeError):
            repo._truncate(None)

    @pytest.mark.asyncio
    async def test_store_empty_user_id(self):
        repo = MemoryRepository()
        with pytest.raises(ValueError):
            await repo.store("", [{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_store_missing_messages_pair(self, mock_mongo_adapter):
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Another user message"},
        ]
        await repo.store("user123", messages)
        mock_mongo_adapter.insert_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_mongo_error(self, mock_mongo_adapter):
        mock_mongo_adapter.delete_all.side_effect = Exception("Delete error")
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        await repo.delete("user123")
        mock_mongo_adapter.delete_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_zep_memory_error(self, mock_zep):
        mock_zep.thread.delete.side_effect = Exception("Memory delete error")
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        await repo.delete("user123")
        mock_zep.user.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_zep_user_error(self, mock_zep):
        mock_zep.user.delete.side_effect = Exception("User delete error")
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        await repo.delete("user123")
        mock_zep.thread.delete.assert_called_once()

    def test_find_mongo_error(self, mock_mongo_adapter):
        mock_mongo_adapter.find.side_effect = Exception("Find error")
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        result = repo.find("conversations", {})
        assert result == []
        mock_mongo_adapter.find.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_mongo_only_handles_no_results(self, mock_mongo_adapter):
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        mock_mongo_adapter.find.return_value = []
        result = await repo.retrieve("user123")
        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_mongo_only_handles_missing_fields(self, mock_mongo_adapter):
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        mock_mongo_adapter.find.return_value = [{"user_message": "Hello"}]
        result = await repo.retrieve("user123")
        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_mongo_returns_recent_conversation(self, mock_mongo_adapter):
        """Test that retrieve returns formatted recent conversation from Mongo."""
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        mock_mongo_adapter.find.return_value = [
            {"user_message": "Hello", "assistant_message": "Hi there"},
            {"user_message": "How are you?", "assistant_message": "I'm doing well!"},
        ]
        result = await repo.retrieve("user123")
        assert "## Recent Conversation" in result
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result
        assert "User: How are you?" in result
        assert "Assistant: I'm doing well!" in result

    @pytest.mark.asyncio
    async def test_retrieve_zep_only_returns_context(self, mock_zep):
        """Test that retrieve returns Zep context without Mongo."""
        repo = MemoryRepository(zep_api_key="test_key")
        repo.zep = mock_zep
        mock_memory = MagicMock()
        mock_memory.context = "User preferences: likes Python"
        mock_zep.thread.get_user_context.return_value = mock_memory
        result = await repo.retrieve("user123")
        assert result == "User preferences: likes Python"

    @pytest.mark.asyncio
    async def test_retrieve_combines_zep_and_mongo(self, mock_mongo_adapter, mock_zep):
        """Test that retrieve combines Zep long-term memory with Mongo recent history."""
        repo = MemoryRepository(
            mongo_adapter=mock_mongo_adapter, zep_api_key="test_key"
        )
        repo.zep = mock_zep

        # Setup Zep response
        mock_memory = MagicMock()
        mock_memory.context = "User preferences: likes Python programming"
        mock_zep.thread.get_user_context.return_value = mock_memory

        # Setup Mongo response
        mock_mongo_adapter.find.return_value = [
            {
                "user_message": "Tell me about Python",
                "assistant_message": "Python is great!",
            },
        ]

        result = await repo.retrieve("user123")
        assert "## Long-term Memory" in result
        assert "User preferences: likes Python programming" in result
        assert "## Recent Conversation" in result
        assert "User: Tell me about Python" in result
        assert "Assistant: Python is great!" in result

    @pytest.mark.asyncio
    async def test_retrieve_mongo_skips_incomplete_turns(self, mock_mongo_adapter):
        """Test that retrieve skips turns with missing user or assistant message."""
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        mock_mongo_adapter.find.return_value = [
            {"user_message": "Hello", "assistant_message": "Hi"},
            {"user_message": "Missing assistant", "assistant_message": ""},
            {"user_message": "", "assistant_message": "Missing user"},
            {"user_message": "Complete", "assistant_message": "Also complete"},
        ]
        result = await repo.retrieve("user123")
        assert "User: Hello" in result
        assert "Assistant: Hi" in result
        assert "User: Complete" in result
        assert "Assistant: Also complete" in result
        assert "Missing assistant" not in result
        assert "Missing user" not in result

    @pytest.mark.asyncio
    async def test_retrieve_respects_limit_of_10(self, mock_mongo_adapter):
        """Test that retrieve requests only 10 most recent messages."""
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        mock_mongo_adapter.find.return_value = []
        await repo.retrieve("user123")
        mock_mongo_adapter.find.assert_called_once_with(
            "conversations",
            {"user_id": "user123"},
            sort=[("timestamp", -1)],
            limit=10,
        )

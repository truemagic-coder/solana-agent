"""
Tests for MongoMemoryRepository implementation.

This module contains unit tests for the MongoDB-based memory repository.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from solana_agent.repositories.mongo_memory import MongoMemoryRepository
from solana_agent.domains import MemoryInsight


@pytest.fixture
def mock_db_adapter():
    """Create a mock database adapter."""
    adapter = Mock()
    adapter.create_collection = Mock()
    adapter.create_index = Mock()
    adapter.insert_one = Mock(return_value="mock_id")
    adapter.find = Mock()
    adapter.delete_one = Mock()
    return adapter


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock()
    store.store_vectors = Mock()
    store.search_vectors = Mock()
    return store


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = Mock()
    provider.generate_embedding = Mock(return_value=[0.1, 0.2, 0.3])
    return provider


@pytest.fixture
def memory_repo(mock_db_adapter):
    """Create a basic memory repository without vector store or LLM provider."""
    return MongoMemoryRepository(mock_db_adapter)


@pytest.fixture
def full_memory_repo(mock_db_adapter, mock_vector_store, mock_llm_provider):
    """Create a memory repository with all optional components."""
    return MongoMemoryRepository(
        mock_db_adapter,
        vector_store=mock_vector_store,
        llm_provider=mock_llm_provider
    )


@pytest.fixture
def sample_insight():
    """Create a sample memory insight for testing."""
    return MemoryInsight(
        category="preference",  # Changed from insight_type to category
        content="User prefers dark mode UI",
        source="conversation",
        confidence=0.95,
        timestamp="2025-03-15T14:30:00Z",
        metadata={"conversation_id": "conv123"}
    )


class TestMongoMemoryRepository:
    """Tests for the MongoMemoryRepository implementation."""

    def test_init(self, mock_db_adapter):
        """Test repository initialization."""
        repo = MongoMemoryRepository(mock_db_adapter)

        # Verify collections are created
        mock_db_adapter.create_collection.assert_any_call("memory_insights")
        mock_db_adapter.create_collection.assert_any_call(
            "conversation_history")
        assert mock_db_adapter.create_collection.call_count == 2

        # Verify indexes are created
        assert mock_db_adapter.create_index.call_count == 4
        mock_db_adapter.create_index.assert_any_call(
            "memory_insights", [("user_id", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "memory_insights", [("created_at", -1)])
        mock_db_adapter.create_index.assert_any_call(
            "conversation_history", [("user_id", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "conversation_history", [("timestamp", -1)])

    def test_store_insight_basic(self, memory_repo, mock_db_adapter, sample_insight):
        """Test storing an insight without vector store."""
        user_id = "user123"

        # Store insight
        memory_repo.store_insight(user_id, sample_insight)

        # Verify DB operations
        mock_db_adapter.insert_one.assert_called_once()
        collection, data = mock_db_adapter.insert_one.call_args[0]

        assert collection == "memory_insights"
        assert data["user_id"] == user_id
        assert data["content"] == sample_insight.content
        # Changed from insight_type to category
        assert data["category"] == "preference"
        assert data["source"] == "conversation"

    def test_store_insight_with_vector(self, full_memory_repo, mock_db_adapter,
                                       mock_vector_store, mock_llm_provider, sample_insight):
        """Test storing an insight with vector store and LLM provider."""
        user_id = "user123"

        # Store insight
        full_memory_repo.store_insight(user_id, sample_insight)

        # Verify DB operations
        mock_db_adapter.insert_one.assert_called_once()

        # Verify embedding generation
        mock_llm_provider.generate_embedding.assert_called_once_with(
            sample_insight.content)

        # Verify vector storage
        mock_vector_store.store_vectors.assert_called_once()
        vectors = mock_vector_store.store_vectors.call_args[0][0]
        namespace = mock_vector_store.store_vectors.call_args[1]['namespace']

        assert len(vectors) == 1
        assert vectors[0]["id"] == "mock_id"
        assert vectors[0]["values"] == [0.1, 0.2, 0.3]
        assert vectors[0]["metadata"]["user_id"] == user_id
        assert vectors[0]["metadata"]["content"] == sample_insight.content
        assert namespace == "memory_insights"

    def test_search_with_vector_store(self, full_memory_repo, mock_vector_store, mock_llm_provider):
        """Test searching with vector store."""
        query = "dark mode"
        limit = 5

        # Configure mock vector store response
        mock_vector_store.search_vectors.return_value = [
            {
                "id": "result1",
                "score": 0.95,
                "metadata": {
                    "content": "User prefers dark mode UI",
                    "category": "preference",
                    "user_id": "user123",
                    "created_at": "2025-03-15T14:30:00Z"
                }
            },
            {
                "id": "result2",
                "score": 0.85,
                "metadata": {
                    "content": "User asked about theme settings",
                    "category": "question",
                    "user_id": "user123",
                    "created_at": "2025-03-15T14:35:00Z"
                }
            }
        ]

        # Search
        results = full_memory_repo.search(query, limit)

        # Verify embedding generation
        mock_llm_provider.generate_embedding.assert_called_once_with(query)

        # Verify vector search
        mock_vector_store.search_vectors.assert_called_once_with(
            [0.1, 0.2, 0.3],
            namespace="memory_insights",
            limit=limit
        )

        # Verify results
        assert len(results) == 2
        assert results[0]["content"] == "User prefers dark mode UI"
        assert results[0]["score"] == 0.95
        assert results[1]["content"] == "User asked about theme settings"
        assert results[1]["score"] == 0.85

    def test_search_without_vector_store(self, memory_repo, mock_db_adapter):
        """Test searching without vector store (falls back to MongoDB text search)."""
        query = "dark mode"
        limit = 5

        # Configure mock DB response
        mock_db_adapter.find.return_value = [
            {
                "content": "User prefers dark mode UI",
                "user_id": "user123",
                "created_at": datetime.now()
            }
        ]

        # Search
        results = memory_repo.search(query, limit)

        # Verify MongoDB text search is used
        mock_db_adapter.find.assert_called_once()
        collection, query_filter, *args = mock_db_adapter.find.call_args[0]

        assert collection == "memory_insights"
        assert "$text" in query_filter
        assert query_filter["$text"]["$search"] == query

        # Verify results
        assert len(results) == 1
        assert results[0]["content"] == "User prefers dark mode UI"

    def test_search_fallback_to_regex(self, memory_repo, mock_db_adapter):
        """Test searching falls back to regex search when text search fails."""
        query = "dark mode"
        limit = 5

        # Configure mock to fail on first call and succeed on second
        mock_db_adapter.find.side_effect = [
            Exception("Text search failed"),
            [{"content": "User prefers dark mode UI", "user_id": "user123"}]
        ]

        # Search
        results = memory_repo.search(query, limit)

        # Verify regex search is used as fallback
        assert mock_db_adapter.find.call_count == 2
        collection, query_filter, *args = mock_db_adapter.find.call_args[0]

        assert collection == "memory_insights"
        assert "$regex" in query_filter["content"]
        assert query_filter["content"]["$regex"] == query

        # Verify results
        assert len(results) == 1
        assert results[0]["content"] == "User prefers dark mode UI"

    def test_get_user_history(self, memory_repo, mock_db_adapter):
        """Test getting user conversation history."""
        user_id = "user123"
        limit = 10

        # Configure mock response
        mock_db_adapter.find.return_value = [
            {
                "user_id": user_id,
                "user_message": "How do I change the theme?",
                "assistant_message": "You can change the theme in settings.",
                "timestamp": datetime.now()
            },
            {
                "user_id": user_id,
                "user_message": "Thanks, I found it.",
                "assistant_message": "You're welcome!",
                "timestamp": datetime.now()
            }
        ]

        # Get history
        history = memory_repo.get_user_history(user_id, limit)

        # Verify DB query
        mock_db_adapter.find.assert_called_once()
        args, kwargs = mock_db_adapter.find.call_args
        collection, query = args

        assert collection == "conversation_history"
        assert query == {"user_id": user_id}
        assert kwargs.get("limit") == limit
        assert kwargs.get("sort") == [("timestamp", -1)]

        # Verify results
        assert len(history) == 2
        assert history[0]["user_message"] == "How do I change the theme?"
        assert history[1]["user_message"] == "Thanks, I found it."

    def test_delete_user_memory(self, memory_repo, mock_db_adapter):
        """Test deleting all memory for a user."""
        user_id = "user123"

        # Delete memory
        result = memory_repo.delete_user_memory(user_id)

        # Verify DB operations
        assert mock_db_adapter.delete_one.call_count == 2
        mock_db_adapter.delete_one.assert_any_call(
            "memory_insights", {"user_id": user_id})
        mock_db_adapter.delete_one.assert_any_call(
            "conversation_history", {"user_id": user_id})

        # Verify result
        assert result is True

    def test_delete_user_memory_with_vector_store(self, full_memory_repo, mock_db_adapter, mock_vector_store):
        """Test deleting memory when vector store is available."""
        user_id = "user123"

        # Delete memory
        result = full_memory_repo.delete_user_memory(user_id)

        # Verify DB operations
        assert mock_db_adapter.delete_one.call_count == 2

        # Note: The current implementation doesn't actually delete from vector store
        # so we're just verifying the MongoDB operations succeed
        assert result is True

    def test_store_conversation_entry(self, memory_repo, mock_db_adapter):
        """Test storing a conversation entry."""
        user_id = "user123"
        user_message = "How can I help improve the product?"
        assistant_message = "You can provide feedback through our survey."

        # Store conversation
        memory_repo.store_conversation_entry(
            user_id, user_message, assistant_message)

        # Verify DB operations
        mock_db_adapter.insert_one.assert_called_once()
        collection, data = mock_db_adapter.insert_one.call_args[0]

        assert collection == "conversation_history"
        assert data["user_id"] == user_id
        assert data["user_message"] == user_message
        assert data["assistant_message"] == assistant_message
        assert "timestamp" in data

    def test_vector_store_error_handling(self, full_memory_repo, mock_vector_store,
                                         mock_llm_provider, sample_insight):
        """Test error handling when vector store operations fail."""
        user_id = "user123"

        # Configure mock to raise exception
        mock_vector_store.store_vectors.side_effect = Exception(
            "Vector store error")

        # Store should not raise exception even if vector store fails
        full_memory_repo.store_insight(user_id, sample_insight)

        # Verify embedding was still attempted
        mock_llm_provider.generate_embedding.assert_called_once()

        # Verify vector store was attempted but failed (no exception raised to caller)
        mock_vector_store.store_vectors.assert_called_once()

    def test_search_error_handling(self, full_memory_repo, mock_vector_store,
                                   mock_llm_provider, mock_db_adapter):
        """Test search error handling when vector search fails."""
        query = "dark mode"

        # Configure mocks
        mock_vector_store.search_vectors.side_effect = Exception(
            "Vector search error")
        mock_db_adapter.find.return_value = [
            {"content": "Fallback result", "user_id": "user123"}
        ]

        # Search should not fail even if vector search fails
        results = full_memory_repo.search(query)

        # Verify vector search was attempted
        mock_vector_store.search_vectors.assert_called_once()

        # Verify fallback to MongoDB
        mock_db_adapter.find.assert_called_once()

        # Verify results from fallback
        assert len(results) == 1
        assert results[0]["content"] == "Fallback result"

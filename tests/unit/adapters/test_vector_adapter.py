"""
Tests for vector database adapter implementations.

This module contains unit tests for PineconeAdapter and QdrantAdapter.
"""
import pytest
from unittest.mock import patch, MagicMock, call
import json

from solana_agent.adapters.vector_adapter import PineconeAdapter, QdrantAdapter


class TestPineconeAdapter:
    """Tests for the Pinecone adapter implementation."""

    @pytest.fixture
    def mock_pinecone(self):
        """Create a mock Pinecone client with mocked Index."""
        with patch('solana_agent.adapters.vector_adapter.Pinecone') as mock_pinecone:
            # Create a mock index
            mock_index = MagicMock()

            # Configure the mock client to return the mock index
            mock_pinecone.return_value.Index.return_value = mock_index

            yield mock_pinecone, mock_index

    @pytest.fixture
    def pinecone_adapter(self, mock_pinecone):
        """Create a PineconeAdapter with mocked dependencies."""
        mock_pinecone_client, _ = mock_pinecone
        adapter = PineconeAdapter(
            api_key="test_api_key", index_name="test_index")
        return adapter

    def test_init(self, mock_pinecone):
        """Test initialization of PineconeAdapter."""
        mock_pinecone_client, _ = mock_pinecone

        adapter = PineconeAdapter(
            api_key="test_api_key", index_name="test_index")

        # Verify client was created with API key
        mock_pinecone_client.assert_called_once_with(api_key="test_api_key")

        # Verify index was retrieved
        mock_pinecone_client.return_value.Index.assert_called_once_with(
            "test_index")

        # Verify default embedding model
        assert adapter.embedding_model == "text-embedding-3-small"

    def test_init_custom_embedding_model(self, mock_pinecone):
        """Test initialization with custom embedding model."""
        mock_pinecone_client, _ = mock_pinecone

        adapter = PineconeAdapter(
            api_key="test_api_key",
            index_name="test_index",
            embedding_model="custom-model"
        )

        assert adapter.embedding_model == "custom-model"

    def test_store_vectors(self, pinecone_adapter, mock_pinecone):
        """Test storing vectors in Pinecone."""
        _, mock_index = mock_pinecone

        # Sample vectors
        vectors = [
            {
                "id": "vec1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"source": "document1"}
            },
            {
                "id": "vec2",
                "values": [0.4, 0.5, 0.6],
                "metadata": {"source": "document2"}
            }
        ]

        # Call the method
        pinecone_adapter.store_vectors(vectors, namespace="test_namespace")

        # Verify the upsert was called with correct parameters
        mock_index.upsert.assert_called_once_with(
            vectors=vectors,
            namespace="test_namespace"
        )

    def test_search_vectors(self, pinecone_adapter, mock_pinecone):
        """Test searching for vectors in Pinecone."""
        _, mock_index = mock_pinecone

        # Set up mock response
        mock_match1 = MagicMock()
        mock_match1.id = "vec1"
        mock_match1.score = 0.95
        mock_match1.metadata = {"source": "document1"}

        mock_match2 = MagicMock()
        mock_match2.id = "vec2"
        mock_match2.score = 0.85
        mock_match2.metadata = {"source": "document2"}

        mock_results = MagicMock()
        mock_results.matches = [mock_match1, mock_match2]

        mock_index.query.return_value = mock_results

        # Sample query vector
        query_vector = [0.1, 0.2, 0.3]

        # Call the method
        results = pinecone_adapter.search_vectors(
            query_vector=query_vector,
            namespace="test_namespace",
            limit=2
        )

        # Verify query was called with correct parameters
        mock_index.query.assert_called_once_with(
            vector=query_vector,
            top_k=2,
            include_metadata=True,
            namespace="test_namespace"
        )

        # Verify results were processed correctly
        assert len(results) == 2
        assert results[0]["id"] == "vec1"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"] == {"source": "document1"}
        assert results[1]["id"] == "vec2"
        assert results[1]["score"] == 0.85
        assert results[1]["metadata"] == {"source": "document2"}

    def test_search_vectors_no_metadata(self, pinecone_adapter, mock_pinecone):
        """Test searching vectors when matches have no metadata."""
        _, mock_index = mock_pinecone

        # Set up mock response with no metadata
        mock_match1 = MagicMock()
        mock_match1.id = "vec1"
        mock_match1.score = 0.95
        mock_match1.metadata = None

        mock_results = MagicMock()
        mock_results.matches = [mock_match1]

        mock_index.query.return_value = mock_results

        # Call the method
        results = pinecone_adapter.search_vectors(
            query_vector=[0.1, 0.2, 0.3],
            namespace="test_namespace"
        )

        # Verify we don't include results without metadata
        assert len(results) == 0

    def test_search_vectors_no_matches(self, pinecone_adapter, mock_pinecone):
        """Test searching vectors when there are no matches."""
        _, mock_index = mock_pinecone

        # Set up mock response with no matches attribute
        mock_results = MagicMock()
        # No matches attribute

        mock_index.query.return_value = mock_results

        # Call the method
        results = pinecone_adapter.search_vectors(
            query_vector=[0.1, 0.2, 0.3],
            namespace="test_namespace"
        )

        # Verify we handle no matches gracefully
        assert results == []

    def test_delete_vector(self, pinecone_adapter, mock_pinecone):
        """Test deleting a vector from Pinecone."""
        _, mock_index = mock_pinecone

        # Call the method
        pinecone_adapter.delete_vector("vec1", namespace="test_namespace")

        # Verify delete was called with correct parameters
        mock_index.delete.assert_called_once_with(
            ids=["vec1"],
            namespace="test_namespace"
        )


class TestQdrantAdapter:
    """Tests for the Qdrant adapter implementation."""

    @pytest.fixture
    def mock_qdrant_imports(self):
        """Mock the Qdrant client imports."""
        mock_client = MagicMock()
        mock_models = MagicMock()

        with patch.dict('sys.modules', {
            'qdrant_client': MagicMock(),
            'qdrant_client.http': MagicMock(),
            'qdrant_client.http.models': mock_models
        }):
            with patch('qdrant_client.QdrantClient', return_value=mock_client):
                yield mock_client, mock_models

    @pytest.fixture
    def qdrant_adapter(self, mock_qdrant_imports):
        """Create a QdrantAdapter with mocked dependencies."""
        mock_client, _ = mock_qdrant_imports

        # Mock collections to simulate empty collection list
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        adapter = QdrantAdapter(
            url="http://localhost:6333",
            api_key="test_api_key",
            collection_name="test_collection"
        )
        return adapter

    def test_init_new_collection(self, mock_qdrant_imports):
        """Test initialization when collection doesn't exist."""
        mock_client, mock_models = mock_qdrant_imports

        # Mock collections to simulate empty collection list
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # Create distance enum
        mock_models.Distance.COSINE = "cosine"

        # Create adapter
        QdrantAdapter(
            url="http://localhost:6333",
            api_key="test_api_key",
            collection_name="test_collection",
            vector_size=768
        )

        # Verify client creation
        from qdrant_client import QdrantClient
        QdrantClient.assert_called_once_with(
            url="http://localhost:6333",
            api_key="test_api_key"
        )

        # Verify collection was created
        mock_client.create_collection.assert_called_once()
        args, kwargs = mock_client.create_collection.call_args
        assert kwargs['collection_name'] == "test_collection"

    def test_init_existing_collection(self, mock_qdrant_imports):
        """Test initialization when collection already exists."""
        mock_client, _ = mock_qdrant_imports

        # Mock collections to simulate collection already exists
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"

        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        # Create adapter
        adapter = QdrantAdapter(
            url="http://localhost:6333",
            collection_name="test_collection"
        )

        # Verify collection was not created again
        mock_client.create_collection.assert_not_called()

        # Verify default values
        assert adapter.embedding_model == "text-embedding-3-small"
        assert adapter.vector_size == 1536

    def test_init_error_handling(self, mock_qdrant_imports):
        """Test initialization error handling."""
        mock_client, _ = mock_qdrant_imports

        # Make get_collections raise an exception
        mock_client.get_collections.side_effect = Exception("Connection error")

        # Should not raise exception, just print error
        adapter = QdrantAdapter(collection_name="test_collection")

        # Client should still be created
        assert adapter.client == mock_client
        assert adapter.collection_name == "test_collection"

    def test_store_vectors(self, qdrant_adapter, mock_qdrant_imports):
        """Test storing vectors in Qdrant."""
        mock_client, _ = mock_qdrant_imports

        # Sample vectors
        vectors = [
            {
                "id": "vec1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"source": "document1"}
            },
            {
                "id": "vec2",
                "values": [0.4, 0.5, 0.6],
                "metadata": {"source": "document2"}
            }
        ]

        # Call the method
        qdrant_adapter.store_vectors(vectors, namespace="test_namespace")

        # Verify that upsert was called at least once
        assert mock_client.upsert.call_count > 0

        # Check that collection name was passed correctly
        args, kwargs = mock_client.upsert.call_args
        assert kwargs['collection_name'] == "test_collection"

    def test_search_vectors(self, qdrant_adapter, mock_qdrant_imports):
        """Test searching for vectors in Qdrant."""
        mock_client, mock_models = mock_qdrant_imports

        # Configure mock model classes
        mock_models.Filter.return_value = "mock_filter"
        mock_models.FieldCondition.return_value = "mock_condition"
        mock_models.MatchValue.return_value = "mock_match"

        # Set up mock response
        mock_result1 = MagicMock()
        mock_result1.id = "vec1"
        mock_result1.score = 0.95
        mock_result1.payload = {"source": "document1",
                                "namespace": "test_namespace"}

        mock_result2 = MagicMock()
        mock_result2.id = "vec2"
        mock_result2.score = 0.85
        mock_result2.payload = {"source": "document2",
                                "namespace": "test_namespace"}

        mock_client.search.return_value = [mock_result1, mock_result2]

        # Sample query vector
        query_vector = [0.1, 0.2, 0.3]

        # Call the method
        results = qdrant_adapter.search_vectors(
            query_vector=query_vector,
            namespace="test_namespace",
            limit=2
        )

        # Verify search was called
        mock_client.search.assert_called_once()
        args, kwargs = mock_client.search.call_args
        assert kwargs['collection_name'] == "test_collection"
        assert kwargs['query_vector'] == query_vector
        assert kwargs['limit'] == 2

        # Verify results were processed correctly
        assert len(results) == 2
        assert results[0]["id"] == "vec1"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"] == {
            "source": "document1", "namespace": "test_namespace"}

    def test_search_vectors_error_handling(self, qdrant_adapter, mock_qdrant_imports):
        """Test search error handling in Qdrant."""
        mock_client, _ = mock_qdrant_imports

        # Make search raise an exception
        mock_client.search.side_effect = Exception("Search error")

        # Call the method - should handle exception gracefully
        results = qdrant_adapter.search_vectors(
            query_vector=[0.1, 0.2, 0.3],
            namespace="test_namespace"
        )

        # Should return empty results on error
        assert results == []

    def test_delete_vector(self, qdrant_adapter, mock_qdrant_imports):
        """Test deleting a vector from Qdrant."""
        mock_client, _ = mock_qdrant_imports

        # Call the method
        qdrant_adapter.delete_vector("vec1", namespace="test_namespace")

        # Verify delete was called with correct parameters
        mock_client.delete.assert_called_once()
        args, kwargs = mock_client.delete.call_args

        assert kwargs['collection_name'] == "test_collection"

    def test_delete_vector_error_handling(self, qdrant_adapter, mock_qdrant_imports):
        """Test delete error handling in Qdrant."""
        mock_client, _ = mock_qdrant_imports

        # Make delete raise an exception
        mock_client.delete.side_effect = Exception("Delete error")

        # Call the method - should handle exception gracefully
        qdrant_adapter.delete_vector("vec1", namespace="test_namespace")

        # No assertion needed - we're testing that no exception is raised

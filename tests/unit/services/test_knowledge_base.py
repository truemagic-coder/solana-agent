"""
Tests for the KnowledgeBase service implementation.

This module provides comprehensive test coverage for the KnowledgeBase service
which combines Pinecone for vector search and MongoDB for document storage.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone
import uuid

from solana_agent.services.knowledge_base import KnowledgeBase
from solana_agent.adapters.pinecone_adapter import PineconeAdapter
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter


@pytest.fixture
def mock_pinecone_adapter():
    """Create a mock Pinecone adapter."""
    adapter = AsyncMock(spec=PineconeAdapter)
    adapter.upsert_text = AsyncMock()
    adapter.query_text = AsyncMock()
    adapter.delete = AsyncMock()
    return adapter


@pytest.fixture
def mock_mongo_adapter():
    """Create a mock MongoDB adapter."""
    adapter = MagicMock(spec=MongoDBAdapter)
    adapter.collection_exists = MagicMock(return_value=False)
    adapter.create_collection = MagicMock()
    adapter.create_index = MagicMock()
    adapter.insert_one = MagicMock(return_value="test_doc_id")
    adapter.find_one = MagicMock()
    adapter.find = MagicMock(return_value=[])
    adapter.update_one = MagicMock(return_value=True)
    adapter.delete_one = MagicMock(return_value=True)
    return adapter


@pytest.fixture
def knowledge_base(mock_pinecone_adapter, mock_mongo_adapter):
    """Create a KnowledgeBase instance with mock adapters."""
    kb = KnowledgeBase(
        pinecone_adapter=mock_pinecone_adapter,
        mongodb_adapter=mock_mongo_adapter,
        collection_name="test_kb",
        rerank_results=False,
        rerank_top_k=3
    )
    return kb


@pytest.fixture
def sample_document():
    """Sample document data for testing."""
    return {
        "text": "Solana is a high-performance blockchain supporting builders around the world.",
        "metadata": {
            "title": "About Solana",
            "source": "website",
            "url": "https://solana.com",
            "tags": ["blockchain", "crypto"]
        }
    }


@pytest.fixture
def sample_pinecone_results():
    """Sample Pinecone query results for testing."""
    return [
        {"id": "doc1", "score": 0.95, "metadata": {
            "document_id": "doc1", "source": "website"}},
        {"id": "doc2", "score": 0.85, "metadata": {
            "document_id": "doc2", "source": "blog"}}
    ]


@pytest.fixture
def sample_mongo_docs():
    """Sample MongoDB documents for testing."""
    return [
        {
            "document_id": "doc1",
            "content": "Solana is a high-performance blockchain.",
            "title": "About Solana",
            "source": "website",
            "tags": ["blockchain"]
        },
        {
            "document_id": "doc2",
            "content": "Solana supports high transaction throughput.",
            "title": "Solana Performance",
            "source": "blog",
            "tags": ["blockchain", "performance"]
        }
    ]


class TestKnowledgeBase:
    """Test suite for KnowledgeBase service."""

    def test_init_creates_indexes(self, mock_pinecone_adapter, mock_mongo_adapter):
        """Test that initialization creates collection and indexes."""
        kb = KnowledgeBase(mock_pinecone_adapter, mock_mongo_adapter)

        mock_mongo_adapter.create_collection.assert_called_once()
        assert mock_mongo_adapter.create_index.call_count == 4

    def test_init_with_existing_collection(self, mock_pinecone_adapter, mock_mongo_adapter):
        """Test initialization with an existing collection."""
        mock_mongo_adapter.collection_exists.return_value = True

        kb = KnowledgeBase(mock_pinecone_adapter, mock_mongo_adapter)

        mock_mongo_adapter.create_collection.assert_not_called()
        assert mock_mongo_adapter.create_index.call_count == 4

    @pytest.mark.asyncio
    async def test_add_document_with_provided_id(self, knowledge_base, sample_document):
        """Test adding a document with a provided ID."""
        doc_id = "test-doc-123"

        result = await knowledge_base.add_document(
            text=sample_document["text"],
            metadata=sample_document["metadata"],
            document_id=doc_id
        )

        assert result == doc_id

        # Check MongoDB insertion
        knowledge_base.mongo.insert_one.assert_called_once()
        args = knowledge_base.mongo.insert_one.call_args[0]
        assert args[0] == "test_kb"
        assert args[1]["document_id"] == doc_id
        assert args[1]["content"] == sample_document["text"]
        assert args[1]["title"] == sample_document["metadata"]["title"]

        # Check Pinecone insertion
        knowledge_base.pinecone.upsert_text.assert_called_once()
        kwargs = knowledge_base.pinecone.upsert_text.call_args[1]
        assert kwargs["texts"] == [sample_document["text"]]
        assert kwargs["ids"] == [doc_id]
        assert kwargs["metadatas"][0]["document_id"] == doc_id

    @pytest.mark.asyncio
    async def test_add_document_with_generated_id(self, knowledge_base, sample_document):
        """Test adding a document with an auto-generated ID."""
        with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
            result = await knowledge_base.add_document(
                text=sample_document["text"],
                metadata=sample_document["metadata"]
            )

        assert result == "12345678-1234-5678-1234-567812345678"

        # Check both MongoDB and Pinecone were called
        knowledge_base.mongo.insert_one.assert_called_once()
        knowledge_base.pinecone.upsert_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_no_results(self, knowledge_base):
        """Test querying when no results are returned from Pinecone."""
        knowledge_base.pinecone.query_text.return_value = []

        results = await knowledge_base.query(query_text="test query")

        assert results == []
        knowledge_base.pinecone.query_text.assert_called_once_with(
            query_text="test query",
            filter=None,
            top_k=5,
            namespace=None,
            include_values=False,
            include_metadata=True
        )
        knowledge_base.mongo.find.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_with_results(self, knowledge_base, sample_pinecone_results, sample_mongo_docs):
        """Test querying with results from both Pinecone and MongoDB."""
        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results
        knowledge_base.mongo.find.return_value = sample_mongo_docs

        results = await knowledge_base.query(
            query_text="blockchain performance",
            top_k=2
        )

        assert len(results) == 2
        assert results[0]["document_id"] == "doc1"
        assert results[0]["score"] == 0.95
        assert results[0]["content"] == "Solana is a high-performance blockchain."
        assert "metadata" in results[0]
        assert results[0]["metadata"]["title"] == "About Solana"

        knowledge_base.mongo.find.assert_called_once_with(
            "test_kb",
            {"document_id": {"$in": ["doc1", "doc2"]}}
        )

    @pytest.mark.asyncio
    async def test_query_content_exclusion(self, knowledge_base, sample_pinecone_results, sample_mongo_docs):
        """Test querying with content excluded from results."""
        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results
        knowledge_base.mongo.find.return_value = sample_mongo_docs

        results = await knowledge_base.query(
            query_text="blockchain",
            include_content=False,
            include_metadata=True
        )

        assert len(results) == 2
        assert "content" not in results[0]
        assert "metadata" in results[0]

    @pytest.mark.asyncio
    async def test_query_metadata_exclusion(self, knowledge_base, sample_pinecone_results, sample_mongo_docs):
        """Test querying with metadata excluded from results."""
        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results
        knowledge_base.mongo.find.return_value = sample_mongo_docs

        results = await knowledge_base.query(
            query_text="blockchain",
            include_content=True,
            include_metadata=False
        )

        assert len(results) == 2
        assert "content" in results[0]
        assert "metadata" not in results[0]

    @pytest.mark.asyncio
    async def test_query_with_reranking(self, knowledge_base, sample_pinecone_results, sample_mongo_docs):
        """Test querying with reranking enabled."""
        # Enable reranking
        knowledge_base.rerank_results = True
        knowledge_base.rerank_top_k = 2

        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results
        knowledge_base.mongo.find.return_value = sample_mongo_docs

        results = await knowledge_base.query(query_text="blockchain", top_k=5)

        # Verify rerank_top_k was used instead of provided top_k
        knowledge_base.pinecone.query_text.assert_called_once_with(
            query_text="blockchain",
            filter=None,
            top_k=2,  # Should use rerank_top_k
            namespace=None,
            include_values=False,
            include_metadata=True
        )

    @pytest.mark.asyncio
    async def test_query_missing_mongo_docs(self, knowledge_base, sample_pinecone_results):
        """Test querying when MongoDB doesn't have matching documents."""
        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results
        knowledge_base.mongo.find.return_value = []  # No matching MongoDB docs

        results = await knowledge_base.query(query_text="blockchain")

        # Should return empty list when no mongo docs found
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_delete_document_success(self, knowledge_base):
        """Test successful document deletion."""
        knowledge_base.mongo.delete_one.return_value = True

        result = await knowledge_base.delete_document("doc1")

        assert result is True
        knowledge_base.pinecone.delete.assert_called_once_with(
            ids=["doc1"], namespace=None)
        knowledge_base.mongo.delete_one.assert_called_once_with(
            "test_kb", {"document_id": "doc1"})

    @pytest.mark.asyncio
    async def test_delete_document_pinecone_error(self, knowledge_base):
        """Test document deletion when Pinecone raises an error."""
        knowledge_base.pinecone.delete.side_effect = Exception(
            "Pinecone error")

        result = await knowledge_base.delete_document("doc1")

        assert result is False
        knowledge_base.mongo.delete_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_metadata_only(self, knowledge_base):
        """Test updating only document metadata."""
        # Setup
        knowledge_base.mongo.find_one.return_value = {
            "document_id": "doc1",
            "content": "Original content",
            "title": "Original title",
            "tags": ["original"]
        }

        # Call the method with only metadata updates
        result = await knowledge_base.update_document(
            document_id="doc1",
            metadata={"title": "Updated title", "tags": ["updated"]}
        )

        # Assert results
        assert result is True

        # Check MongoDB was updated
        knowledge_base.mongo.update_one.assert_called_once()
        update_args = knowledge_base.mongo.update_one.call_args[0]
        update_dict = update_args[2]["$set"]
        assert update_dict["title"] == "Updated title"
        assert update_dict["tags"] == ["updated"]
        assert "updated_at" in update_dict

        # Verify Pinecone was NOT updated since content didn't change
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_content_and_metadata(self, knowledge_base):
        """Test updating both document content and metadata."""
        # Setup
        knowledge_base.mongo.find_one.return_value = {
            "document_id": "doc1",
            "content": "Original content",
            "title": "Original title",
            "source": "website",
            "tags": ["original"]
        }

        # Call the method with content and metadata updates
        result = await knowledge_base.update_document(
            document_id="doc1",
            text="Updated content",
            metadata={"title": "Updated title"}
        )

        # Assert results
        assert result is True

        # Check MongoDB was updated with new content and metadata
        knowledge_base.mongo.update_one.assert_called_once()
        update_args = knowledge_base.mongo.update_one.call_args[0]
        update_dict = update_args[2]["$set"]
        assert update_dict["content"] == "Updated content"
        assert update_dict["title"] == "Updated title"

        # Verify Pinecone was updated with new text
        knowledge_base.pinecone.upsert_text.assert_called_once()
        upsert_kwargs = knowledge_base.pinecone.upsert_text.call_args[1]
        assert upsert_kwargs["texts"] == ["Updated content"]
        assert upsert_kwargs["ids"] == ["doc1"]
        assert upsert_kwargs["metadatas"][0]["document_id"] == "doc1"
        # Preserved from original
        assert upsert_kwargs["metadatas"][0]["source"] == "website"

    @pytest.mark.asyncio
    async def test_update_document_not_found(self, knowledge_base):
        """Test updating a document that doesn't exist."""
        # Setup - document not found
        knowledge_base.mongo.find_one.return_value = None

        # Call the method
        result = await knowledge_base.update_document(
            document_id="nonexistent",
            text="New content"
        )

        # Assert results
        assert result is False
        knowledge_base.mongo.update_one.assert_not_called()
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_documents_batch(self, knowledge_base):
        """Test adding multiple documents in a batch."""
        # Setup
        batch_docs = [
            {
                "document_id": "batch1",
                "text": "Batch document 1",
                "metadata": {"title": "Batch 1", "source": "test"}
            },
            {
                "document_id": "batch2",
                "text": "Batch document 2",
                "metadata": {"title": "Batch 2", "source": "test"}
            }
        ]

        # Call the method
        result = await knowledge_base.add_documents_batch(documents=batch_docs)

        # Assert results
        assert result == ["batch1", "batch2"]

        # Check MongoDB insertions
        assert knowledge_base.mongo.insert_one.call_count == 2

        # Check Pinecone batch insertion
        knowledge_base.pinecone.upsert_text.assert_called_once()
        upsert_kwargs = knowledge_base.pinecone.upsert_text.call_args[1]
        assert len(upsert_kwargs["texts"]) == 2
        assert upsert_kwargs["ids"] == ["batch1", "batch2"]
        assert len(upsert_kwargs["metadatas"]) == 2

    @pytest.mark.asyncio
    async def test_add_documents_batch_with_auto_ids(self, knowledge_base):
        """Test adding batch documents without explicit IDs."""
        # Setup
        batch_docs = [
            {
                "text": "Auto ID document 1",
                "metadata": {"title": "Auto 1"}
            },
            {
                "text": "Auto ID document 2",
                "metadata": {"title": "Auto 2"}
            }
        ]

        # Mock UUID generation
        with patch('uuid.uuid4', side_effect=[
            uuid.UUID('aaaaaaaa-1111-2222-3333-444444444444'),
            uuid.UUID('bbbbbbbb-1111-2222-3333-444444444444')
        ]):
            result = await knowledge_base.add_documents_batch(documents=batch_docs)

        # Assert results
        assert result == [
            "aaaaaaaa-1111-2222-3333-444444444444",
            "bbbbbbbb-1111-2222-3333-444444444444"
        ]

    @pytest.mark.asyncio
    async def test_batch_with_custom_batch_size(self, knowledge_base):
        """Test adding documents with a custom batch size."""
        # Setup - create 5 documents
        docs = [{"text": f"Doc {i}", "metadata": {"title": f"Title {i}"}}
                for i in range(5)]

        # Call with batch_size=2
        with patch('asyncio.sleep') as mock_sleep:
            result = await knowledge_base.add_documents_batch(documents=docs, batch_size=2)

        # Should process in 3 batches (2, 2, 1)
        assert knowledge_base.pinecone.upsert_text.call_count == 3

        # Should sleep between batches
        assert mock_sleep.call_count == 2  # Called after first and second batch

    @pytest.mark.asyncio
    async def test_mongodb_insert_error(self, knowledge_base, sample_document):
        """Test error handling when MongoDB insert fails."""
        # Setup MongoDB to raise an error
        knowledge_base.mongo.insert_one.side_effect = Exception(
            "MongoDB insert error")

        with pytest.raises(Exception, match="MongoDB insert error"):
            await knowledge_base.add_document(
                text=sample_document["text"],
                metadata=sample_document["metadata"]
            )

        # Verify Pinecone was not called (MongoDB insert happens first)
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_with_filter(self, knowledge_base, sample_pinecone_results):
        """Test querying with filter criteria."""
        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results
        knowledge_base.mongo.find.return_value = []

        filter_criteria = {"source": "website"}
        await knowledge_base.query(
            query_text="test",
            filter=filter_criteria
        )

        # Verify filter was passed to Pinecone
        knowledge_base.pinecone.query_text.assert_called_once()
        call_kwargs = knowledge_base.pinecone.query_text.call_args[1]
        assert call_kwargs["filter"] == filter_criteria

    @pytest.mark.asyncio
    async def test_query_pinecone_error(self, knowledge_base):
        """Test error handling when Pinecone query fails."""
        # Setup Pinecone to raise an error
        knowledge_base.pinecone.query_text.side_effect = Exception(
            "Pinecone query error")

        # Should return empty results on error
        results = await knowledge_base.query(query_text="test query")
        assert results == []

        # MongoDB should not be called
        knowledge_base.mongo.find.assert_not_called()

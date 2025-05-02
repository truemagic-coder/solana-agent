import pytest
import uuid
import io
import logging
import pypdf  # Import pypdf for error testing
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from datetime import datetime, timedelta, timezone

# Assuming LlamaDocument and NodeWithScore are structured appropriately for mocking
from llama_index.core import Document as LlamaDocument
from llama_index.core.schema import TextNode

# Import the class to test
from solana_agent.services.knowledge_base import KnowledgeBaseService
from solana_agent.adapters.pinecone_adapter import PineconeAdapter
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter

# --- Fixtures ---


@pytest.fixture
def mock_pinecone_adapter():
    adapter = MagicMock(spec=PineconeAdapter)
    adapter.query = AsyncMock(return_value=[])
    adapter.upsert = AsyncMock()
    adapter.delete = AsyncMock()
    adapter.rerank = AsyncMock()  # Add mock for rerank method
    # Set default attributes expected by the service
    adapter.embedding_dimensions = 3072  # Default for large model
    adapter.use_reranking = False
    adapter.rerank_text_field = "text_content_for_rerank"
    adapter.initial_query_top_k_multiplier = 5
    return adapter


@pytest.fixture
def mock_mongodb_adapter():
    adapter = MagicMock(spec=MongoDBAdapter)
    adapter.collection_exists = MagicMock(return_value=True)
    adapter.create_collection = MagicMock()
    adapter.create_index = MagicMock()
    adapter.insert_one = MagicMock(return_value=MagicMock(inserted_id="mock_mongo_id"))
    adapter.insert_many = MagicMock(
        return_value=MagicMock(inserted_ids=["mock_mongo_id_1", "mock_mongo_id_2"])
    )
    adapter.find_one = MagicMock(return_value=None)  # Default to not found
    adapter.find = MagicMock(return_value=[])  # Default to empty list
    adapter.update_one = MagicMock(return_value=MagicMock(modified_count=1))
    adapter.delete_many = MagicMock(return_value=MagicMock(deleted_count=1))
    return adapter


@pytest.fixture
def mock_openai_embedding():
    with patch(
        "solana_agent.services.knowledge_base.OpenAIEmbedding", autospec=True
    ) as mock_embed_class:
        instance = mock_embed_class.return_value
        instance.model = "mock-embedding-model"
        instance.aget_text_embedding = AsyncMock(return_value=[0.1] * 3072)
        instance.aget_query_embedding = AsyncMock(return_value=[0.2] * 3072)
        instance.aget_text_embedding_batch = AsyncMock(
            side_effect=lambda texts, **kwargs: [
                [0.1 + i * 0.01] * 3072 for i in range(len(texts))
            ]
        )
        instance.dimensions = 3072
        yield mock_embed_class  # Yield the mock class


@pytest.fixture
def mock_semantic_splitter():
    with patch(
        "solana_agent.services.knowledge_base.SemanticSplitterNodeParser", autospec=True
    ) as mock_splitter_class:
        instance = mock_splitter_class.return_value
        instance.get_nodes_from_documents = MagicMock(return_value=[])
        # We will check the call_args later, no need to mock embed_model here
        yield mock_splitter_class  # Yield the mock class


@pytest.fixture
def mock_uuid():
    # Single UUID generation
    test_uuid = uuid.UUID("00000001-1111-1111-1111-111111111111")
    with patch("uuid.uuid4", return_value=test_uuid) as mock_uuid_patch:
        yield mock_uuid_patch


@pytest.fixture
def mock_uuid_multiple():
    # Provide more UUIDs to avoid StopIteration
    uuids = [uuid.UUID(f"{i:08x}-1111-1111-1111-111111111111") for i in range(1, 11)]
    with patch("uuid.uuid4") as mock_uuid_patch:
        # Use iterator to handle state correctly across calls
        mock_uuid_patch.side_effect = iter(uuids)
        yield mock_uuid_patch


@pytest.fixture
def mock_datetime():
    # Mock datetime.now() to return a fixed, timezone-aware datetime
    fixed_now = datetime(2025, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    with patch("solana_agent.services.knowledge_base.dt") as mock_dt:
        mock_dt.now.return_value = fixed_now
        # Make sure the class itself is available if needed
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        yield mock_dt


@pytest.fixture
def mock_pypdf_reader():
    # Mock pypdf.PdfReader
    with patch("pypdf.PdfReader", autospec=True) as mock_reader_class:
        instance = mock_reader_class.return_value
        # Default mock pages with some text
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 extracted text. "
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 extracted text."
        instance.pages = [mock_page1, mock_page2]
        yield mock_reader_class


@pytest.fixture
def mock_open():
    # Mock the built-in open function
    with patch("builtins.open", new_callable=MagicMock) as mock_open_func:
        # Configure the mock file object returned by open
        mock_file = MagicMock(spec=io.BytesIO)
        mock_file.read.return_value = b"mock pdf content"
        mock_file.__enter__.return_value = mock_file  # For context manager usage
        mock_open_func.return_value = mock_file
        yield mock_open_func


@pytest.fixture
def mock_asyncio_to_thread():
    # Mock asyncio.to_thread used for running sync splitter
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # Default behavior: return the result of the function passed to it
        async def side_effect_func(func, *args, **kwargs):
            # If the function is the splitter's method, return its configured return_value
            # This might be needed if the splitter mock setup changes
            # if func == mock_semantic_splitter.return_value.get_nodes_from_documents:
            #      return mock_semantic_splitter.return_value.get_nodes_from_documents.return_value
            return func(*args, **kwargs)  # Execute other functions normally

        mock_to_thread.side_effect = side_effect_func
        yield mock_to_thread


# --- Service Fixtures ---


@pytest.fixture
def kb_service_default(
    mock_pinecone_adapter,
    mock_mongodb_adapter,
    mock_openai_embedding,  # Mock class
    mock_semantic_splitter,  # Mock class
    mock_uuid,
    mock_datetime,
):
    # Get the mock instances that will be passed
    mock_embedding_instance = mock_openai_embedding.return_value

    # Initialize service - relies on patches applied by other fixtures
    service = KnowledgeBaseService(
        pinecone_adapter=mock_pinecone_adapter,
        mongodb_adapter=mock_mongodb_adapter,
        openai_api_key="test-key-default",
        openai_model_name="text-embedding-3-large",
        collection_name="test_kb_default",
        rerank_results=False,
    )

    # --- Verification ---
    # Verify the service holds the main mock instances
    assert service.pinecone is mock_pinecone_adapter
    assert service.mongo is mock_mongodb_adapter
    # Verify the service holds the instance created by the mocked splitter class
    assert service.semantic_splitter is mock_semantic_splitter.return_value

    # Verify the mock splitter CLASS was called correctly during service init
    mock_semantic_splitter.assert_called_once()
    call_args, call_kwargs = mock_semantic_splitter.call_args
    assert "embed_model" in call_kwargs
    assert (
        call_kwargs["embed_model"] is mock_embedding_instance
    )  # Check instance passed to constructor

    # --- Explicitly set attribute on the mock instance ---
    # Ensure the mock splitter instance has the embed_model attribute for later use by service code
    service.semantic_splitter.embed_model = mock_embedding_instance

    return service


@pytest.fixture
def kb_service_custom(
    mock_pinecone_adapter,
    mock_mongodb_adapter,
    mock_openai_embedding,  # Mock class
    mock_semantic_splitter,  # Mock class
    mock_uuid,
    mock_datetime,
):
    # Configure mocks for custom settings before service init
    mock_pinecone_adapter.embedding_dimensions = 1536
    mock_pinecone_adapter.use_reranking = True
    # Configure the mock embedding INSTANCE
    mock_embedding_instance = mock_openai_embedding.return_value
    mock_embedding_instance.dimensions = 1536
    mock_embedding_instance.aget_text_embedding.return_value = [0.1] * 1536
    mock_embedding_instance.aget_query_embedding.return_value = [0.2] * 1536
    mock_embedding_instance.aget_text_embedding_batch.side_effect = (
        lambda texts, **kwargs: [[0.1 + i * 0.01] * 1536 for i in range(len(texts))]
    )

    # Initialize service - relies on patches applied by other fixtures
    service = KnowledgeBaseService(
        pinecone_adapter=mock_pinecone_adapter,
        mongodb_adapter=mock_mongodb_adapter,
        openai_api_key="test-key-custom",
        openai_model_name="text-embedding-3-small",
        collection_name="test_kb_custom",
        rerank_results=True,
        rerank_top_k=2,
        splitter_buffer_size=2,
        splitter_breakpoint_percentile=90,
    )

    # --- Verification ---
    # Verify the service holds the main mock instances
    assert service.pinecone is mock_pinecone_adapter
    assert service.mongo is mock_mongodb_adapter
    # Verify the service holds the instance created by the mocked splitter class
    assert service.semantic_splitter is mock_semantic_splitter.return_value

    # Verify the mock splitter CLASS was called correctly during service init
    mock_semantic_splitter.assert_called_once()
    call_args, call_kwargs = mock_semantic_splitter.call_args
    assert "embed_model" in call_kwargs
    assert (
        call_kwargs["embed_model"] is mock_embedding_instance
    )  # Check instance passed to constructor

    # --- Explicitly set attribute on the mock instance ---
    # Ensure the mock splitter instance has the embed_model attribute for later use by service code
    service.semantic_splitter.embed_model = mock_embedding_instance

    return service


# --- Test Classes ---


class TestKnowledgeBaseServiceInitialization:
    # These tests remain largely the same, checking constructor logic
    def test_init_successful_defaults(
        self,
        mock_pinecone_adapter,
        mock_mongodb_adapter,
        mock_openai_embedding,
        mock_semantic_splitter,
    ):
        api_key = "test-key-init-default"
        # Service init relies on patches from fixtures
        service = KnowledgeBaseService(
            mock_pinecone_adapter,
            mock_mongodb_adapter,
            openai_api_key=api_key,
        )
        assert service.pinecone == mock_pinecone_adapter
        assert service.mongo == mock_mongodb_adapter
        assert service.collection == "knowledge_documents"
        assert service.rerank_results is False
        assert service.openai_model_name == "text-embedding-3-large"

        # Check that the correct classes were called with the right args during init
        mock_openai_embedding.assert_called_once_with(
            model="text-embedding-3-large", api_key=api_key
        )
        # Check that the splitter was initialized with the mock embedding instance
        mock_semantic_splitter.assert_called_once_with(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=mock_openai_embedding.return_value,  # Check embed_model passed
        )
        # Check collection setup calls
        mock_mongodb_adapter.collection_exists.assert_called_with("knowledge_documents")
        mock_mongodb_adapter.create_index.assert_called()

    def test_init_successful_custom_args(
        self,
        mock_pinecone_adapter,
        mock_mongodb_adapter,
        mock_openai_embedding,
        mock_semantic_splitter,
    ):
        api_key = "test-key-init-custom"
        # Configure pinecone mock before service init
        mock_pinecone_adapter.embedding_dimensions = 1536

        # Service init relies on patches from fixtures
        service = KnowledgeBaseService(
            mock_pinecone_adapter,
            mock_mongodb_adapter,
            openai_api_key=api_key,
            openai_model_name="text-embedding-3-small",
            collection_name="custom_coll",
            rerank_results=True,
            rerank_top_k=2,
            splitter_buffer_size=3,
            splitter_breakpoint_percentile=80,
        )
        assert service.collection == "custom_coll"
        assert service.rerank_results is True
        assert service.rerank_top_k == 2
        assert service.openai_model_name == "text-embedding-3-small"

        # Check that the correct classes were called with the right args during init
        mock_openai_embedding.assert_called_once_with(
            model="text-embedding-3-small", api_key=api_key
        )
        # Check that the splitter was initialized with the mock embedding instance
        mock_semantic_splitter.assert_called_once_with(
            buffer_size=3,
            breakpoint_percentile_threshold=80,
            embed_model=mock_openai_embedding.return_value,  # Check embed_model passed
        )
        mock_mongodb_adapter.collection_exists.assert_called_with("custom_coll")

    def test_init_missing_api_key_raises_error(
        self, mock_pinecone_adapter, mock_mongodb_adapter
    ):
        with pytest.raises(ValueError, match="OpenAI API key not provided"):
            KnowledgeBaseService(
                mock_pinecone_adapter, mock_mongodb_adapter, openai_api_key=""
            )

    def test_init_unknown_model_no_pinecone_dim_raises_error(
        self, mock_pinecone_adapter, mock_mongodb_adapter
    ):
        # Ensure pinecone adapter mock doesn't have the dimension attribute
        del mock_pinecone_adapter.embedding_dimensions
        api_key = "test-key-unknown-no-dim"
        with pytest.raises(
            ValueError, match="Cannot determine dimension for unknown OpenAI model"
        ):
            KnowledgeBaseService(
                mock_pinecone_adapter,
                mock_mongodb_adapter,
                openai_api_key=api_key,
                openai_model_name="unknown-model",
            )

    def test_init_unknown_model_uses_pinecone_dim_warning(
        self,
        mock_pinecone_adapter,
        mock_mongodb_adapter,
        mock_openai_embedding,
        caplog,  # Use caplog fixture
    ):
        mock_pinecone_adapter.embedding_dimensions = 1024
        api_key = "test-key-unknown"
        # Capture logs at WARNING level
        with caplog.at_level(logging.WARNING):
            KnowledgeBaseService(
                mock_pinecone_adapter,
                mock_mongodb_adapter,
                openai_api_key=api_key,
                openai_model_name="unknown-model",
            )
        mock_openai_embedding.assert_called_once_with(
            model="unknown-model", api_key=api_key
        )
        # Check log output using caplog
        assert "Unknown OpenAI model 'unknown-model'" in caplog.text
        assert "Using dimension 1024 from Pinecone config" in caplog.text

    def test_init_openai_embedding_error(
        self, mock_pinecone_adapter, mock_mongodb_adapter, mock_openai_embedding
    ):
        mock_openai_embedding.side_effect = Exception("OpenAI Init Failed")
        with pytest.raises(Exception, match="OpenAI Init Failed"):
            KnowledgeBaseService(
                mock_pinecone_adapter, mock_mongodb_adapter, openai_api_key="test-key"
            )

    def test_ensure_collection_exists(self, kb_service_default, mock_mongodb_adapter):
        # Reset mocks to simulate collection existing for this specific check
        mock_mongodb_adapter.collection_exists.reset_mock()
        mock_mongodb_adapter.create_collection.reset_mock()
        mock_mongodb_adapter.create_index.reset_mock()
        mock_mongodb_adapter.collection_exists.return_value = True

        # Re-initialize service to trigger _ensure_collection during init
        # Use the already mocked adapters from kb_service_default
        KnowledgeBaseService(
            kb_service_default.pinecone,
            kb_service_default.mongo,
            openai_api_key="test-key-ensure-exists",
            collection_name=kb_service_default.collection,
        )

        mock_mongodb_adapter.collection_exists.assert_called_once_with(
            kb_service_default.collection
        )
        mock_mongodb_adapter.create_collection.assert_not_called()
        # Indexes should still be checked/created even if collection exists
        mock_mongodb_adapter.create_index.assert_called()

    def test_ensure_collection_does_not_exist(
        self,
        mock_pinecone_adapter,
        mock_mongodb_adapter,
        mock_openai_embedding,
        mock_semantic_splitter,  # Needed for service init
    ):
        # Reset mocks and set collection_exists to False for this specific test
        mock_mongodb_adapter.collection_exists.reset_mock()
        mock_mongodb_adapter.create_collection.reset_mock()
        mock_mongodb_adapter.create_index.reset_mock()
        mock_mongodb_adapter.collection_exists.return_value = False

        # Re-initialize service to trigger _ensure_collection
        collection_name = "new_collection"
        KnowledgeBaseService(
            mock_pinecone_adapter,
            mock_mongodb_adapter,
            openai_api_key="test-key-ensure",
            collection_name=collection_name,
        )

        mock_mongodb_adapter.collection_exists.assert_called_once_with(collection_name)
        mock_mongodb_adapter.create_collection.assert_called_once_with(collection_name)
        mock_mongodb_adapter.create_index.assert_called()


@pytest.mark.asyncio
class TestKnowledgeBaseServiceAddDocument:
    # Tests for adding plain text documents
    async def test_add_document_success_rerank_off(
        self,
        kb_service_default,  # Uses the corrected fixture
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_uuid,
        mock_datetime,
    ):
        service = kb_service_default
        text = "This is a test document."
        metadata = {"source": "test_source", "tags": ["tag1", "tag2"]}
        expected_id = str(mock_uuid.return_value)
        # Access embed model via service instance which should have the correct mock
        expected_embedding = (
            service.semantic_splitter.embed_model.aget_text_embedding.return_value
        )
        now = mock_datetime.now.return_value

        doc_id = await service.add_document(text, metadata, namespace="ns1")

        assert doc_id == expected_id
        # Check Mongo insert
        mock_mongodb_adapter.insert_one.assert_called_once_with(
            service.collection,
            {
                "document_id": expected_id,
                "content": text,
                "is_chunk": False,
                "parent_document_id": None,
                "source": "test_source",
                "tags": ["tag1", "tag2"],
                "created_at": now,
                "updated_at": now,
            },
        )
        # Check embedding call via service instance
        service.semantic_splitter.embed_model.aget_text_embedding.assert_awaited_once_with(
            text
        )
        # Check Pinecone upsert
        mock_pinecone_adapter.upsert.assert_awaited_once()
        upsert_call_kwargs = mock_pinecone_adapter.upsert.await_args.kwargs
        assert upsert_call_kwargs["namespace"] == "ns1"
        vector = upsert_call_kwargs["vectors"][0]
        assert vector["id"] == expected_id
        assert vector["values"] == expected_embedding
        assert vector["metadata"] == {
            "document_id": expected_id,
            "is_chunk": False,
            "parent_document_id": None,
            "source": "test_source",
            "tags": ["tag1", "tag2"],
        }
        # Ensure rerank field is NOT present
        assert service.pinecone.rerank_text_field not in vector["metadata"]

    async def test_add_document_success_rerank_on(
        self,
        kb_service_custom,  # Uses the corrected fixture
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_uuid,
        mock_datetime,
    ):
        service = kb_service_custom
        text = "Another test document for reranking."
        metadata = {"source": "rerank_source"}
        expected_id = str(mock_uuid.return_value)
        # Access embed model via service instance
        expected_embedding = (
            service.semantic_splitter.embed_model.aget_text_embedding.return_value
        )
        now = mock_datetime.now.return_value

        doc_id = await service.add_document(text, metadata)

        assert doc_id == expected_id
        # Check Mongo insert
        mock_mongodb_adapter.insert_one.assert_called_once_with(
            service.collection,
            {
                "document_id": expected_id,
                "content": text,
                "is_chunk": False,
                "parent_document_id": None,
                "source": "rerank_source",
                "tags": [],  # Default tags if not provided
                "created_at": now,
                "updated_at": now,
            },
        )
        # Check embedding call via service instance
        service.semantic_splitter.embed_model.aget_text_embedding.assert_awaited_once_with(
            text
        )
        # Check Pinecone upsert
        mock_pinecone_adapter.upsert.assert_awaited_once()
        upsert_call_kwargs = mock_pinecone_adapter.upsert.await_args.kwargs
        assert upsert_call_kwargs["namespace"] is None
        vector = upsert_call_kwargs["vectors"][0]
        assert vector["id"] == expected_id
        assert vector["values"] == expected_embedding
        assert vector["metadata"] == {
            "document_id": expected_id,
            "is_chunk": False,
            "parent_document_id": None,
            "source": "rerank_source",
            "tags": [],
            service.pinecone.rerank_text_field: text,
        }

    async def test_add_document_with_provided_id(
        self,
        kb_service_default,
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_datetime,
    ):
        service = kb_service_default
        text = "Doc with provided ID."
        metadata = {"source": "provided_id_src"}
        provided_id = "my-custom-doc-id-123"
        now = mock_datetime.now.return_value

        doc_id = await service.add_document(text, metadata, document_id=provided_id)

        assert doc_id == provided_id
        # Check Mongo insert used the provided ID
        mongo_call_args = mock_mongodb_adapter.insert_one.call_args[0][1]
        assert mongo_call_args["document_id"] == provided_id
        # Check Pinecone upsert used the provided ID
        pinecone_call_args = mock_pinecone_adapter.upsert.await_args.kwargs["vectors"][
            0
        ]
        assert pinecone_call_args["id"] == provided_id
        assert pinecone_call_args["metadata"]["document_id"] == provided_id

    async def test_add_document_mongo_error(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, caplog
    ):
        service = kb_service_default
        mock_mongodb_adapter.insert_one.side_effect = Exception("Mongo insert failed")

        with pytest.raises(Exception, match="Mongo insert failed"):
            await service.add_document("text", {})

        mock_mongodb_adapter.insert_one.assert_called_once()
        # Embedding and Pinecone should not be called if Mongo fails first
        service.semantic_splitter.embed_model.aget_text_embedding.assert_not_awaited()
        mock_pinecone_adapter.upsert.assert_not_awaited()
        assert "Error inserting document" in caplog.text

    async def test_add_document_embedding_error(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, caplog
    ):
        service = kb_service_default
        service.semantic_splitter.embed_model.aget_text_embedding.side_effect = (
            Exception("Embedding failed")
        )

        with pytest.raises(Exception, match="Embedding failed"):
            await service.add_document("text", {})

        mock_mongodb_adapter.insert_one.assert_called_once()  # Mongo insert succeeded
        service.semantic_splitter.embed_model.aget_text_embedding.assert_awaited_once()  # Embedding attempted
        mock_pinecone_adapter.upsert.assert_not_awaited()  # Pinecone not called
        assert "Error embedding document" in caplog.text

    async def test_add_document_pinecone_error(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, caplog
    ):
        service = kb_service_default
        mock_pinecone_adapter.upsert.side_effect = Exception("Pinecone upsert failed")

        with pytest.raises(Exception, match="Pinecone upsert failed"):
            await service.add_document("text", {})

        mock_mongodb_adapter.insert_one.assert_called_once()  # Mongo insert succeeded
        service.semantic_splitter.embed_model.aget_text_embedding.assert_awaited_once()  # Embedding succeeded
        mock_pinecone_adapter.upsert.assert_awaited_once()  # Pinecone attempted
        assert "Error upserting vector" in caplog.text


@pytest.mark.asyncio
class TestKnowledgeBaseServiceAddPdfDocument:
    # Tests for PDF documents, focusing on chunking and storage changes
    async def test_add_pdf_document_bytes_success_rerank_on(
        self,
        kb_service_custom,  # Rerank on, small model
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_pypdf_reader,
        mock_semantic_splitter,
        mock_uuid,
        mock_datetime,
        mock_asyncio_to_thread,
    ):
        service = kb_service_custom
        pdf_bytes = b"fake pdf content"
        metadata = {"source": "pdf_bytes_test", "year": 2024}
        expected_parent_id = str(mock_uuid.return_value)
        expected_extracted_text = "Page 1 extracted text. Page 2 extracted text."
        expected_chunk_texts = ["Chunk 1 text.", "Chunk 2 text."]
        # Mock embeddings for small model (1536 dims)
        expected_chunk_embeddings = [
            [0.1 + i * 0.01] * 1536 for i in range(len(expected_chunk_texts))
        ]
        service.semantic_splitter.embed_model.aget_text_embedding_batch.return_value = (
            expected_chunk_embeddings
        )

        # Mock splitter to return 2 nodes
        splitter_instance = mock_semantic_splitter.return_value
        mock_nodes = [
            TextNode(text=t, id_=f"node_{i}")
            for i, t in enumerate(expected_chunk_texts)
        ]
        splitter_instance.get_nodes_from_documents.return_value = mock_nodes
        now = mock_datetime.now.return_value

        doc_id = await service.add_pdf_document(pdf_bytes, metadata)

        assert doc_id == expected_parent_id
        mock_pypdf_reader.assert_called_once_with(
            ANY
        )  # Check it was called with BytesIO
        assert isinstance(mock_pypdf_reader.call_args[0][0], io.BytesIO)

        # Check Parent Doc Insertion in Mongo
        mock_mongodb_adapter.insert_one.assert_called_once_with(
            service.collection,
            {
                "document_id": expected_parent_id,
                "content": expected_extracted_text,  # Full extracted text stored
                "is_chunk": False,
                "parent_document_id": None,
                "source": "pdf_bytes_test",
                "year": 2024,
                "created_at": now,
                "updated_at": now,
            },
        )

        # Check Splitter Call (via asyncio.to_thread)
        mock_asyncio_to_thread.assert_awaited_once()
        # Check the underlying sync function was called correctly
        splitter_instance.get_nodes_from_documents.assert_called_once()
        assert isinstance(
            splitter_instance.get_nodes_from_documents.call_args[0][0][0], LlamaDocument
        )
        assert (
            splitter_instance.get_nodes_from_documents.call_args[0][0][0].text
            == expected_extracted_text
        )

        # Check Chunk Embedding Call (single batch call)
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_awaited_once_with(
            expected_chunk_texts, show_progress=True
        )

        # Check Chunk Docs Insertion in Mongo (single call for all chunks)
        mock_mongodb_adapter.insert_many.assert_called_once()
        chunk_mongo_call_args = mock_mongodb_adapter.insert_many.call_args[0][1]
        assert len(chunk_mongo_call_args) == 2
        assert (
            chunk_mongo_call_args[0]["document_id"] == f"{expected_parent_id}_chunk_0"
        )
        assert chunk_mongo_call_args[0]["parent_document_id"] == expected_parent_id
        assert chunk_mongo_call_args[0]["is_chunk"] is True
        assert chunk_mongo_call_args[0]["content"] == expected_chunk_texts[0]
        assert chunk_mongo_call_args[0]["source"] == "pdf_bytes_test"  # Inherited
        assert (
            chunk_mongo_call_args[1]["document_id"] == f"{expected_parent_id}_chunk_1"
        )
        assert chunk_mongo_call_args[1]["chunk_index"] == 1

        # Check Pinecone Upsert (single call as batch size > num chunks)
        mock_pinecone_adapter.upsert.assert_awaited_once()
        upsert_call_kwargs = mock_pinecone_adapter.upsert.await_args.kwargs
        assert upsert_call_kwargs["namespace"] is None
        vectors = upsert_call_kwargs["vectors"]
        assert len(vectors) == 2

        # Vector 0
        assert vectors[0]["id"] == f"{expected_parent_id}_chunk_0"
        assert vectors[0]["values"] == expected_chunk_embeddings[0]
        assert vectors[0]["metadata"] == {
            "document_id": f"{expected_parent_id}_chunk_0",
            "parent_document_id": expected_parent_id,
            "chunk_index": 0,
            "is_chunk": True,
            "source": "pdf_bytes_test",
            "tags": [],  # Inherited default
            "year": 2024,  # Inherited from parent metadata
            service.pinecone.rerank_text_field: expected_chunk_texts[
                0
            ],  # Rerank text included
        }
        # Vector 1
        assert vectors[1]["id"] == f"{expected_parent_id}_chunk_1"
        assert vectors[1]["values"] == expected_chunk_embeddings[1]
        assert (
            vectors[1]["metadata"][service.pinecone.rerank_text_field]
            == expected_chunk_texts[1]
        )

    async def test_add_pdf_document_path_success_rerank_off(
        self,
        kb_service_default,  # Reranking disabled
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_pypdf_reader,
        mock_semantic_splitter,
        mock_uuid,
        mock_datetime,
        mock_open,
        mock_asyncio_to_thread,
    ):
        service = kb_service_default
        pdf_path = "/path/to/my.pdf"
        metadata = {"source": "pdf_path", "tags": ["report"]}
        expected_parent_id = str(mock_uuid.return_value)
        expected_chunk_texts = ["Chunk 1 text.", "Chunk 2 text."]
        # Mock embeddings for large model (3072 dims)
        expected_chunk_embeddings = [
            [0.1 + i * 0.01] * 3072 for i in range(len(expected_chunk_texts))
        ]
        service.semantic_splitter.embed_model.aget_text_embedding_batch.return_value = (
            expected_chunk_embeddings
        )

        splitter_instance = mock_semantic_splitter.return_value
        mock_nodes = [
            TextNode(text=t, id_=f"node_{i}")
            for i, t in enumerate(expected_chunk_texts)
        ]
        splitter_instance.get_nodes_from_documents.return_value = mock_nodes

        doc_id = await service.add_pdf_document(pdf_path, metadata, namespace="pdf-ns")

        assert doc_id == expected_parent_id
        mock_open.assert_called_once_with(pdf_path, "rb")
        mock_pypdf_reader.assert_called_once_with(ANY)
        mock_mongodb_adapter.insert_one.assert_called_once()  # Parent doc
        mock_asyncio_to_thread.assert_awaited_once()
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_awaited_once()
        mock_mongodb_adapter.insert_many.assert_called_once()  # Chunk docs

        # Check Pinecone upsert namespace and lack of rerank text
        mock_pinecone_adapter.upsert.assert_awaited_once()
        upsert_call_kwargs = mock_pinecone_adapter.upsert.await_args.kwargs
        assert upsert_call_kwargs["namespace"] == "pdf-ns"
        vectors = upsert_call_kwargs["vectors"]
        assert len(vectors) == 2
        assert service.pinecone.rerank_text_field not in vectors[0]["metadata"]
        assert service.pinecone.rerank_text_field not in vectors[1]["metadata"]
        # Check inherited metadata
        assert vectors[0]["metadata"]["source"] == "pdf_path"
        assert vectors[0]["metadata"]["tags"] == ["report"]

    async def test_add_pdf_document_invalid_type_error(self, kb_service_default):
        with pytest.raises(
            ValueError, match="pdf_data must be bytes or a file path string"
        ):
            await kb_service_default.add_pdf_document(12345, {})

    async def test_add_pdf_document_pdf_read_error(
        self, kb_service_default, mock_pypdf_reader, mock_mongodb_adapter, caplog
    ):
        # Use pypdf.errors specific error if available, else generic Exception
        pdf_error = getattr(pypdf.errors, "PdfReadError", Exception)
        mock_pypdf_reader.side_effect = pdf_error("Invalid PDF")

        with (
            pytest.raises(pdf_error, match="Invalid PDF"),
            caplog.at_level(logging.ERROR),
        ):
            await kb_service_default.add_pdf_document(b"bad-bytes", {})

        # Ensure Mongo insert wasn't called if reading failed early
        mock_mongodb_adapter.insert_one.assert_not_called()
        assert "Error reading or extracting text from PDF" in caplog.text

    async def test_add_pdf_document_empty_pdf_no_chunks(
        self,
        kb_service_default,
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_pypdf_reader,
        mock_semantic_splitter,
        mock_uuid,
        mock_datetime,
        mock_asyncio_to_thread,
        caplog,  # Use caplog fixture
    ):
        service = kb_service_default
        reader_instance = mock_pypdf_reader.return_value
        mock_page_empty = MagicMock()
        mock_page_empty.extract_text.return_value = "  \n  "  # Whitespace only
        reader_instance.pages = [mock_page_empty]
        expected_parent_id = str(mock_uuid.return_value)
        now = mock_datetime.now.return_value

        # Capture logs at WARNING level
        with caplog.at_level(logging.WARNING):
            doc_id = await service.add_pdf_document(b"empty", {"source": "empty"})

        assert doc_id == expected_parent_id
        # Parent doc still inserted, but with empty extracted text
        mock_mongodb_adapter.insert_one.assert_called_once_with(
            service.collection,
            {
                "document_id": expected_parent_id,
                "content": "  \n  ",  # Extracted whitespace
                "is_chunk": False,
                "parent_document_id": None,
                "source": "empty",
                "created_at": now,
                "updated_at": now,
            },
        )
        # Ensure splitter, embedding, chunk insert, and upsert were not called
        mock_asyncio_to_thread.assert_not_awaited()
        mock_semantic_splitter.return_value.get_nodes_from_documents.assert_not_called()
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_not_awaited()
        mock_mongodb_adapter.insert_many.assert_not_called()
        mock_pinecone_adapter.upsert.assert_not_awaited()
        # Check for warning log
        assert f"No text extracted from PDF {expected_parent_id}" in caplog.text

    async def test_add_pdf_document_no_chunks_generated(
        self,
        kb_service_default,
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_pypdf_reader,  # Returns text
        mock_semantic_splitter,  # Returns no nodes
        mock_uuid,
        mock_datetime,
        mock_asyncio_to_thread,
        caplog,  # Use caplog fixture
    ):
        """Test case where text is extracted, but splitter returns no nodes."""
        service = kb_service_default
        expected_parent_id = str(mock_uuid.return_value)
        expected_extracted_text = "Page 1 extracted text. Page 2 extracted text."

        # Mock splitter to return empty list
        splitter_instance = mock_semantic_splitter.return_value
        splitter_instance.get_nodes_from_documents.return_value = []

        with caplog.at_level(logging.INFO):  # Check info logs too
            doc_id = await service.add_pdf_document(b"data", {"source": "no_nodes"})

        assert doc_id == expected_parent_id
        mock_mongodb_adapter.insert_one.assert_called_once()  # Parent inserted
        assert (
            mock_mongodb_adapter.insert_one.call_args[0][1]["content"]
            == expected_extracted_text
        )

        # Splitter was called
        mock_asyncio_to_thread.assert_awaited_once()
        splitter_instance.get_nodes_from_documents.assert_called_once()
        assert "Generated 0 semantic chunks" in caplog.text

        # Embedding, chunk insert, and upsert should NOT be called
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_not_awaited()
        mock_mongodb_adapter.insert_many.assert_not_called()
        mock_pinecone_adapter.upsert.assert_not_awaited()

    async def test_add_pdf_document_splitter_error(
        self,
        kb_service_default,
        mock_mongodb_adapter,
        mock_pypdf_reader,
        mock_semantic_splitter,
        mock_asyncio_to_thread,
        mock_pinecone_adapter,
        caplog,
    ):
        """Test error during the semantic splitting process."""
        service = kb_service_default
        splitter_instance = mock_semantic_splitter.return_value
        splitter_instance.get_nodes_from_documents.side_effect = Exception(
            "Splitter crashed"
        )

        # Mock to_thread to raise the exception when the sync function is called
        async def raise_splitter_error(func, *args, **kwargs):
            raise Exception("Splitter crashed")

        mock_asyncio_to_thread.side_effect = raise_splitter_error

        with (
            pytest.raises(Exception, match="Splitter crashed"),
            caplog.at_level(logging.ERROR),
        ):
            await service.add_pdf_document(b"data", {})

        mock_mongodb_adapter.insert_one.assert_called_once()  # Parent insert succeeded
        mock_asyncio_to_thread.assert_awaited_once()  # Splitter call attempted
        # Embedding, chunk insert, and upsert should not happen
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_not_awaited()
        mock_mongodb_adapter.insert_many.assert_not_called()
        mock_pinecone_adapter.upsert.assert_not_awaited()
        assert "Error during semantic chunking" in caplog.text

    async def test_add_pdf_document_embedding_batch_error(
        self,
        kb_service_default,
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_pypdf_reader,
        mock_semantic_splitter,
        mock_asyncio_to_thread,
        caplog,
    ):
        """Test error during the batch embedding of chunks."""
        service = kb_service_default
        # Mock splitter to return nodes
        splitter_instance = mock_semantic_splitter.return_value
        mock_nodes = [TextNode(text="Chunk 1"), TextNode(text="Chunk 2")]
        splitter_instance.get_nodes_from_documents.return_value = mock_nodes
        # Mock embedding batch to fail
        service.semantic_splitter.embed_model.aget_text_embedding_batch.side_effect = (
            Exception("Batch embed failed")
        )

        with (
            pytest.raises(Exception, match="Batch embed failed"),
            caplog.at_level(logging.ERROR),
        ):
            await service.add_pdf_document(b"data", {})

        mock_mongodb_adapter.insert_one.assert_called_once()  # Parent insert succeeded
        mock_asyncio_to_thread.assert_awaited_once()  # Splitter succeeded
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_awaited_once()  # Embedding attempted
        # Mongo chunk insert and Pinecone upsert should not happen
        mock_mongodb_adapter.insert_many.assert_not_called()
        mock_pinecone_adapter.upsert.assert_not_awaited()
        assert "Error embedding chunks" in caplog.text

    async def test_add_pdf_document_mongo_chunk_insert_error(
        self,
        kb_service_default,
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_pypdf_reader,
        mock_semantic_splitter,
        mock_asyncio_to_thread,
        caplog,  # Use caplog
    ):
        """Test error during MongoDB chunk insertion, Pinecone should still be attempted."""
        service = kb_service_default
        # Mock splitter to return nodes
        splitter_instance = mock_semantic_splitter.return_value
        mock_nodes = [TextNode(text="Chunk 1"), TextNode(text="Chunk 2")]
        splitter_instance.get_nodes_from_documents.return_value = mock_nodes
        # Mock embedding to succeed
        service.semantic_splitter.embed_model.aget_text_embedding_batch.return_value = [
            [0.1] * 3072,
            [0.2] * 3072,
        ]
        # Mock Mongo insert_many to fail
        mock_mongodb_adapter.insert_many.side_effect = Exception(
            "Mongo chunk insert failed"
        )

        # Should not raise, but log error and continue to Pinecone
        with caplog.at_level(logging.ERROR):
            await service.add_pdf_document(b"data", {})

        mock_mongodb_adapter.insert_one.assert_called_once()  # Parent insert
        mock_asyncio_to_thread.assert_awaited_once()  # Splitter
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_awaited_once()  # Embedding
        mock_mongodb_adapter.insert_many.assert_called_once()  # Mongo chunk insert attempted
        mock_pinecone_adapter.upsert.assert_awaited_once()  # Pinecone upsert still attempted

        # Check log output
        assert "Error inserting chunks into MongoDB" in caplog.text

    async def test_add_pdf_document_pinecone_batch_error(
        self,
        kb_service_default,
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_pypdf_reader,
        mock_semantic_splitter,
        mock_asyncio_to_thread,
        caplog,  # Use caplog
    ):
        service = kb_service_default
        # Mock splitter to return nodes
        splitter_instance = mock_semantic_splitter.return_value
        mock_nodes = [TextNode(text="Chunk 1"), TextNode(text="Chunk 2")]
        splitter_instance.get_nodes_from_documents.return_value = mock_nodes
        # Mock embedding to succeed
        service.semantic_splitter.embed_model.aget_text_embedding_batch.return_value = [
            [0.1] * 3072,
            [0.2] * 3072,
        ]
        # Mock Mongo insert_many to succeed
        mock_mongodb_adapter.insert_many.return_value = MagicMock(
            inserted_ids=["m1", "m2"]
        )
        # Mock Pinecone upsert to fail
        mock_pinecone_adapter.upsert.side_effect = Exception("Pinecone chunk failed")

        # Should not raise, but log errors
        # Capture logs at WARNING and ERROR levels
        with caplog.at_level(logging.WARNING):
            await service.add_pdf_document(
                b"data", {}, chunk_batch_size=1
            )  # Use batch size 1 to trigger multiple calls

        mock_mongodb_adapter.insert_one.assert_called_once()  # Parent insert
        mock_asyncio_to_thread.assert_awaited_once()  # Splitter
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_awaited_once()  # Embedding
        mock_mongodb_adapter.insert_many.assert_called_once()  # Mongo chunk insert
        # Pinecone upsert attempted multiple times due to batch size
        assert mock_pinecone_adapter.upsert.await_count == 2

        # Check log output
        assert "Error upserting vector batch" in caplog.text
        assert "Some errors occurred during Pinecone vector upsert" in caplog.text

    async def test_add_pdf_document_multiple_batches(
        self,
        kb_service_custom,
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_pypdf_reader,
        mock_semantic_splitter,
        mock_uuid,  # Correct: Only one UUID needed for the parent doc
        mock_datetime,
        mock_asyncio_to_thread,
    ):
        service = kb_service_custom
        pdf_bytes = b"multi-batch-pdf"
        metadata = {"source": "multi_batch"}
        expected_parent_id = str(mock_uuid.return_value)  # Use the single mock UUID
        now = mock_datetime.now.return_value

        # Simulate 5 chunks
        expected_chunk_texts = [f"Chunk {i} text." for i in range(5)]
        # Mock splitter to return 5 nodes
        splitter_instance = mock_semantic_splitter.return_value
        mock_nodes = [
            TextNode(text=t, id_=f"node_{i}")
            for i, t in enumerate(expected_chunk_texts)
        ]
        splitter_instance.get_nodes_from_documents.return_value = mock_nodes

        # Mock embeddings (single call for all 5 chunks)
        expected_chunk_embeddings = [
            [0.1 + i * 0.01] * 1536
            for i in range(len(expected_chunk_texts))  # Custom service uses 1536
        ]
        service.semantic_splitter.embed_model.aget_text_embedding_batch.return_value = (
            expected_chunk_embeddings
        )

        # Set batch size for the test call
        test_batch_size = 2

        doc_id = await service.add_pdf_document(
            pdf_bytes, metadata, chunk_batch_size=test_batch_size
        )

        assert doc_id == expected_parent_id
        mock_pypdf_reader.assert_called_once()
        mock_mongodb_adapter.insert_one.assert_called_once()  # Parent doc

        # Check Splitter Call
        mock_asyncio_to_thread.assert_awaited_once()

        # Check Chunk Embedding Call (single batch call)
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_awaited_once_with(
            expected_chunk_texts, show_progress=True
        )

        # Check Chunk Docs Insertion in Mongo (single call for all chunks)
        mock_mongodb_adapter.insert_many.assert_called_once()
        chunk_mongo_call_args = mock_mongodb_adapter.insert_many.call_args[0][1]
        assert len(chunk_mongo_call_args) == 5
        assert (
            chunk_mongo_call_args[0]["document_id"] == f"{expected_parent_id}_chunk_0"
        )
        assert (
            chunk_mongo_call_args[4]["document_id"] == f"{expected_parent_id}_chunk_4"
        )

        # Check Pinecone Upsert (multiple calls due to batch size)
        assert (
            mock_pinecone_adapter.upsert.await_count == 3
        )  # 5 chunks, batch size 2 -> 3 calls

        # Check first batch call (chunks 0, 1)
        call1_kwargs = mock_pinecone_adapter.upsert.await_args_list[0].kwargs
        assert len(call1_kwargs["vectors"]) == 2
        assert call1_kwargs["vectors"][0]["id"] == f"{expected_parent_id}_chunk_0"
        assert call1_kwargs["vectors"][1]["id"] == f"{expected_parent_id}_chunk_1"

        # Check second batch call (chunks 2, 3)
        call2_kwargs = mock_pinecone_adapter.upsert.await_args_list[1].kwargs
        assert len(call2_kwargs["vectors"]) == 2
        assert call2_kwargs["vectors"][0]["id"] == f"{expected_parent_id}_chunk_2"
        assert call2_kwargs["vectors"][1]["id"] == f"{expected_parent_id}_chunk_3"

        # Check third batch call (chunk 4)
        call3_kwargs = mock_pinecone_adapter.upsert.await_args_list[2].kwargs
        assert len(call3_kwargs["vectors"]) == 1
        assert call3_kwargs["vectors"][0]["id"] == f"{expected_parent_id}_chunk_4"


@pytest.mark.asyncio
class TestKnowledgeBaseServiceQuery:
    # Tests for querying, including interaction between Pinecone and Mongo
    async def test_query_success_plain_doc_and_chunk(
        self,
        kb_service_default,  # Uses corrected fixture
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_datetime,
    ):
        service = kb_service_default
        query = "find my documents"
        top_k = 5
        namespace = "test-ns"
        # Access embed model via service instance
        expected_query_vector = (
            service.semantic_splitter.embed_model.aget_query_embedding.return_value
        )
        now = mock_datetime.now.return_value

        # Mock Pinecone response (mix of plain doc and chunk)
        pinecone_res = [
            # ... (pinecone response setup) ...
            {
                "id": "plain_doc_1",
                "score": 0.9,
                "metadata": {
                    "document_id": "plain_doc_1",
                    "is_chunk": False,
                    "source": "pinecone_s1",
                },
            },
            {
                "id": "pdf1_chunk_0",
                "score": 0.8,
                "metadata": {
                    "document_id": "pdf1_chunk_0",
                    "parent_document_id": "pdf1",
                    "is_chunk": True,
                    "chunk_index": 0,
                    "source": "pinecone_pdf_source",
                },
            },
        ]
        mock_pinecone_adapter.query.return_value = pinecone_res

        # Mock Mongo response
        mongo_docs = [
            # ... (mongo response setup) ...
            {
                "_id": "mongo_plain_1",
                "document_id": "plain_doc_1",
                "content": "Content of plain doc 1.",
                "is_chunk": False,
                "parent_document_id": None,
                "source": "mongo_s1",
                "tags": ["mongo_tag"],
                "created_at": now - timedelta(days=1),
                "updated_at": now,
            },
            {
                "_id": "mongo_chunk_0",
                "document_id": "pdf1_chunk_0",
                "content": "Content of PDF 1 chunk 0.",
                "is_chunk": True,
                "parent_document_id": "pdf1",
                "chunk_index": 0,
                "source": "mongo_pdf_source",
                "created_at": now - timedelta(days=2),
                "updated_at": now - timedelta(days=1),
            },
            {
                "_id": "mongo_parent_1",
                "document_id": "pdf1",
                "content": "Full extracted text of PDF 1.",
                "is_chunk": False,
                "parent_document_id": None,
                "source": "mongo_pdf_source",
                "year": 2023,
                "created_at": now - timedelta(days=2),
                "updated_at": now - timedelta(days=1),
            },
        ]
        mock_mongodb_adapter.find.return_value = mongo_docs

        results = await service.query(query, top_k=top_k, namespace=namespace)

        # Check embedding call
        service.semantic_splitter.embed_model.aget_query_embedding.assert_awaited_once_with(
            query
        )
        # Check Pinecone query call
        mock_pinecone_adapter.query.assert_awaited_once_with(
            vector=expected_query_vector,
            top_k=top_k,  # No rerank, so top_k is passed directly
            filter=None,
            namespace=namespace,
            include_metadata=True,
        )
        # Check Mongo find call (should query for all unique IDs: plain_doc_1, pdf1_chunk_0, pdf1)
        mock_mongodb_adapter.find.assert_called_once()
        mongo_query_filter = mock_mongodb_adapter.find.call_args[0][1]
        assert mongo_query_filter["document_id"]["$in"] == [
            "plain_doc_1",
            "pdf1_chunk_0",
            "pdf1",
        ]

        # Check results structure and content
        assert len(results) == 2

        # Result 1 (Plain Doc)
        assert results[0].document_id == "plain_doc_1"
        assert results[0].content == "Content of plain doc 1."
        assert results[0].score == 0.9
        assert results[0].metadata["source"] == "mongo_s1"  # Mongo source used
        assert results[0].metadata["tags"] == ["mongo_tag"]
        assert results[0].metadata["is_chunk"] is False
        assert (
            "parent_document_id" not in results[0].metadata
        )  # Should be None or absent

        # Result 2 (Chunk)
        assert results[1].document_id == "pdf1_chunk_0"
        assert results[1].content == "Content of PDF 1 chunk 0."
        assert results[1].score == 0.8
        assert results[1].metadata["source"] == "mongo_pdf_source"  # Mongo source used
        assert results[1].metadata["is_chunk"] is True
        assert results[1].metadata["parent_document_id"] == "pdf1"
        assert results[1].metadata["chunk_index"] == 0
        assert results[1].metadata["year"] == 2023  # Inherited from parent

        # more assertions can be added here as needed
        assert len(results) == 2
        assert results[0].document_id == "plain_doc_1"
        assert results[1].document_id == "pdf1_chunk_0"

    async def test_query_success_rerank_on(
        self,
        kb_service_custom,
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        mock_datetime,
    ):
        service = kb_service_custom  # Rerank is ON
        query = "find reranked docs"
        top_k = 2  # Final desired top_k
        namespace = "rerank-ns"
        expected_query_vector = (
            service.semantic_splitter.embed_model.aget_query_embedding.return_value
        )
        now = mock_datetime.now.return_value

        # Mock Pinecone response (more results initially due to multiplier)
        initial_top_k = (
            top_k * service.pinecone.initial_query_top_k_multiplier
        )  # 2 * 5 = 10
        pinecone_res = [
            {  # Will be reranked lower
                "id": "doc_low",
                "score": 0.9,
                "metadata": {
                    "document_id": "doc_low",
                    "is_chunk": False,
                    service.pinecone.rerank_text_field: "Irrelevant text",
                },
            },
            {  # Will be reranked higher
                "id": "doc_high",
                "score": 0.8,
                "metadata": {
                    "document_id": "doc_high",
                    "is_chunk": False,
                    service.pinecone.rerank_text_field: "Highly relevant text for reranking",
                },
            },
            {  # Will be kept
                "id": "doc_mid",
                "score": 0.85,
                "metadata": {
                    "document_id": "doc_mid",
                    "is_chunk": False,
                    service.pinecone.rerank_text_field: "Medium relevant text",
                },
            },
            # ... potentially more results up to initial_top_k
        ]
        mock_pinecone_adapter.query.return_value = pinecone_res[
            :3
        ]  # Simulate Pinecone returning 3

        # Mock Mongo response
        mongo_docs = [
            {
                "_id": "m1",
                "document_id": "doc_low",
                "content": "Low content",
                "is_chunk": False,
                "source": "s1",
            },
            {
                "_id": "m2",
                "document_id": "doc_high",
                "content": "High content",
                "is_chunk": False,
                "source": "s2",
            },
            {
                "_id": "m3",
                "document_id": "doc_mid",
                "content": "Mid content",
                "is_chunk": False,
                "source": "s3",
            },
        ]
        mock_mongodb_adapter.find.return_value = mongo_docs

        # Mock the reranker call (assuming it's part of the service or adapter)
        # Let's assume rerank is a method on the pinecone adapter for simplicity here
        async def mock_rerank(query, docs, top_n):
            # Simulate reranking based on the rerank_text_field content
            # Return top_n results, reordered and rescored
            reranked = sorted(
                docs,
                key=lambda d: d.metadata.get(
                    service.pinecone.rerank_text_field, ""
                ).find("relevant"),
                reverse=True,
            )
            # Assign new scores (example logic)
            for i, doc in enumerate(reranked):
                doc.score = 1.0 - (i * 0.1)
            return reranked[:top_n]

        mock_pinecone_adapter.rerank = AsyncMock(side_effect=mock_rerank)

        results = await service.query(query, top_k=top_k, namespace=namespace)

        # Check embedding call
        service.semantic_splitter.embed_model.aget_query_embedding.assert_awaited_once_with(
            query
        )
        # Check Pinecone query call used initial_top_k
        mock_pinecone_adapter.query.assert_awaited_once_with(
            vector=expected_query_vector,
            top_k=initial_top_k,  # Used multiplier
            filter=None,
            namespace=namespace,
            include_metadata=True,  # Metadata needed for reranking text
        )
        # Check Mongo find call
        mock_mongodb_adapter.find.assert_called_once()
        assert mock_mongodb_adapter.find.call_args[0][1]["document_id"]["$in"] == [
            "doc_low",
            "doc_high",
            "doc_mid",
        ]

        # Check rerank call
        mock_pinecone_adapter.rerank.assert_awaited_once()
        rerank_args = mock_pinecone_adapter.rerank.await_args
        assert rerank_args.args[0] == query  # Query text passed
        assert len(rerank_args.args[1]) == 3  # Docs retrieved from Mongo passed
        assert rerank_args.kwargs["top_n"] == top_k  # Final top_k passed

        # Check final results (should be reranked and limited to top_k)
        assert len(results) == top_k  # Should be 2
        assert results[0].document_id == "doc_high"  # Reranked first
        assert results[0].score == 1.0  # New score from reranker
        assert results[1].document_id == "doc_mid"  # Reranked second
        assert results[1].score == 0.9  # New score from reranker

    async def test_query_with_filter(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter
    ):
        service = kb_service_default
        query = "find specific docs"
        top_k = 3
        query_filter = {"source": "specific_source", "year": {"$gte": 2024}}
        expected_query_vector = (
            service.semantic_splitter.embed_model.aget_query_embedding.return_value
        )

        # Mock Pinecone to return something (filter applied there)
        mock_pinecone_adapter.query.return_value = [
            {
                "id": "doc_filtered",
                "score": 0.7,
                "metadata": {"document_id": "doc_filtered", "is_chunk": False},
            }
        ]
        # Mock Mongo
        mock_mongodb_adapter.find.return_value = [
            {
                "_id": "m_filt",
                "document_id": "doc_filtered",
                "content": "Filtered content",
                "is_chunk": False,
                "source": "specific_source",
                "year": 2024,
            }
        ]

        results = await service.query(query, top_k=top_k, filter=query_filter)

        # Check Pinecone query call included the filter
        mock_pinecone_adapter.query.assert_awaited_once_with(
            vector=expected_query_vector,
            top_k=top_k,
            filter=query_filter,  # Filter passed to Pinecone
            namespace=None,
            include_metadata=True,
        )
        # Check Mongo find call used the IDs from Pinecone result
        mock_mongodb_adapter.find.assert_called_once()
        assert mock_mongodb_adapter.find.call_args[0][1]["document_id"]["$in"] == [
            "doc_filtered"
        ]

        assert len(results) == 1
        assert results[0].document_id == "doc_filtered"

    async def test_query_pinecone_miss_mongo_hit_ignored(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter
    ):
        """Pinecone returns IDs, but Mongo doesn't find corresponding docs."""
        service = kb_service_default
        query = "missing mongo docs"
        top_k = 5

        # Mock Pinecone response
        pinecone_res = [
            {"id": "missing_1", "score": 0.9, "metadata": {"document_id": "missing_1"}},
            {"id": "missing_2", "score": 0.8, "metadata": {"document_id": "missing_2"}},
        ]
        mock_pinecone_adapter.query.return_value = pinecone_res
        # Mock Mongo find to return empty list
        mock_mongodb_adapter.find.return_value = []

        results = await service.query(query, top_k=top_k)

        mock_pinecone_adapter.query.assert_awaited_once()
        # Mongo was queried with the IDs from Pinecone
        mock_mongodb_adapter.find.assert_called_once()
        assert mock_mongodb_adapter.find.call_args[0][1]["document_id"]["$in"] == [
            "missing_1",
            "missing_2",
        ]
        # Final results should be empty
        assert len(results) == 0

    async def test_query_pinecone_returns_nothing(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter
    ):
        service = kb_service_default
        query = "no results query"
        top_k = 5
        # Mock Pinecone query to return empty list
        mock_pinecone_adapter.query.return_value = []

        results = await service.query(query, top_k=top_k)

        mock_pinecone_adapter.query.assert_awaited_once()
        # Mongo find should NOT be called if Pinecone returns nothing
        mock_mongodb_adapter.find.assert_not_called()
        assert len(results) == 0

    async def test_query_embedding_error(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, caplog
    ):
        service = kb_service_default
        service.semantic_splitter.embed_model.aget_query_embedding.side_effect = (
            Exception("Query embed failed")
        )

        # Should return empty list and log error
        with caplog.at_level(logging.ERROR):
            results = await service.query("test query", top_k=5)

        assert results == []  # Expect empty list
        service.semantic_splitter.embed_model.aget_query_embedding.assert_awaited_once()
        mock_pinecone_adapter.query.assert_not_awaited()
        mock_mongodb_adapter.find.assert_not_called()
        assert "Error embedding query" in caplog.text

    async def test_query_pinecone_error(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, caplog
    ):
        service = kb_service_default
        mock_pinecone_adapter.query.side_effect = Exception("Pinecone query failed")

        # Should return empty list and log error
        with caplog.at_level(logging.ERROR):
            results = await service.query("test query", top_k=5)

        assert results == []  # Expect empty list
        service.semantic_splitter.embed_model.aget_query_embedding.assert_awaited_once()
        mock_pinecone_adapter.query.assert_awaited_once()
        mock_mongodb_adapter.find.assert_not_called()
        assert "Error querying Pinecone" in caplog.text

    async def test_query_mongo_error(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, caplog
    ):
        service = kb_service_default
        # Mock Pinecone to return results
        mock_pinecone_adapter.query.return_value = [
            {"id": "doc1", "score": 0.9, "metadata": {"document_id": "doc1"}}
        ]
        # Mock Mongo find to raise an error
        mock_mongodb_adapter.find.side_effect = Exception("Mongo find failed")

        # Should return empty list and log error
        with caplog.at_level(logging.ERROR):
            results = await service.query("test query", top_k=5)

        assert results == []  # Expect empty list
        service.semantic_splitter.embed_model.aget_query_embedding.assert_awaited_once()
        mock_pinecone_adapter.query.assert_awaited_once()
        mock_mongodb_adapter.find.assert_called_once()  # Mongo find was attempted
        assert "Error fetching documents from MongoDB" in caplog.text


@pytest.mark.asyncio
class TestKnowledgeBaseServiceDeleteDocument:
    async def test_delete_plain_text_document_success(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter
    ):
        service = kb_service_default
        doc_id = "plain_doc_to_delete"
        namespace = "delete-ns"

        # Mock initial find_one check (not a chunk)
        mock_mongodb_adapter.find_one.return_value = {
            "document_id": doc_id,
            "is_chunk": False,
        }
        # Mock find to return only the parent doc
        mock_mongodb_adapter.find.return_value = [{"document_id": doc_id}]
        # Mock Mongo delete_many to succeed
        mock_mongodb_adapter.delete_many.return_value = MagicMock(deleted_count=1)
        # Mock Pinecone delete to succeed
        mock_pinecone_adapter.delete.return_value = {}

        success = await service.delete_document(doc_id, namespace=namespace)

        assert success is True
        # Check initial find_one call
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id}, projection={"is_chunk": 1}
        )
        # Check find call to get all related IDs
        mock_mongodb_adapter.find.assert_called_once_with(
            service.collection,
            {"$or": [{"document_id": doc_id}, {"parent_document_id": doc_id}]},
            projection={"document_id": 1},
        )
        # Check Mongo delete_many call (should target only the doc_id found)
        mock_mongodb_adapter.delete_many.assert_called_once_with(
            service.collection, {"document_id": {"$in": [doc_id]}}
        )
        # Check Pinecone delete call
        mock_pinecone_adapter.delete.assert_awaited_once_with(
            ids=[doc_id], namespace=namespace
        )

    async def test_delete_pdf_document_success(
        self,
        kb_service_default,
        mock_mongodb_adapter,
        mock_pinecone_adapter,
        # mock_uuid_multiple, # Not needed for delete
    ):
        service = kb_service_default
        parent_doc_id = "pdf_to_delete"
        chunk_ids = [f"{parent_doc_id}_chunk_{i}" for i in range(3)]
        all_mongo_docs_found = [{"document_id": parent_doc_id}] + [
            {"document_id": chunk_id} for chunk_id in chunk_ids
        ]
        all_ids_to_delete = [parent_doc_id] + chunk_ids
        namespace = "delete-pdf-ns"

        # Mock initial find_one check (parent is not a chunk)
        mock_mongodb_adapter.find_one.return_value = {
            "document_id": parent_doc_id,
            "is_chunk": False,
        }
        # Mock find to return parent and chunk docs
        mock_mongodb_adapter.find.return_value = all_mongo_docs_found
        # Mock Mongo delete_many to succeed
        mock_mongodb_adapter.delete_many.return_value = MagicMock(
            deleted_count=len(all_ids_to_delete)
        )
        # Mock Pinecone delete to succeed
        mock_pinecone_adapter.delete.return_value = {}

        success = await service.delete_document(parent_doc_id, namespace=namespace)

        assert success is True
        # Check initial find_one call
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection,
            {"document_id": parent_doc_id},
            projection={"is_chunk": 1},
        )
        # Check find call to get all related IDs
        mock_mongodb_adapter.find.assert_called_once_with(
            service.collection,
            {
                "$or": [
                    {"document_id": parent_doc_id},
                    {"parent_document_id": parent_doc_id},
                ]
            },
            projection={"document_id": 1},
        )
        # Check Mongo delete_many call (should target all found IDs)
        mock_mongodb_adapter.delete_many.assert_called_once()
        delete_filter = mock_mongodb_adapter.delete_many.call_args[0][1]
        assert sorted(delete_filter["document_id"]["$in"]) == sorted(all_ids_to_delete)

        # Check Pinecone delete call (should include all found IDs)
        mock_pinecone_adapter.delete.assert_awaited_once()
        pinecone_delete_args = mock_pinecone_adapter.delete.await_args.kwargs
        assert sorted(pinecone_delete_args["ids"]) == sorted(all_ids_to_delete)
        assert pinecone_delete_args["namespace"] == namespace

    async def test_delete_chunk_document_fails(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, caplog
    ):
        service = kb_service_default
        doc_id = "cannot_delete_chunk_directly"
        # Mock initial find_one check to return a chunk document
        mock_mongodb_adapter.find_one.return_value = {
            "document_id": doc_id,
            "is_chunk": True,
        }

        with caplog.at_level(logging.WARNING):
            success = await service.delete_document(doc_id)

        assert success is False  # Should return False immediately
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id}, projection={"is_chunk": 1}
        )
        # Subsequent find and delete operations should not be called
        mock_mongodb_adapter.find.assert_not_called()
        mock_mongodb_adapter.delete_many.assert_not_called()
        mock_pinecone_adapter.delete.assert_not_awaited()
        assert (
            f"Cannot delete chunk {doc_id} directly. Delete the parent document"
            in caplog.text
        )

    async def test_delete_non_existent_document(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, caplog
    ):
        service = kb_service_default
        doc_id = "does_not_exist_del"
        # Mock initial find_one to return None
        mock_mongodb_adapter.find_one.return_value = None
        # Mock find to return empty list
        mock_mongodb_adapter.find.return_value = []

        with caplog.at_level(logging.WARNING):
            success = await service.delete_document(doc_id)

        assert success is False  # Document wasn't found initially
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id}, projection={"is_chunk": 1}
        )
        # Find should be called to check for related docs (even though parent not found)
        mock_mongodb_adapter.find.assert_called_once()
        # Delete operations might be called for cleanup, but the result is False
        mock_mongodb_adapter.delete_many.assert_not_called()  # No mongo IDs found
        # Pinecone delete might be called with just the non-existent ID if find fails/is empty
        # Depending on exact flow, let's assert it might be called for cleanup attempt
        # mock_pinecone_adapter.delete.assert_awaited_once_with(ids=[doc_id], namespace=None) # Check if cleanup is attempted
        assert f"Document {doc_id} not found for deletion" in caplog.text

    async def test_delete_mongo_error(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, caplog
    ):
        service = kb_service_default
        doc_id = "plain_doc_mongo_fail_del"
        # Mock initial find_one (not a chunk)
        mock_mongodb_adapter.find_one.return_value = {
            "document_id": doc_id,
            "is_chunk": False,
        }
        # Mock find to return the doc
        mock_mongodb_adapter.find.return_value = [{"document_id": doc_id}]
        # Mock Pinecone delete to succeed
        mock_pinecone_adapter.delete.return_value = {}
        # Mock delete_many to fail
        mock_mongodb_adapter.delete_many.side_effect = Exception("Mongo delete failed")

        with caplog.at_level(logging.ERROR):
            success = await service.delete_document(doc_id)

        assert success is False  # Failed because Mongo errored
        mock_mongodb_adapter.find_one.assert_called_once()
        mock_mongodb_adapter.find.assert_called_once()
        mock_pinecone_adapter.delete.assert_awaited_once_with(
            ids=[doc_id], namespace=None
        )  # Pinecone attempted
        mock_mongodb_adapter.delete_many.assert_called_once()  # Mongo delete attempted
        assert "Error deleting documents from MongoDB" in caplog.text

    async def test_delete_pinecone_error(
        self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, caplog
    ):
        service = kb_service_default
        doc_id = "plain_doc_pinecone_fail_del"
        # Mock initial find_one (not a chunk)
        mock_mongodb_adapter.find_one.return_value = {
            "document_id": doc_id,
            "is_chunk": False,
        }
        # Mock find to return the doc
        mock_mongodb_adapter.find.return_value = [{"document_id": doc_id}]
        # Mock delete_many to succeed
        mock_mongodb_adapter.delete_many.return_value = MagicMock(deleted_count=1)
        # Mock Pinecone delete to fail
        mock_pinecone_adapter.delete.side_effect = Exception("Pinecone delete failed")

        with caplog.at_level(logging.ERROR):
            success = await service.delete_document(doc_id)

        assert success is False  # Failed because Pinecone errored
        mock_mongodb_adapter.find_one.assert_called_once()
        mock_mongodb_adapter.find.assert_called_once()
        mock_mongodb_adapter.delete_many.assert_called_once()  # Mongo succeeded
        mock_pinecone_adapter.delete.assert_awaited_once()  # Pinecone attempted
        assert "Error deleting vectors from Pinecone" in caplog.text

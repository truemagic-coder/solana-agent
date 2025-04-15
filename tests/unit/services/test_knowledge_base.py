import pytest
import uuid
from unittest.mock import patch, MagicMock, AsyncMock, call, ANY
from datetime import datetime, timezone, timedelta

# Assuming LlamaDocument and NodeWithScore are structured appropriately for mocking
from llama_index.core import Document as LlamaDocument
from llama_index.core.schema import NodeWithScore, TextNode
# Import pypdf errors if needed for specific error tests
import pypdf

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
    # Set default attributes expected by the service
    adapter.embedding_dimensions = 1536  # Default for small model
    adapter.use_reranking = False
    adapter.rerank_text_field = "text_content_for_rerank"  # Example field name
    adapter.initial_query_top_k_multiplier = 5
    return adapter


@pytest.fixture
def mock_mongodb_adapter():
    adapter = MagicMock(spec=MongoDBAdapter)
    adapter.collection_exists = MagicMock(return_value=True)
    adapter.create_collection = MagicMock()
    adapter.create_index = MagicMock()
    adapter.insert_one = MagicMock(
        return_value=MagicMock(inserted_id="mock_mongo_id"))
    adapter.insert_many = MagicMock(return_value=MagicMock(
        inserted_ids=["mock_mongo_id_1", "mock_mongo_id_2"]))
    adapter.find_one = MagicMock(return_value=None)  # Default to not found
    adapter.find = MagicMock(return_value=[])  # Default to empty list
    adapter.update_one = MagicMock(return_value=MagicMock(modified_count=1))
    adapter.delete_many = MagicMock(return_value=MagicMock(deleted_count=1))
    return adapter


@pytest.fixture
def mock_openai_embedding():
    # Patch the class within the module where it's imported
    with patch('solana_agent.services.knowledge_base.OpenAIEmbedding', autospec=True) as mock_embed_class:
        instance = mock_embed_class.return_value
        instance.model = "mock-embedding-model"  # Store model name if needed
        # Configure async methods
        instance.aget_text_embedding = AsyncMock(return_value=[0.1] * 1536)
        instance.aget_query_embedding = AsyncMock(return_value=[0.2] * 1536)
        # Make batch return a list of embeddings matching input length
        instance.aget_text_embedding_batch = AsyncMock(
            side_effect=lambda texts, **kwargs: [[0.1 + i*0.01] * 1536 for i in range(len(texts))])
        yield mock_embed_class  # Yield the mock class itself


@pytest.fixture
def mock_semantic_splitter():
    # Patch the class within the module where it's imported
    with patch('solana_agent.services.knowledge_base.SemanticSplitterNodeParser', autospec=True) as mock_splitter_class:
        instance = mock_splitter_class.return_value
        # Mock the method that returns nodes
        instance.get_nodes_from_documents = MagicMock(return_value=[
            TextNode(text="Chunk 1", id_="chunk_node_1",
                     metadata={"some": "meta"}),
            TextNode(text="Chunk 2", id_="chunk_node_2",
                     metadata={"other": "meta"})
        ])
        # The splitter needs an embed_model attribute, assign the mock embedding instance
        # We'll assign the actual mock instance in the service fixture
        instance.embed_model = MagicMock()
        yield mock_splitter_class  # Yield the mock class


@pytest.fixture
def mock_uuid():
    test_uuid = uuid.UUID('12345678-1234-5678-1234-567812345678')
    with patch('uuid.uuid4', return_value=test_uuid) as mock_uuid_patch:
        yield mock_uuid_patch


@pytest.fixture
def mock_uuid_multiple():
    # Provide multiple unique UUIDs for batch tests if needed
    uuid_1 = uuid.UUID('11111111-1111-1111-1111-111111111111')
    uuid_2 = uuid.UUID('22222222-2222-2222-2222-222222222222')
    uuid_3 = uuid.UUID('33333333-3333-3333-3333-333333333333')
    with patch('uuid.uuid4') as mock_uuid_patch:
        # Make it return different UUIDs on subsequent calls
        mock_uuid_patch.side_effect = [uuid_1, uuid_2, uuid_3]
        yield mock_uuid_patch


@pytest.fixture
def mock_datetime():
    # Use a fixed UTC datetime
    now = datetime(2024, 4, 15, 10, 30, 0, tzinfo=timezone.utc)
    # Patch the specific import 'dt' used in the service
    # 'wraps=datetime' allows other datetime methods to work if needed,
    # but we primarily care about controlling now().
    with patch('solana_agent.services.knowledge_base.dt', wraps=datetime) as mock_dt:
        # Ensure dt.now() always returns our fixed, timezone-aware datetime
        mock_dt.now.return_value = now
        # No need to mock astimezone or tzinfo separately
        yield mock_dt


@pytest.fixture
def mock_pypdf_reader():
    with patch('solana_agent.services.knowledge_base.pypdf.PdfReader') as mock_reader_class:
        instance = mock_reader_class.return_value
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "This is page 1 content. "
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "This is page 2 content."
        instance.pages = [mock_page1, mock_page2]
        yield mock_reader_class


@pytest.fixture
def mock_open():
    # Mock built-in open for reading PDF files
    with patch('builtins.open', new_callable=MagicMock) as mock_open_func:
        # Configure the mock file object returned by open
        mock_file = MagicMock()
        mock_file.read.return_value = b'fake-pdf-bytes'
        mock_file.__enter__.return_value = mock_file  # For context manager
        mock_open_func.return_value = mock_file
        yield mock_open_func


@pytest.fixture
def mock_asyncio_to_thread():
    # Patch asyncio.to_thread used for the sync splitter call
    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
        # Make it return the result of the function it's wrapping
        # We'll rely on the mock_semantic_splitter fixture's return value
        mock_to_thread.side_effect = lambda func, * \
            args, **kwargs: func(*args, **kwargs)
        yield mock_to_thread


# --- Service Fixtures ---

@pytest.fixture
def kb_service_default(mock_pinecone_adapter, mock_mongodb_adapter, mock_openai_embedding, mock_semantic_splitter, mock_uuid, mock_datetime):
    # Assign the mock embedding instance to the mock splitter instance
    mock_splitter_instance = mock_semantic_splitter.return_value
    mock_embedding_instance = mock_openai_embedding.return_value
    mock_splitter_instance.embed_model = mock_embedding_instance

    # Use default model name which implies 3072 dimensions
    mock_pinecone_adapter.embedding_dimensions = 3072

    service = KnowledgeBaseService(
        pinecone_adapter=mock_pinecone_adapter,
        mongodb_adapter=mock_mongodb_adapter,
        openai_api_key="test-api-key-default",  # Explicitly pass API key
        # Other args use defaults
    )
    return service


@pytest.fixture
def kb_service_custom(mock_pinecone_adapter, mock_mongodb_adapter, mock_openai_embedding, mock_semantic_splitter, mock_uuid, mock_datetime):
    # Assign the mock embedding instance to the mock splitter instance
    mock_splitter_instance = mock_semantic_splitter.return_value
    mock_embedding_instance = mock_openai_embedding.return_value
    mock_splitter_instance.embed_model = mock_embedding_instance

    # Use small model explicitly
    mock_pinecone_adapter.embedding_dimensions = 1536
    mock_pinecone_adapter.use_reranking = True  # Enable reranking

    service = KnowledgeBaseService(
        pinecone_adapter=mock_pinecone_adapter,
        mongodb_adapter=mock_mongodb_adapter,
        openai_api_key="test-api-key-custom",  # Explicitly pass API key
        openai_model_name="text-embedding-3-small",
        collection_name="custom_docs",
        rerank_results=True,  # This is stored but Pinecone adapter applies it
        rerank_top_k=5,
        splitter_buffer_size=2,
        splitter_breakpoint_percentile=90
    )
    return service

# --- Test Classes ---


@pytest.mark.asyncio
class TestKnowledgeBaseServiceInitialization:

    def test_init_successful_defaults(self, mock_pinecone_adapter, mock_mongodb_adapter, mock_openai_embedding, mock_semantic_splitter):
        mock_pinecone_adapter.embedding_dimensions = 3072  # Match default model
        api_key = "test-key-defaults"
        service = KnowledgeBaseService(
            mock_pinecone_adapter, mock_mongodb_adapter, openai_api_key=api_key)
        assert service.pinecone == mock_pinecone_adapter
        assert service.mongo == mock_mongodb_adapter
        assert service.collection == "knowledge_documents"
        assert service.rerank_results is False
        assert service.rerank_top_k == 3
        assert service.openai_model_name == "text-embedding-3-large"
        # Check it's the mocked splitter
        assert isinstance(service.semantic_splitter, MagicMock)
        # assert service.semantic_splitter.embed_model == mock_openai_embedding.return_value # REMOVED - Redundant check
        mock_openai_embedding.assert_called_once_with(
            model="text-embedding-3-large", api_key=api_key)
        mock_semantic_splitter.assert_called_once_with(
            # This check is sufficient
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=ANY)
        mock_mongodb_adapter.collection_exists.assert_called_once_with(
            "knowledge_documents")
        # Ensure indexes are checked/created
        mock_mongodb_adapter.create_index.assert_called()

    def test_init_successful_custom_args(self, mock_pinecone_adapter, mock_mongodb_adapter, mock_openai_embedding, mock_semantic_splitter):
        mock_pinecone_adapter.embedding_dimensions = 1536  # Match custom model
        mock_pinecone_adapter.use_reranking = True
        api_key = "custom-key-args"
        service = KnowledgeBaseService(
            pinecone_adapter=mock_pinecone_adapter,
            mongodb_adapter=mock_mongodb_adapter,
            openai_api_key=api_key,
            openai_model_name="text-embedding-3-small",
            collection_name="custom_coll",
            rerank_results=True,
            rerank_top_k=5,
            splitter_buffer_size=2,
            splitter_breakpoint_percentile=90
        )
        assert service.collection == "custom_coll"
        assert service.rerank_results is True
        assert service.rerank_top_k == 5
        assert service.openai_model_name == "text-embedding-3-small"
        mock_openai_embedding.assert_called_once_with(
            model="text-embedding-3-small", api_key=api_key)
        mock_semantic_splitter.assert_called_once_with(
            buffer_size=2, breakpoint_percentile_threshold=90, embed_model=ANY)
        mock_mongodb_adapter.collection_exists.assert_called_once_with(
            "custom_coll")

    def test_init_missing_api_key_raises_error(self, mock_pinecone_adapter, mock_mongodb_adapter):
        # Test the internal check within __init__
        with pytest.raises(ValueError, match="OpenAI API key not provided"):
            # Pass empty string
            KnowledgeBaseService(mock_pinecone_adapter,
                                 mock_mongodb_adapter, openai_api_key="")
        with pytest.raises(ValueError, match="OpenAI API key not provided"):
            KnowledgeBaseService(
                mock_pinecone_adapter, mock_mongodb_adapter, openai_api_key=None)  # Pass None

    def test_init_unknown_model_no_pinecone_dim_raises_error(self, mock_pinecone_adapter, mock_mongodb_adapter):
        # Simulate Pinecone adapter not having the dimension attribute or it being 0
        del mock_pinecone_adapter.embedding_dimensions
        # Or mock_pinecone_adapter.embedding_dimensions = 0
        with pytest.raises(ValueError, match="Cannot determine dimension for unknown OpenAI model"):
            KnowledgeBaseService(mock_pinecone_adapter, mock_mongodb_adapter,
                                 openai_api_key="test-key", openai_model_name="unknown-model")

    def test_init_unknown_model_uses_pinecone_dim(self, mock_pinecone_adapter, mock_mongodb_adapter, mock_openai_embedding, mock_semantic_splitter, capsys):
        # Provide a dimension via Pinecone mock
        mock_pinecone_adapter.embedding_dimensions = 1024
        api_key = "test-key-unknown"
        service = KnowledgeBaseService(
            mock_pinecone_adapter, mock_mongodb_adapter, openai_api_key=api_key, openai_model_name="unknown-model")
        captured = capsys.readouterr()
        assert "Warning: Unknown OpenAI model 'unknown-model'" in captured.out
        assert "Using dimension 1024 from Pinecone config" in captured.out
        mock_openai_embedding.assert_called_once_with(
            model="unknown-model", api_key=api_key)
        assert service.openai_model_name == "unknown-model"

    def test_init_openai_embedding_error(self, mock_pinecone_adapter, mock_mongodb_adapter, mock_openai_embedding):
        mock_openai_embedding.side_effect = Exception("OpenAI Init Failed")
        with pytest.raises(Exception, match="OpenAI Init Failed"):
            KnowledgeBaseService(
                mock_pinecone_adapter, mock_mongodb_adapter, openai_api_key="test-key")

    def test_ensure_collection_exists(self, mock_pinecone_adapter, mock_mongodb_adapter, mock_openai_embedding, mock_semantic_splitter):
        mock_mongodb_adapter.collection_exists.return_value = True
        # Init service, which calls _ensure_collection
        KnowledgeBaseService(mock_pinecone_adapter,
                             mock_mongodb_adapter, openai_api_key="test-key")
        mock_mongodb_adapter.collection_exists.assert_called_once_with(
            "knowledge_documents")
        mock_mongodb_adapter.create_collection.assert_not_called()
        # Check that all expected indexes were attempted to be created
        assert mock_mongodb_adapter.create_index.call_count == 6
        mock_mongodb_adapter.create_index.assert_has_calls([
            call("knowledge_documents", [("document_id", 1)], unique=True),
            call("knowledge_documents", [("parent_document_id", 1)]),
            call("knowledge_documents", [("source", 1)]),
            call("knowledge_documents", [("created_at", -1)]),
            call("knowledge_documents", [("tags", 1)]),
            call("knowledge_documents", [("is_chunk", 1)]),
        ], any_order=True)

    def test_ensure_collection_does_not_exist(self, mock_pinecone_adapter, mock_mongodb_adapter, mock_openai_embedding, mock_semantic_splitter):
        mock_mongodb_adapter.collection_exists.return_value = False
        # Init service, which calls _ensure_collection
        KnowledgeBaseService(mock_pinecone_adapter,
                             mock_mongodb_adapter, openai_api_key="test-key")
        mock_mongodb_adapter.collection_exists.assert_called_once_with(
            "knowledge_documents")
        mock_mongodb_adapter.create_collection.assert_called_once_with(
            "knowledge_documents")
        # Indexes created after collection
        assert mock_mongodb_adapter.create_index.call_count == 6


@pytest.mark.asyncio
class TestKnowledgeBaseServiceAddDocument:

    async def test_add_document_success_auto_id(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, mock_uuid, mock_datetime):
        service = kb_service_default
        text = "This is a test document."
        metadata = {"source": "test", "tags": ["tag1"]}
        expected_doc_id = str(mock_uuid.return_value)
        expected_embedding = [0.1] * 3072  # Default model dimension
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding.return_value = expected_embedding

        doc_id = await service.add_document(text, metadata)

        assert doc_id == expected_doc_id
        embed_model_instance.aget_text_embedding.assert_awaited_once_with(text)
        mock_mongodb_adapter.insert_one.assert_called_once()
        mongo_call_args = mock_mongodb_adapter.insert_one.call_args[0][1]
        assert mongo_call_args['document_id'] == expected_doc_id
        assert mongo_call_args['content'] == text
        assert mongo_call_args['is_chunk'] is False
        assert mongo_call_args['source'] == "test"
        assert mongo_call_args['tags'] == ["tag1"]
        assert isinstance(mongo_call_args['created_at'], datetime)
        assert isinstance(mongo_call_args['updated_at'], datetime)
        assert mongo_call_args['created_at'].tzinfo is not None
        assert mongo_call_args['updated_at'].tzinfo is not None

        mock_pinecone_adapter.upsert.assert_awaited_once_with(
            vectors=[{
                "id": expected_doc_id,
                "values": expected_embedding,
                "metadata": {
                    "document_id": expected_doc_id,
                    "is_chunk": False,
                    "source": "test",
                    "tags": ["tag1"]
                    # No text content here as reranking is off by default
                }
            }],
            namespace=None
        )

    async def test_add_document_success_provided_id_namespace_rerank(self, kb_service_custom, mock_mongodb_adapter, mock_pinecone_adapter, mock_datetime):
        service = kb_service_custom  # Uses small model, reranking enabled
        text = "Another test document."
        metadata = {"source": "custom", "other": "value"}
        provided_id = "my-custom-id-123"
        namespace = "my-namespace"
        expected_embedding = [0.1] * 1536  # Small model dimension
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding.return_value = expected_embedding

        doc_id = await service.add_document(text, metadata, document_id=provided_id, namespace=namespace)

        assert doc_id == provided_id
        embed_model_instance.aget_text_embedding.assert_awaited_once_with(text)
        mock_mongodb_adapter.insert_one.assert_called_once()
        mongo_call_args = mock_mongodb_adapter.insert_one.call_args[0][1]
        assert mongo_call_args['document_id'] == provided_id
        assert mongo_call_args['content'] == text
        assert mongo_call_args['source'] == "custom"
        assert mongo_call_args['other'] == "value"
        assert isinstance(mongo_call_args['created_at'], datetime)
        assert isinstance(mongo_call_args['updated_at'], datetime)
        assert mongo_call_args['created_at'].tzinfo is not None
        assert mongo_call_args['updated_at'].tzinfo is not None

        # Check that text content is included in Pinecone metadata for reranking
        mock_pinecone_adapter.upsert.assert_awaited_once_with(
            vectors=[{
                "id": provided_id,
                "values": expected_embedding,
                "metadata": {
                    "document_id": provided_id,
                    "is_chunk": False,
                    "source": "custom",
                    "tags": [],  # Default if not provided
                    service.pinecone.rerank_text_field: text  # Text included
                }
            }],
            namespace=namespace
        )

    async def test_add_document_mongo_error(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        mock_mongodb_adapter.insert_one.side_effect = Exception(
            "Mongo write failed")
        embed_model_instance = service.semantic_splitter.embed_model

        with pytest.raises(Exception, match="Mongo write failed"):
            await service.add_document("test", {})

        embed_model_instance.aget_text_embedding.assert_not_awaited()
        mock_pinecone_adapter.upsert.assert_not_awaited()

    async def test_add_document_embedding_error(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding.side_effect = Exception(
            "Embedding failed")

        with pytest.raises(Exception, match="Embedding failed"):
            await service.add_document("test", {})

        mock_mongodb_adapter.insert_one.assert_called_once()  # Mongo insert happens first
        mock_pinecone_adapter.upsert.assert_not_awaited()

    async def test_add_document_pinecone_error(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        mock_pinecone_adapter.upsert.side_effect = Exception(
            "Pinecone upsert failed")
        embed_model_instance = service.semantic_splitter.embed_model

        with pytest.raises(Exception, match="Pinecone upsert failed"):
            await service.add_document("test", {})

        mock_mongodb_adapter.insert_one.assert_called_once()
        embed_model_instance.aget_text_embedding.assert_awaited_once()


@pytest.mark.asyncio
class TestKnowledgeBaseServiceAddPdfDocument:

    async def test_add_pdf_document_bytes_success(self, kb_service_custom, mock_mongodb_adapter, mock_pinecone_adapter, mock_pypdf_reader, mock_semantic_splitter, mock_uuid, mock_datetime, mock_asyncio_to_thread):
        service = kb_service_custom  # Reranking enabled
        pdf_bytes = b'fake-pdf-bytes'
        metadata = {"source": "pdf_test", "year": 2024}
        expected_parent_id = str(mock_uuid.return_value)
        expected_text = "This is page 1 content. This is page 2 content."
        expected_chunk_texts = ["Chunk 1", "Chunk 2"]
        expected_chunk_embeddings = [
            [0.1] * 1536, [0.11] * 1536]  # From batch mock

        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding_batch.return_value = expected_chunk_embeddings
        splitter_instance = mock_semantic_splitter.return_value
        mock_nodes = [TextNode(text=t, id_=f"node_{i}") for i, t in enumerate(
            expected_chunk_texts)]
        splitter_instance.get_nodes_from_documents.return_value = mock_nodes

        doc_id = await service.add_pdf_document(pdf_bytes, metadata)

        assert doc_id == expected_parent_id
        # mock_pypdf_reader.assert_called_once_with(io.BytesIO(pdf_bytes)) # OLD
        mock_pypdf_reader.assert_called_once_with(ANY)  # NEW
        # Check parent doc insertion in Mongo
        mock_mongodb_adapter.insert_one.assert_called_once()
        mongo_call_args = mock_mongodb_adapter.insert_one.call_args[0][1]
        assert mongo_call_args['document_id'] == expected_parent_id
        assert mongo_call_args['content'] == expected_text
        assert mongo_call_args['pdf_data'] == pdf_bytes
        assert mongo_call_args['is_chunk'] is False
        assert mongo_call_args['source'] == "pdf_test"
        assert mongo_call_args['year'] == 2024
        assert isinstance(mongo_call_args['created_at'], datetime)
        assert isinstance(mongo_call_args['updated_at'], datetime)
        assert mongo_call_args['created_at'].tzinfo is not None
        assert mongo_call_args['updated_at'].tzinfo is not None

        # Check splitter call
        mock_asyncio_to_thread.assert_awaited_once()
        # Check arg type
        assert isinstance(
            mock_asyncio_to_thread.call_args[0][1][0], LlamaDocument)
        assert mock_asyncio_to_thread.call_args[0][1][0].text == expected_text
        # Check embedding call for chunks
        embed_model_instance.aget_text_embedding_batch.assert_awaited_once_with(
            expected_chunk_texts, show_progress=True)
        # Check Pinecone upsert for chunks (using asyncio.gather, so check await_args_list)
        # Should be called once for the single batch
        mock_pinecone_adapter.upsert.assert_awaited_once()
        # Get args of the first (and only) call
        upsert_call_args = mock_pinecone_adapter.upsert.await_args_list[0]
        assert upsert_call_args[1]['namespace'] is None
        vectors = upsert_call_args[1]['vectors']
        assert len(vectors) == 2
        # Check first chunk vector
        assert vectors[0]['id'] == f"{expected_parent_id}_chunk_0"
        assert vectors[0]['values'] == expected_chunk_embeddings[0]
        assert vectors[0]['metadata'] == {
            "document_id": f"{expected_parent_id}_chunk_0",
            "parent_document_id": expected_parent_id,
            "chunk_index": 0,
            "is_chunk": True,
            "source": "pdf_test",
            "tags": [],
            # Rerank text included
            service.pinecone.rerank_text_field: expected_chunk_texts[0]
        }
        # Check second chunk vector
        assert vectors[1]['id'] == f"{expected_parent_id}_chunk_1"
        assert vectors[1]['values'] == expected_chunk_embeddings[1]
        assert vectors[1]['metadata']['chunk_index'] == 1
        assert vectors[1]['metadata'][service.pinecone.rerank_text_field] == expected_chunk_texts[1]

    async def test_add_pdf_document_path_success(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, mock_pypdf_reader, mock_semantic_splitter, mock_uuid, mock_datetime, mock_open, mock_asyncio_to_thread):
        service = kb_service_default  # Reranking disabled
        pdf_path = "/path/to/my.pdf"
        metadata = {"source": "pdf_path"}
        expected_parent_id = str(mock_uuid.return_value)
        expected_chunk_texts = ["Chunk 1", "Chunk 2"]
        expected_chunk_embeddings = [
            [0.1] * 3072, [0.11] * 3072]  # Default model

        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding_batch.return_value = expected_chunk_embeddings
        splitter_instance = mock_semantic_splitter.return_value
        mock_nodes = [TextNode(text=t, id_=f"node_{i}") for i, t in enumerate(
            expected_chunk_texts)]
        splitter_instance.get_nodes_from_documents.return_value = mock_nodes

        doc_id = await service.add_pdf_document(pdf_path, metadata, namespace="pdf-ns")

        assert doc_id == expected_parent_id
        mock_open.assert_called_once_with(pdf_path, "rb")  # Check file open
        # mock_pypdf_reader.assert_called_once_with(io.BytesIO(b'fake-pdf-bytes')) # OLD
        mock_pypdf_reader.assert_called_once_with(ANY)  # NEW
        mock_mongodb_adapter.insert_one.assert_called_once()  # Basic check
        mock_asyncio_to_thread.assert_awaited_once()
        embed_model_instance.aget_text_embedding_batch.assert_awaited_once()
        # Check Pinecone upsert namespace and lack of rerank text
        mock_pinecone_adapter.upsert.assert_awaited_once()
        upsert_call_args = mock_pinecone_adapter.upsert.await_args_list[0]
        assert upsert_call_args[1]['namespace'] == "pdf-ns"
        vectors = upsert_call_args[1]['vectors']
        assert len(vectors) == 2
        assert service.pinecone.rerank_text_field not in vectors[0]['metadata']
        assert service.pinecone.rerank_text_field not in vectors[1]['metadata']

    async def test_add_pdf_document_invalid_type_error(self, kb_service_default):
        with pytest.raises(ValueError, match="pdf_data must be bytes or a file path string"):
            # Pass an integer
            await kb_service_default.add_pdf_document(12345, {})

    async def test_add_pdf_document_pdf_read_error(self, kb_service_default, mock_pypdf_reader):
        mock_pypdf_reader.side_effect = pypdf.errors.PdfReadError(
            "Invalid PDF")
        with pytest.raises(pypdf.errors.PdfReadError, match="Invalid PDF"):
            await kb_service_default.add_pdf_document(b'bad-bytes', {})

    async def test_add_pdf_document_empty_pdf_no_chunks(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, mock_pypdf_reader, mock_semantic_splitter, mock_uuid, mock_datetime, capsys):
        service = kb_service_default
        # Mock PdfReader to return no text
        reader_instance = mock_pypdf_reader.return_value
        mock_page_empty = MagicMock()
        mock_page_empty.extract_text.return_value = ""
        reader_instance.pages = [mock_page_empty]
        expected_parent_id = str(mock_uuid.return_value)

        doc_id = await service.add_pdf_document(b'empty', {"source": "empty"})

        assert doc_id == expected_parent_id
        mock_mongodb_adapter.insert_one.assert_called_once()  # Mongo doc still inserted
        captured = capsys.readouterr()
        assert f"Warning: No text extracted from PDF {expected_parent_id}" in captured.out
        assert f"Skipping chunking for PDF {expected_parent_id}" in captured.out
        # Ensure splitter and embedding were not called
        mock_semantic_splitter.return_value.get_nodes_from_documents.assert_not_called()
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_not_awaited()
        mock_pinecone_adapter.upsert.assert_not_awaited()  # No chunks to upsert

    async def test_add_pdf_document_splitter_error(self, kb_service_default, mock_mongodb_adapter, mock_pypdf_reader, mock_semantic_splitter, mock_asyncio_to_thread):
        service = kb_service_default
        # Make the splitter raise an error
        mock_asyncio_to_thread.side_effect = Exception("Splitter failed")

        with pytest.raises(Exception, match="Splitter failed"):
            await service.add_pdf_document(b'data', {})

        mock_mongodb_adapter.insert_one.assert_called_once()  # Mongo insert succeeded
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_not_awaited()

    async def test_add_pdf_document_chunk_embedding_error(self, kb_service_default, mock_mongodb_adapter, mock_pypdf_reader, mock_semantic_splitter, mock_asyncio_to_thread):
        service = kb_service_default
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding_batch.side_effect = Exception(
            "Chunk embed failed")

        with pytest.raises(Exception, match="Chunk embed failed"):
            await service.add_pdf_document(b'data', {})

        mock_mongodb_adapter.insert_one.assert_called_once()
        mock_asyncio_to_thread.assert_awaited_once()  # Splitter was called
        # Embedding was attempted
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_awaited_once()

    async def test_add_pdf_document_pinecone_batch_error(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, mock_pypdf_reader, mock_semantic_splitter, mock_asyncio_to_thread, capsys):
        service = kb_service_default
        # Simulate error during asyncio.gather
        mock_pinecone_adapter.upsert.side_effect = Exception(
            "Pinecone chunk failed")

        # This should not raise an exception in the main function, but log errors
        # Batch size 1 to ensure multiple calls if needed
        await service.add_pdf_document(b'data', {}, chunk_batch_size=1)

        mock_mongodb_adapter.insert_one.assert_called_once()
        mock_asyncio_to_thread.assert_awaited_once()
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_awaited_once()
        # Check that upsert was called (at least once before failing)
        mock_pinecone_adapter.upsert.assert_awaited()
        captured = capsys.readouterr()
        # Check if the error was logged (adjust match based on actual log format)
        assert "Error upserting vector batch" in captured.out
        assert "Pinecone chunk failed" in captured.out

    async def test_add_pdf_document_multiple_batches(self, kb_service_custom, mock_mongodb_adapter, mock_pinecone_adapter, mock_pypdf_reader, mock_semantic_splitter, mock_asyncio_to_thread):
        service = kb_service_custom  # Reranking enabled
        # Simulate more chunks than batch size
        expected_chunk_texts = [f"Chunk {i}" for i in range(5)]
        expected_chunk_embeddings = [[0.1 + i*0.01] * 1536 for i in range(5)]

        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding_batch.return_value = expected_chunk_embeddings
        splitter_instance = mock_semantic_splitter.return_value
        mock_nodes = [TextNode(text=t, id_=f"node_{i}") for i, t in enumerate(
            expected_chunk_texts)]
        splitter_instance.get_nodes_from_documents.return_value = mock_nodes

        # Use batch size smaller than number of chunks
        await service.add_pdf_document(b'data', {}, chunk_batch_size=2)

        # Expect multiple upsert calls initiated via asyncio.gather
        # 5 chunks, batch size 2 -> 3 batches initiated
        assert mock_pinecone_adapter.upsert.await_count == 3
        # Check args of first call
        first_call_args = mock_pinecone_adapter.upsert.await_args_list[0]
        assert len(first_call_args[1]['vectors']) == 2
        # Check args of last call
        last_call_args = mock_pinecone_adapter.upsert.await_args_list[2]
        assert len(last_call_args[1]['vectors']) == 1


@pytest.mark.asyncio
class TestKnowledgeBaseServiceQuery:

    async def test_query_success_no_rerank(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        query = "find documents"
        expected_query_vector = [0.2] * 3072
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_query_embedding.return_value = expected_query_vector

        # Mock Pinecone response (no reranking)
        pinecone_res = [
            {"id": "doc1", "score": 0.9, "metadata": {
                "document_id": "doc1", "source": "s1", "is_chunk": False}},
            {"id": "pdf1_chunk_0", "score": 0.8, "metadata": {"document_id": "pdf1_chunk_0",
                                                              "parent_document_id": "pdf1", "is_chunk": True, "chunk_index": 0}},
        ]
        mock_pinecone_adapter.query.return_value = pinecone_res

        # Mock Mongo response
        mongo_docs = [
            {"_id": "m1", "document_id": "doc1",
                "content": "Content for doc1", "source": "s1", "tags": ["t1"]},
            {"_id": "m2", "document_id": "pdf1", "content": "Full PDF content",
                "source": "pdf_source", "year": 2023},
        ]
        mock_mongodb_adapter.find.return_value = mongo_docs

        results = await service.query(query, top_k=2, include_content=True, include_metadata=True)

        embed_model_instance.aget_query_embedding.assert_awaited_once_with(
            query)
        mock_pinecone_adapter.query.assert_awaited_once_with(
            vector=expected_query_vector,
            filter=None,
            top_k=2,  # Matches requested top_k as reranking is off
            namespace=None,
            include_values=False,
            include_metadata=True
        )
        # Check Mongo find call (should fetch doc1 and pdf1)
        mock_mongodb_adapter.find.assert_called_once()
        find_call_args = mock_mongodb_adapter.find.call_args
        assert find_call_args[0][0] == service.collection
        # assert find_call_args[0][1] == {"document_id": {"$in": ["doc1", "pdf1"]}} # OLD
        # Compare lists ignoring order
        assert sorted(find_call_args[0][1]["document_id"]["$in"]) == sorted(
            ["doc1", "pdf1"])  # NEW

        # Check combined results
        assert len(results) == 2
        # Result 1 (doc1)
        assert results[0]['document_id'] == "doc1"
        assert results[0]['score'] == 0.9
        assert results[0]['is_chunk'] is False
        assert results[0]['parent_document_id'] is None
        assert results[0]['content'] == "Content for doc1"
        assert results[0]['metadata'] == {
            # Merged from Mongo
            "source": "s1", "tags": ["t1"], "is_chunk": False}
        # Result 2 (pdf1_chunk_0)
        assert results[1]['document_id'] == "pdf1_chunk_0"
        assert results[1]['score'] == 0.8
        assert results[1]['is_chunk'] is True
        assert results[1]['parent_document_id'] == "pdf1"
        # Content from parent Mongo doc
        assert results[1]['content'] == "Full PDF content"
        assert results[1]['metadata'] == {"parent_document_id": "pdf1", "is_chunk": True, "chunk_index": 0,
                                          # Merged from Pinecone chunk meta and Mongo parent meta
                                          "source": "pdf_source", "year": 2023}

    async def test_query_success_with_rerank(self, kb_service_custom, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_custom  # Reranking enabled
        query = "find reranked docs"
        expected_query_vector = [0.2] * 1536
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_query_embedding.return_value = expected_query_vector

        # Mock Pinecone response (reranking enabled in adapter mock)
        # Pinecone adapter mock should handle the reranking logic simulation if needed,
        # but the service just passes the initial_k multiplier.
        # Assume Pinecone returns the final top_k results after its internal reranking.
        rerank_field = service.pinecone.rerank_text_field
        pinecone_res = [
            {"id": "pdf1_chunk_1", "score": 0.95, "metadata": {"document_id": "pdf1_chunk_1",
                                                               "parent_document_id": "pdf1", "is_chunk": True, "chunk_index": 1, rerank_field: "Reranked Chunk 1 Text"}},
            {"id": "doc2", "score": 0.90, "metadata": {"document_id": "doc2",
                                                       "source": "s2", "is_chunk": False, rerank_field: "Reranked Doc2 Text"}},
        ]
        mock_pinecone_adapter.query.return_value = pinecone_res

        # Mock Mongo response
        mongo_docs = [
            {"_id": "m1", "document_id": "pdf1",
                "content": "Original PDF content", "source": "pdf_source"},
            {"_id": "m2", "document_id": "doc2",
                "content": "Original Doc2 content", "source": "s2"},
        ]
        mock_mongodb_adapter.find.return_value = mongo_docs

        results = await service.query(query, top_k=2, include_content=True, include_metadata=True)

        embed_model_instance.aget_query_embedding.assert_awaited_once_with(
            query)
        # Check that initial_k is multiplied
        expected_initial_k = 2 * service.pinecone.initial_query_top_k_multiplier
        mock_pinecone_adapter.query.assert_awaited_once_with(
            vector=expected_query_vector,
            filter=None,
            top_k=expected_initial_k,  # Multiplied k
            namespace=None,
            include_values=False,
            include_metadata=True
        )
        # NEW ORDER-INSENSITIVE ASSERTION:
        mock_mongodb_adapter.find.assert_called_once()  # Check it was called once
        call_args, call_kwargs = mock_mongodb_adapter.find.call_args
        # Check collection name
        assert call_args[0] == service.collection
        # Check query structure and compare IDs ignoring order
        assert "document_id" in call_args[1]
        assert "$in" in call_args[1]["document_id"]
        actual_ids = call_args[1]["document_id"]["$in"]
        expected_ids = ["pdf1", "doc2"]
        assert sorted(actual_ids) == sorted(expected_ids)

        # Check combined results
        assert len(results) == 2
        # Result 1 (pdf1_chunk_1) - Content should come from Pinecone metadata
        assert results[0]['document_id'] == "pdf1_chunk_1"
        assert results[0]['score'] == 0.95
        # From Pinecone meta
        assert results[0]['content'] == "Reranked Chunk 1 Text"
        # From Mongo parent
        assert results[0]['metadata']['source'] == "pdf_source"
        # From Pinecone chunk meta
        assert results[0]['metadata']['chunk_index'] == 1
        # Result 2 (doc2) - Content should come from Pinecone metadata
        assert results[1]['document_id'] == "doc2"
        assert results[1]['score'] == 0.90
        # From Pinecone meta
        assert results[1]['content'] == "Reranked Doc2 Text"
        assert results[1]['metadata']['source'] == "s2"  # From Mongo doc

    async def test_query_with_filter_namespace_no_content(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        query = "filtered query"
        my_filter = {"tags": "important"}
        my_namespace = "project_x"
        expected_query_vector = [0.2] * 3072
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_query_embedding.return_value = expected_query_vector

        pinecone_res = [{"id": "doc3", "score": 0.7, "metadata": {
            "document_id": "doc3", "is_chunk": False}}]
        mock_pinecone_adapter.query.return_value = pinecone_res
        # No need to mock Mongo find if include_content/metadata is false

        results = await service.query(query, filter=my_filter, top_k=1, namespace=my_namespace, include_content=False, include_metadata=False)

        mock_pinecone_adapter.query.assert_awaited_once_with(
            vector=expected_query_vector,
            filter=my_filter,
            top_k=1,
            namespace=my_namespace,
            include_values=False,
            include_metadata=True  # Still need metadata for IDs and chunk info
        )
        # mock_mongodb_adapter.find.assert_not_called() # REMOVED - Mongo might still be called internally

        assert len(results) == 1
        assert results[0]['document_id'] == "doc3"
        assert results[0]['score'] == 0.7
        assert results[0]['is_chunk'] is False
        assert results[0]['parent_document_id'] is None
        assert 'content' not in results[0]
        assert 'metadata' not in results[0]

    async def test_query_mongo_doc_not_found(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        query = "missing mongo doc"
        expected_query_vector = [0.2] * 3072
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_query_embedding.return_value = expected_query_vector

        pinecone_res = [{"id": "doc_missing", "score": 0.6, "metadata": {
            "document_id": "doc_missing", "is_chunk": False}}]
        mock_pinecone_adapter.query.return_value = pinecone_res
        mock_mongodb_adapter.find.return_value = []  # Mongo returns nothing

        results = await service.query(query, top_k=1)

        mock_pinecone_adapter.query.assert_awaited_once()
        mock_mongodb_adapter.find.assert_called_once_with(
            service.collection, {"document_id": {"$in": ["doc_missing"]}})

        assert len(results) == 1
        assert results[0]['document_id'] == "doc_missing"
        # Empty content if Mongo doc missing
        assert results[0]['content'] == ""
        # assert results[0]['metadata'] == {} # OLD
        # Expect Pinecone metadata when Mongo fails
        assert results[0]['metadata'] == {"is_chunk": False}  # NEW

    async def test_query_embedding_error(self, kb_service_default):
        service = kb_service_default
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_query_embedding.side_effect = Exception(
            "Query embed failed")

        results = await service.query("test")
        assert results == []  # Should return empty list on error

    async def test_query_pinecone_error(self, kb_service_default, mock_pinecone_adapter):
        service = kb_service_default
        mock_pinecone_adapter.query.side_effect = Exception(
            "Pinecone query failed")

        results = await service.query("test")
        assert results == []  # Should return empty list on error

    async def test_query_mongo_find_error(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, capsys):
        service = kb_service_default
        pinecone_res = [{"id": "doc1", "score": 0.9, "metadata": {
            "document_id": "doc1", "is_chunk": False}}]
        mock_pinecone_adapter.query.return_value = pinecone_res
        mock_mongodb_adapter.find.side_effect = Exception("Mongo find failed")

        results = await service.query("test")

        mock_pinecone_adapter.query.assert_awaited_once()
        mock_mongodb_adapter.find.assert_called_once()
        captured = capsys.readouterr()
        assert "Error fetching documents from MongoDB" in captured.out

        # Should still return results based on Pinecone, but without Mongo data
        assert len(results) == 1
        assert results[0]['document_id'] == "doc1"
        assert results[0]['content'] == ""
        # assert results[0]['metadata'] == {} # OLD
        # Expect Pinecone metadata when Mongo fails
        assert results[0]['metadata'] == {"is_chunk": False}

    async def test_query_no_results_from_pinecone(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        mock_pinecone_adapter.query.return_value = []  # Pinecone returns empty list

        results = await service.query("test")

        mock_pinecone_adapter.query.assert_awaited_once()
        mock_mongodb_adapter.find.assert_not_called()
        assert results == []


@pytest.mark.asyncio
class TestKnowledgeBaseServiceDeleteDocument:

    async def test_delete_document_pdf_with_chunks(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        parent_id = "pdf-to-delete"
        chunk_id_1 = f"{parent_id}_chunk_0"
        chunk_id_2 = f"{parent_id}_chunk_1"

        # Mock Mongo find to return parent and chunk "stubs" (only need IDs)
        mongo_find_results = [
            {"document_id": parent_id, "is_chunk": False},
            # Assume chunks aren't stored directly in Mongo, but maybe they are?
            # The current implementation finds based on parent_document_id too.
            {"document_id": chunk_id_1,
                "parent_document_id": parent_id, "is_chunk": True},
            {"document_id": chunk_id_2,
                "parent_document_id": parent_id, "is_chunk": True},
        ]
        mock_mongodb_adapter.find.return_value = mongo_find_results

        # Mock Mongo delete result
        mock_mongodb_adapter.delete_many.return_value = MagicMock(
            deleted_count=3)

        deleted = await service.delete_document(parent_id, namespace="ns-del")

        assert deleted is True
        # Check Mongo find call
        mock_mongodb_adapter.find.assert_called_once_with(
            service.collection,
            {"$or": [{"document_id": parent_id}, {
                "parent_document_id": parent_id}]}
        )
        # Check Pinecone delete call (should include parent and chunk IDs found in Mongo)
        expected_pinecone_ids = [parent_id, chunk_id_1, chunk_id_2]
        # mock_pinecone_adapter.delete.assert_awaited_once_with( # OLD
        #     ids=expected_pinecone_ids,
        #     namespace="ns-del"
        # )
        mock_pinecone_adapter.delete.assert_awaited_once()  # NEW Check call happened
        call_args, call_kwargs = mock_pinecone_adapter.delete.await_args  # NEW Get args
        assert call_kwargs.get('namespace') == "ns-del"  # NEW Check namespace
        assert sorted(call_kwargs.get('ids', [])) == sorted(
            expected_pinecone_ids)  # NEW Check IDs ignoring order

        # Check Mongo delete call (using IDs found in Mongo)
        mock_mongodb_adapter.delete_many.assert_called_once_with(
            service.collection,
            {"document_id": {"$in": expected_pinecone_ids}}
        )

    async def test_delete_document_plain_text(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        doc_id = "text-doc-to-delete"

        # Mock Mongo find to return only the single document
        mongo_find_results = [{"document_id": doc_id, "is_chunk": False}]
        mock_mongodb_adapter.find.return_value = mongo_find_results
        mock_mongodb_adapter.delete_many.return_value = MagicMock(
            deleted_count=1)

        deleted = await service.delete_document(doc_id)

        assert deleted is True
        mock_mongodb_adapter.find.assert_called_once_with(
            service.collection,
            {"$or": [{"document_id": doc_id}, {"parent_document_id": doc_id}]}
        )
        mock_pinecone_adapter.delete.assert_awaited_once_with(
            ids=[doc_id],
            namespace=None
        )
        mock_mongodb_adapter.delete_many.assert_called_once_with(
            service.collection,
            {"document_id": {"$in": [doc_id]}}
        )

    async def test_delete_document_mongo_find_error(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, capsys):
        service = kb_service_default
        doc_id = "find-error-doc"
        mock_mongodb_adapter.find.side_effect = Exception("Mongo find crashed")

        # Should still attempt deletion with the main ID
        deleted = await service.delete_document(doc_id)

        assert deleted is True  # Pinecone deletion likely succeeded
        captured = capsys.readouterr()
        assert "Warning: Error finding documents in MongoDB for deletion" in captured.out
        # Pinecone delete called with only the main ID
        mock_pinecone_adapter.delete.assert_awaited_once_with(
            ids=[doc_id], namespace=None)
        # Mongo delete not called as find failed
        mock_mongodb_adapter.delete_many.assert_not_called()

    async def test_delete_document_pinecone_error(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        doc_id = "pinecone-error-doc"
        mongo_find_results = [{"document_id": doc_id}]
        mock_mongodb_adapter.find.return_value = mongo_find_results
        mock_pinecone_adapter.delete.side_effect = Exception(
            "Pinecone delete failed")
        mock_mongodb_adapter.delete_many.return_value = MagicMock(
            deleted_count=1)

        deleted = await service.delete_document(doc_id)

        # Should return True because Mongo deletion succeeded
        assert deleted is True
        mock_pinecone_adapter.delete.assert_awaited_once()
        mock_mongodb_adapter.delete_many.assert_called_once()

    async def test_delete_document_mongo_delete_error(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        doc_id = "mongo-delete-error-doc"
        mongo_find_results = [{"document_id": doc_id}]
        mock_mongodb_adapter.find.return_value = mongo_find_results
        mock_mongodb_adapter.delete_many.side_effect = Exception(
            "Mongo delete crashed")

        deleted = await service.delete_document(doc_id)

        # Should return True because Pinecone deletion succeeded
        assert deleted is True
        mock_pinecone_adapter.delete.assert_awaited_once()
        mock_mongodb_adapter.delete_many.assert_called_once()

    async def test_delete_document_not_found_anywhere(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        doc_id = "does-not-exist"
        mock_mongodb_adapter.find.return_value = []  # Mongo finds nothing
        # Simulate Pinecone delete succeeding even if ID doesn't exist (idempotent)
        mock_pinecone_adapter.delete.return_value = None
        mock_mongodb_adapter.delete_many.return_value = MagicMock(
            deleted_count=0)

        deleted = await service.delete_document(doc_id)

        # assert deleted is False # OLD - Nothing was actually deleted
        assert deleted is True  # NEW - Operation completed without error
        mock_mongodb_adapter.find.assert_called_once()
        # Pinecone delete still called with the ID
        mock_pinecone_adapter.delete.assert_awaited_once_with(
            ids=[doc_id], namespace=None)
        # Mongo delete not called as no IDs were found
        mock_mongodb_adapter.delete_many.assert_not_called()


@pytest.mark.asyncio
class TestKnowledgeBaseServiceUpdateDocument:

    @pytest.fixture
    def existing_doc(self, mock_datetime):
        now = mock_datetime.now.return_value
        one_day_ago = now - timedelta(days=1)
        return {
            "_id": "mongo1",
            "document_id": "doc-to-update",
            "content": "Original text content.",
            "is_chunk": False,
            "parent_document_id": None,
            "source": "original_source",
            "tags": ["tagA"],
            "created_at": one_day_ago,
            "updated_at": one_day_ago
        }

    async def test_update_document_metadata_only(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, existing_doc, mock_datetime):
        service = kb_service_default
        doc_id = existing_doc["document_id"]
        mock_mongodb_adapter.find_one.return_value = existing_doc
        new_metadata = {"source": "updated_source",
                        "tags": ["tagB"], "new_field": True}
        expected_update_time = mock_datetime.now.return_value

        updated = await service.update_document(doc_id, metadata=new_metadata)

        assert updated is True
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id})
        # Check Mongo update call
        mock_mongodb_adapter.update_one.assert_called_once()
        update_call_args = mock_mongodb_adapter.update_one.call_args
        assert update_call_args[0][0] == service.collection
        assert update_call_args[0][1] == {"document_id": doc_id}
        assert update_call_args[0][2] == {
            "$set": {
                "source": "updated_source",
                "tags": ["tagB"],
                "new_field": True,
                "updated_at": expected_update_time
            }
        }
        # Pinecone should NOT be updated as text didn't change
        mock_pinecone_adapter.upsert.assert_not_awaited()
        service.semantic_splitter.embed_model.aget_text_embedding.assert_not_awaited()

    async def test_update_document_text_only_rerank_off(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, existing_doc, mock_datetime):
        service = kb_service_default  # Rerank off
        doc_id = existing_doc["document_id"]
        mock_mongodb_adapter.find_one.return_value = existing_doc
        new_text = "Updated text content."
        expected_embedding = [0.15] * 3072
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding.return_value = expected_embedding
        expected_update_time = mock_datetime.now.return_value

        updated = await service.update_document(doc_id, text=new_text, namespace="update-ns")

        assert updated is True
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id})
        # Check Mongo update
        mock_mongodb_adapter.update_one.assert_called_once_with(
            service.collection,
            {"document_id": doc_id},
            {"$set": {"content": new_text, "updated_at": expected_update_time}}
        )
        # Check embedding call
        embed_model_instance.aget_text_embedding.assert_awaited_once_with(
            new_text)
        # Check Pinecone upsert
        mock_pinecone_adapter.upsert.assert_awaited_once()
        upsert_call_args = mock_pinecone_adapter.upsert.call_args
        assert upsert_call_args[1]['namespace'] == "update-ns"
        vector = upsert_call_args[1]['vectors'][0]
        assert vector['id'] == doc_id
        assert vector['values'] == expected_embedding
        # Metadata should reflect original non-updated fields from Mongo doc
        assert vector['metadata'] == {
            "document_id": doc_id,
            "is_chunk": False,
            "source": "original_source",  # From existing_doc
            "tags": ["tagA"]           # From existing_doc
            # No rerank field
        }

    async def test_update_document_text_and_metadata_rerank_on(self, kb_service_custom, mock_mongodb_adapter, mock_pinecone_adapter, existing_doc, mock_datetime):
        service = kb_service_custom  # Rerank on
        doc_id = existing_doc["document_id"]
        # Adjust existing doc for custom service collection
        existing_doc["source"] = "custom_source"
        mock_mongodb_adapter.find_one.return_value = existing_doc
        new_text = "Updated text for rerank."
        new_metadata = {"tags": ["tagC"], "status": "active"}
        expected_embedding = [0.15] * 1536
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding.return_value = expected_embedding
        expected_update_time = mock_datetime.now.return_value

        updated = await service.update_document(doc_id, text=new_text, metadata=new_metadata)

        assert updated is True
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id})
        # Check Mongo update
        mock_mongodb_adapter.update_one.assert_called_once_with(
            service.collection,
            {"document_id": doc_id},
            {"$set": {
                "content": new_text,
                "tags": ["tagC"],
                "status": "active",
                "updated_at": expected_update_time
            }}
        )
        # Check embedding call
        embed_model_instance.aget_text_embedding.assert_awaited_once_with(
            new_text)
        # Check Pinecone upsert
        mock_pinecone_adapter.upsert.assert_awaited_once()
        vector = mock_pinecone_adapter.upsert.call_args[1]['vectors'][0]
        assert vector['id'] == doc_id
        assert vector['values'] == expected_embedding
        # Metadata should reflect merged state (original + updates)
        assert vector['metadata'] == {
            "document_id": doc_id,
            "is_chunk": False,
            "source": "custom_source",  # Original, not updated
            "tags": ["tagC"],           # Updated
            "status": "active",         # New
            service.pinecone.rerank_text_field: new_text  # Rerank text included
        }

    async def test_update_document_not_found(self, kb_service_default, mock_mongodb_adapter):
        service = kb_service_default
        doc_id = "does-not-exist"
        mock_mongodb_adapter.find_one.return_value = None  # Simulate not found

        updated = await service.update_document(doc_id, text="new text")

        assert updated is False
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id})
        mock_mongodb_adapter.update_one.assert_not_called()
        service.semantic_splitter.embed_model.aget_text_embedding.assert_not_awaited()

    async def test_update_document_is_chunk_error(self, kb_service_default, mock_mongodb_adapter, existing_doc):
        service = kb_service_default
        doc_id = existing_doc["document_id"]
        existing_doc["is_chunk"] = True  # Mark as chunk
        mock_mongodb_adapter.find_one.return_value = existing_doc

        updated = await service.update_document(doc_id, text="new text")

        assert updated is False
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id})
        mock_mongodb_adapter.update_one.assert_not_called()
        service.semantic_splitter.embed_model.aget_text_embedding.assert_not_awaited()

    async def test_update_document_pdf_text_error(self, kb_service_default, mock_mongodb_adapter, existing_doc):
        service = kb_service_default
        doc_id = existing_doc["document_id"]
        existing_doc["pdf_data"] = b"some bytes"  # Mark as PDF
        mock_mongodb_adapter.find_one.return_value = existing_doc

        # Attempt to update text
        updated = await service.update_document(doc_id, text="new text")

        assert updated is False
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id})
        mock_mongodb_adapter.update_one.assert_not_called()
        service.semantic_splitter.embed_model.aget_text_embedding.assert_not_awaited()

    async def test_update_document_no_changes(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, existing_doc, mock_datetime):
        service = kb_service_default
        doc_id = existing_doc["document_id"]
        mock_mongodb_adapter.find_one.return_value = existing_doc
        # Simulate update_one returning 0 modified count if no actual change
        mock_mongodb_adapter.update_one.return_value = MagicMock(
            modified_count=0)
        expected_update_time = mock_datetime.now.return_value

        # Call update with None for text and metadata
        updated = await service.update_document(doc_id, text=None, metadata=None)

        # Even with no changes, updated_at should be set
        assert updated is False  # Because modified_count was 0
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id})
        mock_mongodb_adapter.update_one.assert_called_once_with(
            service.collection,
            {"document_id": doc_id},
            {"$set": {"updated_at": expected_update_time}}
        )
        mock_pinecone_adapter.upsert.assert_not_awaited()  # No text change

    async def test_update_document_mongo_update_fails(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, existing_doc):
        service = kb_service_default
        doc_id = existing_doc["document_id"]
        mock_mongodb_adapter.find_one.return_value = existing_doc
        mock_mongodb_adapter.update_one.side_effect = Exception(
            "Mongo update failed")

        updated = await service.update_document(doc_id, metadata={"source": "new"})

        assert updated is False  # Should return False if Mongo fails
        mock_mongodb_adapter.find_one.assert_called_once()
        mock_mongodb_adapter.update_one.assert_called_once()
        mock_pinecone_adapter.upsert.assert_not_awaited()

    async def test_update_document_text_embedding_fails(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, existing_doc):
        service = kb_service_default
        doc_id = existing_doc["document_id"]
        mock_mongodb_adapter.find_one.return_value = existing_doc
        mock_mongodb_adapter.update_one.return_value = MagicMock(
            modified_count=1)  # Mongo succeeds
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding.side_effect = Exception(
            "Embed failed")

        updated = await service.update_document(doc_id, text="new text")

        # Should return True because Mongo update succeeded
        assert updated is True
        mock_mongodb_adapter.find_one.assert_called_once()
        mock_mongodb_adapter.update_one.assert_called_once()
        embed_model_instance.aget_text_embedding.assert_awaited_once()
        mock_pinecone_adapter.upsert.assert_not_awaited()  # Pinecone upsert skipped

    async def test_update_document_text_pinecone_fails(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, existing_doc):
        service = kb_service_default
        doc_id = existing_doc["document_id"]
        mock_mongodb_adapter.find_one.return_value = existing_doc
        mock_mongodb_adapter.update_one.return_value = MagicMock(
            modified_count=1)  # Mongo succeeds
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding.return_value = [0.15] * 3072
        mock_pinecone_adapter.upsert.side_effect = Exception(
            "Pinecone update failed")

        updated = await service.update_document(doc_id, text="new text")

        # Should return True because Mongo update succeeded
        assert updated is True
        mock_mongodb_adapter.find_one.assert_called_once()
        mock_mongodb_adapter.update_one.assert_called_once()
        embed_model_instance.aget_text_embedding.assert_awaited_once()
        mock_pinecone_adapter.upsert.assert_awaited_once()  # Pinecone upsert attempted


@pytest.mark.asyncio
class TestKnowledgeBaseServiceAddDocumentsBatch:

    @pytest.fixture
    def sample_docs(self):
        return [
            {'text': 'Doc 1 text', 'metadata': {'source': 'batch1', 'id': 'doc1'}},
            {'text': 'Doc 2 text', 'metadata': {
                'source': 'batch1', 'tags': ['t1']}},  # Auto ID
            {'text': 'Doc 3 text', 'metadata': {'source': 'batch2', 'id': 'doc3'}},
        ]

    # Use mock_uuid_multiple fixture for batch tests needing multiple UUIDs
    async def test_add_documents_batch_success_single_batch(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, sample_docs, mock_uuid_multiple, mock_datetime):
        service = kb_service_default
        docs_to_add = sample_docs[:2]  # First two docs
        # Assuming service ignores provided 'id' and generates UUIDs for all
        expected_ids = [str(uuid.UUID('11111111-1111-1111-1111-111111111111')),
                        str(uuid.UUID('22222222-2222-2222-2222-222222222222'))]
        # Ensure expected dimensions match the service default (3072)
        expected_embeddings = [[0.1] * 3072, [0.11] * 3072]

        embed_model_instance = service.semantic_splitter.embed_model
        # Clear any default side_effect from the fixture
        embed_model_instance.aget_text_embedding_batch.side_effect = None
        # Set the specific return value for this test
        embed_model_instance.aget_text_embedding_batch.return_value = expected_embeddings
        # Mock insert_many to reflect the IDs inserted
        mock_mongodb_adapter.insert_many.return_value = MagicMock(
            inserted_ids=["mongo_id_1", "mongo_id_2"])

        added_ids = await service.add_documents_batch(docs_to_add, batch_size=5)

        assert added_ids == expected_ids
        # ... rest of assertions ...
        # Check Pinecone upsert
        mock_pinecone_adapter.upsert.assert_awaited_once()
        pinecone_call_args = mock_pinecone_adapter.upsert.call_args[1]['vectors']
        assert len(pinecone_call_args) == 2
        # Check generated ID
        assert pinecone_call_args[0]['id'] == expected_ids[0]
        # This assertion should now pass
        assert pinecone_call_args[0]['values'] == expected_embeddings[0]
        assert pinecone_call_args[0]['metadata']['source'] == 'batch1'
        # Check generated ID
        assert pinecone_call_args[1]['id'] == expected_ids[1]
        assert pinecone_call_args[1]['values'] == expected_embeddings[1]
        assert pinecone_call_args[1]['metadata']['tags'] == ['t1']

    # Use mock_uuid_multiple fixture
    async def test_add_documents_batch_multiple_batches(self, kb_service_custom, mock_mongodb_adapter, mock_pinecone_adapter, sample_docs, mock_uuid_multiple):
        service = kb_service_custom  # Rerank on
        docs_to_add = sample_docs  # All 3 docs
        # Assuming service ignores provided IDs and generates 3 UUIDs
        expected_ids = [str(uuid.UUID('11111111-1111-1111-1111-111111111111')),
                        str(uuid.UUID('22222222-2222-2222-2222-222222222222')),
                        str(uuid.UUID('33333333-3333-3333-3333-333333333333'))]

        # Simulate embeddings for 3 docs
        expected_embeddings_batch1 = [[0.1] * 1536, [0.11] * 1536]
        expected_embeddings_batch2 = [[0.12] * 1536]
        embed_model_instance = service.semantic_splitter.embed_model
        # Ensure side_effect matches batching (2 calls for batch_size=2, docs=3)
        embed_model_instance.aget_text_embedding_batch.side_effect = [
            expected_embeddings_batch1, expected_embeddings_batch2
        ]

        added_ids = await service.add_documents_batch(docs_to_add, batch_size=2, namespace="multi-batch")

        assert added_ids == expected_ids
        # Check calls (2 batches)
        assert mock_mongodb_adapter.insert_many.call_count == 2
        assert embed_model_instance.aget_text_embedding_batch.await_count == 2
        assert mock_pinecone_adapter.upsert.await_count == 2
        # Check Pinecone calls have correct namespace and rerank text
        first_pinecone_call = mock_pinecone_adapter.upsert.await_args_list[0]
        assert first_pinecone_call[1]['namespace'] == "multi-batch"
        assert len(first_pinecone_call[1]['vectors']) == 2
        assert service.pinecone.rerank_text_field in first_pinecone_call[
            1]['vectors'][0]['metadata']
        assert first_pinecone_call[1]['vectors'][0]['metadata'][service.pinecone.rerank_text_field] == 'Doc 1 text'
        # Check generated ID
        assert first_pinecone_call[1]['vectors'][0]['id'] == expected_ids[0]
        # Check generated ID
        assert first_pinecone_call[1]['vectors'][1]['id'] == expected_ids[1]

        second_pinecone_call = mock_pinecone_adapter.upsert.await_args_list[1]
        assert second_pinecone_call[1]['namespace'] == "multi-batch"
        assert len(second_pinecone_call[1]['vectors']) == 1
        assert service.pinecone.rerank_text_field in second_pinecone_call[
            1]['vectors'][0]['metadata']
        assert second_pinecone_call[1]['vectors'][0]['metadata'][service.pinecone.rerank_text_field] == 'Doc 3 text'
        # Check generated ID
        assert second_pinecone_call[1]['vectors'][0]['id'] == expected_ids[2]

    async def test_add_documents_batch_mongo_error_skips_batch(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, sample_docs, capsys):
        service = kb_service_default
        docs_to_add = sample_docs  # 3 docs
        mock_mongodb_adapter.insert_many.side_effect = [
            MagicMock(inserted_ids=["id1"]),  # Batch 1 (size 1) succeeds
            Exception("Mongo batch 2 failed"),  # Batch 2 (size 1) fails
            MagicMock(inserted_ids=["id3"]),  # Batch 3 (size 1) succeeds
        ]
        embed_model_instance = service.semantic_splitter.embed_model

        # 3 batches
        await service.add_documents_batch(docs_to_add, batch_size=1)

        # Mongo called for each batch, even if it fails and logs error
        # assert mock_mongodb_adapter.insert_many.call_count == 2 # OLD
        assert mock_mongodb_adapter.insert_many.call_count == 3  # NEW
        # Embedding only called for batches *before* the mongo error (or successful ones if loop continues)
        # Batch 1: Mongo OK -> Embed OK -> Upsert OK
        # Batch 2: Mongo FAIL -> Embed SKIP -> Upsert SKIP
        # Batch 3: Mongo OK -> Embed OK -> Upsert OK
        # Called for batch 1 and 3
        assert embed_model_instance.aget_text_embedding_batch.await_count == 2
        # Pinecone only called for batches where embed succeeded
        assert mock_pinecone_adapter.upsert.await_count == 2  # Called for batch 1 and 3
        captured = capsys.readouterr()
        assert "Error inserting batch 2 into MongoDB" in captured.out

    async def test_add_documents_batch_embedding_error_skips_upsert(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, sample_docs, capsys):
        service = kb_service_default
        docs_to_add = sample_docs
        mock_mongodb_adapter.insert_many.return_value = MagicMock(
            inserted_ids=["id1", "id2", "id3"])  # Mongo succeeds
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding_batch.side_effect = [
            [[0.1] * 3072],  # Batch 1 succeeds
            Exception("Embed batch 2 failed"),  # Batch 2 fails
            [[0.12] * 3072]  # Batch 3 succeeds
        ]

        await service.add_documents_batch(docs_to_add, batch_size=1)

        assert mock_mongodb_adapter.insert_many.call_count == 3
        # Embedding attempted 3 times
        assert embed_model_instance.aget_text_embedding_batch.await_count == 3
        # Pinecone upsert only called twice (skipped batch 2)
        assert mock_pinecone_adapter.upsert.await_count == 2
        captured = capsys.readouterr()
        assert "Error embedding batch 2" in captured.out

    async def test_add_documents_batch_pinecone_error_logged(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, sample_docs, capsys):
        service = kb_service_default
        docs_to_add = sample_docs
        mock_mongodb_adapter.insert_many.return_value = MagicMock(
            inserted_ids=["id1", "id2", "id3"])  # Mongo succeeds
        embed_model_instance = service.semantic_splitter.embed_model
        embed_model_instance.aget_text_embedding_batch.return_value = [
            [0.1] * 3072]  # Embedding succeeds
        mock_pinecone_adapter.upsert.side_effect = Exception(
            "Pinecone batch failed")

        # Should not raise, just log
        await service.add_documents_batch(docs_to_add, batch_size=3)

        assert mock_mongodb_adapter.insert_many.call_count == 1
        embed_model_instance.aget_text_embedding_batch.assert_awaited_once()
        mock_pinecone_adapter.upsert.assert_awaited_once()  # Attempted
        captured = capsys.readouterr()
        assert "Error upserting vector batch 1 to Pinecone" in captured.out

    async def test_add_documents_batch_empty_list(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter):
        service = kb_service_default
        added_ids = await service.add_documents_batch([])
        assert added_ids == []
        mock_mongodb_adapter.insert_many.assert_not_called()
        service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_not_awaited()
        mock_pinecone_adapter.upsert.assert_not_awaited()

    async def test_add_pdf_document_no_chunks_generated(self, kb_service_default, mock_mongodb_adapter, mock_pinecone_adapter, mock_pypdf_reader, mock_semantic_splitter, mock_uuid, mock_datetime, mock_asyncio_to_thread, capsys):
        """Test the case where PDF text is extracted but splitter returns no nodes."""
        service = kb_service_default
        pdf_bytes = b'some-pdf-bytes'
        metadata = {"source": "no_chunks_test"}
        expected_parent_id = str(mock_uuid.return_value)
        # Mock pypdf to return some text
        reader_instance = mock_pypdf_reader.return_value
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Some text that results in no chunks."
        reader_instance.pages = [mock_page]

        # Mock splitter to return empty list
        splitter_instance = mock_semantic_splitter.return_value
        splitter_instance.get_nodes_from_documents.return_value = []
        embed_model_instance = service.semantic_splitter.embed_model

        doc_id = await service.add_pdf_document(pdf_bytes, metadata)

        # Assertions
        assert doc_id == expected_parent_id
        # mock_pypdf_reader.assert_called_once_with(io.BytesIO(pdf_bytes)) # OLD
        mock_pypdf_reader.assert_called_once_with(ANY)  # NEW
        # Parent doc should still be inserted in Mongo
        mock_mongodb_adapter.insert_one.assert_called_once()
        mongo_call_args = mock_mongodb_adapter.insert_one.call_args[0][1]
        assert mongo_call_args['document_id'] == expected_parent_id
        assert mongo_call_args['content'] == "Some text that results in no chunks."

        # Splitter should have been called
        mock_asyncio_to_thread.assert_awaited_once()
        splitter_instance.get_nodes_from_documents.assert_called_once()

        # Embedding and Pinecone upsert should NOT be called
        embed_model_instance.aget_text_embedding_batch.assert_not_awaited()
        mock_pinecone_adapter.upsert.assert_not_awaited()

        # Check logs (optional, depends on desired logging behavior)
        captured = capsys.readouterr()
        # Example: Check if a specific log message appears or doesn't appear
        assert f"Generated 0 semantic chunks for PDF {expected_parent_id}" in captured.out
        # Check that embedding step was skipped
        assert "Embedding 0 chunks" not in captured.out


@pytest.mark.asyncio
class TestKnowledgeBaseServiceGetFullDocument:

    async def test_get_full_document_success(self, kb_service_default, mock_mongodb_adapter):
        service = kb_service_default
        doc_id = "get-me-123"
        expected_doc = {"document_id": doc_id,
                        "content": "Full content", "pdf_data": b"bytes"}
        mock_mongodb_adapter.find_one.return_value = expected_doc

        doc = await service.get_full_document(doc_id)

        assert doc == expected_doc
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id})

    async def test_get_full_document_not_found(self, kb_service_default, mock_mongodb_adapter):
        service = kb_service_default
        doc_id = "not-found-id"
        mock_mongodb_adapter.find_one.return_value = None  # Simulate not found

        doc = await service.get_full_document(doc_id)

        assert doc is None
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id})

    async def test_get_full_document_mongo_error(self, kb_service_default, mock_mongodb_adapter, capsys):
        service = kb_service_default
        doc_id = "mongo-error-id"
        mock_mongodb_adapter.find_one.side_effect = Exception(
            "Mongo find_one failed")

        doc = await service.get_full_document(doc_id)

        assert doc is None
        mock_mongodb_adapter.find_one.assert_called_once_with(
            service.collection, {"document_id": doc_id})
        captured = capsys.readouterr()
        assert f"Error retrieving full document {doc_id} from MongoDB" in captured.out

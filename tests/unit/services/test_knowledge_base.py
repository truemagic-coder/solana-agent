import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from datetime import datetime as dt, timezone
import uuid
import io

# Assuming these are correctly importable based on project structure
from solana_agent.adapters.pinecone_adapter import PineconeAdapter
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
from solana_agent.interfaces.providers.llm import LLMProvider
from solana_agent.services.knowledge_base import KnowledgeBase

# Mock LlamaIndex classes if they are complex or have external dependencies
# For simplicity, using MagicMock for now. Replace with more specific mocks if needed.
try:
    from llama_index.core import Document as LlamaDocument
    from llama_index.core.schema import TextNode
    from llama_index.core.node_parser import SemanticSplitterNodeParser
except ImportError:
    LlamaDocument = MagicMock()
    TextNode = MagicMock()
    SemanticSplitterNodeParser = MagicMock()

# Mock pypdf
try:
    import pypdf
except ImportError:
    pypdf = MagicMock()


# --- Fixtures ---

@pytest.fixture
def mock_pinecone_adapter():
    adapter = AsyncMock(spec=PineconeAdapter)
    adapter.use_reranking = False  # Default, can be overridden in tests
    adapter.rerank_text_field = "text_for_rerank"
    adapter.embedding_dimensions = 768  # Example dimension
    return adapter


@pytest.fixture
def mock_mongo_adapter():
    adapter = MagicMock(spec=MongoDBAdapter)
    adapter.collection_exists.return_value = True  # Assume exists by default
    # Mock find/insert etc. as needed per test
    adapter.find.return_value = []
    adapter.find_one.return_value = None
    adapter.insert_one.return_value = MagicMock(inserted_id="mock_mongo_id")
    # FIX: Explicitly create the insert_many mock attribute *after* spec is set
    adapter.insert_many = MagicMock(return_value=MagicMock(
        inserted_ids=["mock_id_1", "mock_id_2"]))
    adapter.update_one.return_value = MagicMock(modified_count=1)
    # FIX: Explicitly create the delete_many mock attribute *after* spec is set
    adapter.delete_many = MagicMock(return_value=MagicMock(deleted_count=1))
    return adapter


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock(spec=LLMProvider)
    # Mock the required embed_text method

    async def mock_embed_text(text, model=None, dimensions=None):
        # Return a dummy embedding of the correct dimension
        dim = dimensions or 768
        return [float(hash(text + str(i)) % 1000 / 1000) for i in range(dim)]
    provider.embed_text = mock_embed_text
    return provider

# Mock for the LlamaIndex embedding adapter used internally


class MockLlamaEmbedding:
    async def _aget_text_embedding(self, text: str) -> list[float]:
        return [0.1] * 768  # Return dummy embedding


@pytest.fixture
def mock_semantic_splitter_nodes():
    # Create mock TextNode objects
    return [
        TextNode(id_="node1", text="Chunk 0 content.",
                 metadata={"doc_id": "temp"}),
        TextNode(id_="node2", text="Chunk 1 content.",
                 metadata={"doc_id": "temp"}),
        TextNode(id_="node3", text="Chunk 2 content.",
                 metadata={"doc_id": "temp"}),
    ]


@pytest.fixture
def mock_pdf_reader(mock_semantic_splitter_nodes):
    # Mock pypdf.PdfReader and its methods/attributes
    mock_reader_instance = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Extracted PDF text. "
    # Combine node texts for simulation
    all_node_text = "".join(
        [node.text for node in mock_semantic_splitter_nodes])
    mock_page1.extract_text.return_value += all_node_text

    mock_reader_instance.pages = [mock_page1]

    # Patch pypdf.PdfReader to return our mock instance
    with patch('pypdf.PdfReader', return_value=mock_reader_instance) as mock_reader_class:
        yield mock_reader_class  # Yield the patched class


@pytest.fixture
def knowledge_base(mock_pinecone_adapter, mock_mongo_adapter, mock_llm_provider):
    """Fixture to provide a KnowledgeBase instance with mocked dependencies."""
    # Patch the SemanticSplitterNodeParser constructor within the KnowledgeBase init
    with patch('solana_agent.services.knowledge_base.SemanticSplitterNodeParser') as MockSplitterClass:
        # Configure the mock splitter instance returned by the patched class
        mock_splitter_instance = MagicMock(spec=SemanticSplitterNodeParser)
        # Mock the method that will be called on the instance
        mock_splitter_instance.get_nodes_from_documents = MagicMock(
            return_value=[])  # Default empty list
        MockSplitterClass.return_value = mock_splitter_instance

        kb = KnowledgeBase(mock_pinecone_adapter, mock_mongo_adapter,
                           mock_llm_provider, collection_name="test_kb")
        # Attach the mock splitter instance to the kb instance for assertion purposes if needed
        kb.mock_splitter_instance = mock_splitter_instance
        yield kb


@pytest.fixture
def sample_pdf_data():
    # Create minimal valid PDF bytes
    return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n140\n%%EOF"


@pytest.fixture
def sample_pdf_metadata():
    return {"author": "Test Author", "source": "test_files", "tags": ["pdf", "testing"], "title": "Test PDF Document"}


@pytest.fixture
def sample_pinecone_results():
    # Simulate results from pinecone.query_text
    return [
        {"id": "doc1", "score": 0.95, "metadata": {"document_id": "doc1",
                                                   "is_chunk": False, "source": "website", "tags": ["blockchain", "crypto"]}},
        {"id": "pdf1_chunk_0", "score": 0.90, "metadata": {"document_id": "pdf1_chunk_0", "parent_document_id": "pdf1",
                                                           "chunk_index": 0, "is_chunk": True, "source": "report.pdf", "tags": ["finance", "pdf"]}},
        {"id": "doc2", "score": 0.85, "metadata": {"document_id": "doc2",
                                                   "is_chunk": False, "source": "blog", "tags": ["solana", "tutorial"]}},
        {"id": "pdf1_chunk_1", "score": 0.80, "metadata": {"document_id": "pdf1_chunk_1", "parent_document_id": "pdf1",
                                                           "chunk_index": 1, "is_chunk": True, "source": "report.pdf", "tags": ["finance", "pdf"]}},
    ]


@pytest.fixture
def sample_mongo_docs_map(sample_pinecone_results):
    # Simulate documents fetched from MongoDB based on IDs in sample_pinecone_results
    now = dt.now(timezone.utc)
    docs = {
        "doc1": {
            "_id": "mongo_id_1",
            "document_id": "doc1",
            "content": "Solana is a high-performance blockchain supporting builders around the world.",
            "title": "About Solana",
            "source": "website",
            "tags": ["blockchain", "crypto"],
            "is_chunk": False,
            "parent_document_id": None,
            "pdf_data": None,
            "created_at": now,
            "updated_at": now
        },
        "doc2": {
            "_id": "mongo_id_2",
            "document_id": "doc2",
            "content": "Learn how to build dApps on Solana.",
            "title": "Solana Tutorial",
            "source": "blog",
            "tags": ["solana", "tutorial"],
            "is_chunk": False,
            "parent_document_id": None,
            "pdf_data": None,
            "created_at": now,
            "updated_at": now
        },
        # Parent doc for the chunks
        "pdf1": {
            "_id": "mongo_id_pdf1",
            "document_id": "pdf1",
            "content": "Extracted PDF text. Chunk 0 content. Chunk 1 content.",  # Full extracted text
            "title": "Financial Report Q1",
            "source": "report.pdf",
            "tags": ["finance", "pdf"],
            "is_chunk": False,
            "parent_document_id": None,
            "pdf_data": b"dummy pdf bytes",  # Representing stored PDF
            "created_at": now,
            "updated_at": now
        }
        # Chunks themselves are NOT stored in Mongo in this design
    }
    return docs


# --- Test Class ---

@pytest.mark.usefixtures("knowledge_base")
class TestKnowledgeBase:

    # --- Initialization Tests ---
    def test_init_success(self, knowledge_base: KnowledgeBase, mock_pinecone_adapter, mock_mongo_adapter, mock_llm_provider):
        """Test successful initialization."""
        assert knowledge_base.pinecone == mock_pinecone_adapter
        assert knowledge_base.mongo == mock_mongo_adapter
        assert knowledge_base.llm_provider == mock_llm_provider
        assert knowledge_base.collection == "test_kb"
        # Check if splitter was initialized (via the mock instance attached in fixture)
        assert hasattr(knowledge_base, 'semantic_splitter')
        assert isinstance(knowledge_base.semantic_splitter,
                          MagicMock)  # Check it's the mock
        # Check if _ensure_collection was called (implicitly via mongo mock calls)
        mock_mongo_adapter.collection_exists.assert_called_once_with("test_kb")
        # Assuming collection exists=True, create_collection is not called
        mock_mongo_adapter.create_collection.assert_not_called()
        # Check if indexes were created
        # Check count matches _ensure_collection
        assert mock_mongo_adapter.create_index.call_count >= 6

    def test_init_creates_collection_if_not_exists(self, mock_pinecone_adapter, mock_mongo_adapter, mock_llm_provider):
        """Test initialization creates collection if it doesn't exist."""
        # FIX: Reset mock before test-specific instantiation
        mock_mongo_adapter.reset_mock()
        mock_mongo_adapter.collection_exists.return_value = False

        # Patch the splitter init for this specific test
        with patch('solana_agent.services.knowledge_base.SemanticSplitterNodeParser') as MockSplitterClass:
            MockSplitterClass.return_value = MagicMock(
                spec=SemanticSplitterNodeParser)
            KnowledgeBase(mock_pinecone_adapter, mock_mongo_adapter,
                          mock_llm_provider, collection_name="new_kb")

        mock_mongo_adapter.collection_exists.assert_called_once_with("new_kb")
        mock_mongo_adapter.create_collection.assert_called_once_with("new_kb")
        # Check indexes are still created
        assert mock_mongo_adapter.create_index.call_count >= 6

    def test_init_with_existing_collection(self, mock_pinecone_adapter, mock_mongo_adapter, mock_llm_provider):
        """Test initialization skips collection creation if it exists, but still creates indexes."""
        # FIX: Reset mock before test-specific instantiation
        mock_mongo_adapter.reset_mock()
        mock_mongo_adapter.collection_exists.return_value = True

        # Patch the SemanticSplitterNodeParser constructor directly
        with patch('solana_agent.services.knowledge_base.SemanticSplitterNodeParser') as MockSplitterClass:
            # We don't need the return value for this test, just prevent the original init
            MockSplitterClass.return_value = MagicMock(
                spec=SemanticSplitterNodeParser)
            kb = KnowledgeBase(mock_pinecone_adapter, mock_mongo_adapter,
                               mock_llm_provider, collection_name="existing_kb")

        mock_mongo_adapter.collection_exists.assert_called_once_with(
            "existing_kb")
        mock_mongo_adapter.create_collection.assert_not_called()
        # Check indexes are still created
        assert mock_mongo_adapter.create_index.call_count >= 6

    def test_init_llm_provider_missing_embed_method(self, mock_pinecone_adapter, mock_mongo_adapter):
        """Test error if LLMProvider doesn't have embed_text."""
        bad_llm_provider = MagicMock()
        # Explicitly remove the method if it somehow exists on the mock
        if hasattr(bad_llm_provider, 'embed_text'):
            del bad_llm_provider.embed_text

        # Patch the splitter initialization parts to isolate the ValueError check
        # FIX: Patch the SemanticSplitterNodeParser constructor directly
        with patch('solana_agent.services.knowledge_base.SemanticSplitterNodeParser'):
            with pytest.raises(ValueError, match="LLMProvider must have an 'embed_text' method"):
                KnowledgeBase(mock_pinecone_adapter,
                              mock_mongo_adapter, bad_llm_provider)

    # --- Add Text Document Tests ---

    @pytest.mark.asyncio
    async def test_add_document_success(self, knowledge_base: KnowledgeBase):
        """Test adding a simple text document."""
        doc_id = "test-doc-1"
        text = "This is the document content."
        metadata = {"source": "test", "tags": ["simple", "text"]}

        result_id = await knowledge_base.add_document(text, metadata, document_id=doc_id, namespace="ns_text")

        assert result_id == doc_id

        # Check MongoDB insert
        knowledge_base.mongo.insert_one.assert_called_once()
        args, kwargs = knowledge_base.mongo.insert_one.call_args
        assert args[0] == knowledge_base.collection
        mongo_doc = args[1]
        assert mongo_doc["document_id"] == doc_id
        assert mongo_doc["content"] == text
        assert mongo_doc["source"] == "test"
        assert mongo_doc["tags"] == ["simple", "text"]
        assert mongo_doc["is_chunk"] is False
        assert mongo_doc["parent_document_id"] is None
        assert "created_at" in mongo_doc
        assert "updated_at" in mongo_doc

        # Check Pinecone upsert
        knowledge_base.pinecone.upsert_text.assert_called_once()
        args, kwargs = knowledge_base.pinecone.upsert_text.call_args
        assert kwargs["texts"] == [text]
        assert kwargs["ids"] == [doc_id]
        assert kwargs["namespace"] == "ns_text"
        pinecone_meta = kwargs["metadatas"][0]
        assert pinecone_meta["document_id"] == doc_id
        assert pinecone_meta["is_chunk"] is False
        assert pinecone_meta["source"] == "test"
        assert pinecone_meta["tags"] == ["simple", "text"]
        # Check rerank field not added by default
        assert knowledge_base.pinecone.rerank_text_field not in pinecone_meta

    @pytest.mark.asyncio
    async def test_add_document_with_rerank(self, knowledge_base: KnowledgeBase):
        """Test adding a text document when reranking is enabled."""
        knowledge_base.pinecone.use_reranking = True  # Enable reranking for this test
        doc_id = "test-doc-rerank"
        text = "Content for reranking."
        metadata = {"source": "rerank_test"}

        await knowledge_base.add_document(text, metadata, document_id=doc_id)

        # Check Pinecone upsert includes the rerank text field
        knowledge_base.pinecone.upsert_text.assert_called_once()
        args, kwargs = knowledge_base.pinecone.upsert_text.call_args
        pinecone_meta = kwargs["metadatas"][0]
        assert knowledge_base.pinecone.rerank_text_field in pinecone_meta
        assert pinecone_meta[knowledge_base.pinecone.rerank_text_field] == text

    @pytest.mark.asyncio
    async def test_add_document_generate_id(self, knowledge_base: KnowledgeBase):
        """Test document ID generation when not provided."""
        text = "Auto generated ID content."
        metadata = {"source": "auto_id"}

        # Patch uuid.uuid4 to control the generated ID
        test_uuid = uuid.UUID('12345678-1234-5678-1234-567812345678')
        with patch('uuid.uuid4', return_value=test_uuid):
            result_id = await knowledge_base.add_document(text, metadata)

        assert result_id == str(test_uuid)
        # Check Mongo and Pinecone calls used the generated ID
        knowledge_base.mongo.insert_one.assert_called_once()
        assert knowledge_base.mongo.insert_one.call_args[0][1]["document_id"] == str(
            test_uuid)
        knowledge_base.pinecone.upsert_text.assert_called_once()
        assert knowledge_base.pinecone.upsert_text.call_args[1]["ids"] == [
            str(test_uuid)]

    @pytest.mark.asyncio
    async def test_add_document_mongo_error(self, knowledge_base: KnowledgeBase):
        """Test error handling when MongoDB insert fails."""
        knowledge_base.mongo.insert_one.side_effect = Exception(
            "Mongo insert error")
        text = "Text that won't be saved."
        metadata = {"source": "error_test"}

        with pytest.raises(Exception, match="Mongo insert error"):
            await knowledge_base.add_document(text, metadata)

        # Ensure Pinecone was not called
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_document_pinecone_error(self, knowledge_base: KnowledgeBase):
        """Test error handling when Pinecone upsert fails (should still insert to Mongo)."""
        knowledge_base.pinecone.upsert_text.side_effect = Exception(
            "Pinecone upsert error")
        doc_id = "doc-pinecone-fail"
        text = "Text saved in Mongo but not Pinecone."
        metadata = {"source": "pinecone_error"}

        # The function should still complete and return the ID, but log the error
        # We expect the Pinecone error to be raised *after* Mongo insert
        with pytest.raises(Exception, match="Pinecone upsert error"):
            await knowledge_base.add_document(text, metadata, document_id=doc_id)

        # Check Mongo insert was called
        knowledge_base.mongo.insert_one.assert_called_once()
        assert knowledge_base.mongo.insert_one.call_args[0][1]["document_id"] == doc_id

    # --- Add PDF Document Tests ---

    @pytest.mark.asyncio
    async def test_add_pdf_document_from_bytes(
        self, knowledge_base: KnowledgeBase, mock_pdf_reader, mock_semantic_splitter_nodes,
        sample_pdf_data, sample_pdf_metadata
    ):
        """Test adding a PDF document from bytes with chunking."""
        parent_doc_id = "pdf-bytes-test"
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.return_value = mock_semantic_splitter_nodes

        result_id = await knowledge_base.add_pdf_document(
            pdf_data=sample_pdf_data,
            metadata=sample_pdf_metadata,
            document_id=parent_doc_id,
            namespace="ns_pdf",
            chunk_batch_size=2  # Test batching
        )

        assert result_id == parent_doc_id

        # 1. Check PDF reading mock was used
        mock_pdf_reader.assert_called_once()
        # Check it was called with BytesIO
        assert isinstance(mock_pdf_reader.call_args[0][0], io.BytesIO)

        # 2. Check MongoDB insert of parent doc
        knowledge_base.mongo.insert_one.assert_called_once()
        args, kwargs = knowledge_base.mongo.insert_one.call_args
        assert args[0] == knowledge_base.collection
        mongo_doc = args[1]
        assert mongo_doc["document_id"] == parent_doc_id
        # Check raw bytes stored
        assert mongo_doc["pdf_data"] == sample_pdf_data
        assert mongo_doc["source"] == sample_pdf_metadata["source"]
        assert mongo_doc["title"] == sample_pdf_metadata["title"]
        assert mongo_doc["is_chunk"] is False
        assert "content" in mongo_doc  # Extracted text
        assert "Extracted PDF text" in mongo_doc["content"]
        assert "Chunk 0 content" in mongo_doc["content"]

        # 3. Check Semantic Splitter was called
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.assert_called_once()
        # Check it was called with a LlamaDocument containing extracted text
        llama_docs_arg = knowledge_base.mock_splitter_instance.get_nodes_from_documents.call_args[
            0][0]
        assert isinstance(llama_docs_arg[0], LlamaDocument)
        assert "Extracted PDF text" in llama_docs_arg[0].text

        # 4. Check Pinecone upsert of chunks (batched)
        assert knowledge_base.pinecone.upsert_text.call_count == 2  # 3 nodes, batch size 2
        calls = knowledge_base.pinecone.upsert_text.call_args_list

        # Call 1 (Batch 1)
        kwargs1 = calls[0][1]
        assert len(kwargs1["ids"]) == 2
        assert kwargs1["ids"] == [
            f"{parent_doc_id}_chunk_0", f"{parent_doc_id}_chunk_1"]
        assert len(kwargs1["texts"]) == 2
        assert kwargs1["texts"] == ["Chunk 0 content.", "Chunk 1 content."]
        assert kwargs1["namespace"] == "ns_pdf"
        meta1 = kwargs1["metadatas"]
        assert len(meta1) == 2
        assert meta1[0]["document_id"] == f"{parent_doc_id}_chunk_0"
        assert meta1[0]["parent_document_id"] == parent_doc_id
        assert meta1[0]["chunk_index"] == 0
        assert meta1[0]["is_chunk"] is True
        assert meta1[0]["source"] == sample_pdf_metadata["source"]  # Inherited
        assert meta1[1]["document_id"] == f"{parent_doc_id}_chunk_1"
        assert meta1[1]["chunk_index"] == 1

        # Call 2 (Batch 2)
        kwargs2 = calls[1][1]
        assert len(kwargs2["ids"]) == 1
        assert kwargs2["ids"] == [f"{parent_doc_id}_chunk_2"]
        assert len(kwargs2["texts"]) == 1
        assert kwargs2["texts"] == ["Chunk 2 content."]
        assert kwargs2["namespace"] == "ns_pdf"
        meta2 = kwargs2["metadatas"]
        assert len(meta2) == 1
        assert meta2[0]["document_id"] == f"{parent_doc_id}_chunk_2"
        assert meta2[0]["chunk_index"] == 2
        assert meta2[0]["is_chunk"] is True

    @pytest.mark.asyncio
    async def test_add_pdf_document_from_path(self, knowledge_base: KnowledgeBase, mock_pdf_reader, mock_semantic_splitter_nodes, sample_pdf_metadata):
        """Test adding a PDF document from a file path."""
        parent_doc_id = "pdf-path-test"
        pdf_path = "/fake/path/to/document.pdf"
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.return_value = mock_semantic_splitter_nodes

        # Mock open() to simulate reading from a file path
        # Use BytesIO to simulate file content matching sample_pdf_data
        # Simulate different content if needed
        mock_file_content = io.BytesIO(b"dummy pdf content from path")
        with patch("builtins.open", new_callable=MagicMock) as mock_open:
            # Configure the mock file handle returned by open()
            mock_file_handle = MagicMock()
            mock_file_handle.read.return_value = mock_file_content.getvalue()
            # Make open().__enter__() return the mock file handle
            mock_open.return_value.__enter__.return_value = mock_file_handle

            result_id = await knowledge_base.add_pdf_document(
                pdf_data=pdf_path,  # Pass the path string
                metadata=sample_pdf_metadata,
                document_id=parent_doc_id
            )

        assert result_id == parent_doc_id
        # Check open was called with the correct path and mode
        mock_open.assert_called_once_with(pdf_path, "rb")
        # Check PdfReader was called with BytesIO containing the read content
        mock_pdf_reader.assert_called_once()
        assert isinstance(mock_pdf_reader.call_args[0][0], io.BytesIO)
        assert mock_pdf_reader.call_args[0][0].getvalue(
        ) == mock_file_content.getvalue()
        # Check Mongo insert contains the read content
        knowledge_base.mongo.insert_one.assert_called_once()
        assert knowledge_base.mongo.insert_one.call_args[0][1]["pdf_data"] == mock_file_content.getvalue(
        )
        # Check splitter and pinecone were called (basic checks)
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.assert_called_once()
        knowledge_base.pinecone.upsert_text.assert_called()  # Called at least once

    @pytest.mark.asyncio
    async def test_add_pdf_document_no_extracted_text(self, knowledge_base: KnowledgeBase, mock_pdf_reader, sample_pdf_data, sample_pdf_metadata):
        """Test adding a PDF where no text could be extracted."""
        parent_doc_id = "pdf-no-text"
        # Configure mock reader to return no text
        mock_reader_instance = mock_pdf_reader.return_value
        # Simulate no text
        mock_reader_instance.pages[0].extract_text.return_value = ""

        result_id = await knowledge_base.add_pdf_document(
            pdf_data=sample_pdf_data,
            metadata=sample_pdf_metadata,
            document_id=parent_doc_id
        )

        assert result_id == parent_doc_id
        # Check Mongo insert still happened
        knowledge_base.mongo.insert_one.assert_called_once()
        mongo_doc = knowledge_base.mongo.insert_one.call_args[0][1]
        assert mongo_doc["document_id"] == parent_doc_id
        assert mongo_doc["content"] == ""  # Empty extracted text
        assert mongo_doc["pdf_data"] == sample_pdf_data
        # Check splitter and pinecone were NOT called
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.assert_not_called()
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_pdf_document_invalid_input(self, knowledge_base: KnowledgeBase, sample_pdf_metadata):
        """Test error handling for invalid pdf_data type."""
        with pytest.raises(ValueError, match="pdf_data must be bytes or a file path string"):
            await knowledge_base.add_pdf_document(pdf_data=12345, metadata=sample_pdf_metadata)

    @pytest.mark.asyncio
    async def test_add_pdf_document_pdf_read_error(self, knowledge_base: KnowledgeBase, mock_pdf_reader, sample_pdf_data, sample_pdf_metadata):
        """Test error handling when pypdf fails to read the PDF."""
        # Configure mock reader to raise an error
        mock_pdf_reader.side_effect = pypdf.errors.PdfReadError(
            "Failed to read PDF")

        with pytest.raises(pypdf.errors.PdfReadError, match="Failed to read PDF"):
            await knowledge_base.add_pdf_document(pdf_data=sample_pdf_data, metadata=sample_pdf_metadata)

        # Ensure nothing was inserted or processed further
        knowledge_base.mongo.insert_one.assert_not_called()
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.assert_not_called()
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_pdf_document_mongo_insert_error(self, knowledge_base: KnowledgeBase, mock_pdf_reader, sample_pdf_data, sample_pdf_metadata):
        """Test error handling when MongoDB insert of parent doc fails."""
        knowledge_base.mongo.insert_one.side_effect = Exception(
            "Mongo insert parent error")

        with pytest.raises(Exception, match="Mongo insert parent error"):
            await knowledge_base.add_pdf_document(pdf_data=sample_pdf_data, metadata=sample_pdf_metadata)

        knowledge_base.mongo.insert_one.assert_called_once()  # Called before error
        # Should not proceed to splitting or upserting
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.assert_not_called()
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_pdf_document_splitter_error(self, knowledge_base: KnowledgeBase, mock_pdf_reader, sample_pdf_data, sample_pdf_metadata):
        """Test error handling when semantic splitter fails."""
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.side_effect = Exception(
            "Splitter error")

        with pytest.raises(Exception, match="Splitter error"):
            await knowledge_base.add_pdf_document(pdf_data=sample_pdf_data, metadata=sample_pdf_metadata)

        # Parent doc inserted before splitting attempt
        knowledge_base.mongo.insert_one.assert_called_once()
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.assert_called_once()
        # Should fail before upserting
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_pdf_document_pinecone_chunk_error(self, knowledge_base: KnowledgeBase, mock_pdf_reader, mock_semantic_splitter_nodes, sample_pdf_data, sample_pdf_metadata):
        """Test error handling when Pinecone upsert fails during chunking."""
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.return_value = mock_semantic_splitter_nodes
        # Simulate error on the first Pinecone upsert call
        knowledge_base.pinecone.upsert_text.side_effect = Exception(
            "Pinecone chunk upsert error")

        # FIX: Test the actual behavior - it logs error and returns parent ID
        parent_doc_id = "pdf-chunk-error-id"
        result = await knowledge_base.add_pdf_document(
            pdf_data=sample_pdf_data,
            metadata=sample_pdf_metadata,
            document_id=parent_doc_id,
            chunk_batch_size=5
        )
        assert result == parent_doc_id  # Should still return the parent ID

        # Check logs (implicitly via stdout capture or explicitly mock print)
        # Check calls were made up to the point of error
        knowledge_base.mongo.insert_one.assert_called_once()  # Parent doc inserted
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.assert_called_once()
        # Called for the first batch before erroring
        knowledge_base.pinecone.upsert_text.assert_called_once()

    # --- Query Tests ---

    @pytest.mark.asyncio
    async def test_query_no_results(self, knowledge_base: KnowledgeBase):
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
        knowledge_base.mongo.find.assert_not_called()  # No IDs to fetch from Mongo

    @pytest.mark.asyncio
    async def test_query_with_filter_and_namespace(self, knowledge_base: KnowledgeBase):
        """Test querying with a filter and namespace."""
        knowledge_base.pinecone.query_text.return_value = [
        ]  # No results needed for this check
        test_filter = {"source": "website"}
        test_namespace = "my_namespace"

        await knowledge_base.query(
            query_text="test query",
            filter=test_filter,
            namespace=test_namespace,
            top_k=10
        )

        knowledge_base.pinecone.query_text.assert_called_once_with(
            query_text="test query",
            filter=test_filter,
            top_k=10,
            namespace=test_namespace,
            include_values=False,
            include_metadata=True
        )

    @pytest.mark.asyncio
    async def test_query_with_results_mixed(self, knowledge_base: KnowledgeBase, sample_pinecone_results, sample_mongo_docs_map):
        """Test querying with mixed results (docs and chunks) from Pinecone and MongoDB."""
        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results
        # Mock mongo.find to return docs based on IDs from Pinecone results
        pinecone_ids = [r['id'] for r in sample_pinecone_results]

        # Mock mongo.find to return only the docs whose IDs are in pinecone_ids (non-chunks)
        def mock_find(*args, **kwargs):
            query = args[1]
            ids_in_query = query.get("document_id", {}).get("$in", [])
            # Return only non-chunk docs found in the map
            return [sample_mongo_docs_map[id] for id in ids_in_query if id in sample_mongo_docs_map and not sample_mongo_docs_map[id].get("is_chunk")]
        knowledge_base.mongo.find.side_effect = mock_find

        # Mock mongo.find_one used to get parent metadata AND content for chunks
        def mock_find_one(*args, **kwargs):
            query = args[1]
            doc_id = query.get("document_id")
            # Return the parent doc 'pdf1' when asked
            # Will return pdf1 or None
            return sample_mongo_docs_map.get(doc_id)
        knowledge_base.mongo.find_one.side_effect = mock_find_one

        results = await knowledge_base.query(
            query_text="blockchain performance",
            top_k=4  # Requesting 4 results
        )

        assert len(results) == 4
        knowledge_base.pinecone.query_text.assert_called_once_with(
            query_text="blockchain performance", filter=None, top_k=4, namespace=None, include_values=False, include_metadata=True
        )
        # Check mongo.find was called with the IDs from Pinecone
        knowledge_base.mongo.find.assert_called_once_with(
            knowledge_base.collection,
            {"document_id": {"$in": pinecone_ids}}
        )
        # Check mongo.find_one was called to get parent metadata/content for chunks
        # FIX: Called twice per chunk (content + metadata) = 4 calls total
        assert knowledge_base.mongo.find_one.call_count == 4
        # FIX: Expect 4 calls
        knowledge_base.mongo.find_one.assert_has_calls([
            call(knowledge_base.collection, {"document_id": "pdf1"}),
            call(knowledge_base.collection, {"document_id": "pdf1"}),
            call(knowledge_base.collection, {"document_id": "pdf1"}),
            call(knowledge_base.collection, {"document_id": "pdf1"})
        ], any_order=True)  # Order depends on internal loop

        # Check structure of results
        # Result 1 (doc1)
        assert results[0]["document_id"] == "doc1"
        assert results[0]["score"] == 0.95
        assert results[0]["is_chunk"] is False
        assert results[0]["parent_document_id"] is None
        # Content from Mongo doc (via find)
        assert results[0]["content"] == sample_mongo_docs_map["doc1"]["content"]
        assert "metadata" in results[0]
        # Merged from Mongo (via find)
        assert results[0]["metadata"]["title"] == sample_mongo_docs_map["doc1"]["title"]
        assert results[0]["metadata"]["source"] == sample_mongo_docs_map["doc1"]["source"]
        # From Pinecone meta
        assert results[0]["metadata"]["is_chunk"] is False

        # Result 2 (pdf1_chunk_0)
        assert results[1]["document_id"] == "pdf1_chunk_0"
        assert results[1]["score"] == 0.90
        assert results[1]["is_chunk"] is True
        assert results[1]["parent_document_id"] == "pdf1"
        # Content for chunks: Fetched from parent Mongo doc (via find_one)
        assert results[1]["content"] == sample_mongo_docs_map["pdf1"]["content"]
        assert "metadata" in results[1]
        # Merged from parent Mongo (via find_one)
        assert results[1]["metadata"]["title"] == sample_mongo_docs_map["pdf1"]["title"]
        assert results[1]["metadata"]["source"] == sample_mongo_docs_map["pdf1"]["source"]
        assert results[1]["metadata"]["chunk_index"] == 0  # From Pinecone meta
        assert results[1]["metadata"]["is_chunk"] is True  # From Pinecone meta

        # Result 3 (doc2)
        assert results[2]["document_id"] == "doc2"
        assert results[2]["score"] == 0.85
        assert results[2]["is_chunk"] is False
        # Content from Mongo doc (via find)
        assert results[2]["content"] == sample_mongo_docs_map["doc2"]["content"]
        assert "metadata" in results[2]
        # Merged from Mongo (via find)
        assert results[2]["metadata"]["title"] == sample_mongo_docs_map["doc2"]["title"]
        # From Pinecone meta
        assert results[2]["metadata"]["is_chunk"] is False

        # Result 4 (pdf1_chunk_1)
        assert results[3]["document_id"] == "pdf1_chunk_1"
        assert results[3]["score"] == 0.80
        assert results[3]["is_chunk"] is True
        assert results[3]["parent_document_id"] == "pdf1"
        # Content from parent Mongo doc (via find_one)
        assert results[3]["content"] == sample_mongo_docs_map["pdf1"]["content"]
        assert "metadata" in results[3]
        # Merged from parent Mongo (via find_one)
        assert results[3]["metadata"]["title"] == sample_mongo_docs_map["pdf1"]["title"]
        assert results[3]["metadata"]["chunk_index"] == 1  # From Pinecone meta
        assert results[3]["metadata"]["is_chunk"] is True  # From Pinecone meta

    @pytest.mark.asyncio
    async def test_query_content_exclusion(self, knowledge_base: KnowledgeBase, sample_pinecone_results, sample_mongo_docs_map):
        """Test querying with content excluded from results."""
        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results
        # Simplified mock find/find_one for this test
        knowledge_base.mongo.find.return_value = [
            sample_mongo_docs_map["doc1"], sample_mongo_docs_map["doc2"]
        ]
        knowledge_base.mongo.find_one.side_effect = lambda *args, **kwargs: sample_mongo_docs_map.get(
            args[1].get("document_id"))  # Return parent if asked

        results = await knowledge_base.query(
            query_text="blockchain",
            include_content=False,  # Exclude content
            include_metadata=True
        )

        assert len(results) == 4
        assert "content" not in results[0]
        assert "content" not in results[1]  # Check chunk result too
        assert "metadata" in results[0]
        assert "metadata" in results[1]

    @pytest.mark.asyncio
    async def test_query_metadata_exclusion(self, knowledge_base: KnowledgeBase, sample_pinecone_results, sample_mongo_docs_map):
        """Test querying with metadata excluded from results."""
        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results
        knowledge_base.mongo.find.return_value = [
            sample_mongo_docs_map["doc1"], sample_mongo_docs_map["doc2"]
        ]
        knowledge_base.mongo.find_one.side_effect = lambda *args, **kwargs: sample_mongo_docs_map.get(
            args[1].get("document_id"))

        results = await knowledge_base.query(
            query_text="blockchain",
            include_content=True,
            include_metadata=False  # Exclude metadata
        )

        assert len(results) == 4
        assert "content" in results[0]
        assert "content" in results[1]
        assert "metadata" not in results[0]
        assert "metadata" not in results[1]

    @pytest.mark.asyncio
    async def test_query_with_reranking_enabled(self, knowledge_base: KnowledgeBase, sample_pinecone_results, sample_mongo_docs_map):
        """Test querying uses rerank_top_k when reranking is enabled in constructor."""
        # Enable reranking on the fixture instance
        knowledge_base.rerank_results = True
        knowledge_base.rerank_top_k = 2  # Set specific rerank_top_k

        # Pinecone returns results
        # Assume pinecone returns only rerank_top_k
        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results[:2]
        knowledge_base.mongo.find.return_value = [
            # Only doc1 needed based on reduced pinecone results
            sample_mongo_docs_map["doc1"]
        ]
        knowledge_base.mongo.find_one.side_effect = lambda *args, **kwargs: sample_mongo_docs_map.get(
            args[1].get("document_id"))  # Return pdf1 if asked

        # User requests 10
        await knowledge_base.query(query_text="blockchain", top_k=10)

        # Verify effective_top_k used rerank_top_k (2) for the Pinecone query
        knowledge_base.pinecone.query_text.assert_called_once_with(
            query_text="blockchain",
            filter=None,
            top_k=2,  # Should use rerank_top_k from constructor override
            namespace=None,
            include_values=False,
            include_metadata=True
        )

    @pytest.mark.asyncio
    async def test_query_pinecone_error(self, knowledge_base: KnowledgeBase):
        """Test error handling when Pinecone query fails."""
        knowledge_base.pinecone.query_text.side_effect = Exception(
            "Pinecone query error")

        # The function should catch the error and return empty list
        results = await knowledge_base.query(query_text="test query")
        assert results == []
        knowledge_base.mongo.find.assert_not_called()

    @pytest.mark.asyncio
    # FIX: Add sample_mongo_docs_map fixture
    async def test_query_mongo_find_error(self, knowledge_base: KnowledgeBase, sample_pinecone_results, sample_mongo_docs_map):
        """Test error handling when MongoDB find fails."""
        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results
        knowledge_base.mongo.find.side_effect = Exception("Mongo find error")
        # Mock find_one to still work (or fail separately)
        # FIX: Use the fixture value in the lambda
        knowledge_base.mongo.find_one.side_effect = lambda *args, **kwargs: sample_mongo_docs_map.get(
            args[1].get("document_id"))

        # The query should handle the error and return partial results
        results = await knowledge_base.query(query_text="test query")

        assert len(results) == len(sample_pinecone_results)
        # Check that results contain only Pinecone data + parent data from find_one if applicable
        for i, res in enumerate(results):
            assert res["document_id"] == sample_pinecone_results[i]["id"]
            assert res["score"] == sample_pinecone_results[i]["score"]
            assert res["is_chunk"] == sample_pinecone_results[i]["metadata"]["is_chunk"]
            if res["is_chunk"]:
                assert res["parent_document_id"] == sample_pinecone_results[i]["metadata"]["parent_document_id"]
                # Content/Metadata from parent (find_one) should still be attempted
                assert "content" in res  # Content fetch from parent attempted
                assert "metadata" in res  # Metadata fetch from parent attempted
                # Check parent metadata was fetched (if find_one worked)
                assert res["metadata"].get(
                    "title") == sample_mongo_docs_map["pdf1"]["title"]
            else:
                assert res["parent_document_id"] is None
                # Content/Metadata from mongo.find failed
                assert "content" not in res or res["content"] == ""
                # FIX: Metadata should exist (from Pinecone) but lack Mongo-specific fields
                assert "metadata" in res
                assert res["metadata"]  # Should not be empty
                # Check that a field only present in Mongo doc is missing
                assert "title" not in res["metadata"]
                # Check that fields from Pinecone meta are present
                assert res["metadata"]["is_chunk"] is False
                assert res["metadata"]["source"] == sample_pinecone_results[i]["metadata"]["source"]

        knowledge_base.mongo.find.assert_called_once()
        # find_one should still be called for chunks
        assert knowledge_base.mongo.find_one.call_count > 0

    @pytest.mark.asyncio
    async def test_query_mongo_find_one_error(self, knowledge_base: KnowledgeBase, sample_pinecone_results, sample_mongo_docs_map):
        """Test error handling when MongoDB find_one fails for parent metadata/content."""
        knowledge_base.pinecone.query_text.return_value = sample_pinecone_results
        # Mock find to return relevant docs (non-chunks)
        knowledge_base.mongo.find.return_value = [
            sample_mongo_docs_map["doc1"], sample_mongo_docs_map["doc2"]
        ]
        # Mock find_one to fail
        knowledge_base.mongo.find_one.side_effect = Exception(
            "Mongo find_one error")

        # FIX: The query should handle the error and return partial results
        results = await knowledge_base.query(query_text="test query")

        assert len(results) == len(sample_pinecone_results)
        # Check results: non-chunks should be fine, chunks should lack parent data
        # Non-chunk ok (content/meta from mongo.find)
        assert results[0]["content"] == sample_mongo_docs_map["doc1"]["content"]
        assert results[0]["metadata"]["title"] == sample_mongo_docs_map["doc1"]["title"]
        # Non-chunk ok (content/meta from mongo.find)
        assert results[2]["content"] == sample_mongo_docs_map["doc2"]["content"]
        assert results[2]["metadata"]["title"] == sample_mongo_docs_map["doc2"]["title"]

        # Chunk results should have Pinecone metadata but lack merged parent data/content
        assert results[1]["document_id"] == "pdf1_chunk_0"
        assert results[1]["is_chunk"] is True
        assert results[1]["parent_document_id"] == "pdf1"
        # Content from parent failed
        assert "content" not in results[1] or results[1]["content"] == ""
        assert "metadata" in results[1]  # Metadata dict exists
        # Original chunk meta ok
        assert results[1]["metadata"]["chunk_index"] == 0
        # Parent title merge failed
        assert "title" not in results[1]["metadata"]

        assert results[3]["document_id"] == "pdf1_chunk_1"
        assert "content" not in results[3] or results[3]["content"] == ""
        assert "metadata" in results[3]
        assert "title" not in results[3]["metadata"]

        knowledge_base.mongo.find.assert_called_once()
        # find_one was called for chunks before erroring
        assert knowledge_base.mongo.find_one.call_count > 0

    # --- Delete Document Tests ---
    @pytest.mark.asyncio
    async def test_delete_document_plain_text_success(self, knowledge_base: KnowledgeBase):
        """Test successful deletion of a plain text document."""
        doc_id = "doc-to-delete"
        # Mock mongo.find to return only the single document
        knowledge_base.mongo.find.return_value = [
            {"document_id": doc_id, "_id": "mongo_id"}]
        # Mock mongo.delete_many result
        delete_result_mock = MagicMock()
        delete_result_mock.deleted_count = 1
        knowledge_base.mongo.delete_many.return_value = delete_result_mock

        result = await knowledge_base.delete_document(doc_id, namespace="ns1")

        assert result is True
        # Check mongo.find was called correctly
        knowledge_base.mongo.find.assert_called_once_with(
            knowledge_base.collection,
            {"$or": [{"document_id": doc_id}, {"parent_document_id": doc_id}]}
        )
        # Check Pinecone delete was called with the ID found by mongo.find
        knowledge_base.pinecone.delete.assert_called_once_with(
            ids=[doc_id], namespace="ns1")
        # Check mongo.delete_many was called with the ID found by mongo.find
        knowledge_base.mongo.delete_many.assert_called_once_with(
            knowledge_base.collection,
            {"document_id": {"$in": [doc_id]}}
        )

    @pytest.mark.asyncio
    async def test_delete_document_pdf_with_chunks_success(self, knowledge_base: KnowledgeBase):
        """Test successful deletion of a PDF document and its associated vectors."""
        parent_id = "pdf-to-delete"
        # Assume chunks aren't stored in Mongo, only parent
        # Mock mongo.find to simulate finding the parent doc in Mongo.
        knowledge_base.mongo.find.return_value = [
            {"document_id": parent_id, "_id": "mongo_parent"},
        ]
        mongo_ids_found = [parent_id]  # Only parent found in Mongo

        # IMPORTANT: The code currently derives Pinecone IDs only from Mongo results.
        # A better implementation might query Pinecone for chunks based on parent_id.
        # This test reflects the *current* implementation.
        pinecone_ids_to_delete = mongo_ids_found  # Currently derived from Mongo find

        # Mock mongo.delete_many result based on what was found in Mongo
        delete_result_mock = MagicMock()
        delete_result_mock.deleted_count = len(mongo_ids_found)
        knowledge_base.mongo.delete_many.return_value = delete_result_mock

        result = await knowledge_base.delete_document(parent_id, namespace="ns_pdf")

        assert result is True
        # Check mongo.find was called correctly
        knowledge_base.mongo.find.assert_called_once_with(
            knowledge_base.collection,
            {"$or": [{"document_id": parent_id}, {
                "parent_document_id": parent_id}]}
        )
        # Check Pinecone delete was called with the derived Pinecone IDs (currently just parent)
        knowledge_base.pinecone.delete.assert_called_once_with(
            ids=pinecone_ids_to_delete, namespace="ns_pdf")
        # Check mongo.delete_many was called with only the IDs found in Mongo
        knowledge_base.mongo.delete_many.assert_called_once_with(
            knowledge_base.collection,
            {"document_id": {"$in": mongo_ids_found}}
        )

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, knowledge_base: KnowledgeBase):
        """Test deleting a document that doesn't exist."""
        doc_id = "doc-not-found"
        knowledge_base.mongo.find.return_value = []  # Simulate not found in Mongo
        # Mock mongo.delete_many result (0 deleted)
        delete_result_mock = MagicMock()
        delete_result_mock.deleted_count = 0
        knowledge_base.mongo.delete_many.return_value = delete_result_mock

        result = await knowledge_base.delete_document(doc_id)

        # Should return False as nothing was deleted
        assert result is False
        knowledge_base.mongo.find.assert_called_once_with(
            knowledge_base.collection,
            {"$or": [{"document_id": doc_id}, {"parent_document_id": doc_id}]}
        )
        # Pinecone delete shouldn't be called if no IDs found
        knowledge_base.pinecone.delete.assert_not_called()
        # Mongo delete shouldn't be called if no IDs found
        knowledge_base.mongo.delete_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_document_pinecone_error(self, knowledge_base: KnowledgeBase):
        """Test document deletion continues with Mongo if Pinecone fails."""
        doc_id = "doc-delete-pinecone-fail"
        knowledge_base.mongo.find.return_value = [
            {"document_id": doc_id, "_id": "mongo_id"}]
        knowledge_base.pinecone.delete.side_effect = Exception(
            "Pinecone delete error")
        # Mock mongo.delete_many result
        delete_result_mock = MagicMock()
        delete_result_mock.deleted_count = 1
        knowledge_base.mongo.delete_many.return_value = delete_result_mock

        result = await knowledge_base.delete_document(doc_id)

        # Should return True because Mongo deletion succeeded (logs Pinecone error)
        assert result is True
        knowledge_base.pinecone.delete.assert_called_once_with(
            ids=[doc_id], namespace=None)
        knowledge_base.mongo.delete_many.assert_called_once_with(
            knowledge_base.collection, {"document_id": {"$in": [doc_id]}}
        )
        # FIX: Removed incorrect 'assert result is False'

    @pytest.mark.asyncio
    async def test_delete_document_mongo_error(self, knowledge_base: KnowledgeBase):
        """Test document deletion handles Mongo delete error."""
        doc_id = "doc-delete-mongo-fail"
        knowledge_base.mongo.find.return_value = [
            {"document_id": doc_id, "_id": "mongo_id"}]
        knowledge_base.mongo.delete_many.side_effect = Exception(
            "Mongo delete error")
        # Assume Pinecone delete succeeds or happens first
        knowledge_base.pinecone.delete = AsyncMock()  # Mock successful delete

        result = await knowledge_base.delete_document(doc_id)

        # Should return True because Pinecone deletion succeeded (logs Mongo error)
        # The function returns pinecone_deleted or mongo_deleted_count > 0
        assert result is True  # Pinecone succeeded
        knowledge_base.pinecone.delete.assert_called_once_with(
            ids=[doc_id], namespace=None)
        knowledge_base.mongo.delete_many.assert_called_once_with(
            knowledge_base.collection, {"document_id": {"$in": [doc_id]}}
        )

    # --- Update Document Tests ---

    @pytest.mark.asyncio
    async def test_update_document_metadata_only(self, knowledge_base: KnowledgeBase):
        """Test updating only document metadata for a plain text document."""
        doc_id = "doc-update-meta"
        original_doc = {
            "_id": "mongo_id", "document_id": doc_id, "content": "Original content",
            "title": "Original title", "tags": ["original"], "is_chunk": False, "pdf_data": None,
            "source": "web", "created_at": dt.now(timezone.utc)
        }
        knowledge_base.mongo.find_one.return_value = original_doc
        # Mock update result
        update_result_mock = MagicMock()
        update_result_mock.modified_count = 1
        knowledge_base.mongo.update_one.return_value = update_result_mock

        update_meta = {"title": "Updated title", "tags": ["updated"]}
        result = await knowledge_base.update_document(
            document_id=doc_id,
            metadata=update_meta,
            namespace="ns_update"
        )

        assert result is True
        knowledge_base.mongo.find_one.assert_called_once_with(
            knowledge_base.collection, {"document_id": doc_id})
        knowledge_base.mongo.update_one.assert_called_once()
        args, kwargs = knowledge_base.mongo.update_one.call_args
        assert args[0] == knowledge_base.collection
        assert args[1] == {"document_id": doc_id}
        update_set = args[2]["$set"]
        assert update_set["title"] == "Updated title"
        assert update_set["tags"] == ["updated"]
        assert "content" not in update_set  # Content not updated
        assert "updated_at" in update_set
        # Pinecone should NOT be called as text didn't change
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_content_and_metadata(self, knowledge_base: KnowledgeBase):
        """Test updating both content and metadata for a plain text document."""
        doc_id = "doc-update-all"
        original_doc = {
            "_id": "mongo_id", "document_id": doc_id, "content": "Original content",
            "title": "Original title", "source": "web", "tags": ["original"], "is_chunk": False, "pdf_data": None,
            "created_at": dt.now(timezone.utc)
        }
        knowledge_base.mongo.find_one.return_value = original_doc
        update_result_mock = MagicMock()
        update_result_mock.modified_count = 1
        knowledge_base.mongo.update_one.return_value = update_result_mock

        new_text = "Updated content"
        update_meta = {"title": "Updated title"}  # Only update title
        result = await knowledge_base.update_document(
            document_id=doc_id,
            text=new_text,
            metadata=update_meta,
            namespace="ns_update"
        )

        assert result is True
        knowledge_base.mongo.find_one.assert_called_once_with(
            knowledge_base.collection, {"document_id": doc_id})
        # Check Mongo update
        knowledge_base.mongo.update_one.assert_called_once()
        update_set = knowledge_base.mongo.update_one.call_args[0][2]["$set"]
        assert update_set["content"] == new_text
        # Check updated meta field
        assert update_set["title"] == "Updated title"
        assert "updated_at" in update_set
        # Check Pinecone update (since text changed)
        knowledge_base.pinecone.upsert_text.assert_called_once()
        args, kwargs = knowledge_base.pinecone.upsert_text.call_args
        assert kwargs["texts"] == [new_text]
        assert kwargs["ids"] == [doc_id]
        assert kwargs["namespace"] == "ns_update"
        pinecone_meta = kwargs["metadatas"][0]
        assert pinecone_meta["document_id"] == doc_id
        assert pinecone_meta["is_chunk"] is False
        # Preserved from original doc via final_metadata merge
        assert pinecone_meta["source"] == "web"
        # Updated via final_metadata merge (FIX Check Key)
        assert pinecone_meta["title"] == "Updated title"
        # Preserved from original doc via final_metadata merge
        assert pinecone_meta["tags"] == ["original"]

    @pytest.mark.asyncio
    async def test_update_document_not_found(self, knowledge_base: KnowledgeBase):
        """Test updating a document that doesn't exist in MongoDB."""
        knowledge_base.mongo.find_one.return_value = None  # Simulate not found

        result = await knowledge_base.update_document(document_id="not-found", text="new text")

        assert result is False
        knowledge_base.mongo.find_one.assert_called_once_with(
            knowledge_base.collection, {"document_id": "not-found"})
        knowledge_base.mongo.update_one.assert_not_called()
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_cannot_update_chunk(self, knowledge_base: KnowledgeBase):
        """Test that updating a chunk directly is disallowed."""
        chunk_id = "pdf1_chunk_0"
        original_chunk_doc = {  # Simulate finding a chunk doc (though not stored)
            "_id": "mongo_chunk_id", "document_id": chunk_id, "is_chunk": True,
            "parent_document_id": "pdf1", "content": None, "pdf_data": None
        }
        knowledge_base.mongo.find_one.return_value = original_chunk_doc

        result = await knowledge_base.update_document(document_id=chunk_id, text="new chunk text")

        assert result is False
        knowledge_base.mongo.find_one.assert_called_once_with(
            knowledge_base.collection, {"document_id": chunk_id})
        # Ensure no update attempts were made
        knowledge_base.mongo.update_one.assert_not_called()
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_cannot_update_pdf_content(self, knowledge_base: KnowledgeBase):
        """Test that updating PDF content via update_document is disallowed."""
        pdf_doc_id = "pdf-to-update"
        original_pdf_doc = {
            "_id": "mongo_pdf_id", "document_id": pdf_doc_id, "content": "Original PDF text",
            "pdf_data": b"original pdf bytes", "is_chunk": False, "title": "Original PDF"
        }
        knowledge_base.mongo.find_one.return_value = original_pdf_doc

        result = await knowledge_base.update_document(document_id=pdf_doc_id, text="new pdf text")

        assert result is False
        knowledge_base.mongo.find_one.assert_called_once_with(
            knowledge_base.collection, {"document_id": pdf_doc_id})
        # Ensure no update attempts were made
        knowledge_base.mongo.update_one.assert_not_called()
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_mongo_update_error(self, knowledge_base: KnowledgeBase):
        """Test error handling when MongoDB update fails."""
        doc_id = "doc-update-mongo-fail"
        original_doc = {
            "_id": "mongo_id", "document_id": doc_id, "content": "Original content",
            "is_chunk": False, "pdf_data": None
        }
        knowledge_base.mongo.find_one.return_value = original_doc
        knowledge_base.mongo.update_one.side_effect = Exception(
            "Mongo update error")

        # Update only metadata (no Pinecone call expected)
        with pytest.raises(Exception, match="Mongo update error"):
            await knowledge_base.update_document(document_id=doc_id, metadata={"new_key": "new_value"})

        knowledge_base.mongo.find_one.assert_called_once_with(
            knowledge_base.collection, {"document_id": doc_id})
        knowledge_base.mongo.update_one.assert_called_once()  # Called before error
        knowledge_base.pinecone.upsert_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_pinecone_update_error(self, knowledge_base: KnowledgeBase):
        """Test handling when Pinecone update fails after successful Mongo update."""
        doc_id = "doc-update-pinecone-fail"
        original_doc = {
            "_id": "mongo_id", "document_id": doc_id, "content": "Original content",
            # Add source for pinecone meta check
            "is_chunk": False, "pdf_data": None, "source": "web"
        }
        knowledge_base.mongo.find_one.return_value = original_doc
        # Mock successful Mongo update
        update_result_mock = MagicMock()
        update_result_mock.modified_count = 1
        knowledge_base.mongo.update_one.return_value = update_result_mock
        # Mock Pinecone failure
        knowledge_base.pinecone.upsert_text.side_effect = Exception(
            "Pinecone update error")

        new_text = "Updated content for pinecone fail"
        # The function should catch the Pinecone error, log it, but return True because Mongo succeeded
        result = await knowledge_base.update_document(document_id=doc_id, text=new_text)

        assert result is True  # Mongo update succeeded
        knowledge_base.mongo.find_one.assert_called_once_with(
            knowledge_base.collection, {"document_id": doc_id})
        knowledge_base.mongo.update_one.assert_called_once()
        knowledge_base.pinecone.upsert_text.assert_called_once()  # Called before error

    # --- Batch Add Document Tests ---
    @pytest.mark.asyncio
    async def test_add_documents_batch_success(self, knowledge_base: KnowledgeBase):
        """Test adding multiple documents in a batch."""
        docs_to_add = [
            {"text": "Batch doc 1", "metadata": {
                "source": "batch", "doc_num": 1, "document_id": "batch-1"}},
            {"text": "Batch doc 2", "metadata": {
                "source": "batch", "doc_num": 2, "document_id": "batch-2"}},
            {"text": "Batch doc 3", "metadata": {
                "source": "batch", "doc_num": 3, "document_id": "batch-3"}},
        ]
        expected_ids = ["batch-1", "batch-2", "batch-3"]

        # Mock insert_many to return mock IDs
        knowledge_base.mongo.insert_many.return_value = MagicMock(
            inserted_ids=["mongo_batch_1", "mongo_batch_2", "mongo_batch_3"])

        result_ids = await knowledge_base.add_documents_batch(docs_to_add, namespace="ns_batch", batch_size=2)

        assert result_ids == expected_ids

        # Check Mongo insert_many calls (batched)
        assert knowledge_base.mongo.insert_many.call_count == 2
        # Call 1
        args1, _ = knowledge_base.mongo.insert_many.call_args_list[0]
        assert args1[0] == knowledge_base.collection
        assert len(args1[1]) == 2  # Batch size 2
        assert args1[1][0]["document_id"] == "batch-1"
        assert args1[1][1]["document_id"] == "batch-2"
        # Call 2
        args2, _ = knowledge_base.mongo.insert_many.call_args_list[1]
        assert len(args2[1]) == 1  # Remaining doc
        assert args2[1][0]["document_id"] == "batch-3"

        # Check Pinecone upsert_text calls (batched)
        assert knowledge_base.pinecone.upsert_text.call_count == 2
        # Call 1
        _, kwargs1 = knowledge_base.pinecone.upsert_text.call_args_list[0]
        assert kwargs1["ids"] == ["batch-1", "batch-2"]
        assert kwargs1["texts"] == ["Batch doc 1", "Batch doc 2"]
        assert kwargs1["namespace"] == "ns_batch"
        assert len(kwargs1["metadatas"]) == 2
        assert kwargs1["metadatas"][0]["source"] == "batch"
        # Call 2
        _, kwargs2 = knowledge_base.pinecone.upsert_text.call_args_list[1]
        assert kwargs2["ids"] == ["batch-3"]
        assert kwargs2["texts"] == ["Batch doc 3"]
        assert len(kwargs2["metadatas"]) == 1

    @pytest.mark.asyncio
    async def test_add_documents_batch_generate_ids(self, knowledge_base: KnowledgeBase):
        """Test batch adding generates IDs if not provided."""
        docs_to_add = [
            {"text": "Batch gen id 1", "metadata": {"source": "gen_batch"}},
            {"text": "Batch gen id 2", "metadata": {"source": "gen_batch"}},
        ]
        # Mock uuid
        uuid1 = uuid.UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
        uuid2 = uuid.UUID('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb')
        with patch('uuid.uuid4', side_effect=[uuid1, uuid2]):
            result_ids = await knowledge_base.add_documents_batch(docs_to_add, batch_size=3)

        assert result_ids == [str(uuid1), str(uuid2)]
        # Check Mongo used generated IDs
        knowledge_base.mongo.insert_many.assert_called_once()
        mongo_docs = knowledge_base.mongo.insert_many.call_args[0][1]
        assert mongo_docs[0]["document_id"] == str(uuid1)
        assert mongo_docs[1]["document_id"] == str(uuid2)
        # Check Pinecone used generated IDs
        knowledge_base.pinecone.upsert_text.assert_called_once()
        assert knowledge_base.pinecone.upsert_text.call_args[1]["ids"] == [
            str(uuid1), str(uuid2)]

    @pytest.mark.asyncio
    async def test_add_documents_batch_pinecone_error(self, knowledge_base: KnowledgeBase):
        """Test batch add continues if one Pinecone batch fails."""
        docs_to_add = [
            {"text": "Batch ok 1", "metadata": {"document_id": "pok1"}},
            {"text": "Batch fail 1", "metadata": {
                "document_id": "pfail1"}},  # Batch 1 (size 2)
            {"text": "Batch fail 2", "metadata": {"document_id": "pfail2"}},
            {"text": "Batch ok 2", "metadata": {
                "document_id": "pok2"}},   # Batch 2 (size 2)
        ]
        expected_ids = ["pok1", "pfail1", "pfail2", "pok2"]

        # Make second Pinecone call fail
        knowledge_base.pinecone.upsert_text.side_effect = [
            AsyncMock(),  # First call succeeds
            Exception("Pinecone batch error")  # Second call fails
        ]

        result_ids = await knowledge_base.add_documents_batch(docs_to_add, batch_size=2)

        assert result_ids == expected_ids
        # Check Mongo insert_many was called for both batches
        assert knowledge_base.mongo.insert_many.call_count == 2
        # Check Pinecone upsert_text was called twice (attempted both batches)
        assert knowledge_base.pinecone.upsert_text.call_count == 2

    # --- Get Full Document Test ---
    @pytest.mark.asyncio
    async def test_get_full_document_success(self, knowledge_base: KnowledgeBase):
        """Test retrieving a full document from MongoDB."""
        doc_id = "doc-to-get"
        expected_doc = {
            "_id": "mongo_get_id", "document_id": doc_id, "content": "Full content",
            "title": "Gettable Doc", "pdf_data": b"pdf bytes if applicable"
        }
        knowledge_base.mongo.find_one.return_value = expected_doc

        result_doc = await knowledge_base.get_full_document(doc_id)

        assert result_doc == expected_doc
        knowledge_base.mongo.find_one.assert_called_once_with(
            knowledge_base.collection, {"document_id": doc_id}
        )

    @pytest.mark.asyncio
    async def test_get_full_document_not_found(self, knowledge_base: KnowledgeBase):
        """Test retrieving a document that doesn't exist."""
        doc_id = "doc-get-not-found"
        knowledge_base.mongo.find_one.return_value = None  # Simulate not found

        result_doc = await knowledge_base.get_full_document(doc_id)

        assert result_doc is None
        knowledge_base.mongo.find_one.assert_called_once_with(
            knowledge_base.collection, {"document_id": doc_id}
        )

    @pytest.mark.asyncio
    async def test_update_document_content_with_rerank(self, knowledge_base: KnowledgeBase):
        """Test updating document content adds rerank field to Pinecone meta when enabled."""
        doc_id = "doc-update-rerank"
        original_doc = {
            "_id": "mongo_id_rerank", "document_id": doc_id, "content": "Original content for rerank test",
            "title": "Rerank Test Doc", "source": "rerank", "is_chunk": False, "pdf_data": None,
            "created_at": dt.now(timezone.utc)
        }
        knowledge_base.mongo.find_one.return_value = original_doc
        update_result_mock = MagicMock()
        update_result_mock.modified_count = 1
        knowledge_base.mongo.update_one.return_value = update_result_mock

        # Enable reranking for this test
        knowledge_base.pinecone.use_reranking = True
        rerank_field = knowledge_base.pinecone.rerank_text_field  # Get the field name

        new_text = "Updated content specifically for rerank field check."
        result = await knowledge_base.update_document(
            document_id=doc_id,
            text=new_text,
            namespace="ns_rerank_update"
        )

        assert result is True
        # Check Mongo update happened
        knowledge_base.mongo.update_one.assert_called_once()
        # Check Pinecone update happened because text changed
        knowledge_base.pinecone.upsert_text.assert_called_once()

        # Verify the rerank field was added to Pinecone metadata
        args, kwargs = knowledge_base.pinecone.upsert_text.call_args
        assert kwargs["texts"] == [new_text]
        assert kwargs["ids"] == [doc_id]
        assert kwargs["namespace"] == "ns_rerank_update"
        pinecone_meta = kwargs["metadatas"][0]

        assert rerank_field in pinecone_meta
        assert pinecone_meta[rerank_field] == new_text
        # Check other standard fields are still there
        assert pinecone_meta["document_id"] == doc_id
        assert pinecone_meta["is_chunk"] is False
        # From original doc merged meta
        assert pinecone_meta["title"] == "Rerank Test Doc"
        assert pinecone_meta["source"] == "rerank"

    @pytest.mark.asyncio
    async def test_query_uses_rerank_field_for_content(self, knowledge_base: KnowledgeBase, sample_mongo_docs_map):
        """Test query uses content from Pinecone rerank field when enabled."""
        # Enable reranking on the KB's pinecone adapter for this test
        knowledge_base.pinecone.use_reranking = True
        rerank_field = knowledge_base.pinecone.rerank_text_field
        doc_id_rerank = "doc_with_rerank_field"
        rerank_content = "This content comes from the Pinecone rerank field."

        # Simulate Pinecone result with the rerank field in metadata
        pinecone_results_rerank = [
            {"id": doc_id_rerank, "score": 0.98, "metadata": {
                "document_id": doc_id_rerank,
                "is_chunk": False,
                rerank_field: rerank_content,  # <<< Key part: rerank field present
                "source": "rerank_source"
            }},
            # Add another result without the rerank field for comparison
            {"id": "doc1", "score": 0.95, "metadata": {
                "document_id": "doc1",
                "is_chunk": False,
                "source": "website"
                # No rerank field here
            }}
        ]
        knowledge_base.pinecone.query_text.return_value = pinecone_results_rerank

        # Mock Mongo find to return docs, including one with DIFFERENT content
        # for the doc_id_rerank to prove the override works.
        mongo_doc_rerank = {
            "_id": "mongo_rerank", "document_id": doc_id_rerank,
            "content": "This is the ORIGINAL content from MongoDB.",  # <<< Different content
            "source": "rerank_source", "is_chunk": False
        }
        # Use sample_mongo_docs_map for the other doc ('doc1')

        def mock_find(*args, **kwargs):
            ids_in_query = args[1].get("document_id", {}).get("$in", [])
            found_docs = []
            if doc_id_rerank in ids_in_query:
                found_docs.append(mongo_doc_rerank)
            if "doc1" in ids_in_query and "doc1" in sample_mongo_docs_map:
                found_docs.append(sample_mongo_docs_map["doc1"])
            return found_docs
        knowledge_base.mongo.find.side_effect = mock_find
        # No chunks involved, find_one shouldn't be needed unless logic changes
        knowledge_base.mongo.find_one.return_value = None

        # --- Act ---
        results = await knowledge_base.query(
            query_text="query text",
            include_content=True,
            include_metadata=True  # Include metadata to check source etc.
        )

        # --- Assert ---
        assert len(results) == 2

        # Check the first result (with rerank field)
        result_rerank = results[0]
        assert result_rerank["document_id"] == doc_id_rerank
        # Verify content comes from the Pinecone rerank field
        assert result_rerank["content"] == rerank_content
        # Verify metadata is still populated correctly
        assert "metadata" in result_rerank
        assert result_rerank["metadata"]["source"] == "rerank_source"
        # Ensure the rerank field itself isn't duplicated in the final metadata output
        assert rerank_field not in result_rerank["metadata"]

        # Check the second result (without rerank field)
        result_no_rerank = results[1]
        assert result_no_rerank["document_id"] == "doc1"
        # Verify content comes from MongoDB (via the mock find)
        assert result_no_rerank["content"] == sample_mongo_docs_map["doc1"]["content"]
        assert "metadata" in result_no_rerank
        assert result_no_rerank["metadata"]["source"] == "website"

    @pytest.mark.asyncio
    async def test_add_documents_batch_with_rerank(self, knowledge_base: KnowledgeBase):
        """Test batch adding includes rerank field in Pinecone meta when enabled."""
        docs_to_add = [
            {"text": "Batch rerank doc 1", "metadata": {
                "source": "batch_rerank", "doc_num": 1, "document_id": "br1"}},
            {"text": "Batch rerank doc 2", "metadata": {
                "source": "batch_rerank", "doc_num": 2, "document_id": "br2"}},
        ]
        expected_ids = ["br1", "br2"]

        # Enable reranking for this test
        knowledge_base.pinecone.use_reranking = True
        rerank_field = knowledge_base.pinecone.rerank_text_field

        # Mock insert_many
        knowledge_base.mongo.insert_many.return_value = MagicMock(
            inserted_ids=["mongo_br1", "mongo_br2"])

        result_ids = await knowledge_base.add_documents_batch(docs_to_add, namespace="ns_batch_rerank", batch_size=3)

        assert result_ids == expected_ids

        # Check Mongo insert_many call
        knowledge_base.mongo.insert_many.assert_called_once()
        mongo_docs = knowledge_base.mongo.insert_many.call_args[0][1]
        assert len(mongo_docs) == 2

        # Check Pinecone upsert_text call
        knowledge_base.pinecone.upsert_text.assert_called_once()
        args, kwargs = knowledge_base.pinecone.upsert_text.call_args
        assert kwargs["ids"] == expected_ids
        assert kwargs["texts"] == [
            docs_to_add[0]["text"], docs_to_add[1]["text"]]
        assert kwargs["namespace"] == "ns_batch_rerank"

        # Verify the rerank field in Pinecone metadata for each doc
        pinecone_metadatas = kwargs["metadatas"]
        assert len(pinecone_metadatas) == 2

        # Doc 1
        assert rerank_field in pinecone_metadatas[0]
        assert pinecone_metadatas[0][rerank_field] == docs_to_add[0]["text"]
        assert pinecone_metadatas[0]["document_id"] == expected_ids[0]
        assert pinecone_metadatas[0]["source"] == "batch_rerank"

        # Doc 2
        assert rerank_field in pinecone_metadatas[1]
        assert pinecone_metadatas[1][rerank_field] == docs_to_add[1]["text"]
        assert pinecone_metadatas[1]["document_id"] == expected_ids[1]
        assert pinecone_metadatas[1]["source"] == "batch_rerank"

    @pytest.mark.asyncio
    async def test_add_pdf_document_with_rerank(
        self, knowledge_base: KnowledgeBase, mock_pdf_reader, mock_semantic_splitter_nodes,
        sample_pdf_data, sample_pdf_metadata
    ):
        """Test adding PDF includes rerank field in chunk metadata when enabled."""
        parent_doc_id = "pdf-rerank-test"
        # Enable reranking for this test
        knowledge_base.pinecone.use_reranking = True
        rerank_field = knowledge_base.pinecone.rerank_text_field

        # Configure mocks
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.return_value = mock_semantic_splitter_nodes

        await knowledge_base.add_pdf_document(
            pdf_data=sample_pdf_data,
            metadata=sample_pdf_metadata,
            document_id=parent_doc_id,
            namespace="ns_pdf_rerank",
            chunk_batch_size=2  # Use batching to test across calls
        )

        # Check Pinecone upsert calls
        assert knowledge_base.pinecone.upsert_text.call_count == 2  # 3 nodes, batch size 2
        calls = knowledge_base.pinecone.upsert_text.call_args_list

        # Verify rerank field in metadata for each chunk across batches
        all_upserted_metadatas = []
        for call_args in calls:
            all_upserted_metadatas.extend(call_args.kwargs["metadatas"])

        assert len(all_upserted_metadatas) == len(mock_semantic_splitter_nodes)

        for i, meta in enumerate(all_upserted_metadatas):
            expected_chunk_text = mock_semantic_splitter_nodes[i].get_content()
            assert rerank_field in meta
            assert meta[rerank_field] == expected_chunk_text
            # Also check other standard fields
            assert meta["document_id"] == f"{parent_doc_id}_chunk_{i}"
            assert meta["parent_document_id"] == parent_doc_id
            assert meta["is_chunk"] is True

    @pytest.mark.asyncio
    async def test_add_pdf_document_no_chunks_generated(
        self, knowledge_base: KnowledgeBase, mock_pdf_reader,
        sample_pdf_data, sample_pdf_metadata
    ):
        """Test adding PDF returns early if semantic splitter generates no nodes."""
        parent_doc_id = "pdf-no-chunks-test"

        # Configure splitter mock to return empty list
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.return_value = []

        result_id = await knowledge_base.add_pdf_document(
            pdf_data=sample_pdf_data,
            metadata=sample_pdf_metadata,
            document_id=parent_doc_id,
            namespace="ns_pdf_no_chunks"
        )

        # Assert the parent ID is returned
        assert result_id == parent_doc_id

        # Check PDF reading and Mongo insert happened
        mock_pdf_reader.assert_called_once()
        knowledge_base.mongo.insert_one.assert_called_once()
        mongo_doc = knowledge_base.mongo.insert_one.call_args[0][1]
        assert mongo_doc["document_id"] == parent_doc_id

        # Check splitter was called
        knowledge_base.mock_splitter_instance.get_nodes_from_documents.assert_called_once()

        # Check Pinecone upsert was NOT called because nodes list was empty
        knowledge_base.pinecone.upsert_text.assert_not_called()

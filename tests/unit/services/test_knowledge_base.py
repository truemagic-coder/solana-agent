import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from solana_agent.services.knowledge_base import KnowledgeBaseService
from solana_agent.adapters.pinecone_adapter import PineconeAdapter
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
import uuid
from datetime import datetime as dt, timezone

# Import the class to test using relative import

# Define dummy document IDs
PARENT_DOC_ID = "parent-doc-123"
CHUNK_DOC_ID_1 = f"{PARENT_DOC_ID}_chunk_0"
CHUNK_DOC_ID_2 = f"{PARENT_DOC_ID}_chunk_1"
PLAIN_DOC_ID = "plain-doc-456"
NON_EXISTENT_DOC_ID = "non-existent-doc-789"
CHUNK_ONLY_ID = (
    "some-chunk-id"  # ID representing a chunk to test direct deletion attempt
)


# Fixture for mocked PineconeAdapter
@pytest.fixture
def mock_pinecone_adapter():
    adapter = MagicMock(spec=PineconeAdapter)
    adapter.delete = AsyncMock(return_value=None)  # Simulate successful async deletion
    # Add attributes potentially checked during init or other methods if needed
    adapter.use_reranking = False
    adapter.embedding_dimensions = 3072  # Example dimension for init logic
    return adapter


# Fixture for mocked MongoDBAdapter
@pytest.fixture
def mock_mongodb_adapter():
    adapter = MagicMock(spec=MongoDBAdapter)
    adapter.find_one = MagicMock(return_value=None)  # Default: not found
    adapter.find = MagicMock(return_value=[])  # Default: no associated docs
    # Mock the return value of delete_many to have a deleted_count attribute
    mock_delete_result = MagicMock()
    mock_delete_result.deleted_count = 0
    adapter.delete_many = MagicMock(return_value=mock_delete_result)
    # Mock methods called during __init__
    adapter.collection_exists = MagicMock(return_value=True)
    adapter.create_collection = MagicMock()
    adapter.create_index = MagicMock()
    return adapter


# Fixture for KnowledgeBaseService instance with mocked dependencies
@pytest.fixture
@patch("solana_agent.services.knowledge_base.OpenAIEmbedding", autospec=True)
@patch("solana_agent.services.knowledge_base.SemanticSplitterNodeParser", autospec=True)
def knowledge_base_service(
    mock_splitter_parser,
    mock_openai_embedding,
    mock_pinecone_adapter,
    mock_mongodb_adapter,
):
    # Mock the embedding model instance needed by the splitter
    mock_embed_model_instance = MagicMock()
    mock_openai_embedding.return_value = mock_embed_model_instance

    # Mock the splitter instance
    mock_splitter_instance = MagicMock()
    mock_splitter_instance.embed_model = (
        mock_embed_model_instance  # Assign embed model to splitter
    )
    mock_splitter_parser.return_value = mock_splitter_instance

    # Prevent actual DB setup during initialization for tests
    with patch.object(KnowledgeBaseService, "_ensure_collection", return_value=None):
        service = KnowledgeBaseService(
            pinecone_adapter=mock_pinecone_adapter,
            mongodb_adapter=mock_mongodb_adapter,
            openai_api_key="fake-key",  # Required by init
            openai_model_name="text-embedding-3-small",  # Match dimension logic
        )
    return service


@pytest.mark.asyncio
class TestKnowledgeBaseServiceDeleteDocument:
    async def test_delete_document_plain_success(
        self,
        knowledge_base_service: KnowledgeBaseService,
        mock_mongodb_adapter: MagicMock,
        mock_pinecone_adapter: MagicMock,
    ):
        """Test successful deletion of a plain document (no chunks)."""
        # Arrange: Mock Mongo to find the plain document
        mock_mongodb_adapter.find_one.return_value = {
            "document_id": PLAIN_DOC_ID,
            "is_chunk": False,
        }
        mock_mongodb_adapter.find.return_value = [{"document_id": PLAIN_DOC_ID}]
        mock_delete_result = MagicMock()
        mock_delete_result.deleted_count = 1
        mock_mongodb_adapter.delete_many.return_value = mock_delete_result

        # Act
        result = await knowledge_base_service.delete_document(PLAIN_DOC_ID)

        # Assert
        assert result is True
        mock_mongodb_adapter.find_one.assert_called_once_with(
            knowledge_base_service.collection,
            {"document_id": PLAIN_DOC_ID},
        )
        mock_mongodb_adapter.find.assert_called_once_with(
            knowledge_base_service.collection,
            {
                "$or": [
                    {"document_id": PLAIN_DOC_ID},
                    {"parent_document_id": PLAIN_DOC_ID},
                ]
            },
        )
        mock_mongodb_adapter.delete_many.assert_called_once_with(
            knowledge_base_service.collection, {"document_id": {"$in": [PLAIN_DOC_ID]}}
        )
        mock_pinecone_adapter.delete.assert_called_once_with(
            ids=[PLAIN_DOC_ID], namespace=None
        )

    async def test_delete_document_with_chunks_success(
        self,
        knowledge_base_service: KnowledgeBaseService,
        mock_mongodb_adapter: MagicMock,
        mock_pinecone_adapter: MagicMock,
    ):
        """Test successful deletion of a parent document and its chunks."""
        # Arrange: Mock Mongo to find parent and chunks
        mock_mongodb_adapter.find_one.return_value = {
            "document_id": PARENT_DOC_ID,
            "is_chunk": False,
        }
        mock_mongodb_adapter.find.return_value = [
            {"document_id": PARENT_DOC_ID},
            {"document_id": CHUNK_DOC_ID_1},
            {"document_id": CHUNK_DOC_ID_2},
        ]
        mock_delete_result = MagicMock()
        mock_delete_result.deleted_count = 3
        mock_mongodb_adapter.delete_many.return_value = mock_delete_result
        expected_ids_to_delete = sorted([PARENT_DOC_ID, CHUNK_DOC_ID_1, CHUNK_DOC_ID_2])

        # Act
        result = await knowledge_base_service.delete_document(PARENT_DOC_ID)

        # Assert
        assert result is True
        mock_mongodb_adapter.find_one.assert_called_once_with(
            knowledge_base_service.collection,
            {"document_id": PARENT_DOC_ID},
        )
        mock_mongodb_adapter.find.assert_called_once_with(
            knowledge_base_service.collection,
            {
                "$or": [
                    {"document_id": PARENT_DOC_ID},
                    {"parent_document_id": PARENT_DOC_ID},
                ]
            },
        )
        # Assert delete_many called with all IDs
        call_args, _ = mock_mongodb_adapter.delete_many.call_args
        assert call_args[0] == knowledge_base_service.collection
        assert sorted(call_args[1]["document_id"]["$in"]) == expected_ids_to_delete

        # Assert pinecone delete called with all IDs
        mock_pinecone_adapter.delete.assert_called_once()
        call_kwargs = mock_pinecone_adapter.delete.call_args.kwargs
        assert sorted(call_kwargs.get("ids", [])) == expected_ids_to_delete
        assert call_kwargs.get("namespace") is None

    async def test_delete_document_attempt_delete_chunk(
        self,
        knowledge_base_service: KnowledgeBaseService,
        mock_mongodb_adapter: MagicMock,
        mock_pinecone_adapter: MagicMock,
    ):
        """Test attempting to delete a chunk directly fails."""
        # Arrange: Mock Mongo find_one to return a chunk document
        mock_mongodb_adapter.find_one.return_value = {
            "document_id": CHUNK_ONLY_ID,
            "is_chunk": True,
        }

        # Act
        result = await knowledge_base_service.delete_document(CHUNK_ONLY_ID)

        # Assert
        assert result is False
        mock_mongodb_adapter.find_one.assert_called_once_with(
            knowledge_base_service.collection,
            {"document_id": CHUNK_ONLY_ID},
        )
        # Ensure other operations were not called
        mock_mongodb_adapter.find.assert_not_called()
        mock_mongodb_adapter.delete_many.assert_not_called()
        mock_pinecone_adapter.delete.assert_not_called()

    async def test_delete_document_not_found(
        self,
        knowledge_base_service: KnowledgeBaseService,
        mock_mongodb_adapter: MagicMock,
        mock_pinecone_adapter: MagicMock,
    ):
        """Test deleting a document that does not exist."""
        # Arrange: Mock Mongo find_one and find to return nothing
        mock_mongodb_adapter.find_one.return_value = None
        mock_mongodb_adapter.find.return_value = []

        # Act
        result = await knowledge_base_service.delete_document(NON_EXISTENT_DOC_ID)

        # Assert
        assert result is False
        mock_mongodb_adapter.find_one.assert_called_once_with(
            knowledge_base_service.collection,
            {"document_id": NON_EXISTENT_DOC_ID},
        )
        # find is still called to check for potential chunks even if find_one fails
        mock_mongodb_adapter.find.assert_called_once_with(
            knowledge_base_service.collection,
            {
                "$or": [
                    {"document_id": NON_EXISTENT_DOC_ID},
                    {"parent_document_id": NON_EXISTENT_DOC_ID},
                ]
            },
        )
        # Ensure delete operations were not called as nothing was found
        mock_mongodb_adapter.delete_many.assert_not_called()
        mock_pinecone_adapter.delete.assert_not_called()

    async def test_delete_document_mongodb_delete_error(
        self,
        knowledge_base_service: KnowledgeBaseService,
        mock_mongodb_adapter: MagicMock,
        mock_pinecone_adapter: MagicMock,
    ):
        """Test deletion fails if MongoDB delete_many raises an error."""
        # Arrange: Mock Mongo find, but make delete_many raise an error
        mock_mongodb_adapter.find_one.return_value = {
            "document_id": PLAIN_DOC_ID,
            "is_chunk": False,
        }
        mock_mongodb_adapter.find.return_value = [{"document_id": PLAIN_DOC_ID}]
        mock_mongodb_adapter.delete_many.side_effect = Exception("Mongo DB Error")

        # Act
        result = await knowledge_base_service.delete_document(PLAIN_DOC_ID)

        # Assert
        assert result is False
        mock_mongodb_adapter.find_one.assert_called_once()
        mock_mongodb_adapter.find.assert_called_once()
        mock_mongodb_adapter.delete_many.assert_called_once()  # It was called
        mock_pinecone_adapter.delete.assert_called_once_with(
            ids=[PLAIN_DOC_ID], namespace=None
        )  # Pinecone delete still attempted

    async def test_delete_document_pinecone_delete_error(
        self,
        knowledge_base_service: KnowledgeBaseService,
        mock_mongodb_adapter: MagicMock,
        mock_pinecone_adapter: MagicMock,
    ):
        """Test deletion fails if Pinecone delete raises an error."""
        # Arrange: Mock successful Mongo find/delete, but make Pinecone delete raise error
        mock_mongodb_adapter.find_one.return_value = {
            "document_id": PLAIN_DOC_ID,
            "is_chunk": False,
        }
        mock_mongodb_adapter.find.return_value = [{"document_id": PLAIN_DOC_ID}]
        mock_delete_result = MagicMock()
        mock_delete_result.deleted_count = 1
        mock_mongodb_adapter.delete_many.return_value = mock_delete_result
        mock_pinecone_adapter.delete.side_effect = Exception("Pinecone Error")

        # Act
        result = await knowledge_base_service.delete_document(PLAIN_DOC_ID)

        # Assert
        assert result is False
        mock_mongodb_adapter.find_one.assert_called_once()
        mock_mongodb_adapter.find.assert_called_once()
        mock_mongodb_adapter.delete_many.assert_called_once()  # Mongo delete was attempted (and succeeded mock)
        mock_pinecone_adapter.delete.assert_called_once_with(
            ids=[PLAIN_DOC_ID], namespace=None
        )  # Pinecone delete was called

    async def test_delete_document_mongodb_find_one_error(
        self,
        knowledge_base_service: KnowledgeBaseService,
        mock_mongodb_adapter: MagicMock,
        mock_pinecone_adapter: MagicMock,
    ):
        """Test deletion fails if the initial MongoDB find_one check raises an error."""
        # Arrange
        mock_mongodb_adapter.find_one.side_effect = Exception("Mongo Find One Error")

        # Act
        result = await knowledge_base_service.delete_document(PLAIN_DOC_ID)

        # Assert
        assert result is False
        mock_mongodb_adapter.find_one.assert_called_once()
        # Ensure other operations were not called
        mock_mongodb_adapter.find.assert_not_called()
        mock_mongodb_adapter.delete_many.assert_not_called()
        mock_pinecone_adapter.delete.assert_not_called()

    async def test_delete_document_with_namespace(
        self,
        knowledge_base_service: KnowledgeBaseService,
        mock_mongodb_adapter: MagicMock,
        mock_pinecone_adapter: MagicMock,
    ):
        """Test successful deletion uses the provided namespace for Pinecone."""
        # Arrange
        namespace = "test-namespace"
        mock_mongodb_adapter.find_one.return_value = {
            "document_id": PLAIN_DOC_ID,
            "is_chunk": False,
        }
        mock_mongodb_adapter.find.return_value = [{"document_id": PLAIN_DOC_ID}]
        mock_delete_result = MagicMock()
        mock_delete_result.deleted_count = 1
        mock_mongodb_adapter.delete_many.return_value = mock_delete_result

        # Act
        result = await knowledge_base_service.delete_document(
            PLAIN_DOC_ID, namespace=namespace
        )

        # Assert
        assert result is True
        mock_mongodb_adapter.delete_many.assert_called_once()
        mock_pinecone_adapter.delete.assert_called_once_with(
            ids=[PLAIN_DOC_ID], namespace=namespace
        )  # Check namespace
        # filepath: /Users/bevanhunt/solana-agent/tests/unit/services/test_knowledge_base.py

        # Import the class to test using relative import if needed, or adjust path
        # Assuming the test file structure allows this import:
        # from solana_agent.services.knowledge_base import KnowledgeBaseService
        # from solana_agent.adapters.pinecone_adapter import PineconeAdapter
        # from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
        # Note: Imports might already exist in the actual file, ensure they are correct.

        # Define dummy document IDs (assuming these are defined globally in the test file)
        # PARENT_DOC_ID = "parent-doc-123"
        # CHUNK_DOC_ID_1 = f"{PARENT_DOC_ID}_chunk_0"
        # CHUNK_DOC_ID_2 = f"{PARENT_DOC_ID}_chunk_1"
        # PLAIN_DOC_ID = "plain-doc-456"
        # NON_EXISTENT_DOC_ID = "non-existent-doc-789"
        # CHUNK_ONLY_ID = "some-chunk-id"

        @pytest.mark.asyncio
        class TestKnowledgeBaseServiceDeleteDocument:
            # Assuming fixtures mock_mongodb_adapter, mock_pinecone_adapter,
            # and knowledge_base_service are defined as in the example.

            async def test_delete_document_plain_success(
                self,
                knowledge_base_service: KnowledgeBaseService,
                mock_mongodb_adapter: MagicMock,
                mock_pinecone_adapter: MagicMock,
            ):
                """Test successful deletion of a plain document (no chunks)."""
                # Arrange: Mock Mongo to find the plain document (not a chunk)
                mock_mongodb_adapter.find_one.return_value = {
                    "document_id": PLAIN_DOC_ID,
                    "is_chunk": False,
                }
                # Mock find to return only the document itself
                mock_mongodb_adapter.find.return_value = [{"document_id": PLAIN_DOC_ID}]
                # Mock successful deletion in Mongo
                mock_delete_result = MagicMock()
                mock_delete_result.deleted_count = 1
                mock_mongodb_adapter.delete_many.return_value = mock_delete_result
                # Mock successful deletion in Pinecone
                mock_pinecone_adapter.delete = AsyncMock(return_value=None)

                # Act
                result = await knowledge_base_service.delete_document(PLAIN_DOC_ID)

                # Assert
                assert result is True
                # Check initial find_one call
                mock_mongodb_adapter.find_one.assert_called_once_with(
                    knowledge_base_service.collection,
                    {"document_id": PLAIN_DOC_ID},
                )
                # Check find call for associated docs
                mock_mongodb_adapter.find.assert_called_once_with(
                    knowledge_base_service.collection,
                    {
                        "$or": [
                            {"document_id": PLAIN_DOC_ID},
                            {"parent_document_id": PLAIN_DOC_ID},
                        ]
                    },
                )
                # Check Mongo delete call
                mock_mongodb_adapter.delete_many.assert_called_once_with(
                    knowledge_base_service.collection,
                    {"document_id": {"$in": [PLAIN_DOC_ID]}},
                )
                # Check Pinecone delete call
                mock_pinecone_adapter.delete.assert_called_once_with(
                    ids=[PLAIN_DOC_ID], namespace=None
                )

            async def test_delete_document_with_chunks_success(
                self,
                knowledge_base_service: KnowledgeBaseService,
                mock_mongodb_adapter: MagicMock,
                mock_pinecone_adapter: MagicMock,
            ):
                """Test successful deletion of a parent document and its chunks."""
                # Arrange: Mock Mongo find_one for parent
                mock_mongodb_adapter.find_one.return_value = {
                    "document_id": PARENT_DOC_ID,
                    "is_chunk": False,
                }
                # Mock find to return parent and chunks
                expected_ids_to_delete = sorted(
                    [PARENT_DOC_ID, CHUNK_DOC_ID_1, CHUNK_DOC_ID_2]
                )
                mock_mongodb_adapter.find.return_value = [
                    {"document_id": PARENT_DOC_ID},
                    {"document_id": CHUNK_DOC_ID_1},
                    {"document_id": CHUNK_DOC_ID_2},
                ]
                # Mock successful deletion in Mongo
                mock_delete_result = MagicMock()
                mock_delete_result.deleted_count = 3
                mock_mongodb_adapter.delete_many.return_value = mock_delete_result
                # Mock successful deletion in Pinecone
                mock_pinecone_adapter.delete = AsyncMock(return_value=None)

                # Act
                result = await knowledge_base_service.delete_document(PARENT_DOC_ID)

                # Assert
                assert result is True
                # Check initial find_one call
                mock_mongodb_adapter.find_one.assert_called_once_with(
                    knowledge_base_service.collection,
                    {"document_id": PARENT_DOC_ID},
                )
                # Check find call for associated docs
                mock_mongodb_adapter.find.assert_called_once_with(
                    knowledge_base_service.collection,
                    {
                        "$or": [
                            {"document_id": PARENT_DOC_ID},
                            {"parent_document_id": PARENT_DOC_ID},
                        ]
                    },
                )
                # Assert delete_many called with all IDs
                call_args, _ = mock_mongodb_adapter.delete_many.call_args
                assert call_args[0] == knowledge_base_service.collection
                assert (
                    sorted(call_args[1]["document_id"]["$in"]) == expected_ids_to_delete
                )

                # Assert pinecone delete called with all IDs
                mock_pinecone_adapter.delete.assert_called_once()
                call_kwargs = mock_pinecone_adapter.delete.call_args.kwargs
                assert sorted(call_kwargs.get("ids", [])) == expected_ids_to_delete
                assert call_kwargs.get("namespace") is None

            async def test_delete_document_attempt_delete_chunk(
                self,
                knowledge_base_service: KnowledgeBaseService,
                mock_mongodb_adapter: MagicMock,
                mock_pinecone_adapter: MagicMock,
            ):
                """Test attempting to delete a chunk directly fails."""
                # Arrange: Mock Mongo find_one to return a chunk document
                mock_mongodb_adapter.find_one.return_value = {
                    "document_id": CHUNK_ONLY_ID,
                    "is_chunk": True,
                }
                # Reset other mocks to ensure they aren't called
                mock_mongodb_adapter.find.reset_mock()
                mock_mongodb_adapter.delete_many.reset_mock()
                mock_pinecone_adapter.delete = (
                    AsyncMock()
                )  # Use AsyncMock for awaitable methods

                # Act
                result = await knowledge_base_service.delete_document(CHUNK_ONLY_ID)

                # Assert
                assert result is False
                # Check initial find_one call
                mock_mongodb_adapter.find_one.assert_called_once_with(
                    knowledge_base_service.collection,
                    {"document_id": CHUNK_ONLY_ID},
                )
                # Ensure other operations were not called
                mock_mongodb_adapter.find.assert_not_called()
                mock_mongodb_adapter.delete_many.assert_not_called()
                mock_pinecone_adapter.delete.assert_not_called()

            async def test_delete_document_not_found(
                self,
                knowledge_base_service: KnowledgeBaseService,
                mock_mongodb_adapter: MagicMock,
                mock_pinecone_adapter: MagicMock,
            ):
                """Test deleting a document that does not exist."""
                # Arrange: Mock Mongo find_one and find to return nothing
                mock_mongodb_adapter.find_one.return_value = None
                mock_mongodb_adapter.find.return_value = []
                # Reset other mocks
                mock_mongodb_adapter.delete_many.reset_mock()
                mock_pinecone_adapter.delete = AsyncMock()

                # Act
                result = await knowledge_base_service.delete_document(
                    NON_EXISTENT_DOC_ID
                )

                # Assert
                assert result is False
                # Check initial find_one call
                mock_mongodb_adapter.find_one.assert_called_once_with(
                    knowledge_base_service.collection,
                    {"document_id": NON_EXISTENT_DOC_ID},
                )
                # Check find call (it's still called even if find_one returns None)
                mock_mongodb_adapter.find.assert_called_once_with(
                    knowledge_base_service.collection,
                    {
                        "$or": [
                            {"document_id": NON_EXISTENT_DOC_ID},
                            {"parent_document_id": NON_EXISTENT_DOC_ID},
                        ]
                    },
                )
                # Ensure delete operations were not called as nothing was found
                mock_mongodb_adapter.delete_many.assert_not_called()
                mock_pinecone_adapter.delete.assert_not_called()

            async def test_delete_document_mongodb_delete_error(
                self,
                knowledge_base_service: KnowledgeBaseService,
                mock_mongodb_adapter: MagicMock,
                mock_pinecone_adapter: MagicMock,
            ):
                """Test deletion fails if MongoDB delete_many raises an error."""
                # Arrange: Mock find_one and find to succeed
                mock_mongodb_adapter.find_one.return_value = {
                    "document_id": PLAIN_DOC_ID,
                    "is_chunk": False,
                }
                mock_mongodb_adapter.find.return_value = [{"document_id": PLAIN_DOC_ID}]
                # Mock delete_many to raise an error
                mock_mongodb_adapter.delete_many.side_effect = Exception(
                    "Mongo DB Error"
                )
                # Mock pinecone delete to succeed (it should still be called)
                mock_pinecone_adapter.delete = AsyncMock(return_value=None)

                # Act
                result = await knowledge_base_service.delete_document(PLAIN_DOC_ID)

                # Assert
                assert result is False
                mock_mongodb_adapter.find_one.assert_called_once()
                mock_mongodb_adapter.find.assert_called_once()
                mock_mongodb_adapter.delete_many.assert_called_once()  # It was called
                mock_pinecone_adapter.delete.assert_called_once_with(
                    ids=[PLAIN_DOC_ID], namespace=None
                )  # Pinecone delete still attempted

            async def test_delete_document_pinecone_delete_error(
                self,
                knowledge_base_service: KnowledgeBaseService,
                mock_mongodb_adapter: MagicMock,
                mock_pinecone_adapter: MagicMock,
            ):
                """Test deletion fails if Pinecone delete raises an error."""
                # Arrange: Mock successful Mongo find/delete
                mock_mongodb_adapter.find_one.return_value = {
                    "document_id": PLAIN_DOC_ID,
                    "is_chunk": False,
                }
                mock_mongodb_adapter.find.return_value = [{"document_id": PLAIN_DOC_ID}]
                mock_delete_result = MagicMock()
                mock_delete_result.deleted_count = 1
                mock_mongodb_adapter.delete_many.return_value = mock_delete_result
                # Mock Pinecone delete to raise error
                mock_pinecone_adapter.delete = AsyncMock(
                    side_effect=Exception("Pinecone Error")
                )

                # Act
                result = await knowledge_base_service.delete_document(PLAIN_DOC_ID)

                # Assert
                assert result is False
                mock_mongodb_adapter.find_one.assert_called_once()
                mock_mongodb_adapter.find.assert_called_once()
                mock_mongodb_adapter.delete_many.assert_called_once()  # Mongo delete was attempted
                mock_pinecone_adapter.delete.assert_called_once_with(
                    ids=[PLAIN_DOC_ID], namespace=None
                )  # Pinecone delete was called

            async def test_delete_document_mongodb_find_one_error(
                self,
                knowledge_base_service: KnowledgeBaseService,
                mock_mongodb_adapter: MagicMock,
                mock_pinecone_adapter: MagicMock,
            ):
                """Test deletion fails if the initial MongoDB find_one check raises an error."""
                # Arrange: Mock find_one to raise error
                mock_mongodb_adapter.find_one.side_effect = Exception(
                    "Mongo Find One Error"
                )
                # Reset other mocks
                mock_mongodb_adapter.find.reset_mock()
                mock_mongodb_adapter.delete_many.reset_mock()
                mock_pinecone_adapter.delete = AsyncMock()

                # Act
                result = await knowledge_base_service.delete_document(PLAIN_DOC_ID)

                # Assert
                assert result is False
                mock_mongodb_adapter.find_one.assert_called_once()
                # Ensure other operations were not called
                mock_mongodb_adapter.find.assert_not_called()
                mock_mongodb_adapter.delete_many.assert_not_called()
                mock_pinecone_adapter.delete.assert_not_called()

            async def test_delete_document_mongodb_find_error(
                self,
                knowledge_base_service: KnowledgeBaseService,
                mock_mongodb_adapter: MagicMock,
                mock_pinecone_adapter: MagicMock,
            ):
                """Test deletion fails if MongoDB find (for associated docs) raises an error, but cleanup is attempted."""
                # Arrange: Mock find_one success, but find raises error
                mock_mongodb_adapter.find_one.return_value = {
                    "document_id": PLAIN_DOC_ID,
                    "is_chunk": False,
                }
                mock_mongodb_adapter.find.side_effect = Exception("Mongo Find Error")
                # Mock successful deletion for the fallback ID cleanup
                mock_delete_result = MagicMock()
                mock_delete_result.deleted_count = 1
                mock_mongodb_adapter.delete_many.return_value = mock_delete_result
                mock_pinecone_adapter.delete = AsyncMock(return_value=None)

                # Act
                result = await knowledge_base_service.delete_document(PLAIN_DOC_ID)

                # Assert
                assert result is False  # Should fail because find errored
                mock_mongodb_adapter.find_one.assert_called_once()
                mock_mongodb_adapter.find.assert_called_once()  # find was called
                # Assert cleanup was attempted with only the original ID due to the error
                mock_mongodb_adapter.delete_many.assert_called_once_with(
                    knowledge_base_service.collection,
                    {"document_id": {"$in": [PLAIN_DOC_ID]}},
                )
                mock_pinecone_adapter.delete.assert_called_once_with(
                    ids=[PLAIN_DOC_ID], namespace=None
                )

            async def test_delete_document_with_namespace(
                self,
                knowledge_base_service: KnowledgeBaseService,
                mock_mongodb_adapter: MagicMock,
                mock_pinecone_adapter: MagicMock,
            ):
                """Test successful deletion uses the provided namespace for Pinecone."""
                # Arrange
                PLAIN_DOC_ID = "plain-doc-456"
                namespace = "test-namespace"
                mock_mongodb_adapter.find_one.return_value = {
                    "document_id": PLAIN_DOC_ID,
                    "is_chunk": False,
                }
                mock_mongodb_adapter.find.return_value = [{"document_id": PLAIN_DOC_ID}]
                mock_delete_result = MagicMock()
                mock_delete_result.deleted_count = 1
                mock_mongodb_adapter.delete_many.return_value = mock_delete_result
                mock_pinecone_adapter.delete = AsyncMock(return_value=None)

                # Act
                result = await knowledge_base_service.delete_document(
                    PLAIN_DOC_ID, namespace=namespace
                )

                # Assert
                assert result is True
                mock_mongodb_adapter.find_one.assert_called_once()
                mock_mongodb_adapter.find.assert_called_once()
                mock_mongodb_adapter.delete_many.assert_called_once()
                # Check Pinecone delete call with namespace
                mock_pinecone_adapter.delete.assert_called_once_with(
                    ids=[PLAIN_DOC_ID], namespace=namespace
                )

                # Assuming these imports are correctly handled relative to the test file location
                from llama_index.core.schema import (
                    TextNode,
                )  # For mocking splitter output

                # Define dummy document IDs (assuming these are defined globally)
                PARENT_DOC_ID = "parent-doc-123"
                CHUNK_DOC_ID_1 = f"{PARENT_DOC_ID}_chunk_0"
                PLAIN_DOC_ID = "plain-doc-456"
                NON_EXISTENT_DOC_ID = "non-existent-doc-789"
                FIXED_UUID = "fixed-uuid-for-tests"
                FIXED_DATETIME = dt(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

                # --- Existing Fixtures (Ensure they are present and correct) ---
                # @pytest.fixture
                # def mock_pinecone_adapter(): ...
                # @pytest.fixture
                # def mock_mongodb_adapter(): ...
                # @pytest.fixture
                # @patch(...)
                # def knowledge_base_service(...): ...
                # --- End Existing Fixtures ---

                # --- Add/Update Fixtures if needed ---
                @pytest.fixture(autouse=True)
                def mock_datetime_uuid(monkeypatch):
                    """Auto-mock datetime.now and uuid.uuid4 for consistent tests."""
                    mock_dt = MagicMock(spec=dt)
                    mock_dt.now.return_value = FIXED_DATETIME
                    monkeypatch.setattr(
                        "solana_agent.services.knowledge_base.dt", mock_dt
                    )

                    mock_uuid = MagicMock(spec=uuid)
                    mock_uuid.uuid4.return_value = FIXED_UUID
                    monkeypatch.setattr(
                        "solana_agent.services.knowledge_base.uuid", mock_uuid
                    )

                # --- Test Class for delete_document (Complete/Refined) ---
                @pytest.mark.asyncio
                class TestKnowledgeBaseServiceDeleteDocument:
                    # Tests provided in the previous turn are assumed here.
                    # Add any missing cases or refine existing ones based on the implementation.

                    async def test_delete_document_not_found_but_exists_in_mongo_find(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                    ):
                        """Test deletion when find_one fails but find returns associated docs (inconsistency)."""
                        # Arrange: Mock find_one to return None, but find returns docs
                        mock_mongodb_adapter.find_one.return_value = None
                        mock_mongodb_adapter.find.return_value = [
                            {
                                "document_id": NON_EXISTENT_DOC_ID
                            },  # Simulate finding the doc later
                            {"document_id": f"{NON_EXISTENT_DOC_ID}_chunk_0"},
                        ]
                        mock_delete_result = MagicMock()
                        mock_delete_result.deleted_count = 2
                        mock_mongodb_adapter.delete_many.return_value = (
                            mock_delete_result
                        )
                        mock_pinecone_adapter.delete = AsyncMock(return_value=None)
                        expected_ids = sorted(
                            [NON_EXISTENT_DOC_ID, f"{NON_EXISTENT_DOC_ID}_chunk_0"]
                        )

                        # Act
                        result = await knowledge_base_service.delete_document(
                            NON_EXISTENT_DOC_ID
                        )

                        # Assert
                        assert (
                            result is False
                        )  # Should be False because initial find_one failed (document_found=False)
                        mock_mongodb_adapter.find_one.assert_called_once()
                        mock_mongodb_adapter.find.assert_called_once()
                        # Assert cleanup was attempted based on find results
                        mock_mongodb_adapter.delete_many.assert_called_once()
                        call_args, _ = mock_mongodb_adapter.delete_many.call_args
                        assert (
                            sorted(call_args[1]["document_id"]["$in"]) == expected_ids
                        )
                        mock_pinecone_adapter.delete.assert_called_once()
                        call_kwargs = mock_pinecone_adapter.delete.call_args.kwargs
                        assert sorted(call_kwargs.get("ids", [])) == expected_ids

                # --- Test Class for add_document ---
                @pytest.mark.asyncio
                class TestKnowledgeBaseServiceAddDocument:
                    async def test_add_document_success_no_id_no_rerank(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                    ):
                        # Arrange
                        text = "This is a test document."
                        metadata = {"source": "test_source", "tags": ["tag1"]}
                        namespace = "test_namespace"
                        expected_doc_id = FIXED_UUID
                        mock_embedding = [0.1] * 3072  # Match dimension
                        knowledge_base_service.pinecone.use_reranking = (
                            False  # Ensure rerank is off
                        )

                        # Mock dependencies
                        mock_mongodb_adapter.insert_one = MagicMock()
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding = AsyncMock(
                            return_value=mock_embedding
                        )
                        mock_pinecone_adapter.upsert = AsyncMock()

                        # Act
                        doc_id = await knowledge_base_service.add_document(
                            text, metadata, namespace=namespace
                        )

                        # Assert
                        assert doc_id == expected_doc_id
                        # Mongo insert check
                        mock_mongodb_adapter.insert_one.assert_called_once()
                        mongo_call_args = mock_mongodb_adapter.insert_one.call_args[0]
                        assert mongo_call_args[0] == knowledge_base_service.collection
                        expected_mongo_doc = {
                            "document_id": expected_doc_id,
                            "content": text,
                            "is_chunk": False,
                            "parent_document_id": None,
                            "source": "test_source",
                            "tags": ["tag1"],
                            "created_at": FIXED_DATETIME,
                            "updated_at": FIXED_DATETIME,
                        }
                        assert mongo_call_args[1] == expected_mongo_doc
                        # Embedding check
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding.assert_awaited_once_with(
                            text
                        )
                        # Pinecone upsert check
                        mock_pinecone_adapter.upsert.assert_awaited_once()
                        pinecone_call_kwargs = (
                            mock_pinecone_adapter.upsert.call_args.kwargs
                        )
                        expected_pinecone_metadata = {
                            "document_id": expected_doc_id,
                            "is_chunk": False,
                            "parent_document_id": None,
                            "source": "test_source",
                            "tags": ["tag1"],
                        }
                        expected_pinecone_vector = {
                            "id": expected_doc_id,
                            "values": mock_embedding,
                            "metadata": expected_pinecone_metadata,
                        }
                        assert pinecone_call_kwargs["vectors"] == [
                            expected_pinecone_vector
                        ]
                        assert pinecone_call_kwargs["namespace"] == namespace

                    async def test_add_document_success_with_id_with_rerank(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                    ):
                        # Arrange
                        text = "Another test document."
                        metadata = {"source": "rerank_source"}
                        provided_doc_id = "provided-doc-id-789"
                        knowledge_base_service.pinecone.use_reranking = True
                        knowledge_base_service.pinecone.rerank_text_field = (
                            "text_for_rerank"
                        )
                        mock_embedding = [0.2] * 3072

                        # Mock dependencies
                        mock_mongodb_adapter.insert_one = MagicMock()
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding = AsyncMock(
                            return_value=mock_embedding
                        )
                        mock_pinecone_adapter.upsert = AsyncMock()

                        # Act
                        doc_id = await knowledge_base_service.add_document(
                            text, metadata, document_id=provided_doc_id
                        )

                        # Assert
                        assert doc_id == provided_doc_id
                        # Mongo insert check (only check ID and content for brevity)
                        mock_mongodb_adapter.insert_one.assert_called_once()
                        mongo_call_args = mock_mongodb_adapter.insert_one.call_args[0]
                        assert mongo_call_args[1]["document_id"] == provided_doc_id
                        assert mongo_call_args[1]["content"] == text
                        # Embedding check
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding.assert_awaited_once_with(
                            text
                        )
                        # Pinecone upsert check (check metadata includes rerank text)
                        mock_pinecone_adapter.upsert.assert_awaited_once()
                        pinecone_call_kwargs = (
                            mock_pinecone_adapter.upsert.call_args.kwargs
                        )
                        pinecone_metadata = pinecone_call_kwargs["vectors"][0][
                            "metadata"
                        ]
                        assert pinecone_metadata["document_id"] == provided_doc_id
                        assert pinecone_metadata["source"] == "rerank_source"
                        assert (
                            pinecone_metadata[
                                knowledge_base_service.pinecone.rerank_text_field
                            ]
                            == text
                        )
                        assert (
                            pinecone_call_kwargs["namespace"] is None
                        )  # Default namespace

                    async def test_add_document_mongo_insert_fails(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                    ):
                        # Arrange
                        mock_mongodb_adapter.insert_one.side_effect = Exception(
                            "Mongo Error"
                        )
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding = AsyncMock()
                        mock_pinecone_adapter.upsert = AsyncMock()

                        # Act & Assert
                        with pytest.raises(Exception, match="Mongo Error"):
                            await knowledge_base_service.add_document("text", {})
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding.assert_not_awaited()
                        mock_pinecone_adapter.upsert.assert_not_awaited()

                    async def test_add_document_embedding_fails(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                    ):
                        # Arrange
                        mock_mongodb_adapter.insert_one = MagicMock()  # Mongo succeeds
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding.side_effect = Exception(
                            "Embedding Error"
                        )
                        mock_pinecone_adapter.upsert = AsyncMock()

                        # Act & Assert
                        with pytest.raises(Exception, match="Embedding Error"):
                            await knowledge_base_service.add_document("text", {})
                        mock_mongodb_adapter.insert_one.assert_called_once()  # Mongo was called
                        mock_pinecone_adapter.upsert.assert_not_awaited()  # Pinecone was not

                    async def test_add_document_pinecone_upsert_fails(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                    ):
                        # Arrange
                        mock_mongodb_adapter.insert_one = MagicMock()  # Mongo succeeds
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding = AsyncMock(
                            return_value=[0.1] * 3072
                        )  # Embedding succeeds
                        mock_pinecone_adapter.upsert.side_effect = Exception(
                            "Pinecone Error"
                        )

                        # Act & Assert
                        with pytest.raises(Exception, match="Pinecone Error"):
                            await knowledge_base_service.add_document("text", {})
                        mock_mongodb_adapter.insert_one.assert_called_once()
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding.assert_awaited_once()
                        mock_pinecone_adapter.upsert.assert_awaited_once()  # Pinecone was called

                # --- Test Class for add_pdf_document ---
                @pytest.mark.asyncio
                @patch("solana_agent.services.knowledge_base.pypdf.PdfReader")
                @patch("solana_agent.services.knowledge_base.io.BytesIO")
                @patch("solana_agent.services.knowledge_base.asyncio.to_thread")
                class TestKnowledgeBaseServiceAddPdfDocument:
                    @pytest.fixture
                    def mock_pdf_reader_instance(self):
                        mock_reader = MagicMock()
                        mock_page1 = MagicMock()
                        mock_page1.extract_text.return_value = (
                            "This is the first chunk. "
                        )
                        mock_page2 = MagicMock()
                        mock_page2.extract_text.return_value = (
                            "This is the second chunk."
                        )
                        mock_reader.pages = [mock_page1, mock_page2]
                        return mock_reader

                    @pytest.fixture
                    def mock_splitter_nodes(self):
                        node1 = TextNode(
                            text="This is the first chunk.",
                            id_=f"{PARENT_DOC_ID}_chunk_0",
                        )
                        node2 = TextNode(
                            text="This is the second chunk.",
                            id_=f"{PARENT_DOC_ID}_chunk_1",
                        )
                        return [node1, node2]

                    async def test_add_pdf_document_bytes_success(
                        self,
                        mock_to_thread,
                        mock_bytesio,
                        mock_pdfreader_cls,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                        mock_pdf_reader_instance,
                        mock_splitter_nodes,
                    ):
                        # Arrange
                        pdf_bytes = b"dummy pdf bytes"
                        metadata = {"source": "pdf_source", "tags": ["pdf"]}
                        namespace = "pdf_namespace"
                        expected_parent_id = FIXED_UUID
                        mock_embeddings = [[0.1] * 3072, [0.2] * 3072]

                        # Mock dependencies
                        mock_bytesio.return_value = (
                            MagicMock()
                        )  # Mock the BytesIO object
                        mock_pdfreader_cls.return_value = mock_pdf_reader_instance
                        mock_to_thread.return_value = mock_splitter_nodes  # Mock the result of to_thread(splitter...)
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding_batch = AsyncMock(
                            return_value=mock_embeddings
                        )
                        mock_mongodb_adapter.insert_one = MagicMock()  # For parent doc
                        mock_mongodb_adapter.insert_many = MagicMock()  # For chunks
                        mock_pinecone_adapter.upsert = AsyncMock()  # For chunks

                        # Act
                        parent_doc_id = await knowledge_base_service.add_pdf_document(
                            pdf_bytes, metadata, namespace=namespace
                        )

                        # Assert
                        assert parent_doc_id == expected_parent_id
                        # PDF Reading
                        mock_bytesio.assert_called_once_with(pdf_bytes)
                        mock_pdfreader_cls.assert_called_once_with(
                            mock_bytesio.return_value
                        )
                        # Mongo Parent Insert
                        mock_mongodb_adapter.insert_one.assert_called_once()
                        parent_mongo_call_args = (
                            mock_mongodb_adapter.insert_one.call_args[0]
                        )
                        assert (
                            parent_mongo_call_args[1]["document_id"]
                            == expected_parent_id
                        )
                        assert parent_mongo_call_args[1]["is_chunk"] is False
                        assert (
                            parent_mongo_call_args[1]["content"]
                            == "This is the first chunk. This is the second chunk."
                        )
                        assert parent_mongo_call_args[1]["source"] == "pdf_source"
                        # Chunking
                        mock_to_thread.assert_awaited_once()  # Check that splitter was called via to_thread
                        # Embedding
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_awaited_once_with(
                            ["This is the first chunk.", "This is the second chunk."],
                            show_progress=True,
                        )
                        # Mongo Chunk Insert
                        mock_mongodb_adapter.insert_many.assert_called_once()
                        chunk_mongo_call_args = (
                            mock_mongodb_adapter.insert_many.call_args[0]
                        )
                        assert len(chunk_mongo_call_args[1]) == 2
                        assert (
                            chunk_mongo_call_args[1][0]["document_id"]
                            == f"{expected_parent_id}_chunk_0"
                        )
                        assert (
                            chunk_mongo_call_args[1][0]["parent_document_id"]
                            == expected_parent_id
                        )
                        assert chunk_mongo_call_args[1][0]["is_chunk"] is True
                        assert (
                            chunk_mongo_call_args[1][0]["content"]
                            == "This is the first chunk."
                        )
                        assert (
                            chunk_mongo_call_args[1][1]["document_id"]
                            == f"{expected_parent_id}_chunk_1"
                        )
                        assert (
                            chunk_mongo_call_args[1][1]["content"]
                            == "This is the second chunk."
                        )
                        # Pinecone Chunk Upsert (assuming batch size >= 2)
                        mock_pinecone_adapter.upsert.assert_awaited_once()
                        pinecone_call_kwargs = (
                            mock_pinecone_adapter.upsert.call_args.kwargs
                        )
                        assert len(pinecone_call_kwargs["vectors"]) == 2
                        assert (
                            pinecone_call_kwargs["vectors"][0]["id"]
                            == f"{expected_parent_id}_chunk_0"
                        )
                        assert (
                            pinecone_call_kwargs["vectors"][0]["values"]
                            == mock_embeddings[0]
                        )
                        assert (
                            pinecone_call_kwargs["vectors"][0]["metadata"][
                                "parent_document_id"
                            ]
                            == expected_parent_id
                        )
                        assert (
                            pinecone_call_kwargs["vectors"][0]["metadata"]["is_chunk"]
                            is True
                        )
                        assert (
                            pinecone_call_kwargs["vectors"][1]["id"]
                            == f"{expected_parent_id}_chunk_1"
                        )
                        assert (
                            pinecone_call_kwargs["vectors"][1]["values"]
                            == mock_embeddings[1]
                        )
                        assert pinecone_call_kwargs["namespace"] == namespace

                    @patch("builtins.open", new_callable=MagicMock)
                    async def test_add_pdf_document_path_success(
                        self,
                        mock_open,
                        mock_to_thread,
                        mock_bytesio,
                        mock_pdfreader_cls,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                        mock_pdf_reader_instance,
                        mock_splitter_nodes,
                    ):
                        # Arrange
                        pdf_path = "/path/to/dummy.pdf"
                        pdf_bytes = b"dummy pdf bytes from file"
                        metadata = {"source": "pdf_path_source"}
                        expected_parent_id = FIXED_UUID
                        mock_embeddings = [[0.3] * 3072, [0.4] * 3072]

                        # Mock file reading
                        mock_file = MagicMock()
                        mock_file.read.return_value = pdf_bytes
                        mock_open.return_value.__enter__.return_value = mock_file

                        # Mock other dependencies
                        mock_bytesio.return_value = MagicMock()
                        mock_pdfreader_cls.return_value = mock_pdf_reader_instance
                        mock_to_thread.return_value = mock_splitter_nodes
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding_batch = AsyncMock(
                            return_value=mock_embeddings
                        )
                        mock_mongodb_adapter.insert_one = MagicMock()
                        mock_mongodb_adapter.insert_many = MagicMock()
                        mock_pinecone_adapter.upsert = AsyncMock()

                        # Act
                        parent_doc_id = await knowledge_base_service.add_pdf_document(
                            pdf_path, metadata
                        )

                        # Assert
                        assert parent_doc_id == expected_parent_id
                        mock_open.assert_called_once_with(
                            pdf_path, "rb"
                        )  # Check file open
                        mock_bytesio.assert_called_once_with(
                            pdf_bytes
                        )  # Check BytesIO with read data
                        # Other assertions would be similar to the bytes test

                    async def test_add_pdf_document_no_text_extracted(
                        self,
                        mock_to_thread,
                        mock_bytesio,
                        mock_pdfreader_cls,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                    ):
                        # Arrange
                        pdf_bytes = b"empty pdf"
                        metadata = {"source": "empty_pdf"}
                        expected_parent_id = FIXED_UUID

                        # Mock PDF reader to return no text
                        mock_empty_reader = MagicMock()
                        mock_empty_page = MagicMock()
                        mock_empty_page.extract_text.return_value = ""  # No text
                        mock_empty_reader.pages = [mock_empty_page]
                        mock_bytesio.return_value = MagicMock()
                        mock_pdfreader_cls.return_value = mock_empty_reader

                        # Mock other dependencies
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding_batch = AsyncMock()
                        mock_mongodb_adapter.insert_one = MagicMock()
                        mock_mongodb_adapter.insert_many = MagicMock()
                        mock_pinecone_adapter.upsert = AsyncMock()

                        # Act
                        parent_doc_id = await knowledge_base_service.add_pdf_document(
                            pdf_bytes, metadata
                        )

                        # Assert
                        assert parent_doc_id == expected_parent_id
                        # Parent Mongo doc should still be inserted
                        mock_mongodb_adapter.insert_one.assert_called_once()
                        parent_mongo_call_args = (
                            mock_mongodb_adapter.insert_one.call_args[0]
                        )
                        assert (
                            parent_mongo_call_args[1]["document_id"]
                            == expected_parent_id
                        )
                        assert (
                            parent_mongo_call_args[1]["content"] == ""
                        )  # Empty content stored
                        assert parent_mongo_call_args[1]["source"] == "empty_pdf"
                        # Chunking, Embedding, Chunk DB/Vector operations should NOT be called
                        mock_to_thread.assert_not_awaited()
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_not_awaited()
                        mock_mongodb_adapter.insert_many.assert_not_called()
                        mock_pinecone_adapter.upsert.assert_not_awaited()

                    async def test_add_pdf_document_pdf_read_error(
                        self,
                        mock_to_thread,
                        mock_bytesio,
                        mock_pdfreader_cls,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                    ):
                        # Arrange
                        mock_bytesio.side_effect = Exception(
                            "BytesIO Error"
                        )  # Error during BytesIO creation

                        # Act & Assert
                        with pytest.raises(Exception, match="BytesIO Error"):
                            await knowledge_base_service.add_pdf_document(
                                b"bad pdf", {}
                            )
                        mock_mongodb_adapter.insert_one.assert_not_called()
                        mock_to_thread.assert_not_awaited()

                    async def test_add_pdf_document_chunking_error(
                        self,
                        mock_to_thread,
                        mock_bytesio,
                        mock_pdfreader_cls,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                        mock_pdf_reader_instance,  # Need successful read
                    ):
                        # Arrange
                        mock_bytesio.return_value = MagicMock()
                        mock_pdfreader_cls.return_value = mock_pdf_reader_instance
                        mock_to_thread.side_effect = Exception(
                            "Chunking Error"
                        )  # Error during chunking
                        mock_mongodb_adapter.insert_one = (
                            MagicMock()
                        )  # Parent insert succeeds

                        # Act & Assert
                        with pytest.raises(Exception, match="Chunking Error"):
                            await knowledge_base_service.add_pdf_document(
                                b"good pdf", {}
                            )
                        mock_mongodb_adapter.insert_one.assert_called_once()  # Parent was inserted
                        # Embedding and subsequent steps should not happen
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_not_awaited()
                        mock_mongodb_adapter.insert_many.assert_not_called()
                        mock_pinecone_adapter.upsert.assert_not_awaited()

                    async def test_add_pdf_document_embedding_error(
                        self,
                        mock_to_thread,
                        mock_bytesio,
                        mock_pdfreader_cls,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                        mock_pdf_reader_instance,
                        mock_splitter_nodes,  # Need successful read & chunk
                    ):
                        # Arrange
                        mock_bytesio.return_value = MagicMock()
                        mock_pdfreader_cls.return_value = mock_pdf_reader_instance
                        mock_to_thread.return_value = mock_splitter_nodes
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding_batch.side_effect = Exception(
                            "Embedding Batch Error"
                        )
                        mock_mongodb_adapter.insert_one = (
                            MagicMock()
                        )  # Parent insert succeeds

                        # Act & Assert
                        with pytest.raises(Exception, match="Embedding Batch Error"):
                            await knowledge_base_service.add_pdf_document(
                                b"good pdf", {}
                            )
                        mock_mongodb_adapter.insert_one.assert_called_once()  # Parent was inserted
                        mock_to_thread.assert_awaited_once()  # Chunking happened
                        # Mongo/Pinecone chunk operations should not happen
                        mock_mongodb_adapter.insert_many.assert_not_called()
                        mock_pinecone_adapter.upsert.assert_not_awaited()

                    async def test_add_pdf_document_mongo_chunk_insert_error(
                        self,
                        mock_to_thread,
                        mock_bytesio,
                        mock_pdfreader_cls,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                        mock_pdf_reader_instance,
                        mock_splitter_nodes,
                    ):
                        # Arrange - Simulate successful steps up to Mongo chunk insert, which fails
                        pdf_bytes = b"dummy pdf bytes"
                        metadata = {"source": "pdf_source"}
                        namespace = "pdf_namespace"
                        expected_parent_id = FIXED_UUID
                        mock_embeddings = [[0.1] * 3072, [0.2] * 3072]

                        mock_bytesio.return_value = MagicMock()
                        mock_pdfreader_cls.return_value = mock_pdf_reader_instance
                        mock_to_thread.return_value = mock_splitter_nodes
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding_batch = AsyncMock(
                            return_value=mock_embeddings
                        )
                        mock_mongodb_adapter.insert_one = MagicMock()  # Parent succeeds
                        mock_mongodb_adapter.insert_many.side_effect = Exception(
                            "Mongo Chunk Insert Error"
                        )  # Chunks fail
                        mock_pinecone_adapter.upsert = (
                            AsyncMock()
                        )  # Pinecone upsert mock

                        # Act - Should not raise, but log error and continue to Pinecone
                        parent_doc_id = await knowledge_base_service.add_pdf_document(
                            pdf_bytes, metadata, namespace=namespace
                        )

                        # Assert
                        assert (
                            parent_doc_id == expected_parent_id
                        )  # Still returns parent ID
                        mock_mongodb_adapter.insert_one.assert_called_once()
                        mock_to_thread.assert_awaited_once()
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_awaited_once()
                        mock_mongodb_adapter.insert_many.assert_called_once()  # Attempted chunk insert
                        # Pinecone upsert should still be attempted
                        mock_pinecone_adapter.upsert.assert_awaited_once()
                        pinecone_call_kwargs = (
                            mock_pinecone_adapter.upsert.call_args.kwargs
                        )
                        assert (
                            len(pinecone_call_kwargs["vectors"]) == 2
                        )  # Correct vectors prepared
                        assert pinecone_call_kwargs["namespace"] == namespace

                    async def test_add_pdf_document_pinecone_upsert_error(
                        self,
                        mock_to_thread,
                        mock_bytesio,
                        mock_pdfreader_cls,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                        mock_pdf_reader_instance,
                        mock_splitter_nodes,
                    ):
                        # Arrange - Simulate successful steps up to Pinecone upsert, which fails
                        pdf_bytes = b"dummy pdf bytes"
                        metadata = {"source": "pdf_source"}
                        namespace = "pdf_namespace"
                        expected_parent_id = FIXED_UUID
                        mock_embeddings = [[0.1] * 3072, [0.2] * 3072]

                        mock_bytesio.return_value = MagicMock()
                        mock_pdfreader_cls.return_value = mock_pdf_reader_instance
                        mock_to_thread.return_value = mock_splitter_nodes
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding_batch = AsyncMock(
                            return_value=mock_embeddings
                        )
                        mock_mongodb_adapter.insert_one = MagicMock()  # Parent succeeds
                        mock_mongodb_adapter.insert_many = MagicMock()  # Chunks succeed
                        mock_pinecone_adapter.upsert.side_effect = Exception(
                            "Pinecone Upsert Error"
                        )  # Pinecone fails

                        # Act - Should not raise, but log error and return parent ID
                        parent_doc_id = await knowledge_base_service.add_pdf_document(
                            pdf_bytes, metadata, namespace=namespace
                        )

                        # Assert
                        assert (
                            parent_doc_id == expected_parent_id
                        )  # Still returns parent ID
                        mock_mongodb_adapter.insert_one.assert_called_once()
                        mock_to_thread.assert_awaited_once()
                        knowledge_base_service.semantic_splitter.embed_model.aget_text_embedding_batch.assert_awaited_once()
                        mock_mongodb_adapter.insert_many.assert_called_once()
                        mock_pinecone_adapter.upsert.assert_awaited_once()  # Pinecone upsert was attempted

                # --- Test Class for query ---
                @pytest.mark.asyncio
                class TestKnowledgeBaseServiceQuery:
                    @pytest.fixture
                    def mock_pinecone_query_results(self):
                        # Simulate results for one chunk and one plain doc
                        return [
                            {
                                "id": CHUNK_DOC_ID_1,
                                "score": 0.9,
                                "metadata": {
                                    "document_id": CHUNK_DOC_ID_1,
                                    "is_chunk": True,
                                    "parent_document_id": PARENT_DOC_ID,
                                    "chunk_index": 0,
                                    "source": "pdf_source",
                                    "tags": ["pdf", "test"],
                                },
                            },
                            {
                                "id": PLAIN_DOC_ID,
                                "score": 0.8,
                                "metadata": {
                                    "document_id": PLAIN_DOC_ID,
                                    "is_chunk": False,
                                    "parent_document_id": None,
                                    "source": "text_source",
                                    "tags": ["text"],
                                },
                            },
                        ]

                    @pytest.fixture
                    def mock_mongo_find_results(self):
                        # Corresponding Mongo docs for the Pinecone results + the parent
                        return [
                            {  # Chunk doc
                                "_id": "mongo_chunk_1",
                                "document_id": CHUNK_DOC_ID_1,
                                "content": "Content of chunk 1.",
                                "is_chunk": True,
                                "parent_document_id": PARENT_DOC_ID,
                                "chunk_index": 0,
                                "source": "pdf_source",  # May differ slightly from pinecone meta if needed
                                "tags": ["pdf", "test"],
                                "created_at": FIXED_DATETIME,
                            },
                            {  # Plain doc
                                "_id": "mongo_plain_1",
                                "document_id": PLAIN_DOC_ID,
                                "content": "Content of plain doc.",
                                "is_chunk": False,
                                "parent_document_id": None,
                                "source": "text_source",
                                "tags": ["text"],
                                "created_at": FIXED_DATETIME,
                                "other_meta": "value",
                            },
                            {  # Parent doc for the chunk
                                "_id": "mongo_parent_1",
                                "document_id": PARENT_DOC_ID,
                                "content": "Full PDF text.",  # Not usually used in result meta
                                "is_chunk": False,
                                "parent_document_id": None,
                                "source": "pdf_source",
                                "tags": ["pdf", "test"],
                                "created_at": FIXED_DATETIME,
                                "parent_only_meta": "parent_value",
                            },
                        ]

                    async def test_query_success_default_options(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                        mock_pinecone_query_results,
                        mock_mongo_find_results,
                    ):
                        # Arrange
                        query_text = "search for something"
                        query_vector = [0.5] * 3072
                        top_k = 2
                        knowledge_base_service.pinecone.use_reranking = (
                            False  # No reranking
                        )

                        # Mock dependencies
                        knowledge_base_service.semantic_splitter.embed_model.aget_query_embedding = AsyncMock(
                            return_value=query_vector
                        )
                        mock_pinecone_adapter.query = AsyncMock(
                            return_value=mock_pinecone_query_results
                        )
                        mock_mongodb_adapter.find = MagicMock(
                            return_value=mock_mongo_find_results
                        )

                        # Act
                        results = await knowledge_base_service.query(
                            query_text, top_k=top_k
                        )

                        # Assert
                        # Embedding
                        knowledge_base_service.semantic_splitter.embed_model.aget_query_embedding.assert_awaited_once_with(
                            query_text
                        )
                        # Pinecone Query
                        mock_pinecone_adapter.query.assert_awaited_once_with(
                            vector=query_vector,
                            filter=None,
                            top_k=top_k,  # Initial k matches top_k when no rerank
                            namespace=None,
                            include_values=False,
                            include_metadata=True,
                        )
                        # Mongo Find
                        mock_mongodb_adapter.find.assert_called_once()
                        mongo_call_args = mock_mongodb_adapter.find.call_args[0]
                        expected_mongo_ids = sorted(
                            [CHUNK_DOC_ID_1, PLAIN_DOC_ID, PARENT_DOC_ID]
                        )
                        assert (
                            sorted(mongo_call_args[1]["document_id"]["$in"])
                            == expected_mongo_ids
                        )

                        # Results structure
                        assert len(results) == 2
                        # Chunk result
                        assert results[0]["document_id"] == CHUNK_DOC_ID_1
                        assert results[0]["score"] == 0.9
                        assert results[0]["is_chunk"] is True
                        assert results[0]["parent_document_id"] == PARENT_DOC_ID
                        assert results[0]["content"] == "Content of chunk 1."
                        assert (
                            "parent_only_meta" in results[0]["metadata"]
                        )  # Inherited from parent
                        assert (
                            results[0]["metadata"]["parent_only_meta"] == "parent_value"
                        )
                        assert (
                            "chunk_index" in results[0]["metadata"]
                        )  # From Pinecone meta
                        assert results[0]["metadata"]["chunk_index"] == 0
                        assert (
                            results[0]["metadata"]["source"] == "pdf_source"
                        )  # From Pinecone meta
                        # Plain doc result
                        assert results[1]["document_id"] == PLAIN_DOC_ID
                        assert results[1]["score"] == 0.8
                        assert results[1]["is_chunk"] is False
                        assert results[1]["parent_document_id"] is None
                        assert results[1]["content"] == "Content of plain doc."
                        assert (
                            "other_meta" in results[1]["metadata"]
                        )  # From its own Mongo doc
                        assert results[1]["metadata"]["other_meta"] == "value"
                        assert (
                            results[1]["metadata"]["source"] == "text_source"
                        )  # From Pinecone meta

                    async def test_query_success_with_rerank_filter_namespace_no_content(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                        mock_pinecone_query_results,
                        mock_mongo_find_results,
                    ):
                        # Arrange
                        query_text = "search with options"
                        query_vector = [0.6] * 3072
                        top_k = 1  # Final desired K
                        namespace = "query_ns"
                        filter = {"tags": "test"}
                        knowledge_base_service.pinecone.use_reranking = True
                        knowledge_base_service.pinecone.initial_query_top_k_multiplier = 3  # Example multiplier
                        initial_k = (
                            top_k
                            * knowledge_base_service.pinecone.initial_query_top_k_multiplier
                        )  # 1 * 3 = 3

                        # Mock dependencies (Pinecone adapter handles reranking internally, so mock returns final results)
                        knowledge_base_service.semantic_splitter.embed_model.aget_query_embedding = AsyncMock(
                            return_value=query_vector
                        )
                        # Assume Pinecone adapter returns only the top_k=1 result after reranking
                        mock_pinecone_adapter.query = AsyncMock(
                            return_value=mock_pinecone_query_results[:1]
                        )
                        mock_mongodb_adapter.find = MagicMock(
                            return_value=mock_mongo_find_results
                        )  # Mongo still needs all potential docs

                        # Act
                        results = await knowledge_base_service.query(
                            query_text,
                            filter=filter,
                            top_k=top_k,
                            namespace=namespace,
                            include_content=False,
                        )

                        # Assert
                        # Embedding
                        knowledge_base_service.semantic_splitter.embed_model.aget_query_embedding.assert_awaited_once_with(
                            query_text
                        )
                        # Pinecone Query
                        mock_pinecone_adapter.query.assert_awaited_once_with(
                            vector=query_vector,
                            filter=filter,
                            top_k=initial_k,  # Called with initial_k
                            namespace=namespace,
                            include_values=False,
                            include_metadata=True,
                        )
                        # Mongo Find (only needs IDs for the single result + its parent)
                        mock_mongodb_adapter.find.assert_called_once()
                        mongo_call_args = mock_mongodb_adapter.find.call_args[0]
                        expected_mongo_ids = sorted([CHUNK_DOC_ID_1, PARENT_DOC_ID])
                        assert (
                            sorted(mongo_call_args[1]["document_id"]["$in"])
                            == expected_mongo_ids
                        )

                        # Results structure
                        assert len(results) == 1
                        assert results[0]["document_id"] == CHUNK_DOC_ID_1
                        assert results[0]["score"] == 0.9
                        assert "content" not in results[0]  # include_content=False
                        assert (
                            "metadata" in results[0]
                        )  # include_metadata=True (default)

                    async def test_query_embedding_fails(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                    ):
                        # Arrange
                        knowledge_base_service.semantic_splitter.embed_model.aget_query_embedding.side_effect = Exception(
                            "Query Embedding Error"
                        )
                        mock_pinecone_adapter.query = AsyncMock()

                        # Act
                        results = await knowledge_base_service.query("test")

                        # Assert
                        assert results == []
                        mock_pinecone_adapter.query.assert_not_awaited()

                    async def test_query_pinecone_fails(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                    ):
                        # Arrange
                        knowledge_base_service.semantic_splitter.embed_model.aget_query_embedding = AsyncMock(
                            return_value=[0.1] * 3072
                        )
                        mock_pinecone_adapter.query.side_effect = Exception(
                            "Pinecone Query Error"
                        )
                        mock_mongodb_adapter.find = MagicMock()

                        # Act
                        results = await knowledge_base_service.query("test")

                        # Assert
                        assert results == []
                        knowledge_base_service.semantic_splitter.embed_model.aget_query_embedding.assert_awaited_once()
                        mock_pinecone_adapter.query.assert_awaited_once()
                        mock_mongodb_adapter.find.assert_not_called()

                    async def test_query_mongo_fails(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                        mock_pinecone_query_results,
                    ):
                        # Arrange
                        knowledge_base_service.semantic_splitter.embed_model.aget_query_embedding = AsyncMock(
                            return_value=[0.1] * 3072
                        )
                        mock_pinecone_adapter.query = AsyncMock(
                            return_value=mock_pinecone_query_results
                        )
                        mock_mongodb_adapter.find.side_effect = Exception(
                            "Mongo Find Error"
                        )

                        # Act - Should still proceed but results might be incomplete or empty
                        results = await knowledge_base_service.query("test")

                        # Assert
                        assert (
                            results == []
                        )  # Fails because mongo_docs_map is empty, causing skips
                        knowledge_base_service.semantic_splitter.embed_model.aget_query_embedding.assert_awaited_once()
                        mock_pinecone_adapter.query.assert_awaited_once()
                        mock_mongodb_adapter.find.assert_called_once()  # Mongo find was attempted

                    async def test_query_mongo_doc_missing(
                        self,
                        knowledge_base_service: KnowledgeBaseService,
                        mock_mongodb_adapter: MagicMock,
                        mock_pinecone_adapter: MagicMock,
                        mock_pinecone_query_results,
                        mock_mongo_find_results,
                    ):
                        # Arrange: Remove one doc from the Mongo results to simulate inconsistency
                        missing_mongo_results = [
                            doc
                            for doc in mock_mongo_find_results
                            if doc["document_id"] != PLAIN_DOC_ID
                        ]

                        knowledge_base_service.semantic_splitter.embed_model.aget_query_embedding = AsyncMock(
                            return_value=[0.1] * 3072
                        )
                        mock_pinecone_adapter.query = AsyncMock(
                            return_value=mock_pinecone_query_results
                        )
                        mock_mongodb_adapter.find = MagicMock(
                            return_value=missing_mongo_results
                        )

                        # Act
                        results = await knowledge_base_service.query("test")

                        # Assert
                        assert len(results) == 1  # Only the chunk result should remain
                        assert results[0]["document_id"] == CHUNK_DOC_ID_1
                        # Verify calls were made
                        knowledge_base_service.semantic_splitter.embed_model.aget_query_embedding.assert_awaited_once()
                        mock_pinecone_adapter.query.assert_awaited_once()
                        mock_mongodb_adapter.find.assert_called_once()

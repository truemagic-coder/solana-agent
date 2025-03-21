"""
Tests for MongoDB adapter implementation.

This module contains unit tests for MongoDBAdapter.
"""
import pytest
from unittest.mock import MagicMock, patch

from solana_agent.adapters.mongodb_adapter import MongoDBAdapter


@pytest.fixture
def mock_mongo_client():
    """Create a mock MongoDB client."""
    with patch('solana_agent.adapters.mongodb_adapter.MongoClient') as mock_client:
        # Setup mock database and collections
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db

        # Return both the client and database mocks for flexibility in tests
        yield mock_client, mock_db


@pytest.fixture
def mongodb_adapter(mock_mongo_client):
    """Create a MongoDB adapter with mocked client."""
    mock_client, _ = mock_mongo_client
    return MongoDBAdapter(connection_string="mongodb://localhost:27017", database_name="test_db")


def test_init(mock_mongo_client):
    """Test MongoDBAdapter initialization."""
    mock_client, _ = mock_mongo_client

    adapter = MongoDBAdapter(
        connection_string="mongodb://localhost:27017", database_name="test_db")

    # Verify client was created with correct connection string
    mock_client.assert_called_once_with("mongodb://localhost:27017")

    # Verify database was accessed with correct name
    mock_client.return_value.__getitem__.assert_called_once_with("test_db")


def test_create_collection(mongodb_adapter, mock_mongo_client):
    """Test creating a collection if it doesn't exist."""
    _, mock_db = mock_mongo_client

    # Configure mock to say collection doesn't exist
    mock_db.list_collection_names.return_value = ["existing_collection"]

    # Call method
    mongodb_adapter.create_collection("new_collection")

    # Verify collection creation was called
    mock_db.create_collection.assert_called_once_with("new_collection")


def test_create_collection_already_exists(mongodb_adapter, mock_mongo_client):
    """Test creating a collection that already exists."""
    _, mock_db = mock_mongo_client

    # Configure mock to say collection already exists
    mock_db.list_collection_names.return_value = ["existing_collection"]

    # Call method
    mongodb_adapter.create_collection("existing_collection")

    # Verify collection creation was not called
    mock_db.create_collection.assert_not_called()


def test_collection_exists(mongodb_adapter, mock_mongo_client):
    """Test checking if a collection exists."""
    _, mock_db = mock_mongo_client

    # Configure mock
    mock_db.list_collection_names.return_value = ["existing_collection"]

    # Verify results
    assert mongodb_adapter.collection_exists("existing_collection") is True
    assert mongodb_adapter.collection_exists(
        "non_existing_collection") is False


def test_insert_one_with_id(mongodb_adapter, mock_mongo_client):
    """Test inserting a document with pre-defined ID."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Call method
    document = {"_id": "existing_id", "name": "Test"}
    result_id = mongodb_adapter.insert_one("test_collection", document)

    # Verify results
    mock_collection.insert_one.assert_called_once_with(document)
    assert result_id == "existing_id"


def test_insert_one_without_id(mongodb_adapter, mock_mongo_client):
    """Test inserting a document without pre-defined ID."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Use patch to control UUID generation
    with patch('solana_agent.adapters.mongodb_adapter.uuid.uuid4', return_value="generated_uuid"):
        # Call method
        document = {"name": "Test"}
        result_id = mongodb_adapter.insert_one("test_collection", document)

        # Verify results
        assert "_id" in document
        assert document["_id"] == "generated_uuid"
        mock_collection.insert_one.assert_called_once_with(document)
        assert result_id == "generated_uuid"


def test_find_one(mongodb_adapter, mock_mongo_client):
    """Test finding a single document."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Configure mock response
    expected_doc = {"_id": "doc1", "name": "Test"}
    mock_collection.find_one.return_value = expected_doc

    # Call method
    query = {"name": "Test"}
    result = mongodb_adapter.find_one("test_collection", query)

    # Verify results
    mock_collection.find_one.assert_called_once_with(query)
    assert result == expected_doc


def test_find_one_no_results(mongodb_adapter, mock_mongo_client):
    """Test finding a single document with no results."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Configure mock response for no results
    mock_collection.find_one.return_value = None

    # Call method
    query = {"name": "NonExistent"}
    result = mongodb_adapter.find_one("test_collection", query)

    # Verify results
    mock_collection.find_one.assert_called_once_with(query)
    assert result is None


def test_find(mongodb_adapter, mock_mongo_client):
    """Test finding multiple documents."""
    _, mock_db = mock_mongo_client

    # Create mock collection and cursor
    mock_collection = MagicMock()
    mock_cursor = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_collection.find.return_value = mock_cursor

    # Configure mock cursor to return list of documents
    expected_docs = [{"_id": "doc1", "name": "Test1"},
                     {"_id": "doc2", "name": "Test2"}]
    mock_cursor.__iter__.return_value = iter(expected_docs)

    # Call method
    query = {"category": "test"}
    results = mongodb_adapter.find("test_collection", query)

    # Verify results
    mock_collection.find.assert_called_once_with(query)
    assert results == expected_docs


def test_find_with_sort(mongodb_adapter, mock_mongo_client):
    """Test finding documents with sorting."""
    _, mock_db = mock_mongo_client

    # Create mock collection and cursor
    mock_collection = MagicMock()
    mock_cursor = MagicMock()
    sorted_cursor = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_collection.find.return_value = mock_cursor
    mock_cursor.sort.return_value = sorted_cursor

    # Configure mock cursor to return list of documents
    expected_docs = [{"_id": "doc1", "name": "Test1"},
                     {"_id": "doc2", "name": "Test2"}]
    sorted_cursor.__iter__.return_value = iter(expected_docs)

    # Call method
    query = {"category": "test"}
    sort_params = [("name", 1)]
    results = mongodb_adapter.find("test_collection", query, sort=sort_params)

    # Verify results
    mock_collection.find.assert_called_once_with(query)
    mock_cursor.sort.assert_called_once_with(sort_params)
    assert results == expected_docs


def test_find_with_limit(mongodb_adapter, mock_mongo_client):
    """Test finding documents with limit."""
    _, mock_db = mock_mongo_client

    # Create mock collection and cursor
    mock_collection = MagicMock()
    mock_cursor = MagicMock()
    limited_cursor = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_collection.find.return_value = mock_cursor
    mock_cursor.limit.return_value = limited_cursor

    # Configure mock cursor to return list of documents
    expected_docs = [{"_id": "doc1", "name": "Test1"}]
    limited_cursor.__iter__.return_value = iter(expected_docs)

    # Call method
    query = {"category": "test"}
    results = mongodb_adapter.find("test_collection", query, limit=1)

    # Verify results
    mock_collection.find.assert_called_once_with(query)
    mock_cursor.limit.assert_called_once_with(1)
    assert results == expected_docs


def test_find_with_sort_and_limit(mongodb_adapter, mock_mongo_client):
    """Test finding documents with both sorting and limit."""
    _, mock_db = mock_mongo_client

    # Create mock collection and cursor
    mock_collection = MagicMock()
    mock_cursor = MagicMock()
    sorted_cursor = MagicMock()
    limited_cursor = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_collection.find.return_value = mock_cursor
    mock_cursor.sort.return_value = sorted_cursor
    sorted_cursor.limit.return_value = limited_cursor

    # Configure mock cursor to return list of documents
    expected_docs = [{"_id": "doc1", "name": "Test1"}]
    limited_cursor.__iter__.return_value = iter(expected_docs)

    # Call method
    query = {"category": "test"}
    sort_params = [("name", 1)]
    results = mongodb_adapter.find(
        "test_collection", query, sort=sort_params, limit=1)

    # Verify results
    mock_collection.find.assert_called_once_with(query)
    mock_cursor.sort.assert_called_once_with(sort_params)
    sorted_cursor.limit.assert_called_once_with(1)
    assert results == expected_docs


def test_update_one(mongodb_adapter, mock_mongo_client):
    """Test updating a single document."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Configure mock response
    mock_result = MagicMock()
    mock_result.modified_count = 1
    mock_result.upserted_id = None
    mock_collection.update_one.return_value = mock_result

    # Call method
    query = {"_id": "doc1"}
    update = {"$set": {"status": "updated"}}
    success = mongodb_adapter.update_one("test_collection", query, update)

    # Verify results
    mock_collection.update_one.assert_called_once_with(
        query, update, upsert=False)
    assert success is True


def test_update_one_no_matches(mongodb_adapter, mock_mongo_client):
    """Test updating a single document with no matches."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Configure mock response for no matches
    mock_result = MagicMock()
    mock_result.modified_count = 0
    mock_result.upserted_id = None
    mock_collection.update_one.return_value = mock_result

    # Call method
    query = {"_id": "nonexistent"}
    update = {"$set": {"status": "updated"}}
    success = mongodb_adapter.update_one("test_collection", query, update)

    # Verify results
    mock_collection.update_one.assert_called_once_with(
        query, update, upsert=False)
    assert success is False


def test_update_one_with_upsert(mongodb_adapter, mock_mongo_client):
    """Test updating a single document with upsert."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Configure mock response for upsert
    mock_result = MagicMock()
    mock_result.modified_count = 0
    mock_result.upserted_id = "new_doc_id"
    mock_collection.update_one.return_value = mock_result

    # Call method
    query = {"_id": "nonexistent"}
    update = {"$set": {"status": "created"}}
    success = mongodb_adapter.update_one(
        "test_collection", query, update, upsert=True)

    # Verify results
    mock_collection.update_one.assert_called_once_with(
        query, update, upsert=True)
    assert success is True


def test_delete_one(mongodb_adapter, mock_mongo_client):
    """Test deleting a single document."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Configure mock response
    mock_result = MagicMock()
    mock_result.deleted_count = 1
    mock_collection.delete_one.return_value = mock_result

    # Call method
    query = {"_id": "doc_to_delete"}
    success = mongodb_adapter.delete_one("test_collection", query)

    # Verify results
    mock_collection.delete_one.assert_called_once_with(query)
    assert success is True


def test_delete_one_no_matches(mongodb_adapter, mock_mongo_client):
    """Test deleting a single document with no matches."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Configure mock response for no matches
    mock_result = MagicMock()
    mock_result.deleted_count = 0
    mock_collection.delete_one.return_value = mock_result

    # Call method
    query = {"_id": "nonexistent"}
    success = mongodb_adapter.delete_one("test_collection", query)

    # Verify results
    mock_collection.delete_one.assert_called_once_with(query)
    assert success is False


def test_count_documents(mongodb_adapter, mock_mongo_client):
    """Test counting documents in a collection."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Configure mock response
    mock_collection.count_documents.return_value = 5

    # Call method
    query = {"status": "active"}
    count = mongodb_adapter.count_documents("test_collection", query)

    # Verify results
    mock_collection.count_documents.assert_called_once_with(query)
    assert count == 5


def test_aggregate(mongodb_adapter, mock_mongo_client):
    """Test aggregation pipeline execution."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Configure mock cursor to return aggregation results
    expected_results = [{"_id": "group1", "count": 5},
                        {"_id": "group2", "count": 3}]
    mock_cursor = MagicMock()
    mock_cursor.__iter__.return_value = iter(expected_results)
    mock_collection.aggregate.return_value = mock_cursor

    # Call method
    pipeline = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    results = mongodb_adapter.aggregate("test_collection", pipeline)

    # Verify results
    mock_collection.aggregate.assert_called_once_with(pipeline)
    assert results == expected_results


def test_create_index(mongodb_adapter, mock_mongo_client):
    """Test creating an index."""
    _, mock_db = mock_mongo_client

    # Create mock collection
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Call method with additional parameters
    keys = [("name", 1), ("created_at", -1)]
    mongodb_adapter.create_index(
        "test_collection", keys, unique=True, background=True)

    # Verify results
    mock_collection.create_index.assert_called_once_with(
        keys, unique=True, background=True)


def test_find_with_skip(mongodb_adapter, mock_mongo_client):
    """Test finding documents with skip."""
    _, mock_db = mock_mongo_client

    # Create mock collection and cursor chain
    mock_collection = MagicMock()
    mock_cursor = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_collection.find.return_value = mock_cursor
    # Important: cursor returns self for chaining
    mock_cursor.skip.return_value = mock_cursor

    # Configure mock cursor to return list of documents
    expected_docs = [{"_id": "doc2", "name": "Test2"}]
    mock_cursor.__iter__.return_value = iter(expected_docs)

    # Call method
    query = {"category": "test"}
    results = mongodb_adapter.find("test_collection", query, skip=1)

    # Verify results
    mock_collection.find.assert_called_once_with(query)
    mock_cursor.skip.assert_called_once_with(1)
    assert results == expected_docs


def test_find_with_skip_sort_and_limit(mongodb_adapter, mock_mongo_client):
    """Test finding documents with skip, sort, and limit combined."""
    _, mock_db = mock_mongo_client

    # Create mock collection and cursor chain
    mock_collection = MagicMock()
    mock_cursor = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_collection.find.return_value = mock_cursor

    # Configure cursor to return self for method chaining
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.skip.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor

    # Configure mock cursor to return list of documents
    expected_docs = [{"_id": "doc2", "name": "Test2"}]
    mock_cursor.__iter__.return_value = iter(expected_docs)

    # Call method
    query = {"category": "test"}
    sort_params = [("name", 1)]
    results = mongodb_adapter.find(
        "test_collection",
        query,
        sort=sort_params,
        skip=1,
        limit=1
    )

    # Verify cursor method calls in any order
    mock_collection.find.assert_called_once_with(query)
    mock_cursor.sort.assert_called_once_with(sort_params)
    mock_cursor.skip.assert_called_once_with(1)
    mock_cursor.limit.assert_called_once_with(1)
    assert results == expected_docs

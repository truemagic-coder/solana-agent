import os
import uuid
import pytest
import mongomock
from pymongo import MongoClient, ASCENDING, DESCENDING

from solana_agent.adapters.mongodb_adapter import MongoDBAdapter


@pytest.fixture
def mongo_client():
    """Fixture for MongoDB client (mock or real based on env var)."""
    if os.environ.get("MONGODB_REAL") == "1":
        # Use real MongoDB for integration tests
        client = MongoClient("mongodb://localhost:27017/")
        db = client["test_db"]
        yield client, db
        # Cleanup after tests
        client.drop_database("test_db")
    else:
        # Use mongomock for unit tests
        client = mongomock.MongoClient()
        db = client["test_db"]
        yield client, db


@pytest.fixture
def mongodb_adapter(mongo_client):
    """Fixture for MongoDB adapter."""
    client, _ = mongo_client
    adapter = MongoDBAdapter(connection_string=client.HOST, database_name="test_db")
    # Replace the real client with our fixture
    adapter.client = client
    adapter.db = client["test_db"]
    return adapter


@pytest.fixture
def sample_data():
    """Fixture for sample test data."""
    return [
        {"name": "Alice", "age": 30, "tags": ["developer", "python"]},
        {"name": "Bob", "age": 25, "tags": ["designer", "ui"]},
        {"name": "Charlie", "age": 35, "tags": ["manager", "agile"]},
        {"name": "Diana", "age": 28, "tags": ["developer", "javascript"]},
        {"name": "Eve", "age": 32, "tags": ["developer", "python", "data"]},
    ]


class TestMongoDBAdapter:
    """Test suite for MongoDB adapter."""

    def test_init(self, mongo_client):
        """Test initializing the adapter."""
        client, _ = mongo_client
        adapter = MongoDBAdapter(connection_string=client.HOST, database_name="test_db")
        assert adapter.db.name == "test_db"

    def test_create_collection(self, mongodb_adapter):
        """Test creating a collection."""
        mongodb_adapter.create_collection("test_collection")
        assert "test_collection" in mongodb_adapter.db.list_collection_names()

    def test_collection_exists(self, mongodb_adapter):
        """Test checking if a collection exists."""
        mongodb_adapter.create_collection("existing_collection")
        assert mongodb_adapter.collection_exists("existing_collection") is True
        assert mongodb_adapter.collection_exists("non_existing_collection") is False

    def test_insert_one_with_id(self, mongodb_adapter):
        """Test inserting a document with existing ID."""
        document = {"_id": "test_id", "name": "Test"}
        result_id = mongodb_adapter.insert_one("test_collection", document)
        assert result_id == "test_id"
        stored = mongodb_adapter.db["test_collection"].find_one({"_id": "test_id"})
        assert stored["name"] == "Test"

    def test_insert_one_without_id(self, mongodb_adapter):
        """Test inserting a document without ID (should generate UUID)."""
        document = {"name": "Test"}
        result_id = mongodb_adapter.insert_one("test_collection", document)
        assert result_id is not None
        # Verify UUID format
        uuid.UUID(result_id)  # Will raise ValueError if not a valid UUID
        stored = mongodb_adapter.db["test_collection"].find_one({"_id": result_id})
        assert stored["name"] == "Test"

    def test_insert_many(self, mongodb_adapter, sample_data):
        """Test inserting multiple documents."""
        result_ids = mongodb_adapter.insert_many("test_collection", sample_data)
        assert len(result_ids) == len(sample_data)

        # Verify all documents were inserted
        stored_docs = list(mongodb_adapter.db["test_collection"].find())
        assert len(stored_docs) == len(sample_data)

        # Check if the inserted documents match the sample data
        for doc in stored_docs:
            assert doc in sample_data

    def test_insert_many_with_id(self, mongodb_adapter):
        """Test inserting multiple documents with existing IDs."""
        documents = [
            {"_id": "id1", "name": "Test1"},
            {"_id": "id2", "name": "Test2"},
        ]
        result_ids = mongodb_adapter.insert_many("test_collection", documents)
        assert result_ids == ["id1", "id2"]

        # Verify all documents were inserted
        stored_docs = list(mongodb_adapter.db["test_collection"].find())
        assert len(stored_docs) == len(documents)

        # Check if the inserted documents match the sample data
        for doc in stored_docs:
            assert doc in documents

    def test_delete_many(self, mongodb_adapter, sample_data):
        """Test deleting multiple documents."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        # Delete all developers
        result = mongodb_adapter.delete_many("test_collection", {"tags": "developer"})
        assert result.deleted_count == 3

        remaining = mongodb_adapter.find("test_collection", {})
        assert len(remaining) == 2

    def test_find_one_existing(self, mongodb_adapter, sample_data):
        """Test finding a single existing document."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        result = mongodb_adapter.find_one("test_collection", {"name": "Alice"})
        assert result is not None
        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_find_one_not_existing(self, mongodb_adapter):
        """Test finding a non-existent document."""
        result = mongodb_adapter.find_one("test_collection", {"name": "NonExistent"})
        assert result is None

    def test_find_all(self, mongodb_adapter, sample_data):
        """Test finding all documents."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        results = mongodb_adapter.find("test_collection", {})
        assert len(results) == len(sample_data)

    def test_find_with_filter(self, mongodb_adapter, sample_data):
        """Test finding documents with filter."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        results = mongodb_adapter.find("test_collection", {"age": {"$gt": 30}})
        assert len(results) == 2
        assert all(doc["age"] > 30 for doc in results)

    def test_find_with_sort(self, mongodb_adapter, sample_data):
        """Test finding documents with sorting."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        results = mongodb_adapter.find("test_collection", {}, sort=[("age", ASCENDING)])
        assert len(results) == len(sample_data)
        assert results[0]["age"] == 25  # Bob is youngest
        assert results[-1]["age"] == 35  # Charlie is oldest

        # Test descending sort
        results = mongodb_adapter.find(
            "test_collection", {}, sort=[("age", DESCENDING)]
        )
        assert results[0]["age"] == 35  # Charlie is oldest
        assert results[-1]["age"] == 25  # Bob is youngest

    def test_find_with_limit(self, mongodb_adapter, sample_data):
        """Test finding documents with limit."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        results = mongodb_adapter.find("test_collection", {}, limit=3)
        assert len(results) == 3

    def test_find_with_skip(self, mongodb_adapter, sample_data):
        """Test finding documents with skip."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        results = mongodb_adapter.find("test_collection", {}, skip=2)
        assert len(results) == len(sample_data) - 2

    def test_find_with_limit_and_skip(self, mongodb_adapter, sample_data):
        """Test finding documents with both limit and skip."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        results = mongodb_adapter.find("test_collection", {}, limit=2, skip=2)
        assert len(results) == 2

    def test_update_one_existing(self, mongodb_adapter, sample_data):
        """Test updating an existing document."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        result = mongodb_adapter.update_one(
            "test_collection", {"name": "Alice"}, {"$set": {"age": 31}}
        )
        assert result is True

        updated = mongodb_adapter.find_one("test_collection", {"name": "Alice"})
        assert updated["age"] == 31

    def test_update_one_non_existing(self, mongodb_adapter):
        """Test updating a non-existent document."""
        result = mongodb_adapter.update_one(
            "test_collection", {"name": "NonExistent"}, {"$set": {"age": 50}}
        )
        assert result is False

    def test_update_one_with_upsert(self, mongodb_adapter):
        """Test upserting a document."""
        result = mongodb_adapter.update_one(
            "test_collection", {"name": "Frank"}, {"$set": {"age": 40}}, upsert=True
        )
        assert result is True

        upserted = mongodb_adapter.find_one("test_collection", {"name": "Frank"})
        assert upserted is not None
        assert upserted["age"] == 40

    def test_delete_one_existing(self, mongodb_adapter, sample_data):
        """Test deleting an existing document."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        result = mongodb_adapter.delete_one("test_collection", {"name": "Alice"})
        assert result is True

        remaining = mongodb_adapter.find("test_collection", {})
        assert len(remaining) == len(sample_data) - 1
        assert all(doc["name"] != "Alice" for doc in remaining)

    def test_delete_one_non_existing(self, mongodb_adapter):
        """Test deleting a non-existent document."""
        result = mongodb_adapter.delete_one("test_collection", {"name": "NonExistent"})
        assert result is False

    def test_delete_all(self, mongodb_adapter, sample_data):
        """Test deleting multiple documents."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        # Delete all developers
        result = mongodb_adapter.delete_all("test_collection", {"tags": "developer"})
        assert result is True

        remaining = mongodb_adapter.find("test_collection", {})
        assert len(remaining) == 2  # Bob and Charlie are not developers
        assert all("developer" not in doc.get("tags", []) for doc in remaining)

    def test_delete_all_empty_result(self, mongodb_adapter):
        """Test deleting with a query that matches no documents."""
        result = mongodb_adapter.delete_all("test_collection", {"name": "NonExistent"})
        assert result is True  # Still returns True as 0 == 0

    def test_count_documents(self, mongodb_adapter, sample_data):
        """Test counting documents."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        count = mongodb_adapter.count_documents("test_collection", {})
        assert count == len(sample_data)

        count_filtered = mongodb_adapter.count_documents(
            "test_collection", {"age": {"$lt": 30}}
        )
        assert count_filtered == 2  # Bob and Diana

    def test_aggregate(self, mongodb_adapter, sample_data):
        """Test aggregation pipeline."""
        for doc in sample_data:
            mongodb_adapter.insert_one("test_collection", doc)

        # Group by age ranges and count
        pipeline = [
            {
                "$group": {
                    "_id": {"$cond": [{"$lt": ["$age", 30]}, "under_30", "30_or_over"]},
                    "count": {"$sum": 1},
                }
            }
        ]

        results = mongodb_adapter.aggregate("test_collection", pipeline)
        assert len(results) == 2

        # Convert to dict for easier assertion
        results_dict = {result["_id"]: result["count"] for result in results}
        assert results_dict["under_30"] == 2  # Bob and Diana
        assert results_dict["30_or_over"] == 3  # Alice, Charlie, Eve

    def test_create_index(self, mongodb_adapter):
        """Test creating an index."""
        mongodb_adapter.create_index(
            "test_collection", [("name", ASCENDING)], unique=True
        )

        # Verify index was created
        indexes = mongodb_adapter.db["test_collection"].index_information()
        assert len(indexes) > 1  # _id index + our new index

        # Find the name index
        name_index = None
        for idx_name, idx_info in indexes.items():
            if idx_name != "_id_" and "name" in dict(idx_info["key"]):
                name_index = idx_info
                break

        assert name_index is not None
        assert name_index["unique"] is True

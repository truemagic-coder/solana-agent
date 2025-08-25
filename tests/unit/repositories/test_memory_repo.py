import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from solana_agent.repositories import memory as memory_module
from solana_agent.repositories.memory import MemoryRepository
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
from solana_agent.adapters.pinecone_adapter import PineconeAdapter


class FakeEmbData:
    def __init__(self, embedding: List[float]):
        self.embedding = embedding


class FakeEmbResp:
    def __init__(self, embedding: List[float]):
        self.data = [FakeEmbData(embedding)]


class FakeOpenAIClient:
    def __init__(self, should_raise_init: bool = False, embedding: List[float] = None):
        if should_raise_init:
            raise RuntimeError("init failed")
        self._embedding = embedding or [0.01, 0.02, 0.03]
        self.embeddings = self

    # Sync method used via run_in_executor
    def create(self, model: str, input: str):
        return FakeEmbResp(self._embedding)


@pytest.fixture
def mock_mongo():
    m = MagicMock(spec=MongoDBAdapter)
    m.create_collection = MagicMock()
    m.create_index = MagicMock()
    m.insert_one = MagicMock(return_value="mongo-id-1")
    m.delete_all = MagicMock()
    m.find = MagicMock(return_value=[])
    m.find_one = MagicMock(return_value=None)
    m.update_one = MagicMock()
    m.count_documents = MagicMock(return_value=5)
    return m


@pytest.fixture
def mock_pinecone():
    pc = MagicMock(spec=PineconeAdapter)
    pc.upsert = AsyncMock()
    pc.query = AsyncMock(return_value=[])
    return pc


def test_init_without_mongo():
    repo = MemoryRepository()
    assert repo.mongo is None
    assert repo.collection is None
    assert repo.captures_collection == "captures"
    # OpenAI client is None by default when no key supplied
    assert repo._openai_client is None


def test_init_with_mongo_creates_indexes_and_handles_partial_index_error(mock_mongo):
    # Cause only the partial unique index to fail
    def create_index_side_effect(collection, keys, **kwargs):
        if kwargs.get("unique") and kwargs.get("partialFilterExpression"):
            raise Exception("partial unique index error")
        return None

    mock_mongo.create_index.side_effect = create_index_side_effect
    repo = MemoryRepository(mongo_adapter=mock_mongo)
    assert repo.mongo is mock_mongo
    assert repo.collection == "conversations"
    assert repo.captures_collection == "captures"
    # Conversations collection/indexes
    mock_mongo.create_collection.assert_any_call("conversations")
    # Captures collection created
    mock_mongo.create_collection.assert_any_call("captures")


def test_openai_client_init_failure_is_handled(mock_mongo, monkeypatch):
    # Patch OpenAI class in module namespace to raise on init
    class BadOpenAI:
        def __init__(self, api_key: str):
            raise RuntimeError("boom")

    monkeypatch.setattr(memory_module, "OpenAI", BadOpenAI, raising=True)
    repo = MemoryRepository(mongo_adapter=mock_mongo, openai_api_key="key")
    assert repo._openai_client is None  # gracefully handled


@pytest.mark.asyncio
async def test_store_validation_errors():
    repo = MemoryRepository()
    with pytest.raises(ValueError):
        await repo.store("", [{"role": "user", "content": "hi"}])
    with pytest.raises(ValueError):
        await repo.store("u1", None)  # type: ignore
    with pytest.raises(ValueError):
        await repo.store("u1", [])
    with pytest.raises(ValueError):
        await repo.store("u1", [{"role": "user"}])  # missing content
    with pytest.raises(ValueError):
        await repo.store("u1", [{"content": "hi"}])  # missing role
    with pytest.raises(ValueError):
        await repo.store("u1", [{"role": "tool", "content": "x"}])  # invalid role


@pytest.mark.asyncio
async def test_store_inserts_pair_and_indexes(mock_mongo, monkeypatch):
    repo = MemoryRepository(mongo_adapter=mock_mongo)
    # Patch out _maybe_index_pair to observe call
    repo._maybe_index_pair = AsyncMock()

    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    await repo.store("user123", msgs)

    mock_mongo.insert_one.assert_called_once()
    args, kwargs = mock_mongo.insert_one.call_args
    assert args[0] == "conversations"
    doc = args[1]
    assert doc["user_id"] == "user123"
    assert doc["user_message"] == "Hello"
    assert doc["assistant_message"] == "Hi there"
    assert isinstance(doc["timestamp"], datetime)

    repo._maybe_index_pair.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_no_pair_no_insert(mock_mongo):
    repo = MemoryRepository(mongo_adapter=mock_mongo)
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Another user"},
    ]
    await repo.store("user123", msgs)
    mock_mongo.insert_one.assert_not_called()


@pytest.mark.asyncio
async def test_store_mongo_error_is_logged(mock_mongo, caplog):
    mock_mongo.insert_one.side_effect = Exception("db down")
    repo = MemoryRepository(mongo_adapter=mock_mongo)
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    with caplog.at_level("ERROR"):
        await repo.store("user123", msgs)
        assert any("MongoDB storage error" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_retrieve_returns_empty_string():
    repo = MemoryRepository()
    out = await repo.retrieve("anything")
    assert out == ""


@pytest.mark.asyncio
async def test_delete_handles_normal_and_empty_user(mock_mongo):
    repo = MemoryRepository(mongo_adapter=mock_mongo)
    await repo.delete("userA")
    await repo.delete("")
    mock_mongo.delete_all.assert_any_call("conversations", {"user_id": "userA"})
    mock_mongo.delete_all.assert_any_call("conversations", {"user_id": "user123"})


def test_count_documents_with_and_without_mongo(mock_mongo):
    repo_none = MemoryRepository()
    assert repo_none.count_documents("x", {}) == 0
    repo = MemoryRepository(mongo_adapter=mock_mongo)
    mock_mongo.count_documents.return_value = 7
    assert repo.count_documents("x", {}) == 7


def test_truncate_variants():
    repo = MemoryRepository()
    assert repo._truncate("") == ""
    assert repo._truncate("short") == "short"
    # sentence boundary
    long = "First sentence. Second sentence that is longer."
    assert repo._truncate(long, limit=20) == "First sentence."
    # no period before limit -> ellipsis
    text = "a" * 3000
    out = repo._truncate(text)
    assert out.endswith("...")
    with pytest.raises(AttributeError):
        repo._truncate(None)  # type: ignore


@pytest.mark.asyncio
async def test__maybe_index_pair_skips_when_unconfigured(mock_mongo):
    repo = MemoryRepository(mongo_adapter=mock_mongo)  # no OpenAI/Pinecone
    # Should simply return without error
    await repo._maybe_index_pair("u", "uh", "ah", datetime.now(timezone.utc))


@pytest.mark.asyncio
async def test__maybe_index_pair_happy_path(monkeypatch):
    # Configure repo with fake OpenAI and Pinecone
    repo = MemoryRepository(
        pinecone_adapter=MagicMock(spec=PineconeAdapter),
        openai_api_key="KEY",
    )
    # Inject fake OpenAI client directly
    repo._openai_client = FakeOpenAIClient(embedding=[0.1, 0.2, 0.3])
    # Mock pinecone upsert
    pc = MagicMock()
    pc.upsert = AsyncMock()
    repo.pinecone = pc

    ts = datetime.now(timezone.utc)
    await repo._maybe_index_pair("userX", "hello world", "hi user", ts)

    repo.pinecone.upsert.assert_awaited_once()
    args, kwargs = repo.pinecone.upsert.call_args
    vectors = kwargs["vectors"]
    assert len(vectors) == 2
    assert vectors[0]["metadata"]["role"] == "user"
    assert vectors[1]["metadata"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_add_message_validation_errors():
    repo = MemoryRepository()
    with pytest.raises(ValueError):
        await repo.add_message("", "user", "x")
    with pytest.raises(ValueError):
        await repo.add_message("s1", "badrole", "x")
    with pytest.raises(ValueError):
        await repo.add_message("s1", "user", "")


@pytest.mark.asyncio
async def test_add_message_stores_in_mongo_and_schedules_task(mock_mongo, monkeypatch):
    repo = MemoryRepository(mongo_adapter=mock_mongo, store_temporal_in_mongo=True)
    # Ensure messages collection is created; exercise _ensure_messages_collection path
    assert repo._messages_ready is False
    # Intercept create_task to verify call
    created_tasks: Dict[str, Any] = {}

    def fake_create_task(coro):
        created_tasks["task"] = coro
        # Return a dummy Task
        loop = asyncio.get_running_loop()
        return loop.create_task(asyncio.sleep(0))

    monkeypatch.setattr(memory_module.asyncio, "create_task", fake_create_task)

    mongo_id = await repo.add_message("sess1", "user", "I need a laptop.")
    assert mongo_id == "mongo-id-1"
    # Indexes created lazily
    mock_mongo.create_collection.assert_any_call("messages")
    mock_mongo.create_index.assert_any_call(
        "messages", [("session_id", 1), ("timestamp", -1)]
    )
    assert repo._messages_ready is True
    # create_task invoked
    assert "task" in created_tasks


@pytest.mark.asyncio
async def test__embed_and_upsert_message_happy_path(monkeypatch):
    repo = MemoryRepository(
        pinecone_adapter=MagicMock(spec=PineconeAdapter),
        openai_api_key="KEY",
    )
    repo._openai_client = FakeOpenAIClient(embedding=[0.9, 0.8])
    pc = MagicMock()
    pc.upsert = AsyncMock()
    repo.pinecone = pc

    ts = datetime.now(timezone.utc)
    await repo._embed_and_upsert_message(
        session_id="s1",
        role="assistant",
        content="answer",
        ts=ts,
        mongo_id="64f0c0ffee0c0ffee0c0ffee",
    )
    repo.pinecone.upsert.assert_awaited_once()
    args, kwargs = repo.pinecone.upsert.call_args
    vectors = kwargs["vectors"]
    assert vectors[0]["id"] == "64f0c0ffee0c0ffee0c0ffee"
    assert vectors[0]["metadata"]["role"] == "assistant"


@pytest.mark.asyncio
async def test__embed_and_upsert_message_error_is_swallowed(monkeypatch):
    repo = MemoryRepository(
        pinecone_adapter=MagicMock(spec=PineconeAdapter),
        openai_api_key="KEY",
    )
    repo._openai_client = FakeOpenAIClient(embedding=[0.9, 0.8])
    pc = MagicMock()
    pc.upsert = AsyncMock(side_effect=Exception("pc down"))
    repo.pinecone = pc

    ts = datetime.now(timezone.utc)
    # Should not raise
    await repo._embed_and_upsert_message("s1", "user", "text", ts, None)


@pytest.mark.asyncio
async def test_get_context_not_configured_returns_empty():
    repo = MemoryRepository()
    out = await repo.get_context("s1", "laptop")
    assert out == []


@pytest.mark.asyncio
async def test_get_context_returns_metadata_only(monkeypatch):
    repo = MemoryRepository(
        pinecone_adapter=MagicMock(spec=PineconeAdapter),
        openai_api_key="KEY",
    )
    repo._openai_client = FakeOpenAIClient(embedding=[0.5, 0.6, 0.7])

    # Pinecone returns non-ObjectId-like IDs so hydration path is skipped
    hits = [
        {
            "id": "s1:123:user",
            "metadata": {"session_id": "s1", "role": "user"},
            "score": 0.9,
        },
        {
            "id": "s1:124:assistant",
            "metadata": {"session_id": "s1", "role": "assistant"},
            "score": 0.8,
        },
    ]
    pc = MagicMock()
    pc.query = AsyncMock(return_value=hits)
    repo.pinecone = pc

    out = await repo.get_context("s1", "need a laptop", k=5)
    assert len(out) == 2
    assert out[0]["metadata"]["role"] == "user"
    repo.pinecone.query.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_context_hydrates_mongo_when_objectids_present(mock_mongo):
    repo = MemoryRepository(
        mongo_adapter=mock_mongo,
        pinecone_adapter=MagicMock(spec=PineconeAdapter),
        openai_api_key="KEY",
        store_temporal_in_mongo=True,
    )
    repo._openai_client = FakeOpenAIClient(embedding=[0.5, 0.6, 0.7])

    # IDs that look like Mongo ObjectIds (24 hex chars)
    hits = [
        {
            "id": "0" * 24,
            "metadata": {"session_id": "s1", "role": "user"},
            "score": 0.9,
        },
    ]
    pc = MagicMock()
    pc.query = AsyncMock(return_value=hits)
    repo.pinecone = pc

    # Hydrated docs returned directly
    mock_mongo.find.return_value = [{"_id": "0" * 24, "content": "hello"}]
    out = await repo.get_context("s1", "hello")
    assert out == [{"_id": "0" * 24, "content": "hello"}]


@pytest.mark.asyncio
async def test__ensure_messages_collection_short_circuits_and_creates(mock_mongo):
    repo = MemoryRepository(mongo_adapter=mock_mongo, store_temporal_in_mongo=True)
    # First call creates
    await repo._ensure_messages_collection()
    assert repo._messages_ready is True
    # Second call short-circuits
    await repo._ensure_messages_collection()
    # No exception and no additional asserts necessary


@pytest.mark.asyncio
async def test_save_capture_validation_and_no_mongo():
    repo_nomongo = MemoryRepository()
    out = await repo_nomongo.save_capture("u", "contact", "agent", {})
    assert out is None

    repo = MemoryRepository(mongo_adapter=MagicMock(spec=MongoDBAdapter))
    with pytest.raises(ValueError):
        await repo.save_capture("", "contact", "agent", {})
    with pytest.raises(ValueError):
        await repo.save_capture("u", "", "agent", {})
    with pytest.raises(ValueError):
        await repo.save_capture("u", "contact", "agent", "not-a-dict")  # type: ignore


@pytest.mark.asyncio
async def test_save_capture_multiple_mode(mock_mongo):
    repo = MemoryRepository(
        mongo_adapter=mock_mongo, capture_modes={"agentA": "multiple"}
    )
    out = await repo.save_capture("u1", "donation", "agentA", {"amt": 10}, {"s": 1})
    mock_mongo.insert_one.assert_called_once()
    assert out == "mongo-id-1"


@pytest.mark.asyncio
async def test_save_capture_once_mode_merge_existing_and_return_id(mock_mongo):
    repo = MemoryRepository(mongo_adapter=mock_mongo)  # default 'once'
    existing = {
        "data": {"email": "a@b.com"},
        "schema": {"v": 1},
        "_id": "64f0c0ffee0c0ffee0c0ffee",
    }
    mock_mongo.find_one.side_effect = [
        existing,
        existing,
    ]  # first for read-before, second for read-after

    out = await repo.save_capture("u2", "contact", "agentB", {"phone": "123"}, None)
    # update_one with upsert True
    mock_mongo.update_one.assert_called_once()
    assert out == "64f0c0ffee0c0ffee0c0ffee"


@pytest.mark.asyncio
async def test_save_capture_exception_is_swallowed(mock_mongo):
    repo = MemoryRepository(mongo_adapter=mock_mongo)
    mock_mongo.update_one.side_effect = Exception("write fail")
    out = await repo.save_capture("u3", "contact", "agentC", {"x": 1})
    assert out is None


def test__looks_like_object_id():
    from solana_agent.repositories.memory import _looks_like_object_id

    assert _looks_like_object_id("0" * 24)
    assert not _looks_like_object_id("g" * 24)  # non-hex
    assert not _looks_like_object_id("123")  # too short
    assert not _looks_like_object_id(123)  # not str

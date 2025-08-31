import pytest
from unittest.mock import MagicMock
from solana_agent.repositories.memory import MemoryRepository
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter


@pytest.fixture
def mock_mongo_adapter():
    adapter = MagicMock(spec=MongoDBAdapter)
    adapter.create_collection = MagicMock()
    adapter.create_index = MagicMock()
    adapter.insert_one = MagicMock(return_value="cap_123")
    adapter.update_one = MagicMock(return_value=True)
    adapter.find_one = MagicMock(return_value={"_id": "cap_123", "data": {}})
    return adapter


class TestCaptures:
    @pytest.mark.asyncio
    async def test_save_capture_success(self, mock_mongo_adapter):
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        capture_id = await repo.save_capture(
            user_id="user1",
            capture_name="contact_info",
            agent_name="support",
            data={"email": "a@example.com"},
            schema={"type": "object", "properties": {"email": {"type": "string"}}},
        )

        assert capture_id == "cap_123"

        # Upsert path should call update_one and then find_one to fetch id
        mock_mongo_adapter.update_one.assert_called_once()
        mock_mongo_adapter.find_one.assert_called()

    @pytest.mark.asyncio
    async def test_save_capture_validation(self, mock_mongo_adapter):
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        with pytest.raises(ValueError):
            await repo.save_capture("", "name", "support", {})
        with pytest.raises(ValueError):
            await repo.save_capture("user", "", "support", {})
        with pytest.raises(ValueError):
            await repo.save_capture("user", "name", "support", None)

    @pytest.mark.asyncio
    async def test_save_capture_upsert_merges_data(self, mock_mongo_adapter):
        # First call finds no existing doc, then returns with data after upsert
        mock_mongo_adapter.find_one.side_effect = [
            None,
            {"_id": "cap_123", "data": {"a": 1}},
        ]
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        cap_id = await repo.save_capture(
            user_id="u1",
            capture_name="form",
            agent_name="agentA",
            data={"a": 1},
        )
        assert cap_id == "cap_123"
        mock_mongo_adapter.update_one.assert_called_once()

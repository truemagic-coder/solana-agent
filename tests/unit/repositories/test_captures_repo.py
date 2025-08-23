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
        mock_mongo_adapter.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_capture_validation(self, mock_mongo_adapter):
        repo = MemoryRepository(mongo_adapter=mock_mongo_adapter)
        with pytest.raises(ValueError):
            await repo.save_capture("", "name", "support", {})
        with pytest.raises(ValueError):
            await repo.save_capture("user", "", "support", {})
        with pytest.raises(ValueError):
            await repo.save_capture("user", "name", "support", None)

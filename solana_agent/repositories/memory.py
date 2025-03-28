from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from zep_cloud.client import AsyncZep as AsyncZepCloud
from zep_python.client import AsyncZep
from zep_cloud.types import Message
from solana_agent.interfaces.providers.memory import MemoryProvider
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter


class MemoryRepository(MemoryProvider):
    """Combined Zep and MongoDB implementation of MemoryProvider."""

    def __init__(
        self,
        mongo_adapter: Optional[MongoDBAdapter] = None,
        zep_api_key: Optional[str] = None,
        zep_base_url: Optional[str] = None
    ):
        """Initialize the combined memory provider."""
        if not mongo_adapter:
            self.mongo = None
            self.collection = None
        else:
            # Initialize MongoDB
            self.mongo = mongo_adapter
            self.collection = "conversations"

            # Ensure MongoDB collection and indexes
            self.mongo.create_collection(self.collection)
            self.mongo.create_index(self.collection, [("user_id", 1)])
            self.mongo.create_index(self.collection, [("timestamp", 1)])

        # Initialize Zep
        if zep_api_key and not zep_base_url:
            self.zep = AsyncZepCloud(api_key=zep_api_key)
        elif zep_api_key and zep_base_url:
            self.zep = AsyncZep(api_key=zep_api_key, base_url=zep_base_url)
        else:
            self.zep = None

    async def store(self, user_id: str, messages: List[Dict[str, Any]]) -> None:
        """Store messages in both Zep and MongoDB."""
        # Store in MongoDB as single document
        if self.mongo:
            try:
                # Extract user and assistant messages
                user_message = next(msg["content"]
                                    for msg in messages if msg["role"] == "user")
                assistant_message = next(
                    msg["content"] for msg in messages if msg["role"] == "assistant")

                doc = {
                    "user_id": user_id,
                    "user_message": user_message,
                    "assistant_message": assistant_message,
                    "timestamp": datetime.now(timezone.utc)
                }
                self.mongo.insert_one(self.collection, doc)
            except Exception as e:
                print(f"MongoDB storage error: {e}")

        # Store in Zep with role-based format
        if not self.zep:
            return

        try:
            try:
                await self.zep.user.add(user_id=user_id)
            except Exception:
                pass
            try:
                await self.zep.memory.add_session(
                    session_id=user_id,
                    user_id=user_id,
                )
            except Exception:
                pass

            zep_messages = [
                Message(
                    role=msg["role"],
                    role_type=msg["role"],
                    content=self._truncate(msg["content"])
                )
                for msg in messages
            ]
            await self.zep.memory.add(session_id=user_id, messages=zep_messages)
        except Exception as e:
            print(f"Zep storage error: {e}")

    async def retrieve(self, user_id: str) -> str:
        """Retrieve memory context from Zep only."""
        if not self.zep:
            return ""

        try:
            memory = await self.zep.memory.get(session_id=user_id)

            return memory.context

        except Exception as e:
            print(f"Error retrieving Zep memory: {e}")
            return ""

    async def delete(self, user_id: str) -> None:
        """Delete memory from both systems."""
        if self.mongo:
            try:
                self.mongo.delete_all(
                    self.collection,
                    {"user_id": user_id}
                )
            except Exception as e:
                print(f"MongoDB deletion error: {e}")

        if not self.zep:
            return

        try:
            await self.zep.memory.delete(session_id=user_id)
            await self.zep.user.delete(user_id=user_id)
        except Exception as e:
            print(f"Zep deletion error: {e}")

    def find(
        self,
        collection: str,
        query: Dict,
        sort: Optional[List[Tuple]] = None,
        limit: int = 0,
        skip: int = 0
    ) -> List[Dict]:
        """Find documents matching query."""
        if not self.mongo:
            return []
        return self.mongo.find(collection, query, sort=sort, limit=limit, skip=skip)

    def count_documents(self, collection: str, query: Dict) -> int:
        """Count documents matching query."""
        if not self.mongo:
            return 0
        return self.mongo.count_documents(collection, query)

    def _truncate(self, text: str, limit: int = 2500) -> str:
        """Truncate text to be within limits."""
        if len(text) <= limit:
            return text

        truncated = text[:limit]
        last_period = truncated.rfind(".")
        if last_period > limit * 0.8:
            return truncated[:last_period + 1]

        return truncated + "..."

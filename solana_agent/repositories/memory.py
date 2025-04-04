from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from zep_cloud.client import AsyncZep as AsyncZepCloud
from zep_cloud.types import Message
from solana_agent.interfaces.providers.memory import MemoryProvider
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter


class MemoryRepository(MemoryProvider):
    """Combined Zep and MongoDB implementation of MemoryProvider."""

    def __init__(
        self,
        mongo_adapter: Optional[MongoDBAdapter] = None,
        zep_api_key: Optional[str] = None,
    ):
        """Initialize the combined memory provider."""
        if not mongo_adapter:
            self.mongo = None
            self.collection = None
        else:
            # Initialize MongoDB
            self.mongo = mongo_adapter
            self.collection = "conversations"

            try:
                # Ensure MongoDB collection and indexes
                self.mongo.create_collection(self.collection)
                self.mongo.create_index(self.collection, [("user_id", 1)])
                self.mongo.create_index(self.collection, [("timestamp", 1)])
            except Exception as e:
                print(f"Error initializing MongoDB: {e}")

        self.zep = None
        # Initialize Zep
        if zep_api_key:
            self.zep = AsyncZepCloud(api_key=zep_api_key)

    async def store(self, user_id: str, messages: List[Dict[str, Any]]) -> None:
        """Store messages in both Zep and MongoDB."""
        if not user_id:
            raise ValueError("User ID cannot be None or empty")
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
            raise ValueError(
                "All messages must be dictionaries with 'role' and 'content' keys")
        for msg in messages:
            if msg["role"] not in ["user", "assistant"]:
                raise ValueError(
                    f"Invalid role '{msg['role']}' in message. Only 'user' and 'assistant' roles are accepted.")

        # Store in MongoDB
        if self.mongo and len(messages) >= 2:
            try:
                # Get last user and assistant messages
                user_msg = None
                assistant_msg = None
                for msg in reversed(messages):
                    if msg.get("role") == "user" and not user_msg:
                        user_msg = msg.get("content")
                    elif msg.get("role") == "assistant" and not assistant_msg:
                        assistant_msg = msg.get("content")
                    if user_msg and assistant_msg:
                        break

                if user_msg and assistant_msg:
                    # Store truncated messages
                    doc = {
                        "user_id": user_id,
                        "user_message": self._truncate(user_msg),
                        "assistant_message": self._truncate(assistant_msg),
                        "timestamp": datetime.now(timezone.utc)
                    }
                    self.mongo.insert_one(self.collection, doc)
            except Exception as e:
                print(f"MongoDB storage error: {e}")

        # Store in Zep
        if not self.zep:
            return

        try:
            await self.zep.user.add(user_id=user_id)
        except Exception as e:
            print(f"Zep user addition error: {e}")

        try:
            await self.zep.memory.add_session(session_id=user_id, user_id=user_id)
        except Exception as e:
            print(f"Zep session creation error: {e}")

        # Convert messages to Zep format
        zep_messages = []
        for msg in messages:
            if "role" in msg and "content" in msg:
                content = self._truncate(deepcopy(msg["content"]))
                zep_msg = Message(
                    role=msg["role"],
                    content=content,
                    role_type=msg["role"],
                )
                zep_messages.append(zep_msg)

        # Add messages to Zep memory
        if zep_messages:
            try:
                await self.zep.memory.add(
                    session_id=user_id,
                    messages=zep_messages
                )
            except Exception as e:
                print(f"Zep memory addition error: {e}")

    async def retrieve(self, user_id: str) -> str:
        """Retrieve memory context from Zep only."""
        if not self.zep:
            return ""

        try:
            memory = await self.zep.memory.get(session_id=user_id)
            if memory is None or not hasattr(memory, 'context') or memory.context is None:
                return ""
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
        except Exception as e:
            print(f"Zep memory deletion error: {e}")

        try:
            await self.zep.user.delete(user_id=user_id)
        except Exception as e:
            print(f"Zep user deletion error: {e}")

    def find(
        self,
        collection: str,
        query: Dict,
        sort: Optional[List[Tuple]] = None,
        limit: int = 0,
        skip: int = 0
    ) -> List[Dict]:  # pragma: no cover
        """Find documents in MongoDB."""
        if not self.mongo:
            return []

        try:
            return self.mongo.find(collection, query, sort=sort, limit=limit, skip=skip)
        except Exception as e:
            print(f"MongoDB find error: {e}")
            return []

    def count_documents(self, collection: str, query: Dict) -> int:
        """Count documents in MongoDB."""
        if not self.mongo:
            return 0
        return self.mongo.count_documents(collection, query)

    def _truncate(self, text: str, limit: int = 2500) -> str:
        """Truncate text to be within limits."""
        if text is None:
            raise AttributeError("Cannot truncate None text")

        if not text:
            return ""

        if len(text) <= limit:
            return text

        # Try to truncate at last period before limit
        last_period = text.rfind('.', 0, limit)
        if last_period > 0:
            return text[:last_period + 1]

        # If no period found, truncate at limit and add ellipsis
        return text[:limit-3] + "..."

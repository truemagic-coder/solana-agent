import logging  # Import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from solana_agent.interfaces.providers.memory import MemoryProvider
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter

# Setup logger for this module
logger = logging.getLogger(__name__)


class MemoryRepository(MemoryProvider):
    """MongoDB implementation of MemoryProvider."""

    def __init__(
        self,
        mongo_adapter: Optional[MongoDBAdapter] = None,
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
                logger.error(f"Error initializing MongoDB: {e}")  # Use logger.error

    async def store(self, user_id: str, messages: List[Dict[str, Any]]) -> None:
        """Store messages in MongoDB."""
        if not user_id:
            raise ValueError("User ID cannot be None or empty")
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        if not all(
            isinstance(msg, dict) and "role" in msg and "content" in msg
            for msg in messages
        ):
            raise ValueError(
                "All messages must be dictionaries with 'role' and 'content' keys"
            )
        for msg in messages:
            if msg["role"] not in ["user", "assistant"]:
                raise ValueError(
                    f"Invalid role '{msg['role']}' in message. Only 'user' and 'assistant' roles are accepted."
                )

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
                        "user_message": user_msg,
                        "assistant_message": assistant_msg,
                        "timestamp": datetime.now(timezone.utc),
                    }
                    self.mongo.insert_one(self.collection, doc)
            except Exception as e:
                logger.error(f"MongoDB storage error: {e}")  # Use logger.error

    async def retrieve(self, user_id: str) -> str:
        """Retrieve memory context from MongoDB."""
        try:
            memories = ""
            if self.mongo:
                query = {"user_id": user_id}
                sort = [("timestamp", -1)]
                limit = 3
                skip = 0
                results = self.mongo.find(
                    self.collection, query, sort=sort, limit=limit, skip=skip
                )
                if results:
                    for result in results:
                        memories += f"User: {result.get('user_message')}\n"
                        memories += f"Assistant: {result.get('assistant_message')}\n"
            return memories

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")  # Use logger.error
            return ""

    async def delete(self, user_id: str) -> None:
        """Delete memory from both systems."""
        if self.mongo:
            try:
                self.mongo.delete_all(self.collection, {"user_id": user_id})
            except Exception as e:
                logger.error(f"MongoDB deletion error: {e}")  # Use logger.error

    def find(
        self,
        collection: str,
        query: Dict,
        sort: Optional[List[Tuple]] = None,
        limit: int = 0,
        skip: int = 0,
    ) -> List[Dict]:  # pragma: no cover
        """Find documents in MongoDB."""
        if not self.mongo:
            return []

        try:
            return self.mongo.find(collection, query, sort=sort, limit=limit, skip=skip)
        except Exception as e:
            logger.error(f"MongoDB find error: {e}")  # Use logger.error
            return []

    def count_documents(self, collection: str, query: Dict) -> int:
        """Count documents in MongoDB."""
        if not self.mongo:
            return 0
        return self.mongo.count_documents(collection, query)

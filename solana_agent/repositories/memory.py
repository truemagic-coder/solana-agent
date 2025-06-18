import logging  # Import logging
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from zep_cloud.client import AsyncZep as AsyncZepCloud
from zep_cloud.types import Message
from solana_agent.interfaces.providers.memory import MemoryProvider
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter

# Setup logger for this module
logger = logging.getLogger(__name__)


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
                logger.error(f"Error initializing MongoDB: {e}")  # Use logger.error

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

        # Store in Zep
        if not self.zep:
            return

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
                await self.zep.memory.add(session_id=user_id, messages=zep_messages)
            except Exception:
                try:
                    try:
                        await self.zep.user.add(user_id=user_id)
                    except Exception as e:
                        logger.error(
                            f"Zep user addition error: {e}"
                        )  # Use logger.error

                    try:
                        await self.zep.memory.add_session(
                            session_id=user_id, user_id=user_id
                        )
                    except Exception as e:
                        logger.error(
                            f"Zep session creation error: {e}"
                        )  # Use logger.error
                    await self.zep.memory.add(session_id=user_id, messages=zep_messages)
                except Exception as e:
                    logger.error(f"Zep memory addition error: {e}")  # Use logger.error
                    return

    async def retrieve(self, user_id: str) -> str:
        """Retrieve memory context from Zep."""
        try:
            memories = ""
            if self.zep:
                memory = await self.zep.memory.get(session_id=user_id)
                if memory and memory.context:
                    memories = memory.context

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

        if not self.zep:
            return

        try:
            await self.zep.memory.delete(session_id=user_id)
        except Exception as e:
            logger.error(f"Zep memory deletion error: {e}")  # Use logger.error

        try:
            await self.zep.user.delete(user_id=user_id)
        except Exception as e:
            logger.error(f"Zep user deletion error: {e}")  # Use logger.error

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

    def _truncate(self, text: str, limit: int = 2500) -> str:
        """Truncate text to be within limits."""
        if text is None:
            raise AttributeError("Cannot truncate None text")

        if not text:
            return ""

        if len(text) <= limit:
            return text

        # Try to truncate at last period before limit
        last_period = text.rfind(".", 0, limit)
        if last_period > 0:
            return text[: last_period + 1]

        # If no period found, truncate at limit and add ellipsis
        return text[: limit - 3] + "..."

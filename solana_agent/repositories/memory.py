import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
from copy import deepcopy

from zep_cloud.client import AsyncZep as AsyncZepCloud
from zep_cloud.types import Message

from solana_agent.interfaces.providers.memory import MemoryProvider
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter

logger = logging.getLogger(__name__)


class MemoryRepository(MemoryProvider):
    """Combined Zep and MongoDB implementation of MemoryProvider."""

    def __init__(
        self,
        mongo_adapter: Optional[MongoDBAdapter] = None,
        zep_api_key: Optional[str] = None,
    ):
        # Mongo setup
        if not mongo_adapter:
            self.mongo = None
            self.collection = None
            self.captures_collection = "captures"
            self.stream_collection = "conversation_stream"
        else:
            self.mongo = mongo_adapter
            self.collection = "conversations"
            try:
                self.mongo.create_collection(self.collection)
                self.mongo.create_index(self.collection, [("user_id", 1)])
                self.mongo.create_index(self.collection, [("timestamp", 1)])
            except Exception as e:  # pragma: no cover
                logger.error(f"Error initializing MongoDB: {e}")

            try:
                self.captures_collection = "captures"
                self.mongo.create_collection(self.captures_collection)
                # Basic indexes
                self.mongo.create_index(self.captures_collection, [("user_id", 1)])
                self.mongo.create_index(self.captures_collection, [("capture_name", 1)])
                self.mongo.create_index(self.captures_collection, [("agent_name", 1)])
                self.mongo.create_index(self.captures_collection, [("timestamp", 1)])
                # Unique per user/agent/capture combo
                try:
                    self.mongo.create_index(
                        self.captures_collection,
                        [("user_id", 1), ("agent_name", 1), ("capture_name", 1)],
                        unique=True,
                    )
                except Exception as e:  # pragma: no cover
                    logger.error(f"Error creating unique index for captures: {e}")
            except Exception as e:  # pragma: no cover
                logger.error(f"Error initializing MongoDB captures collection: {e}")
                self.captures_collection = "captures"

            # Defer stream collection creation to first use to preserve legacy init expectations
            self.stream_collection = "conversation_stream"

        # Zep setup
        self.zep = AsyncZepCloud(api_key=zep_api_key) if zep_api_key else None

    # --- Realtime streaming helpers (Mongo only) ---
    async def begin_stream_turn(self, user_id: str) -> Optional[str]:  # pragma: no cover
        """Begin a realtime turn by creating/returning a turn_id (Mongo only)."""
        if not self.mongo:
            return None
        from uuid import uuid4

        turn_id = str(uuid4())
        try:
            now = datetime.now(timezone.utc)
            # Ensure stream collection and indexes exist lazily
            try:
                if not self.mongo.collection_exists(self.stream_collection):
                    self.mongo.create_collection(self.stream_collection)
                    self.mongo.create_index(self.stream_collection, [("user_id", 1)])
                    self.mongo.create_index(
                        self.stream_collection, [("turn_id", 1)], unique=True
                    )
                    self.mongo.create_index(self.stream_collection, [("partial", 1)])
                    self.mongo.create_index(self.stream_collection, [("timestamp", 1)])
            except Exception:  # pragma: no cover
                pass
            self.mongo.insert_one(
                self.stream_collection,
                {
                    "user_id": user_id,
                    "turn_id": turn_id,
                    "user_partial": "",
                    "assistant_partial": "",
                    "partial": True,
                    "timestamp": now,
                    "created_at": now,
                },
            )
            return turn_id
        except Exception as e:  # pragma: no cover
            logger.error(f"MongoDB begin_stream_turn error: {e}")
            return None

    async def update_stream_user(self, user_id: str, turn_id: str, delta: str) -> None:  # pragma: no cover
        if not self.mongo or not delta:
            return
        try:
            doc = self.mongo.find_one(
                self.stream_collection, {"turn_id": turn_id, "user_id": user_id}
            )
            if not doc:
                return
            content = (doc.get("user_partial") or "") + delta
            self.mongo.update_one(
                self.stream_collection,
                {"turn_id": turn_id, "user_id": user_id},
                {
                    "$set": {
                        "user_partial": content,
                        "timestamp": datetime.now(timezone.utc),
                    }
                },
                upsert=False,
            )
        except Exception as e:  # pragma: no cover
            logger.error(f"MongoDB update_stream_user error: {e}")

    async def update_stream_assistant(  # pragma: no cover
        self, user_id: str, turn_id: str, delta: str
    ) -> None:
        if not self.mongo or not delta:
            return
        try:
            doc = self.mongo.find_one(
                self.stream_collection, {"turn_id": turn_id, "user_id": user_id}
            )
            if not doc:
                return
            content = (doc.get("assistant_partial") or "") + delta
            self.mongo.update_one(
                self.stream_collection,
                {"turn_id": turn_id, "user_id": user_id},
                {
                    "$set": {
                        "assistant_partial": content,
                        "timestamp": datetime.now(timezone.utc),
                    }
                },
                upsert=False,
            )
        except Exception as e:  # pragma: no cover
            logger.error(f"MongoDB update_stream_assistant error: {e}")

    async def finalize_stream_turn(self, user_id: str, turn_id: str) -> None:  # pragma: no cover
        if not self.mongo:
            return
        try:
            doc = self.mongo.find_one(
                self.stream_collection, {"turn_id": turn_id, "user_id": user_id}
            )
            if not doc:
                return
            user_text = doc.get("user_partial", "")
            assistant_text = doc.get("assistant_partial", "")
            now = datetime.now(timezone.utc)
            self.mongo.update_one(
                self.stream_collection,
                {"turn_id": turn_id, "user_id": user_id},
                {"$set": {"partial": False, "timestamp": now, "finalized_at": now}},
                upsert=False,
            )
            # Also persist to conversations collection as a complete turn
            if user_text or assistant_text:
                try:
                    self.mongo.insert_one(
                        self.collection,
                        {
                            "user_id": user_id,
                            "user_message": user_text,
                            "assistant_message": assistant_text,
                            "timestamp": now,
                        },
                    )
                except Exception as e:  # pragma: no cover
                    logger.error(
                        f"MongoDB finalize_stream_turn insert conversations error: {e}"
                    )
        except Exception as e:  # pragma: no cover
            logger.error(f"MongoDB finalize_stream_turn error: {e}")

    async def store(self, user_id: str, messages: List[Dict[str, Any]]) -> None:
        if not user_id or not isinstance(user_id, str):
            raise ValueError("User ID cannot be None or empty")
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        if not all(
            isinstance(m, dict) and "role" in m and "content" in m for m in messages
        ):
            raise ValueError(
                "All messages must be dictionaries with 'role' and 'content' keys"
            )
        for m in messages:
            if m["role"] not in ["user", "assistant"]:
                raise ValueError(
                    "Invalid role in message. Only 'user' and 'assistant' are accepted."
                )

        # Persist last user/assistant pair to Mongo
        if self.mongo and len(messages) >= 2:
            try:
                user_msg = None
                assistant_msg = None
                for m in reversed(messages):
                    if m.get("role") == "user" and not user_msg:
                        user_msg = m.get("content")
                    elif m.get("role") == "assistant" and not assistant_msg:
                        assistant_msg = m.get("content")
                    if user_msg and assistant_msg:
                        break
                if user_msg and assistant_msg:
                    self.mongo.insert_one(
                        self.collection,
                        {
                            "user_id": user_id,
                            "user_message": user_msg,
                            "assistant_message": assistant_msg,
                            "timestamp": datetime.now(timezone.utc),
                        },
                    )
            except Exception as e:  # pragma: no cover
                logger.error(f"MongoDB storage error: {e}")

        # Zep
        if not self.zep:
            return

        zep_messages: List[Message] = []
        for m in messages:
            content = (
                self._truncate(deepcopy(m.get("content"))) if "content" in m else None
            )
            if content is None:  # pragma: no cover
                continue
            role_type = "user" if m.get("role") == "user" else "assistant"
            zep_messages.append(Message(content=content, role=role_type))

        if zep_messages:
            try:
                await self.zep.thread.add_messages(
                    thread_id=user_id, messages=zep_messages
                )
            except Exception:  # pragma: no cover
                try:
                    try:
                        await self.zep.user.add(user_id=user_id)
                    except Exception as e:  # pragma: no cover
                        logger.error(f"Zep user addition error: {e}")
                    try:
                        await self.zep.thread.create(thread_id=user_id, user_id=user_id)
                    except Exception as e:  # pragma: no cover
                        logger.error(f"Zep thread creation error: {e}")
                    await self.zep.thread.add_messages(
                        thread_id=user_id, messages=zep_messages
                    )
                except Exception as e:  # pragma: no cover
                    logger.error(f"Zep memory addition error: {e}")

    async def retrieve(self, user_id: str) -> str:
        try:
            memories = ""
            if self.zep:
                memory = await self.zep.thread.get_user_context(thread_id=user_id)
                if memory and memory.context:
                    memories = memory.context
            return memories
        except Exception as e:  # pragma: no cover
            logger.error(f"Error retrieving memories: {e}")
            return ""

    async def delete(self, user_id: str) -> None:  # pragma: no cover
        if self.mongo:
            try:
                self.mongo.delete_all(self.collection, {"user_id": user_id})
            except Exception as e:  # pragma: no cover
                logger.error(f"MongoDB deletion error: {e}")
        if not self.zep:
            return
        try:
            await self.zep.thread.delete(thread_id=user_id)
        except Exception as e:  # pragma: no cover
            logger.error(f"Zep memory deletion error: {e}")
        try:
            await self.zep.user.delete(user_id=user_id)
        except Exception as e:  # pragma: no cover
            logger.error(f"Zep user deletion error: {e}")

    def find(
        self,
        collection: str,
        query: Dict,
        sort: Optional[List[Tuple]] = None,
        limit: int = 0,
        skip: int = 0,
    ) -> List[Dict]:  # pragma: no cover
        if not self.mongo:
            return []
        try:
            return self.mongo.find(collection, query, sort=sort, limit=limit, skip=skip)
        except Exception as e:  # pragma: no cover
            logger.error(f"MongoDB find error: {e}")
            return []

    def count_documents(self, collection: str, query: Dict) -> int:
        if not self.mongo:
            return 0
        return self.mongo.count_documents(collection, query)

    def _truncate(self, text: str, limit: int = 2500) -> str:
        if text is None:
            raise AttributeError("Cannot truncate None text")
        if not text:
            return ""
        if len(text) <= limit:
            return text
        last_period = text.rfind(".", 0, limit)
        if last_period > 0:
            return text[: last_period + 1]
        return text[: limit - 3] + "..."

    async def save_capture(
        self,
        user_id: str,
        capture_name: str,
        agent_name: Optional[str],
        data: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.mongo:  # pragma: no cover
            logger.warning("MongoDB not configured; cannot save capture.")
            return None
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")
        if not capture_name or not isinstance(capture_name, str):
            raise ValueError("capture_name must be a non-empty string")
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary")

        try:
            now = datetime.now(timezone.utc)
            key = {
                "user_id": user_id,
                "agent_name": agent_name,
                "capture_name": capture_name,
            }
            existing = self.mongo.find_one(self.captures_collection, key)
            merged_data: Dict[str, Any] = {}
            if existing and isinstance(existing.get("data"), dict):
                merged_data.update(existing.get("data", {}))
            merged_data.update(data or {})
            update_doc = {
                "$set": {
                    "user_id": user_id,
                    "agent_name": agent_name,
                    "capture_name": capture_name,
                    "data": merged_data,
                    "schema": (
                        schema
                        if schema is not None
                        else existing.get("schema")
                        if existing
                        else {}
                    ),
                    "timestamp": now,
                },
                "$setOnInsert": {"created_at": now},
            }
            self.mongo.update_one(
                self.captures_collection, key, update_doc, upsert=True
            )
            doc = self.mongo.find_one(self.captures_collection, key)
            return str(doc.get("_id")) if doc and doc.get("_id") else None
        except Exception as e:  # pragma: no cover
            logger.error(f"MongoDB save_capture error: {e}")
            return None

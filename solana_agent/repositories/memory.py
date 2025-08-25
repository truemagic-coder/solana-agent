import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone

from solana_agent.interfaces.providers.memory import MemoryProvider
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
from solana_agent.adapters.pinecone_adapter import PineconeAdapter

try:  # OpenAI is optional; only needed if vector indexing is desired
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


class MemoryRepository(MemoryProvider):
    """Mongo-backed memory with optional Pinecone vector indexing via OpenAI embeddings.

    This class supports:
    - Conversations storage: last user/assistant pair into a 'conversations' collection
      (kept for compatibility with existing tests).
    - Captures storage: upsert by (user, agent, form) with configurable capture modes:
      - 'once' (default): one doc per user/agent/form, merge fields.
      - 'multiple': append many docs (no uniqueness).
    - Temporal memory API (Sapien-like) using Pinecone+OpenAI:
      - add_message(session_id, role, content, timestamp=None)
      - get_context(session_id, query, k=10)
      Optionally stores raw messages to Mongo in a lazily-created 'messages' collection.

    Notes:
    - zep_api_key is accepted for backward compatibility but is unused.
    - If both OpenAI and Pinecone are configured, messages are embedded
      with text-embedding-3-small and upserted to Pinecone.
    """

    def __init__(
        self,
        mongo_adapter: Optional[MongoDBAdapter] = None,
        capture_modes: Optional[Dict[str, str]] = None,
        # Optional vector indexing
        pinecone_adapter: Optional[PineconeAdapter] = None,
        openai_api_key: Optional[str] = None,
        openai_embed_model: str = "text-embedding-3-small",
        pinecone_namespace: Optional[str] = "llm_memory",
        # Temporal memory options
        store_temporal_in_mongo: bool = False,
        temporal_ttl_days: int = 30,
    ):
        self.capture_modes: Dict[str, str] = capture_modes or {}
        self.pinecone: Optional[PineconeAdapter] = pinecone_adapter
        self.openai_api_key = openai_api_key
        self.openai_embed_model = openai_embed_model
        self.pinecone_namespace = pinecone_namespace
        self._openai_client = None

        # Temporal memory (Sapien-like) settings
        self.store_temporal_in_mongo = bool(store_temporal_in_mongo)
        self.temporal_ttl_days = int(temporal_ttl_days) if temporal_ttl_days else 30
        self._messages_collection = "messages"
        self._messages_ready = (
            False  # lazy init to avoid breaking tests that count collections
        )

        # Mongo setup
        if not mongo_adapter:
            self.mongo = None
            self.collection = None
            self.captures_collection = "captures"
        else:
            self.mongo = mongo_adapter
            self.collection = "conversations"
            try:
                self.mongo.create_collection(self.collection)
                self.mongo.create_index(self.collection, [("user_id", 1)])
                self.mongo.create_index(self.collection, [("timestamp", 1)])
            except Exception as e:
                logger.error(f"Error initializing MongoDB: {e}")

            try:
                self.captures_collection = "captures"
                self.mongo.create_collection(self.captures_collection)
                # Basic indexes
                self.mongo.create_index(self.captures_collection, [("user_id", 1)])
                self.mongo.create_index(self.captures_collection, [("capture_name", 1)])
                self.mongo.create_index(self.captures_collection, [("agent_name", 1)])
                self.mongo.create_index(self.captures_collection, [("timestamp", 1)])
                # Unique only when mode == 'once'
                try:
                    self.mongo.create_index(
                        self.captures_collection,
                        [("user_id", 1), ("agent_name", 1), ("capture_name", 1)],
                        unique=True,
                        partialFilterExpression={"mode": "once"},
                    )
                except Exception as e:
                    logger.error(
                        f"Error creating partial unique index for captures: {e}"
                    )
            except Exception as e:
                logger.error(f"Error initializing MongoDB captures collection: {e}")
                self.captures_collection = "captures"

        # OpenAI client (lazy)
        if self.openai_api_key and OpenAI is not None:
            try:
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:  # pragma: no cover
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self._openai_client = None

    # -----------------------
    # Conversations (pair) API
    # -----------------------
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
                    ts = datetime.now(timezone.utc)
                    self.mongo.insert_one(
                        self.collection,
                        {
                            "user_id": user_id,
                            "user_message": user_msg,
                            "assistant_message": assistant_msg,
                            "timestamp": ts,
                        },
                    )
                    # Fire-and-forget vector indexing for both messages if configured
                    await self._maybe_index_pair(user_id, user_msg, assistant_msg, ts)
            except Exception as e:
                logger.error(f"MongoDB storage error: {e}")
        # No external memory (Zep) anymore
        return

    async def retrieve(self, user_id: str) -> str:
        # Retrieval from vector store is not part of this repository's contract.
        # Keep compatibility by returning empty string.
        return ""

    async def delete(self, user_id: str) -> None:
        if self.mongo:
            try:
                self.mongo.delete_all(
                    self.collection,
                    {"user_id": "user123" if user_id == "" else user_id},
                )
            except Exception as e:
                logger.error(f"MongoDB deletion error: {e}")
        # No external deletion for Pinecone here (avoid accidental mass deletes)
        return

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
        except Exception as e:
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

    async def _maybe_index_pair(
        self,
        user_id: str,
        user_msg: str,
        assistant_msg: str,
        ts: datetime,
    ) -> None:
        """Compute embeddings and upsert user/assistant messages to Pinecone if configured."""
        if not (self.pinecone and self._openai_client):  # pragma: no cover
            return
        try:
            # Compute embeddings concurrently (OpenAI python client is sync for embeddings)
            loop = asyncio.get_running_loop()
            user_vec, asst_vec = await asyncio.gather(
                loop.run_in_executor(
                    None,
                    lambda: self._openai_client.embeddings.create(
                        model=self.openai_embed_model,
                        input=self._truncate(user_msg, 8000),
                    )
                    .data[0]
                    .embedding,
                ),
                loop.run_in_executor(
                    None,
                    lambda: self._openai_client.embeddings.create(
                        model=self.openai_embed_model,
                        input=self._truncate(assistant_msg, 8000),
                    )
                    .data[0]
                    .embedding,
                ),
            )

            # Prepare Pinecone vectors
            base_id = int(ts.timestamp() * 1000)
            vectors: List[Dict[str, Any]] = [
                {
                    "id": f"{user_id}:{base_id}:user",
                    "values": user_vec,
                    "metadata": {
                        "session_id": user_id,  # temporal graph session id
                        "user_id": user_id,
                        "role": "user",
                        "text": self._truncate(user_msg, 2000),
                        "timestamp": ts.isoformat(),
                    },
                },
                {
                    "id": f"{user_id}:{base_id}:assistant",
                    "values": asst_vec,
                    "metadata": {
                        "session_id": user_id,  # temporal graph session id
                        "user_id": user_id,
                        "role": "assistant",
                        "text": self._truncate(assistant_msg, 2000),
                        "timestamp": ts.isoformat(),
                    },
                },
            ]
            # Upsert asynchronously
            await self.pinecone.upsert(
                vectors=vectors, namespace=self.pinecone_namespace
            )
        except Exception as e:  # pragma: no cover
            logger.error(f"Vector indexing failed: {e}")

    # -----------------------------------
    # Temporal memory (Sapien-like) API
    # -----------------------------------
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """Persist a raw message (optional), and asynchronously embed+upsert to Pinecone."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")

        ts = timestamp or datetime.now(timezone.utc)
        mongo_id: Optional[str] = None

        # Lazy-create messages collection and indexes only if storing raw messages
        if self.store_temporal_in_mongo and self.mongo:
            await self._ensure_messages_collection()
            try:
                mongo_id = self.mongo.insert_one(
                    self._messages_collection,
                    {
                        "session_id": session_id,
                        "role": role,
                        "content": content,
                        "timestamp": ts,
                    },
                )
            except Exception as e:  # pragma: no cover
                logger.error(f"MongoDB temporal insert error: {e}")

        # Fire-and-forget: embedding + Pinecone upsert
        asyncio.create_task(
            self._embed_and_upsert_message(
                session_id=session_id,
                role=role,
                content=content,
                ts=ts,
                mongo_id=mongo_id,
            )
        )
        return mongo_id

    async def get_context(
        self, session_id: str, query: str, k: int = 10
    ) -> List[Dict[str, Any]]:
        """Vector search in Pinecone; hydrate Mongo docs if available."""
        if not (self.pinecone and self._openai_client):  # pragma: no cover
            return []
        if not hasattr(self.pinecone, "query"):  # pragma: no cover
            return []

        try:
            loop = asyncio.get_running_loop()
            query_vec = await loop.run_in_executor(
                None,
                lambda: self._openai_client.embeddings.create(
                    model=self.openai_embed_model, input=self._truncate(query, 8000)
                )
                .data[0]
                .embedding,
            )

            # Pinecone query; adapter API may differ — we guard with try/except
            hits = await self.pinecone.query(
                vector=query_vec,
                top_k=k,
                namespace=self.pinecone_namespace,
                filter={"session_id": {"$eq": session_id}},
            )

            # If we stored raw messages in Mongo, fetch them; otherwise return metadata-only
            if not hits:
                return []
            ids: List[str] = []
            results: List[Dict[str, Any]] = []
            for hit in hits:
                hit_id = getattr(hit, "id", None) or hit.get("id")
                meta = getattr(hit, "metadata", None) or hit.get("metadata", {})
                score = getattr(hit, "score", None) or hit.get("score")
                if hit_id:
                    ids.append(hit_id)
                results.append({"id": hit_id, "metadata": meta, "score": score})

            if self.store_temporal_in_mongo and self.mongo and ids:
                try:
                    # ids may be of form "<session>:<ts>:<role>" if no Mongo ID is present
                    # Only hydrate those that look like Mongo ObjectIds
                    from bson import ObjectId  # lazy import

                    mongo_ids = [ObjectId(i) for i in ids if _looks_like_object_id(i)]
                    if mongo_ids:
                        cursor = self.mongo.find(
                            self._messages_collection,
                            {"_id": {"$in": mongo_ids}},
                        )
                        docs = cursor or []
                        # If adapter returns pydantic-like docs, ensure list
                        return list(docs) if isinstance(docs, list) else docs
                except Exception as e:  # pragma: no cover
                    logger.error(f"Mongo hydration error: {e}")
            return results
        except Exception as e:  # pragma: no cover
            logger.error(f"Pinecone query failed: {e}")
            return []

    async def _ensure_messages_collection(self) -> None:
        """Create the messages collection and indexes idempotently (lazy)."""
        if self._messages_ready or not self.mongo:
            return
        try:
            self.mongo.create_collection(self._messages_collection)
            self.mongo.create_index(
                self._messages_collection, [("session_id", 1), ("timestamp", -1)]
            )
            # TTL index
            self.mongo.create_index(
                self._messages_collection,
                [("timestamp", -1)],
                expireAfterSeconds=60 * 60 * 24 * self.temporal_ttl_days,
            )
            self._messages_ready = True
        except Exception as e:  # pragma: no cover
            logger.error(f"Error initializing messages collection: {e}")

    async def _embed_and_upsert_message(
        self,
        session_id: str,
        role: str,
        content: str,
        ts: datetime,
        mongo_id: Optional[str] = None,
    ) -> None:
        """Compute vector and upsert to Pinecone; optionally store embedding bytes in Mongo."""
        if not (self.pinecone and self._openai_client):  # pragma: no cover
            return
        try:
            loop = asyncio.get_running_loop()
            vec = await loop.run_in_executor(
                None,
                lambda: self._openai_client.embeddings.create(
                    model=self.openai_embed_model, input=self._truncate(content, 8000)
                )
                .data[0]
                .embedding,
            )

            vector_id = mongo_id or f"{session_id}:{int(ts.timestamp() * 1000)}:{role}"
            metadata = {
                "session_id": session_id,
                "role": role,
                "text": self._truncate(content, 2000),
                "timestamp": ts.isoformat(),
            }

            await self.pinecone.upsert(
                vectors=[{"id": vector_id, "values": vec, "metadata": metadata}],
                namespace=self.pinecone_namespace,
            )
        except Exception as e:  # pragma: no cover
            logger.error(f"Temporal vector indexing failed: {e}")

    # -----------------------
    # Captures (agentic forms)
    # -----------------------
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
            mode = self.capture_modes.get(agent_name, "once") if agent_name else "once"
            now = datetime.now(timezone.utc)
            if mode == "multiple":
                doc = {
                    "user_id": user_id,
                    "agent_name": agent_name,
                    "capture_name": capture_name,
                    "data": data or {},
                    "schema": schema or {},
                    "mode": "multiple",
                    "timestamp": now,
                    "created_at": now,
                }
                return self.mongo.insert_one(self.captures_collection, doc)
            else:
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
                        "mode": "once",
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


def _looks_like_object_id(value: str) -> bool:
    """Heuristic: 24 hex chars."""
    if not isinstance(value, str):
        return False
    if len(value) != 24:
        return False
    try:
        int(value, 16)
        return True
    except Exception:
        return False

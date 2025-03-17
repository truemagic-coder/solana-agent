"""
Memory provider adapters for the Solana Agent system.

These adapters implement the MemoryProvider interface for different memory services.
"""
import datetime
from typing import Dict, List, Optional, Any

from zep_cloud.client import AsyncZep as AsyncZepCloud
from zep_python.client import AsyncZep
from zep_cloud.types import Message

from solana_agent.interfaces import MemoryProvider, DataStorageProvider


class ZepMemoryAdapter(MemoryProvider):
    """Zep implementation of MemoryProvider."""

    def __init__(self, api_key: str = None, base_url: str = None):
        if api_key and not base_url:
            # Cloud version
            self.client = AsyncZepCloud(api_key=api_key)
        elif api_key and base_url:
            # Self-hosted version with authentication
            self.client = AsyncZep(api_key=api_key, base_url=base_url)
        else:
            # Self-hosted version without authentication
            self.client = AsyncZep(base_url="http://localhost:8000")

    async def store(self, user_id: str, messages: List[Dict[str, Any]]) -> None:
        """Store messages in Zep memory."""
        zep_messages = [
            Message(
                role=msg["role"],
                role_type=msg["role"],
                content=self._truncate(msg["content"], 2500),
            )
            for msg in messages
        ]
        await self.client.memory.add(session_id=user_id, messages=zep_messages)

    async def retrieve(self, user_id: str) -> str:
        """Retrieve memory context for a user."""
        try:
            memory = await self.client.memory.get_session(user_id)
            summary = await self.client.memory.summarize(user_id)

            # Format the memory context
            context = f"Summary: {summary.summary}\n\n"

            # Add most relevant facts if available
            if (
                hasattr(memory, "metadata")
                and memory.metadata
                and "facts" in memory.metadata
            ):
                facts = memory.metadata["facts"]
                if facts:
                    context += "Key facts:\n"
                    for fact in facts[:5]:  # Limit to top 5 facts
                        context += f"- {fact['fact']}\n"

            return context
        except Exception as e:
            return f"Error retrieving memory: {e}"

    async def delete(self, user_id: str) -> None:
        """Delete memory for a user."""
        try:
            await self.client.memory.delete(session_id=user_id)
            await self.client.user.delete(user_id=user_id)
        except Exception as e:
            print(f"Error deleting memory: {e}")

    def _truncate(self, text: str, limit: int = 2500) -> str:
        """Truncate text to be within limits."""
        if len(text) <= limit:
            return text

        # Try to truncate at a sentence boundary
        truncated = text[:limit]
        last_period = truncated.rfind(".")
        if (
            last_period > limit * 0.8
        ):  # Only use period if it's reasonably close to the end
            return truncated[: last_period + 1]

        return truncated + "..."


class MongoMemoryProvider(MemoryProvider):
    """MongoDB implementation of MemoryProvider."""

    def __init__(self, db_adapter: DataStorageProvider):
        self.db = db_adapter
        self.collection = "messages"

        # Ensure collection exists
        self.db.create_collection(self.collection)

        # Create indexes
        self.db.create_index(self.collection, [("user_id", 1)])
        self.db.create_index(self.collection, [("timestamp", 1)])

    async def store(self, user_id: str, messages: List[Dict[str, Any]]) -> None:
        """Store messages in MongoDB."""
        for message in messages:
            doc = {
                "user_id": user_id,
                "role": message["role"],
                "content": message["content"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc)
            }
            self.db.insert_one(self.collection, doc)

    async def retrieve(self, user_id: str) -> str:
        """Retrieve memory context for a user."""
        # Get recent messages
        messages = self.db.find(
            self.collection,
            {"user_id": user_id},
            sort=[("timestamp", 1)],
            limit=10  # Adjust limit as needed
        )

        # Format as context string
        context = ""
        for msg in messages:
            context += f"{msg['role'].upper()}: {msg['content']}\n\n"

        return context

    async def delete(self, user_id: str) -> None:
        """Delete memory for a user."""
        self.db.delete_one(self.collection, {"user_id": user_id})


class DualMemoryProvider(MemoryProvider):
    """Memory provider that stores messages in both MongoDB and optional Zep."""

    def __init__(self, mongo_provider: MongoMemoryProvider, zep_provider: Optional[ZepMemoryAdapter] = None):
        self.mongo_provider = mongo_provider
        self.zep_provider = zep_provider

    async def store(self, user_id: str, messages: List[Dict[str, Any]]) -> None:
        """Store messages in both providers."""
        # Always store in MongoDB for UI history
        await self.mongo_provider.store(user_id, messages)

        # If Zep is configured, also store there for AI context
        if self.zep_provider:
            await self.zep_provider.store(user_id, messages)

    async def retrieve(self, user_id: str) -> str:
        """Retrieve memory context - prefer Zep if available."""
        if self.zep_provider:
            return await self.zep_provider.retrieve(user_id)
        else:
            return await self.mongo_provider.retrieve(user_id)

    async def delete(self, user_id: str) -> None:
        """Delete memory from both providers."""
        await self.mongo_provider.delete(user_id)
        if self.zep_provider:
            await self.zep_provider.delete(user_id)

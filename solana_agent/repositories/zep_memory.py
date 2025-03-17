"""
Zep implementation of the memory repository.

This repository leverages Zep for semantic memory and conversation history.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any

from zep_python.client import AsyncZep
from zep_cloud.client import AsyncZep as AsyncZepCloud
from zep_cloud.types import Message

from solana_agent.interfaces import MemoryRepository
from solana_agent.domains import MemoryInsight


class ZepMemoryRepository(MemoryRepository):
    """Zep implementation of MemoryRepository."""

    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize the Zep memory repository.

        Args:
            api_key: Optional Zep API key for cloud or authenticated self-hosted
            base_url: Optional base URL for self-hosted Zep
        """
        if api_key and not base_url:
            # Cloud version
            self.client = AsyncZepCloud(api_key=api_key)
        elif api_key and base_url:
            # Self-hosted version with authentication
            self.client = AsyncZep(api_key=api_key, base_url=base_url)
        else:
            # Self-hosted version without authentication
            self.client = AsyncZep(
                base_url=base_url or "http://localhost:8000")

        self.insights_collection = "insights"

    async def store_insight(self, user_id: str, insight: MemoryInsight) -> None:
        """Store a memory insight in Zep.

        Args:
            user_id: ID of the user the insight relates to
            insight: Memory insight to store
        """
        try:
            # Format insight as a memory message
            message = Message(
                role="system",
                role_type="insight",
                content=insight.content,
                metadata={
                    "category": insight.category or "general",
                    "confidence": insight.confidence,
                    "source": insight.source,
                    "timestamp": insight.timestamp.isoformat(),
                    **insight.metadata
                }
            )

            # Add to user's memory
            await self.client.memory.add(session_id=user_id, messages=[message])

            # Optionally add to a collection of insights
            # In a real implementation, you might add this to a searchable collection
        except Exception as e:
            print(f"Error storing insight in Zep: {e}")

    async def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory for insights matching a query using Zep's semantic search.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of matching memory items
        """
        try:
            # Use Zep search
            search_results = await self.client.memory.search(query, limit=limit)

            # Convert to consistent format
            results = []
            for result in search_results.results:
                message = result.message
                if not message:
                    continue

                # Extract metadata
                metadata = message.metadata or {}
                results.append({
                    "content": message.content,
                    "category": metadata.get("category", "general"),
                    "confidence": metadata.get("confidence", 0.0),
                    "score": result.score or 0.0,
                    "created_at": metadata.get("timestamp")
                })

            return results
        except Exception as e:
            print(f"Error searching Zep memory: {e}")
            return []

    async def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history for a user from Zep.

        Args:
            user_id: User ID
            limit: Maximum number of items to return

        Returns:
            List of conversation history items
        """
        try:
            # Get memory session
            memory = await self.client.memory.get_session(user_id)

            # Format conversation history
            history = []
            for i, message in enumerate(memory.messages[-limit:] if memory.messages else []):
                if message.role == "user":
                    user_msg = message.content
                    asst_msg = memory.messages[i+1].content if i + \
                        1 < len(memory.messages) else ""
                    history.append({
                        "user_message": user_msg,
                        "assistant_message": asst_msg,
                        "timestamp": datetime.now()  # Zep doesn't provide timestamps directly
                    })

            return history
        except Exception as e:
            print(f"Error retrieving user history from Zep: {e}")
            return []

    async def delete_user_memory(self, user_id: str) -> bool:
        """Delete all memory for a user from Zep.

        Args:
            user_id: User ID

        Returns:
            True if successful
        """
        try:
            # Delete memory session
            await self.client.memory.delete(session_id=user_id)

            # Delete user
            await self.client.user.delete(user_id=user_id)

            return True
        except Exception as e:
            print(f"Error deleting memory from Zep: {e}")
            return False

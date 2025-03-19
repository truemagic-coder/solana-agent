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

    async def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a user.

        Args:
            user_id: User ID
            limit: Maximum number of exchanges to return

        Returns:
            List of conversation exchanges
        """
        try:
            # Get session data from Zep
            session = await self.client.memory.get_session(user_id)

            # Process messages into exchanges (user + assistant pairs)
            exchanges = []
            messages = session.messages

            # Messages might be in reverse chronological order, so we need to process them differently
            # Group user and assistant messages correctly
            i = 0
            while i < len(messages) - 1 and len(exchanges) < limit:
                # Find a user message followed by an assistant message
                if messages[i].role == "user" and messages[i+1].role == "assistant":
                    exchanges.append({
                        "user_message": messages[i].content,
                        "assistant_message": messages[i+1].content,
                        "timestamp": datetime.now()
                    })
                    i += 2  # Move to the next potential pair
                else:
                    i += 1  # Skip this message if it doesn't form a valid pair

            return exchanges

        except Exception as e:
            # Handle errors and log them
            print(f"Error retrieving user history: {str(e)}")
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

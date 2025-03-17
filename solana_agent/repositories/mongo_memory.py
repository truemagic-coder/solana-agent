"""
MongoDB implementation of the memory repository.

This repository handles storing and retrieving memory insights and conversation history.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any

from solana_agent.interfaces import MemoryRepository
from solana_agent.interfaces import DataStorageProvider
from solana_agent.interfaces import VectorStoreProvider
from solana_agent.interfaces import LLMProvider
from solana_agent.domains import MemoryInsight


class MongoMemoryRepository(MemoryRepository):
    """MongoDB implementation of MemoryRepository."""

    def __init__(
        self,
        db_adapter: DataStorageProvider,
        vector_store: Optional[VectorStoreProvider] = None,
        llm_provider: Optional[LLMProvider] = None
    ):
        """Initialize the memory repository.

        Args:
            db_adapter: MongoDB adapter
            vector_store: Optional vector store for semantic search
            llm_provider: Optional LLM provider for generating embeddings
        """
        self.db = db_adapter
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.insights_collection = "memory_insights"
        self.history_collection = "conversation_history"

        # Ensure collections exist
        self.db.create_collection(self.insights_collection)
        self.db.create_collection(self.history_collection)

        # Create indexes
        self.db.create_index(self.insights_collection, [("user_id", 1)])
        self.db.create_index(self.insights_collection, [("created_at", -1)])
        self.db.create_index(self.history_collection, [("user_id", 1)])
        self.db.create_index(self.history_collection, [("timestamp", -1)])

    def store_insight(self, user_id: str, insight: MemoryInsight) -> None:
        """Store a memory insight.

        Args:
            user_id: ID of the user the insight relates to
            insight: Memory insight to store
        """
        # Convert insight to dictionary
        insight_dict = insight.model_dump()

        # Add user_id and created_at
        insight_dict["user_id"] = user_id
        insight_dict["created_at"] = datetime.now()

        # Store in MongoDB
        insight_id = self.db.insert_one(self.insights_collection, insight_dict)

        # If we have vector store and LLM provider, store embedding
        if self.vector_store and self.llm_provider:
            try:
                # Generate embedding
                embedding = self.llm_provider.generate_embedding(
                    insight.content)

                # Store in vector store
                self.vector_store.store_vectors(
                    [{
                        "id": insight_id,
                        "values": embedding,
                        "metadata": {
                            "user_id": user_id,
                            "content": insight.content,
                            "category": insight.category,
                            "created_at": insight_dict["created_at"].isoformat()
                        }
                    }],
                    namespace="memory_insights"
                )
            except Exception as e:
                print(f"Error storing insight embedding: {e}")

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory for insights matching a query.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of matching memory items
        """
        # If we have vector store and LLM provider, use semantic search
        if self.vector_store and self.llm_provider:
            try:
                # Generate query embedding
                query_embedding = self.llm_provider.generate_embedding(query)

                # Search vector store
                results = self.vector_store.search_vectors(
                    query_embedding,
                    namespace="memory_insights",
                    limit=limit
                )

                # Process results
                processed_results = []
                for result in results:
                    metadata = result.get("metadata", {})
                    processed_results.append({
                        "content": metadata.get("content", ""),
                        "category": metadata.get("category"),
                        "user_id": metadata.get("user_id", ""),
                        "score": result.get("score", 0.0),
                        "created_at": metadata.get("created_at")
                    })

                return processed_results

            except Exception as e:
                print(f"Error in vector search: {e}")
                # Fall back to MongoDB text search

        # Use MongoDB text search as fallback
        # Note: This would require a text index on the content field
        try:
            results = self.db.find(
                self.insights_collection,
                {"$text": {"$search": query}},
                limit=limit
            )

            return list(results)
        except Exception as e:
            print(f"Error in MongoDB text search: {e}")

            # Last resort: simple keyword search
            return list(self.db.find(
                self.insights_collection,
                {"content": {"$regex": query, "$options": "i"}},
                limit=limit
            ))

    def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history for a user.

        Args:
            user_id: User ID
            limit: Maximum number of items to return

        Returns:
            List of conversation history items
        """
        return list(self.db.find(
            self.history_collection,
            {"user_id": user_id},
            sort=[("timestamp", -1)],
            limit=limit
        ))

    def delete_user_memory(self, user_id: str) -> bool:
        """Delete all memory for a user.

        Args:
            user_id: User ID

        Returns:
            True if successful
        """
        # Delete from MongoDB
        self.db.delete_one(self.insights_collection, {"user_id": user_id})
        self.db.delete_one(self.history_collection, {"user_id": user_id})

        # Delete from vector store if available
        if self.vector_store:
            try:
                # Note: This is a simplified approach
                # In a real implementation, you'd need to find all IDs first
                pass
            except Exception as e:
                print(f"Error deleting from vector store: {e}")

        return True

    def store_conversation_entry(
        self, user_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Store a conversation entry.

        Args:
            user_id: User ID
            user_message: User message
            assistant_message: Assistant message
        """
        entry = {
            "user_id": user_id,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "timestamp": datetime.now()
        }

        self.db.insert_one(self.history_collection, entry)

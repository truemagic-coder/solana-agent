"""
MongoDB implementation of the memory repository.

This repository handles storing and retrieving memory insights and conversation history.
"""
from datetime import datetime
from typing import Dict, List, Optional

from solana_agent.interfaces.repositories.memory import MemoryRepository
from solana_agent.interfaces.providers.data_storage import DataStorageProvider
from solana_agent.interfaces.providers.vector_storage import VectorStorageProvider
from solana_agent.interfaces.providers.llm import LLMProvider
from solana_agent.domains.memory import MemoryInsight


class MongoMemoryRepository(MemoryRepository):
    """MongoDB implementation of MemoryRepository."""

    def __init__(
        self,
        db_adapter: DataStorageProvider,
        vector_store: Optional[VectorStorageProvider] = None,
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
        self.history_collection = "messages"

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

    def count_user_history(self, user_id: str) -> int:
        """Count the number of conversation pairs for a user.

        Args:
            user_id: User ID to count history for

        Returns:
            Number of conversation pairs (user messages)
        """
        try:
            # Count only user messages to get the number of exchanges
            count = self.db.count_documents(
                self.history_collection,
                {"user_id": user_id, "role": "user"}
            )
            return count
        except Exception as e:
            print(f"Error counting user history: {e}")
            return 0

    def get_user_history_paginated(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 20,
        sort_order: str = "asc"
    ) -> List[Dict]:
        """Get paginated conversation history for a user.

        Args:
            user_id: User ID
            skip: Number of items to skip (for pagination)
            limit: Maximum number of items to return per page
            sort_order: Sort order - 'asc' for oldest first, 'desc' for newest first

        Returns:
            List of conversation history items for the requested page
        """
        try:
            # Determine sort direction based on sort_order parameter
            sort_direction = -1 if sort_order.lower() == "desc" else 1

            # We need to fetch more records to ensure we get complete user-assistant pairs
            fetch_limit = limit * 2  # Fetch twice as many to ensure we get pairs

            # Query the database for records
            results = self.db.find(
                self.history_collection,
                {"user_id": user_id},
                sort=[("timestamp", sort_direction)],
                skip=skip,
                limit=fetch_limit
            )

            # Convert to list of documents
            documents = list(results)

            # Group messages into conversation pairs
            conversation_pairs = []
            user_messages = []
            assistant_messages = []

            # First pass: separate user and assistant messages
            for doc in documents:
                role = doc.get("role")
                content = doc.get("content", "")

                if role == "user":
                    user_messages.append({
                        "content": content,
                        "timestamp": doc.get("timestamp"),
                        "message_id": doc.get("_id")
                    })
                elif role == "assistant":
                    assistant_messages.append({
                        "content": content,
                        "timestamp": doc.get("timestamp"),
                        "message_id": doc.get("_id")
                    })

            # Match user messages with their corresponding assistant responses
            for i, user_msg in enumerate(user_messages):
                if i < len(assistant_messages):
                    # Create a paired conversation entry
                    conversation_pairs.append({
                        "user_id": user_id,
                        "user_message": user_msg["content"],
                        "assistant_message": assistant_messages[i]["content"],
                        "timestamp": user_msg["timestamp"],
                        "user_message_id": user_msg["message_id"],
                        "assistant_message_id": assistant_messages[i]["message_id"]
                    })
                else:
                    # User message without a response
                    conversation_pairs.append({
                        "user_id": user_id,
                        "user_message": user_msg["content"],
                        "assistant_message": "",
                        "timestamp": user_msg["timestamp"],
                        "user_message_id": user_msg["message_id"],
                        "assistant_message_id": None
                    })

            # Limit to the requested number of pairs
            return conversation_pairs[:limit]

        except Exception as e:
            print(f"Error retrieving paginated user history: {e}")
            return []

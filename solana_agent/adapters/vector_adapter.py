"""
Vector database adapters for the Solana Agent system.

These adapters implement the VectorStoreProvider interface for different vector databases.
"""
from typing import Dict, List, Optional, Any

from pinecone import Pinecone

from solana_agent.interfaces import VectorStoreProvider


class PineconeAdapter(VectorStoreProvider):
    """Pinecone implementation of VectorStoreProvider."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.client = Pinecone(api_key=api_key)
        self.index = self.client.Index(index_name)
        self.embedding_model = embedding_model

    def store_vectors(self, vectors: List[Dict], namespace: str) -> None:
        """Store vectors in Pinecone."""
        self.index.upsert(vectors=vectors, namespace=namespace)

    def search_vectors(
        self, query_vector: List[float], namespace: str, limit: int = 5
    ) -> List[Dict]:
        """Search for similar vectors."""
        results = self.index.query(
            vector=query_vector, top_k=limit, include_metadata=True, namespace=namespace
        )

        # Format results
        output = []
        if hasattr(results, "matches"):
            for match in results.matches:
                if hasattr(match, "metadata") and match.metadata:
                    output.append(
                        {
                            "id": match.id,
                            "score": match.score,
                            "metadata": match.metadata,
                        }
                    )

        return output

    def delete_vector(self, id: str, namespace: str) -> None:
        """Delete a vector by ID."""
        self.index.delete(ids=[id], namespace=namespace)


class QdrantAdapter(VectorStoreProvider):
    """Qdrant implementation of VectorStoreProvider."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "solana_agent",
        embedding_model: str = "text-embedding-3-small",
        vector_size: int = 1536,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError:
            raise ImportError(
                "Qdrant support requires the qdrant-client package. Install it with 'pip install qdrant-client'"
            )

        # Initialize Qdrant client
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_size = vector_size

        # Ensure collection exists
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if collection_name not in collection_names:
                from qdrant_client.http import models

                # Create collection with default configuration
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size, distance=models.Distance.COSINE
                    ),
                )
        except Exception as e:
            print(f"Error initializing Qdrant collection: {e}")

    def store_vectors(self, vectors: List[Dict], namespace: str) -> None:
        """Store vectors in Qdrant."""
        try:
            from qdrant_client.http import models

            # Convert input format to Qdrant format
            points = []
            for vector in vectors:
                points.append(
                    models.PointStruct(
                        id=vector["id"],
                        vector=vector["values"],
                        payload={
                            # Add namespace as a metadata field
                            "namespace": namespace,
                            **vector.get("metadata", {}),
                        },
                    )
                )

            # Upsert vectors
            self.client.upsert(
                collection_name=self.collection_name, points=points)
        except Exception as e:
            print(f"Error storing vectors in Qdrant: {e}")

    def search_vectors(
        self, query_vector: List[float], namespace: str, limit: int = 5
    ) -> List[Dict]:
        """Search for similar vectors in specified namespace."""
        try:
            from qdrant_client.http import models

            # Perform search with namespace filter
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="namespace", match=models.MatchValue(value=namespace)
                        )
                    ]
                ),
                limit=limit,
            )

            # Format results to match the expected output format
            output = []
            for result in search_result:
                output.append(
                    {"id": result.id, "score": result.score,
                        "metadata": result.payload}
                )

            return output
        except Exception as e:
            print(f"Error searching vectors in Qdrant: {e}")
            return []

    def delete_vector(self, id: str, namespace: str) -> None:
        """Delete a vector by ID from a specific namespace."""
        try:
            from qdrant_client.http import models

            # Delete with both ID and namespace filter (to ensure we're deleting from the right namespace)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[id],
                ),
                wait=True,
            )
        except Exception as e:
            print(f"Error deleting vector from Qdrant: {e}")

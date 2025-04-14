from datetime import datetime as dt
import datetime
from typing import Dict, List, Any, Optional, Union
import uuid
import asyncio

from solana_agent.adapters.pinecone_adapter import PineconeAdapter
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
from solana_agent.interfaces.services.knowledge_base import KnowledgeService as KnowledgeServiceInterface


class KnowledgeBase(KnowledgeServiceInterface):
    """
    Knowledge Base service that uses Pinecone for vector search and MongoDB for metadata storage.
    Provides an integrated solution for storing, retrieving, and managing knowledge documents.
    """

    def __init__(
        self,
        pinecone_adapter: PineconeAdapter,
        mongodb_adapter: MongoDBAdapter,
        collection_name: str = "knowledge_documents",
        rerank_results: bool = False,
        rerank_top_k: int = 3
    ):
        """
        Initialize the Knowledge Base service.

        Args:
            pinecone_adapter: Configured PineconeAdapter instance
            mongodb_adapter: Configured MongoDBAdapter instance
            collection_name: MongoDB collection for storing document metadata
            rerank_results: Whether to rerank results (requires Pinecone adapter with reranking enabled)
            rerank_top_k: Number of results to return after reranking
        """
        self.pinecone = pinecone_adapter
        self.mongo = mongodb_adapter
        self.collection = collection_name
        self.rerank_results = rerank_results
        self.rerank_top_k = rerank_top_k

        # Ensure the collection exists with proper indexes
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Set up MongoDB collection with appropriate indexes."""
        if not self.mongo.collection_exists(self.collection):
            self.mongo.create_collection(self.collection)

        # Create indexes for faster retrieval
        self.mongo.create_index(
            self.collection, [("document_id", 1)], unique=True)
        self.mongo.create_index(self.collection, [("source", 1)])
        self.mongo.create_index(self.collection, [("created_at", -1)])
        self.mongo.create_index(self.collection, [("tags", 1)])

    async def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> str:
        """
        Add a document to the knowledge base.

        Args:
            text: Document text content to be embedded
            metadata: Document metadata to store in MongoDB
            document_id: Optional document ID (will generate UUID if not provided)
            namespace: Optional Pinecone namespace

        Returns:
            The document ID
        """
        # Generate or use provided document ID
        doc_id = document_id or str(uuid.uuid4())

        # Store metadata in MongoDB
        mongo_doc = {
            "document_id": doc_id,
            "content": text,  # Store full text content
            **metadata,  # Include all provided metadata
            "created_at": metadata.get("created_at", dt.now()),
            "updated_at": dt.now()
        }

        # Insert into MongoDB first to ensure metadata is available
        self.mongo.insert_one(self.collection, mongo_doc)

        # Store embedding in Pinecone with minimal metadata
        pinecone_metadata = {
            "document_id": doc_id,  # This links back to MongoDB
            # Optional source for filtering
            "source": metadata.get("source", "unknown"),
            "tags": metadata.get("tags", [])  # Optional tags for filtering
        }

        # Add embedding to Pinecone
        await self.pinecone.upsert_text(
            texts=[text],
            ids=[doc_id],
            metadatas=[pinecone_metadata],
            namespace=namespace
        )

        return doc_id

    async def query(
        self,
        query_text: str,
        filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        namespace: Optional[str] = None,
        include_content: bool = True,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base with semantic search.

        Args:
            query_text: Search query text
            filter: Optional filter criteria for Pinecone
            top_k: Maximum number of results to return
            namespace: Optional Pinecone namespace
            include_content: Whether to include document content in results
            include_metadata: Whether to include document metadata in results

        Returns:
            List of matching documents with combined data from Pinecone and MongoDB
        """
        # Set the effective top_k based on configuration and parameters
        effective_top_k = self.rerank_top_k if self.rerank_results else top_k

        # Perform semantic search in Pinecone
        try:
            pinecone_results = await self.pinecone.query_text(
                query_text=query_text,
                filter=filter,
                top_k=effective_top_k,
                namespace=namespace,
                include_values=False,
                include_metadata=True
            )
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []  # Return empty list on Pinecone error

        if not pinecone_results:
            return []

        # Extract document IDs from Pinecone results
        doc_ids = [match.get("id") for match in pinecone_results]
        scores = {match.get("id"): match.get("score")
                  for match in pinecone_results}

        # Get full metadata from MongoDB
        mongo_docs = self.mongo.find(
            self.collection,
            {"document_id": {"$in": doc_ids}}
        )

        # Create a lookup map for efficient document retrieval
        mongo_docs_map = {doc["document_id"]: doc for doc in mongo_docs}

        # Combine results in the same order as Pinecone returned them
        results = []
        for doc_id in doc_ids:
            if doc_id in mongo_docs_map:
                mongo_doc = mongo_docs_map[doc_id]
                result = {
                    "document_id": doc_id,
                    "score": scores.get(doc_id, 0.0),
                }

                # Include content if requested
                if include_content:
                    result["content"] = mongo_doc.get("content", "")

                # Include metadata if requested
                if include_metadata:
                    # Copy all fields except internal ones
                    result["metadata"] = {
                        k: v for k, v in mongo_doc.items()
                        if k not in ["_id", "document_id", "content"]
                    }

                results.append(result)

        return results

    async def delete_document(
        self,
        document_id: str,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Delete a document from both Pinecone and MongoDB.

        Args:
            document_id: ID of document to delete
            namespace: Optional Pinecone namespace

        Returns:
            True if document was deleted successfully
        """
        # Delete from Pinecone first
        try:
            await self.pinecone.delete(ids=[document_id], namespace=namespace)
        except Exception as e:
            print(f"Error deleting document {document_id} from Pinecone: {e}")
            return False

        # Then delete from MongoDB
        mongo_deleted = self.mongo.delete_one(
            self.collection,
            {"document_id": document_id}
        )

        return mongo_deleted

    async def update_document(
        self,
        document_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Update an existing document in the knowledge base.

        Args:
            document_id: ID of document to update
            text: Optional new text content (if not provided, only metadata will be updated)
            metadata: Optional metadata to update
            namespace: Optional Pinecone namespace

        Returns:
            True if document was updated successfully
        """
        # Fetch current document to get text if not provided
        current_doc = self.mongo.find_one(
            self.collection, {"document_id": document_id})
        if not current_doc:
            print(f"Document {document_id} not found in MongoDB")
            return False

        update_text = text is not None
        text_content = text if update_text else current_doc.get("content", "")

        # Update MongoDB document
        mongo_update = {}

        if metadata:
            for key, value in metadata.items():
                mongo_update[key] = value

        if update_text:
            mongo_update["content"] = text_content

        mongo_update["updated_at"] = dt.now()

        self.mongo.update_one(
            self.collection,
            {"document_id": document_id},
            {"$set": mongo_update}
        )

        # Only update Pinecone if text content changed (requires re-embedding)
        if update_text:
            # Minimal metadata for Pinecone
            pinecone_metadata = {
                "document_id": document_id,
                "source": metadata.get("source", current_doc.get("source", "unknown")),
                "tags": metadata.get("tags", current_doc.get("tags", []))
            }

            await self.pinecone.upsert_text(
                texts=[text_content],
                ids=[document_id],
                metadatas=[pinecone_metadata],
                namespace=namespace
            )

        return True

    async def add_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 50
    ) -> List[str]:
        """
        Add multiple documents in batches.

        Args:
            documents: List of documents, each with 'text' and 'metadata' fields
            namespace: Optional Pinecone namespace
            batch_size: Number of documents per batch

        Returns:
            List of document IDs
        """
        document_ids = []

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_ids = [doc.get("document_id", str(uuid.uuid4()))
                         for doc in batch]
            document_ids.extend(batch_ids)

            # Prepare MongoDB documents
            mongo_docs = []
            for j, doc in enumerate(batch):
                mongo_doc = {
                    "document_id": batch_ids[j],
                    "content": doc["text"],
                    **doc["metadata"],
                    "created_at": doc["metadata"].get("created_at", dt.now()),
                    "updated_at": dt.now()
                }
                mongo_docs.append(mongo_doc)

            # Insert MongoDB documents
            for doc in mongo_docs:
                self.mongo.insert_one(self.collection, doc)

            # Prepare Pinecone documents
            texts = [doc["text"] for doc in batch]
            pinecone_metadatas = [
                {
                    "document_id": batch_ids[j],
                    "source": doc["metadata"].get("source", "unknown"),
                    "tags": doc["metadata"].get("tags", [])
                }
                for j, doc in enumerate(batch)
            ]

            # Insert into Pinecone
            await self.pinecone.upsert_text(
                texts=texts,
                ids=batch_ids,
                metadatas=pinecone_metadatas,
                namespace=namespace
            )

            # Add small delay between batches, but not after the last batch
            if i + batch_size < len(documents):
                await asyncio.sleep(0.5)

        return document_ids

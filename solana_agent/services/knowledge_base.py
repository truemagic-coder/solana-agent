import logging
from datetime import datetime as dt
from typing import Dict, List, Any, Optional, Union
import uuid
import asyncio
import io

# PDF and Chunking Imports
import pypdf
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

from solana_agent.adapters.pinecone_adapter import PineconeAdapter
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
from solana_agent.interfaces.services.knowledge_base import (
    KnowledgeBaseService as KnowledgeBaseInterface,
)

# Setup logger for this module
logger = logging.getLogger(__name__)


class KnowledgeBaseService(KnowledgeBaseInterface):
    """
    Knowledge Base service using Pinecone for vector search and MongoDB for metadata/chunk storage.
    Supports text documents and PDF semantic chunking using OpenAI embeddings via LlamaIndex.
    PDF binary data is not stored. Chunks are stored individually in MongoDB.
    """

    def __init__(
        self,
        pinecone_adapter: PineconeAdapter,
        mongodb_adapter: MongoDBAdapter,
        openai_api_key: str,
        openai_model_name: str = "text-embedding-3-large",
        collection_name: str = "knowledge_documents",
        rerank_results: bool = False,
        rerank_top_k: int = 3,
        # Semantic Splitter Config
        splitter_buffer_size: int = 1,
        splitter_breakpoint_percentile: int = 95,
    ):
        """
        Initialize the Knowledge Base service.

        Args:
            pinecone_adapter: Configured PineconeAdapter instance.
            mongodb_adapter: Configured MongoDBAdapter instance.
            openai_api_key: OpenAI API key for embedding.
            openai_model_name: OpenAI embedding model name.
            collection_name: MongoDB collection for storing document metadata and chunks.
            rerank_results: Whether PineconeAdapter should rerank results.
            rerank_top_k: Number of results to return after reranking (used by PineconeAdapter).
            splitter_buffer_size: Buffer size for SemanticSplitterNodeParser.
            splitter_breakpoint_percentile: Breakpoint percentile for SemanticSplitterNodeParser.
        """
        self.pinecone = pinecone_adapter
        self.mongo = mongodb_adapter
        self.collection = collection_name
        # Store rerank config for reference, but PineconeAdapter applies it
        self.rerank_results = rerank_results
        self.rerank_top_k = rerank_top_k

        # --- Initialize Semantic Splitter with OpenAIEmbedding ---
        api_key = openai_api_key
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided via argument or OPENAI_API_KEY environment variable."
            )

        # Determine expected embedding dimensions based on model name
        if openai_model_name == "text-embedding-3-large":
            openai_dimensions = 3072
        elif openai_model_name == "text-embedding-3-small":
            openai_dimensions = 1536
        else:
            # Attempt to get dimension from Pinecone config if available, otherwise raise error
            openai_dimensions = getattr(self.pinecone, "embedding_dimensions", 0)
            if openai_dimensions <= 0:
                raise ValueError(
                    f"Cannot determine dimension for unknown OpenAI model '{openai_model_name}' and Pinecone dimension not configured."
                )
            logger.warning(  # Use logger.warning
                f"Unknown OpenAI model '{openai_model_name}'. Using dimension {openai_dimensions} from Pinecone config. Ensure this is correct."
            )

        # Instantiate OpenAIEmbedding
        # Note: LlamaIndex OpenAIEmbedding doesn't directly support reducing dimensions via 'dimensions' param during init
        # like the raw OpenAI client does. It uses the model's default.
        try:
            llama_embed_model = OpenAIEmbedding(
                model=openai_model_name,
                api_key=api_key,
                # embed_batch_size=10 # Optional: Adjust batch size if needed
            )
        except Exception as e:
            logger.error(f"Error initializing OpenAIEmbedding: {e}")  # Use logger.error
            raise

        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=splitter_buffer_size,
            breakpoint_percentile_threshold=splitter_breakpoint_percentile,
            embed_model=llama_embed_model,  # Use the OpenAIEmbedding instance
        )
        # Store model name for logging/reference
        self.openai_model_name = openai_model_name
        # --- End Semantic Splitter Init ---

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Set up MongoDB collection with appropriate indexes."""
        if not self.mongo.collection_exists(self.collection):
            self.mongo.create_collection(self.collection)
            logger.info(
                f"Created MongoDB collection: {self.collection}"
            )  # Use logger.info

        # Indexes for retrieval and filtering
        self.mongo.create_index(self.collection, [("document_id", 1)], unique=True)
        self.mongo.create_index(self.collection, [("parent_document_id", 1)])
        self.mongo.create_index(self.collection, [("source", 1)])
        self.mongo.create_index(self.collection, [("created_at", -1)])
        self.mongo.create_index(self.collection, [("tags", 1)])
        self.mongo.create_index(self.collection, [("is_chunk", 1)])
        logger.info(
            f"Ensured indexes exist for MongoDB collection: {self.collection}"
        )  # Use logger.info

    async def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> str:  # pragma: no cover
        """
        Add a plain text document to the knowledge base. Embeds using OpenAI.

        Args:
            text: Document text content.
            metadata: Document metadata.
            document_id: Optional document ID.
            namespace: Optional Pinecone namespace.

        Returns:
            The document ID.
        """
        doc_id = document_id or str(uuid.uuid4())
        now = dt.now(tz=dt.now().astimezone().tzinfo)

        # Store metadata and content in MongoDB
        mongo_doc = {
            "document_id": doc_id,
            "content": text,
            "is_chunk": False,
            "parent_document_id": None,
            **metadata,
            # Use timezone aware datetime
            "created_at": metadata.get("created_at", now),
            "updated_at": now,
        }
        try:
            self.mongo.insert_one(self.collection, mongo_doc)
        except Exception as e:
            logger.error(
                f"Error inserting document {doc_id} into MongoDB: {e}"
            )  # Use logger.error
            raise

        # Embed text using OpenAIEmbedding
        embed_model: OpenAIEmbedding = self.semantic_splitter.embed_model
        try:
            embedding = await embed_model.aget_text_embedding(text)
        except Exception as e:
            logger.error(  # Use logger.error
                f"Error embedding document {doc_id} using {self.openai_model_name}: {e}"
            )
            # Decide how to handle - Mongo insert succeeded, embedding failed
            raise  # Re-raise for now

        # Prepare Pinecone metadata
        pinecone_metadata = {
            "document_id": doc_id,
            "is_chunk": False,
            "parent_document_id": False,  # Explicitly set for clarity - Pinecone can't use None
            "source": metadata.get("source", "unknown"),
            "tags": metadata.get("tags", []),
        }
        # Add text itself if Pinecone adapter reranking is used
        if self.pinecone.use_reranking:
            pinecone_metadata[self.pinecone.rerank_text_field] = text

        # Upsert vector to Pinecone using the generic upsert method
        try:
            await self.pinecone.upsert(
                vectors=[
                    {"id": doc_id, "values": embedding, "metadata": pinecone_metadata}
                ],
                namespace=namespace,
            )
        except Exception as e:
            logger.error(
                f"Error upserting vector for {doc_id} to Pinecone: {e}"
            )  # Use logger.error
            # Decide how to handle - Mongo insert succeeded, Pinecone failed
            raise  # Re-raise for now

        return doc_id

    async def add_pdf_document(
        self,
        pdf_data: Union[bytes, str],
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        chunk_batch_size: int = 50,
    ) -> str:  # pragma: no cover
        """
        Add a PDF document, performs semantic chunking using OpenAI embeddings,
        stores parent metadata and individual chunks in Mongo, and chunk vectors in Pinecone.
        Full PDF binary is NOT stored.

        Args:
            pdf_data: PDF content as bytes or a path to the PDF file.
            metadata: Document metadata (applies to the parent PDF).
            document_id: Optional parent document ID.
            namespace: Optional Pinecone namespace for chunks.
            chunk_batch_size: Batch size for embedding and upserting chunks.

        Returns:
            The parent document ID.
        """
        parent_doc_id = document_id or str(uuid.uuid4())
        pdf_bytes: bytes
        now = dt.now(tz=dt.now().astimezone().tzinfo)

        # --- 1. Read PDF and Extract Text ---
        try:
            if isinstance(pdf_data, str):
                with open(pdf_data, "rb") as f:
                    pdf_bytes = f.read()
            elif isinstance(pdf_data, bytes):
                pdf_bytes = pdf_data
            else:
                raise ValueError("pdf_data must be bytes or a file path string.")

            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            extracted_text = "".join(page.extract_text() or "" for page in reader.pages)
            if not extracted_text.strip():
                logger.warning(
                    f"No text extracted from PDF {parent_doc_id}."
                )  # Use logger.warning
                # Still store parent metadata even if no text
        except Exception as e:
            logger.error(
                f"Error reading or extracting text from PDF {parent_doc_id}: {e}"
            )  # Use logger.error
            raise

        # --- 2. Store Parent PDF Metadata in MongoDB (NO BINARY) ---
        mongo_parent_doc = {
            "document_id": parent_doc_id,
            "content": None,
            "is_chunk": False,
            "parent_document_id": None,
            **metadata,
            "created_at": metadata.get("created_at", now),
            "updated_at": now,
        }
        try:
            self.mongo.insert_one(self.collection, mongo_parent_doc)
            logger.info(
                f"Stored parent metadata for PDF {parent_doc_id} in MongoDB."
            )  # Use logger.info
        except Exception as e:
            logger.error(
                f"Error inserting parent PDF metadata {parent_doc_id} into MongoDB: {e}"
            )
            raise

        # --- 3. Semantic Chunking ---
        if not extracted_text.strip():
            logger.info(  # Use logger.info
                f"Skipping chunking for PDF {parent_doc_id} due to no extracted text."
            )
            return parent_doc_id  # Return parent ID even if no chunks

        try:
            llama_doc = LlamaDocument(text=extracted_text)
            # Run synchronous splitter in thread pool
            nodes = await asyncio.to_thread(
                self.semantic_splitter.get_nodes_from_documents, [llama_doc]
            )
            logger.info(
                f"Generated {len(nodes)} semantic chunks for PDF {parent_doc_id}."
            )  # Use logger.info
        except Exception as e:
            logger.error(
                f"Error during semantic chunking for PDF {parent_doc_id}: {e}"
            )  # Use logger.error
            # Parent metadata is already stored, decide how to proceed. Raising for now.
            raise

        # --- 4. Embed Chunks and Batch Upsert to Pinecone AND Store Chunks in MongoDB ---
        if not nodes:
            return parent_doc_id  # No chunks generated

        logger.info(
            f"Embedding {len(nodes)} chunks using {self.openai_model_name}..."
        )  # Use logger.info
        chunk_texts = [node.get_content() for node in nodes]
        embed_model: OpenAIEmbedding = self.semantic_splitter.embed_model
        all_chunk_embeddings = []

        # Embed chunks in batches (using embed_model's internal batching)
        try:
            # Use aget_text_embedding_batch for async embedding
            all_chunk_embeddings = await embed_model.aget_text_embedding_batch(
                chunk_texts, show_progress=True
            )
        except Exception as e:
            logger.error(
                f"Error embedding chunks for PDF {parent_doc_id}: {e}"
            )  # Use logger.error
            raise  # Stop if embedding fails

        logger.info(
            "Embedding complete. Preparing vectors for Pinecone and documents for MongoDB."
        )  # Use logger.info
        pinecone_vectors = []
        mongo_chunk_docs = []
        chunk_now = dt.now(
            tz=dt.now().astimezone().tzinfo
        )  # Consistent timestamp for chunks

        for i, node in enumerate(nodes):
            chunk_id = f"{parent_doc_id}_chunk_{i}"
            chunk_text = chunk_texts[i]

            # Prepare Pinecone Vector Metadata
            pinecone_chunk_metadata = {
                "document_id": chunk_id,  # Pinecone ID is the chunk ID
                "parent_document_id": parent_doc_id,
                "chunk_index": i,
                "is_chunk": True,
                "source": metadata.get("source", "unknown"),  # Inherit from parent
                "tags": metadata.get("tags", []),  # Inherit from parent
            }
            # Add chunk text itself if Pinecone adapter reranking is used
            if self.pinecone.use_reranking:
                pinecone_chunk_metadata[self.pinecone.rerank_text_field] = chunk_text

            pinecone_vectors.append(
                {
                    "id": chunk_id,
                    "values": all_chunk_embeddings[i],
                    "metadata": pinecone_chunk_metadata,
                }
            )

            # Prepare MongoDB Chunk Document
            mongo_chunk_doc = {
                "document_id": chunk_id,  # Mongo ID is the chunk ID
                "parent_document_id": parent_doc_id,
                "chunk_index": i,
                "is_chunk": True,
                "content": chunk_text,  # Store chunk text in Mongo
                "source": metadata.get("source", "unknown"),  # Inherit from parent
                "tags": metadata.get("tags", []),  # Inherit from parent
                # Add other relevant parent metadata if needed, avoid duplication if possible
                "created_at": chunk_now,  # Use consistent time for batch
                "updated_at": chunk_now,
            }
            mongo_chunk_docs.append(mongo_chunk_doc)

        # --- 5. Store Chunks in MongoDB ---
        if mongo_chunk_docs:
            try:
                self.mongo.insert_many(self.collection, mongo_chunk_docs)
                logger.info(
                    f"Stored {len(mongo_chunk_docs)} chunks in MongoDB for parent {parent_doc_id}."
                )
            except Exception as e:
                logger.error(
                    f"Error inserting chunks into MongoDB for parent {parent_doc_id}: {e}"
                )
                # Decide how to handle: Pinecone upsert might still proceed or fail.
                # For now, log the error and continue to Pinecone upsert attempt.

        # --- 6. Upsert Chunk Vectors to Pinecone in Batches ---
        if not pinecone_vectors:
            logger.warning(f"No vectors generated to upsert for PDF {parent_doc_id}.")
            return parent_doc_id

        logger.info(  # Use logger.info
            f"Upserting {len(pinecone_vectors)} chunk vectors to Pinecone in batches of {chunk_batch_size}..."
        )
        upsert_tasks = []
        for i in range(0, len(pinecone_vectors), chunk_batch_size):
            batch_vectors = pinecone_vectors[i : i + chunk_batch_size]
            # Create task for each batch upsert
            upsert_tasks.append(
                self.pinecone.upsert(vectors=batch_vectors, namespace=namespace)
            )
            # Optional: Add a small delay between initiating tasks if rate limiting is a concern
            # await asyncio.sleep(0.05)

        # Run upsert tasks concurrently
        results = await asyncio.gather(*upsert_tasks, return_exceptions=True)

        # Check for errors during upsert
        upsert_errors = False
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                upsert_errors = True
                logger.error(
                    f"Error upserting vector batch {idx + 1} to Pinecone for parent {parent_doc_id}: {result}"
                )  # Use logger.error
                # Decide on error handling: log, raise, etc. Consider cleanup?

        if upsert_errors:
            logger.warning(
                f"Some errors occurred during Pinecone vector upsert for {parent_doc_id}."
            )
            # Consider if partial success requires specific handling or cleanup

        logger.info(f"Finished processing PDF {parent_doc_id}.")  # Use logger.info
        return parent_doc_id

    async def query(
        self,
        query_text: str,
        filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        namespace: Optional[str] = None,
        include_content: bool = True,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:  # pragma: no cover
        """
        Query the knowledge base using semantic search with OpenAI embeddings.
        Retrieves chunk or document content and metadata from MongoDB based on Pinecone results.

        Args:
            query_text: The query text.
            filter: Optional Pinecone metadata filter.
            top_k: Number of results to retrieve initially.
            namespace: Optional Pinecone namespace.
            include_content: Whether to include document/chunk content in results.
            include_metadata: Whether to include document/chunk metadata in results.

        Returns:
            List of result dictionaries, potentially reranked by PineconeAdapter.
        """
        # Determine initial K for Pinecone query, considering reranking multiplier
        initial_k = top_k
        if self.pinecone.use_reranking:
            initial_k = top_k * self.pinecone.initial_query_top_k_multiplier

        # --- Embed Query using OpenAIEmbedding ---
        embed_model: OpenAIEmbedding = self.semantic_splitter.embed_model
        try:
            query_vector = await embed_model.aget_query_embedding(query_text)
        except Exception as e:
            logger.error(
                f"Error embedding query text '{query_text}': {e}"
            )  # Use logger.error
            return []

        # --- Query Pinecone using the vector ---
        try:
            # Use the generic query method with the vector
            # PineconeAdapter handles reranking internally if configured
            pinecone_results = await self.pinecone.query(
                vector=query_vector,
                filter=filter,
                top_k=initial_k,  # Fetch more initially if reranking
                namespace=namespace,
                include_values=False,  # Don't need embeddings back
                include_metadata=True,  # Need metadata for linking
            )
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")  # Use logger.error
            return []

        if not pinecone_results:
            return []

        # Extract IDs, scores, and metadata from Pinecone results
        # PineconeAdapter might have already reranked and truncated to final top_k
        result_ids = [
            res["id"] for res in pinecone_results
        ]  # These are chunk IDs or plain doc IDs
        scores = {res["id"]: res["score"] for res in pinecone_results}
        pinecone_metadatas = {
            res["id"]: res.get("metadata", {}) for res in pinecone_results
        }

        # --- Fetch corresponding data from MongoDB ---
        # We need:
        # 1. Chunk documents (using result_ids where is_chunk is True)
        # 2. Parent documents (using parent_document_id from chunk metadata)
        # 3. Plain documents (using result_ids where is_chunk is False)
        chunk_ids_to_fetch = set()
        parent_ids_to_fetch = set()
        plain_doc_ids_to_fetch = set()

        for res_id in result_ids:
            meta = pinecone_metadatas.get(res_id, {})
            if meta.get("is_chunk"):
                chunk_ids_to_fetch.add(res_id)
                parent_id = meta.get("parent_document_id")
                if parent_id:
                    parent_ids_to_fetch.add(parent_id)
            else:
                plain_doc_ids_to_fetch.add(res_id)

        # Fetch all required docs from Mongo in potentially fewer queries
        mongo_docs_map = {}
        ids_to_fetch_mongo = list(
            chunk_ids_to_fetch | parent_ids_to_fetch | plain_doc_ids_to_fetch
        )

        if ids_to_fetch_mongo:
            try:
                mongo_docs = self.mongo.find(
                    self.collection, {"document_id": {"$in": ids_to_fetch_mongo}}
                )
                mongo_docs_map = {doc["document_id"]: doc for doc in mongo_docs}
            except Exception as e:
                logger.error(
                    f"Error fetching documents from MongoDB: {e}"
                )  # Use logger.error
                # Proceed with potentially missing Mongo data

        # --- Combine Results ---
        results = []
        for res_id in result_ids:  # Iterate in Pinecone's score order
            pinecone_meta = pinecone_metadatas.get(res_id, {})
            is_chunk = pinecone_meta.get("is_chunk", False)
            parent_doc_id = pinecone_meta.get("parent_document_id")

            result = {
                "document_id": res_id,  # This is the chunk_id if is_chunk, else the doc_id
                "score": scores.get(res_id, 0.0),
                "is_chunk": is_chunk,
                "parent_document_id": parent_doc_id,  # Null if not a chunk
            }

            mongo_doc = mongo_docs_map.get(
                res_id
            )  # Get the specific chunk or plain doc

            # --- FIX: Skip result if corresponding Mongo doc not found ---
            if not mongo_doc:
                logger.warning(
                    f"Document/chunk {res_id} found in Pinecone but not in MongoDB. Skipping."
                )
                continue
            # --- End FIX ---

            if include_content:
                content = None
                # Priority 1: Reranking field in Pinecone metadata (holds chunk text if reranking)
                # Note: This might be redundant if we fetch from Mongo anyway, but keep for flexibility
                if (
                    self.pinecone.use_reranking
                    and self.pinecone.rerank_text_field in pinecone_meta
                ):
                    content = pinecone_meta[self.pinecone.rerank_text_field]
                # Priority 2: Get content from the fetched Mongo doc (chunk or plain doc)
                elif mongo_doc:
                    content = mongo_doc.get("content")
                result["content"] = content or ""

            if include_metadata:
                combined_meta = {}
                # If it's a chunk, fetch the parent document's metadata
                if is_chunk and parent_doc_id:
                    parent_mongo_doc = mongo_docs_map.get(parent_doc_id)
                    if parent_mongo_doc:
                        # Extract metadata from parent, excluding fields specific to parent/content
                        combined_meta = {
                            k: v
                            for k, v in parent_mongo_doc.items()
                            if k
                            not in [
                                "_id",
                                "document_id",
                                "content",
                                "pdf_data",  # pdf_data removed anyway
                                "is_chunk",
                                "parent_document_id",
                                "created_at",
                                "updated_at",
                                "chunk_index",
                            ]
                        }
                # If it's a plain doc, fetch its own metadata
                elif not is_chunk and mongo_doc:
                    combined_meta = {
                        k: v
                        for k, v in mongo_doc.items()
                        if k
                        not in [
                            "_id",
                            "document_id",
                            "content",
                            "pdf_data",
                            "is_chunk",
                            "parent_document_id",
                            "created_at",
                            "updated_at",
                            "chunk_index",
                        ]
                    }

                # Add/overwrite with chunk-specific info from Pinecone meta (like chunk_index)
                # or specific metadata stored directly on the plain doc in Pinecone
                combined_meta.update(
                    {
                        k: v
                        for k, v in pinecone_meta.items()
                        # Avoid redundancy with already included fields or internal fields
                        if k
                        not in [
                            "document_id",
                            "parent_document_id",
                            "is_chunk",
                            self.pinecone.rerank_text_field,
                        ]
                    }
                )
                result["metadata"] = combined_meta

            results.append(result)

        return results

    async def delete_document(
        self, document_id: str, namespace: Optional[str] = None
    ) -> bool:  # pragma: no cover
        """
        Delete a parent document (plain text or PDF) and all its associated chunks
        from both MongoDB and Pinecone. Cannot delete a chunk directly.

        Args:
            document_id: ID of the parent document to delete.
            namespace: Optional Pinecone namespace.

        Returns:
            True if deletion was successful in both stores (if applicable), False otherwise.
        """
        logger.info(
            f"Attempting to delete document and associated data for ID: {document_id}"
        )
        mongo_delete_error = False
        pinecone_delete_error = False
        document_found = False  # Track if the initial ID exists

        # --- 0. Check if the target ID is a chunk ---
        try:
            target_doc = self.mongo.find_one(
                self.collection,
                {"document_id": document_id},
            )
            if target_doc and target_doc.get("is_chunk"):
                logger.warning(
                    f"Cannot delete chunk {document_id} directly. Delete the parent document."
                )
                return False  # Prevent deleting chunks directly
            if target_doc:
                document_found = True
        except Exception as e:  # pragma: no cover
            logger.error(
                f"Error checking document type for {document_id} in MongoDB: {e}"
            )  # pragma: no cover
            return False  # pragma: no cover # Fail if we can't even check the type

        if not document_found:
            logger.warning(f"Document {document_id} not found for deletion.")
            # Even if not found, attempt cleanup in Pinecone just in case of inconsistency
            # but the overall result should be False as the primary doc wasn't there.
            pass  # Continue to attempt Pinecone cleanup, but final result will be False

        # --- 1. Find all associated document IDs in MongoDB ---
        mongo_ids_to_delete = set()
        pinecone_ids_to_delete = set()
        try:
            # Find parent doc and all chunk docs linked to it
            # Use the ID confirmed not to be a chunk
            docs_to_delete_mongo = list(
                self.mongo.find(
                    self.collection,
                    {
                        "$or": [
                            {"document_id": document_id},
                            {"parent_document_id": document_id},
                        ]
                    },
                )
            )
            if docs_to_delete_mongo:
                document_found = True  # Confirm something was found related to the ID
                for doc in docs_to_delete_mongo:
                    mongo_ids_to_delete.add(doc["document_id"])
                    pinecone_ids_to_delete.add(doc["document_id"])
            elif document_found:  # Parent existed but no chunks found (plain text doc)
                mongo_ids_to_delete.add(document_id)
                pinecone_ids_to_delete.add(document_id)
            # If !document_found initially, sets remain empty unless fallback below happens

        except Exception as e:
            logger.warning(
                f"Error finding associated documents in MongoDB for deletion ({document_id}): {e}. Attempting Pinecone/Mongo deletion with main ID only."
            )
            # Fallback: try deleting the main ID from Pinecone/Mongo
            if document_found:  # Only add if we confirmed the initial doc existed
                pinecone_ids_to_delete.add(document_id)
                mongo_ids_to_delete.add(document_id)

        # Convert sets to lists for deletion methods
        pinecone_ids_list = list(pinecone_ids_to_delete)
        mongo_ids_list = list(mongo_ids_to_delete)

        # If no IDs were found at all, and the initial doc wasn't found, return False
        if not document_found and not mongo_ids_list and not pinecone_ids_list:
            logger.info(f"No trace of document {document_id} found to delete.")
            return False

        # --- 2. Delete from Pinecone ---
        if pinecone_ids_list:
            try:
                await self.pinecone.delete(ids=pinecone_ids_list, namespace=namespace)
                logger.info(
                    f"Attempted deletion of {len(pinecone_ids_list)} vectors from Pinecone for {document_id}."
                )
            except Exception as e:
                logger.error(
                    f"Error deleting vectors from Pinecone for {document_id} (IDs: {pinecone_ids_list}): {e}"
                )
                pinecone_delete_error = True  # Track error

        # --- 3. Delete from MongoDB ---
        mongo_deleted_count = 0
        if mongo_ids_list:
            try:
                delete_result = self.mongo.delete_many(
                    self.collection, {"document_id": {"$in": mongo_ids_list}}
                )
                mongo_deleted_count = delete_result.deleted_count
                if mongo_deleted_count > 0:
                    logger.info(
                        f"Deleted {mongo_deleted_count} documents from MongoDB for {document_id}."
                    )
                # else: # No need to log if count is 0, covered by initial find log
                #     logger.info(f"No documents found to delete in MongoDB for {document_id} with IDs: {mongo_ids_list}")

            except Exception as e:
                logger.error(
                    f"Error deleting documents from MongoDB for {document_id} (IDs: {mongo_ids_list}): {e}"
                )
                mongo_delete_error = True  # Track error

        # Return True only if the document was initially found and no errors occurred during deletion attempts
        # If the document wasn't found initially, return False even if cleanup attempts were made.
        return document_found and not mongo_delete_error and not pinecone_delete_error

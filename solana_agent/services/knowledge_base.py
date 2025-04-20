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


class KnowledgeBaseService(KnowledgeBaseInterface):
    """
    Knowledge Base service using Pinecone for vector search and MongoDB for metadata/full document storage.
    Supports text documents and PDF semantic chunking using OpenAI embeddings via LlamaIndex.
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
            collection_name: MongoDB collection for storing document metadata and full PDFs.
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
            print(
                f"Warning: Unknown OpenAI model '{openai_model_name}'. Using dimension {openai_dimensions} from Pinecone config. Ensure this is correct."
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
            print(f"Error initializing OpenAIEmbedding: {e}")
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
            print(f"Created MongoDB collection: {self.collection}")

        # Indexes for retrieval and filtering
        self.mongo.create_index(self.collection, [("document_id", 1)], unique=True)
        self.mongo.create_index(self.collection, [("parent_document_id", 1)])
        self.mongo.create_index(self.collection, [("source", 1)])
        self.mongo.create_index(self.collection, [("created_at", -1)])
        self.mongo.create_index(self.collection, [("tags", 1)])
        self.mongo.create_index(self.collection, [("is_chunk", 1)])
        print(f"Ensured indexes exist for MongoDB collection: {self.collection}")

    async def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> str:
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

        # Store metadata and content in MongoDB
        mongo_doc = {
            "document_id": doc_id,
            "content": text,
            "is_chunk": False,
            "parent_document_id": None,
            **metadata,
            # Use timezone aware datetime
            "created_at": metadata.get(
                "created_at", dt.now(tz=dt.now().astimezone().tzinfo)
            ),
            "updated_at": dt.now(tz=dt.now().astimezone().tzinfo),
        }
        try:
            self.mongo.insert_one(self.collection, mongo_doc)
        except Exception as e:
            print(f"Error inserting document {doc_id} into MongoDB: {e}")
            raise

        # Embed text using OpenAIEmbedding
        embed_model: OpenAIEmbedding = self.semantic_splitter.embed_model
        try:
            embedding = await embed_model.aget_text_embedding(text)
        except Exception as e:
            print(
                f"Error embedding document {doc_id} using {self.openai_model_name}: {e}"
            )
            # Decide how to handle - Mongo insert succeeded, embedding failed
            raise  # Re-raise for now

        # Prepare Pinecone metadata
        pinecone_metadata = {
            "document_id": doc_id,
            "is_chunk": False,
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
            print(f"Error upserting vector for {doc_id} to Pinecone: {e}")
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
    ) -> str:
        """
        Add a PDF document, performs semantic chunking using OpenAI embeddings,
        stores full PDF in Mongo, and chunk vectors in Pinecone.

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
                print(f"Warning: No text extracted from PDF {parent_doc_id}.")
        except Exception as e:
            print(f"Error reading or extracting text from PDF {parent_doc_id}: {e}")
            raise

        # --- 2. Store Full PDF and Metadata in MongoDB ---
        mongo_parent_doc = {
            "document_id": parent_doc_id,
            "content": extracted_text,
            "pdf_data": pdf_bytes,
            "is_chunk": False,
            "parent_document_id": None,
            **metadata,
            "created_at": metadata.get(
                "created_at", dt.now(tz=dt.now().astimezone().tzinfo)
            ),
            "updated_at": dt.now(tz=dt.now().astimezone().tzinfo),
        }
        try:
            self.mongo.insert_one(self.collection, mongo_parent_doc)
            print(f"Stored full PDF {parent_doc_id} in MongoDB.")
        except Exception as e:  # pragma: no cover
            print(
                f"Error inserting parent PDF {parent_doc_id} into MongoDB: {e}"
            )  # pragma: no cover
            raise  # pragma: no cover

        # --- 3. Semantic Chunking ---
        if not extracted_text.strip():
            print(
                f"Skipping chunking for PDF {parent_doc_id} due to no extracted text."
            )
            return parent_doc_id

        try:
            llama_doc = LlamaDocument(text=extracted_text)
            # Run synchronous splitter in thread pool
            nodes = await asyncio.to_thread(
                self.semantic_splitter.get_nodes_from_documents, [llama_doc]
            )
            print(f"Generated {len(nodes)} semantic chunks for PDF {parent_doc_id}.")
        except Exception as e:
            print(f"Error during semantic chunking for PDF {parent_doc_id}: {e}")
            raise

        # --- 4. Embed Chunks and Batch Upsert to Pinecone ---
        if not nodes:
            return parent_doc_id  # No chunks generated

        print(f"Embedding {len(nodes)} chunks using {self.openai_model_name}...")
        chunk_texts = [node.get_content() for node in nodes]
        embed_model: OpenAIEmbedding = self.semantic_splitter.embed_model
        all_chunk_embeddings = []

        # Embed chunks in batches (using embed_model's internal batching)
        try:
            # Use aget_text_embedding_batch for async embedding
            # Note: LlamaIndex OpenAIEmbedding might handle batch size internally.
            # If large number of nodes, consider explicit batching here if needed.
            all_chunk_embeddings = await embed_model.aget_text_embedding_batch(
                chunk_texts, show_progress=True
            )
        except Exception as e:
            print(f"Error embedding chunks for PDF {parent_doc_id}: {e}")
            raise  # Stop if embedding fails

        print("Embedding complete. Preparing vectors for Pinecone.")
        pinecone_vectors = []
        for i, node in enumerate(nodes):
            chunk_id = f"{parent_doc_id}_chunk_{i}"
            chunk_metadata = {
                "document_id": chunk_id,
                "parent_document_id": parent_doc_id,
                "chunk_index": i,
                "is_chunk": True,
                "source": metadata.get("source", "unknown"),
                "tags": metadata.get("tags", []),
            }
            # Add chunk text itself if Pinecone adapter reranking is used
            if self.pinecone.use_reranking:
                chunk_metadata[self.pinecone.rerank_text_field] = chunk_texts[i]

            pinecone_vectors.append(
                {
                    "id": chunk_id,
                    "values": all_chunk_embeddings[i],
                    "metadata": chunk_metadata,
                }
            )

        # Upsert vectors in batches using the generic upsert method
        print(
            f"Upserting {len(pinecone_vectors)} vectors to Pinecone in batches of {chunk_batch_size}..."
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
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error upserting vector batch {idx + 1} to Pinecone: {result}")
                # Decide on error handling: log, raise, etc.

        print(f"Finished processing PDF {parent_doc_id}.")
        return parent_doc_id

    async def query(
        self,
        query_text: str,
        filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        namespace: Optional[str] = None,
        include_content: bool = True,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base using semantic search with OpenAI embeddings.

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
            print(f"Error embedding query text '{query_text}': {e}")
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
            print(f"Error querying Pinecone: {e}")
            return []

        if not pinecone_results:
            return []

        # Extract IDs, scores, and metadata from Pinecone results
        # PineconeAdapter might have already reranked and truncated to final top_k
        result_ids = [res["id"] for res in pinecone_results]
        scores = {res["id"]: res["score"] for res in pinecone_results}
        pinecone_metadatas = {
            res["id"]: res.get("metadata", {}) for res in pinecone_results
        }

        # --- Fetch corresponding data from MongoDB ---
        mongo_docs_map = {}
        parent_ids_to_fetch = set()
        for res_id in result_ids:
            meta = pinecone_metadatas.get(res_id, {})
            if meta.get("is_chunk"):
                parent_id = meta.get("parent_document_id")
                if parent_id:
                    parent_ids_to_fetch.add(parent_id)
            else:
                # If it's not a chunk, its own ID might be in Mongo
                parent_ids_to_fetch.add(res_id)

        if parent_ids_to_fetch:
            try:
                mongo_docs = self.mongo.find(
                    self.collection, {"document_id": {"$in": list(parent_ids_to_fetch)}}
                )
                mongo_docs_map = {doc["document_id"]: doc for doc in mongo_docs}
            except Exception as e:
                print(f"Error fetching documents from MongoDB: {e}")
                # Proceed with potentially missing Mongo data

        # --- Combine Results ---
        results = []
        for res_id in result_ids:  # Iterate in Pinecone's score order
            pinecone_meta = pinecone_metadatas.get(res_id, {})
            is_chunk = pinecone_meta.get("is_chunk", False)
            parent_doc_id = pinecone_meta.get("parent_document_id")

            # Determine which Mongo doc holds the relevant info
            mongo_doc_for_meta = None
            mongo_doc_for_content = None
            if is_chunk and parent_doc_id:
                mongo_doc_for_meta = mongo_docs_map.get(parent_doc_id)
                mongo_doc_for_content = mongo_doc_for_meta  # Parent holds full content
            else:  # Not a chunk
                mongo_doc_for_meta = mongo_docs_map.get(res_id)
                mongo_doc_for_content = mongo_doc_for_meta

            result = {
                "document_id": res_id,
                "score": scores.get(res_id, 0.0),
                "is_chunk": is_chunk,
                "parent_document_id": parent_doc_id,
            }

            if include_content:
                content = None
                # Priority 1: Reranking field in Pinecone metadata (holds chunk text)
                if (
                    self.pinecone.use_reranking
                    and self.pinecone.rerank_text_field in pinecone_meta
                ):
                    content = pinecone_meta[self.pinecone.rerank_text_field]
                # Priority 2: Get content from the relevant Mongo doc
                elif mongo_doc_for_content:
                    content = mongo_doc_for_content.get("content")
                result["content"] = content or ""

            if include_metadata:
                combined_meta = {}
                # Merge metadata from the relevant Mongo doc (parent or self)
                if mongo_doc_for_meta:
                    combined_meta = {
                        k: v
                        for k, v in mongo_doc_for_meta.items()
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
                        ]
                    }
                # Add/overwrite with chunk-specific info from Pinecone meta
                combined_meta.update(
                    {
                        k: v
                        for k, v in pinecone_meta.items()
                        # Avoid redundancy
                        if k not in ["document_id", self.pinecone.rerank_text_field]
                    }
                )
                result["metadata"] = combined_meta

            results.append(result)

        return results

    async def delete_document(
        self, document_id: str, namespace: Optional[str] = None
    ) -> bool:
        """
        Delete a document (plain text or PDF) and all its associated chunks.

        Args:
            document_id: ID of the parent document (or plain text document).
            namespace: Optional Pinecone namespace.

        Returns:
            True if deletion was successful (or partially successful).
        """
        print(
            f"Attempting to delete document and associated data for ID: {document_id}"
        )
        mongo_deleted_count = 0
        pinecone_deleted = False

        # --- 1. Find all associated document IDs in MongoDB ---
        # This includes the parent doc and potentially chunk metadata if we stored it
        # We primarily need the IDs to delete from Pinecone.
        # A more robust way might be to query Pinecone directly for vectors with parent_document_id == document_id
        # For now, assume IDs in Mongo cover what needs deletion.
        docs_to_delete_mongo = []
        mongo_ids_to_delete = set([document_id])  # Start with the main ID
        try:
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
            for doc in docs_to_delete_mongo:
                mongo_ids_to_delete.add(doc["document_id"])
        except Exception as e:
            print(
                f"Warning: Error finding documents in MongoDB for deletion ({document_id}): {e}. Proceeding with main ID only."
            )

        pinecone_ids_to_delete = list(mongo_ids_to_delete)

        # --- 2. Delete from Pinecone ---
        if pinecone_ids_to_delete:
            try:
                await self.pinecone.delete(
                    ids=pinecone_ids_to_delete, namespace=namespace
                )
                print(
                    f"Deleted {len(pinecone_ids_to_delete)} vectors from Pinecone for parent {document_id}."
                )
                pinecone_deleted = True
            except Exception as e:
                print(f"Error deleting vectors from Pinecone for {document_id}: {e}")

        # --- 3. Delete from MongoDB ---
        # Use the IDs confirmed to be in Mongo
        mongo_ids_found_in_db = [doc["document_id"] for doc in docs_to_delete_mongo]
        if mongo_ids_found_in_db:
            try:
                delete_result = self.mongo.delete_many(
                    self.collection, {"document_id": {"$in": mongo_ids_found_in_db}}
                )
                mongo_deleted_count = delete_result.deleted_count
                print(
                    f"Deleted {mongo_deleted_count} documents from MongoDB for parent {document_id}."
                )
            except Exception as e:
                print(f"Error deleting documents from MongoDB for {document_id}: {e}")

        return pinecone_deleted or mongo_deleted_count > 0

    async def update_document(
        self,
        document_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> bool:
        """
        Update an existing plain text document or metadata. Embeds using OpenAI.
        Updating PDF content requires deleting and re-adding.

        Args:
            document_id: ID of document to update.
            text: Optional new text content (for plain text docs only).
            metadata: Optional metadata to update.
            namespace: Optional Pinecone namespace.

        Returns:
            True if successful.
        """
        current_doc = self.mongo.find_one(self.collection, {"document_id": document_id})
        if not current_doc:
            print(f"Document {document_id} not found for update.")
            return False

        if current_doc.get("is_chunk"):
            print(f"Cannot update chunk {document_id} directly.")
            return False
        if current_doc.get("pdf_data") and text is not None:
            print("Cannot update PDF content via this method. Delete and re-add.")
            return False

        update_text = text is not None and not current_doc.get("pdf_data")
        text_content = text if update_text else current_doc.get("content", "")

        # --- Update MongoDB ---
        mongo_update = {}
        if metadata:
            mongo_update.update(metadata)
        if update_text:
            mongo_update["content"] = text_content
        mongo_update["updated_at"] = dt.now(tz=dt.now().astimezone().tzinfo)

        mongo_updated = False
        if mongo_update:  # Only update if there are changes
            try:
                update_result = self.mongo.update_one(
                    self.collection,
                    {"document_id": document_id},
                    {"$set": mongo_update},
                )
                mongo_updated = update_result.modified_count > 0
            except Exception as e:
                print(f"Error updating document {document_id} in MongoDB: {e}")
                # Decide if we should proceed to Pinecone update if Mongo failed
                return False  # Return False if Mongo update fails

        # --- Update Pinecone (only if text changed) ---
        pinecone_updated = False
        if update_text:
            # Embed updated text
            embed_model: OpenAIEmbedding = self.semantic_splitter.embed_model
            try:
                embedding = await embed_model.aget_text_embedding(text_content)
            except Exception as e:
                print(f"Error embedding updated text for {document_id}: {e}")
                # Mongo update might have succeeded, but embedding failed
                return mongo_updated  # Return based on Mongo success

            # Prepare Pinecone metadata
            final_metadata = {**current_doc, **mongo_update}  # Use updated data
            pinecone_metadata = {"document_id": document_id, "is_chunk": False}
            for key, value in final_metadata.items():
                if key not in [
                    "_id",
                    "content",
                    "pdf_data",
                    "created_at",
                    "updated_at",
                    "document_id",
                    "is_chunk",
                    "parent_document_id",
                ]:
                    pinecone_metadata[key] = value
            if self.pinecone.use_reranking:
                pinecone_metadata[self.pinecone.rerank_text_field] = text_content

            # Upsert vector to Pinecone
            try:
                await self.pinecone.upsert(
                    vectors=[
                        {
                            "id": document_id,
                            "values": embedding,
                            "metadata": pinecone_metadata,
                        }
                    ],
                    namespace=namespace,
                )
                pinecone_updated = True
            except Exception as e:
                print(
                    f"Error upserting updated vector in Pinecone for {document_id}: {e}"
                )
                # Mongo update succeeded, Pinecone failed

        return mongo_updated or pinecone_updated

    async def add_documents_batch(
        self,
        # Expects {'text': ..., 'metadata': ...}
        documents: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 50,
    ) -> List[str]:
        """
        Add multiple plain text documents in batches using OpenAI embeddings.

        Args:
            documents: List of documents, each with 'text' and 'metadata'.
            namespace: Optional Pinecone namespace.
            batch_size: Number of documents per embedding/upsert batch.

        Returns:
            List of added document IDs.
        """
        all_doc_ids = []
        embed_model: OpenAIEmbedding = self.semantic_splitter.embed_model

        for i in range(0, len(documents), batch_size):
            batch_docs_input = documents[i : i + batch_size]
            batch_texts = [doc["text"] for doc in batch_docs_input]
            batch_metadatas = [doc["metadata"] for doc in batch_docs_input]
            # Generate IDs if not provided in metadata
            batch_doc_ids = [
                doc["metadata"].get("document_id") or str(uuid.uuid4())
                for doc in batch_docs_input
            ]
            all_doc_ids.extend(batch_doc_ids)

            # Prepare MongoDB docs
            mongo_batch = []
            for idx, text in enumerate(batch_texts):
                doc_id = batch_doc_ids[idx]
                metadata = batch_metadatas[idx]
                mongo_doc = {
                    "document_id": doc_id,
                    "content": text,
                    "is_chunk": False,
                    "parent_document_id": None,
                    **metadata,
                    "created_at": metadata.get(
                        "created_at", dt.now(tz=dt.now().astimezone().tzinfo)
                    ),
                    "updated_at": dt.now(tz=dt.now().astimezone().tzinfo),
                }
                # Ensure generated ID is in the doc for Mongo
                if "document_id" not in metadata:
                    mongo_doc["document_id"] = doc_id
                mongo_batch.append(mongo_doc)

            # Insert into MongoDB
            if mongo_batch:
                try:
                    self.mongo.insert_many(self.collection, mongo_batch)
                except Exception as e:
                    print(
                        f"Error inserting batch {i // batch_size + 1} into MongoDB: {e}"
                    )
                    # Decide if we should skip Pinecone for this batch
                    continue  # Skip to next batch

            # Embed batch using OpenAIEmbedding
            try:
                batch_embeddings = await embed_model.aget_text_embedding_batch(
                    batch_texts, show_progress=True
                )
            except Exception as e:
                print(
                    f"Error embedding batch {i // batch_size + 1} using {self.openai_model_name}: {e}"
                )
                continue  # Skip Pinecone upsert for this batch

            # Prepare Pinecone vectors
            pinecone_vectors = []
            for idx, doc_id in enumerate(batch_doc_ids):
                metadata = batch_metadatas[idx]
                pinecone_meta = {
                    "document_id": doc_id,
                    "is_chunk": False,
                    "source": metadata.get("source", "unknown"),
                    "tags": metadata.get("tags", []),
                }
                if self.pinecone.use_reranking:
                    pinecone_meta[self.pinecone.rerank_text_field] = batch_texts[idx]

                pinecone_vectors.append(
                    {
                        "id": doc_id,
                        "values": batch_embeddings[idx],
                        "metadata": pinecone_meta,
                    }
                )

            # Upsert vectors to Pinecone
            if pinecone_vectors:
                try:
                    await self.pinecone.upsert(
                        vectors=pinecone_vectors, namespace=namespace
                    )
                except Exception as e:
                    print(
                        f"Error upserting vector batch {i // batch_size + 1} to Pinecone: {e}"
                    )

            # Optional delay
            if i + batch_size < len(documents):
                await asyncio.sleep(0.1)

        return all_doc_ids

    async def get_full_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the full document entry (including PDF data if applicable) from MongoDB.

        Args:
            document_id: The ID of the document (parent ID if it was a PDF).

        Returns:
            The document dictionary from MongoDB, or None if not found.
        """
        try:
            return self.mongo.find_one(self.collection, {"document_id": document_id})
        except Exception as e:
            print(f"Error retrieving full document {document_id} from MongoDB: {e}")
            return None

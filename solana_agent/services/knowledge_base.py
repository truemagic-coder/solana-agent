from datetime import datetime as dt
from typing import Dict, List, Any, Optional, Union
import uuid
import asyncio
import io  # For handling bytes streams

# PDF and Chunking Imports
import pypdf
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.embeddings import BaseEmbedding

from solana_agent.adapters.pinecone_adapter import PineconeAdapter
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
from solana_agent.interfaces.services.knowledge_base import KnowledgeBaseService as KnowledgeBaseInterface
from solana_agent.interfaces.providers.llm import LLMProvider  # Needed for embeddings


class KnowledgeBaseService(KnowledgeBaseInterface):
    """
    Knowledge Base service using Pinecone for vector search and MongoDB for metadata/full document storage.
    Supports text documents and PDF semantic chunking.
    """

    def __init__(
        self,
        pinecone_adapter: PineconeAdapter,
        mongodb_adapter: MongoDBAdapter,
        llm_provider: LLMProvider,  # Required for semantic chunking embeddings
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
            llm_provider: Configured LLMProvider instance (for embeddings).
            collection_name: MongoDB collection for storing document metadata and full PDFs.
            rerank_results: Whether to rerank results via PineconeAdapter.
            rerank_top_k: Number of results to return after reranking.
            splitter_buffer_size: Buffer size for SemanticSplitterNodeParser.
            splitter_breakpoint_percentile: Breakpoint percentile for SemanticSplitterNodeParser.
        """
        self.pinecone = pinecone_adapter
        self.mongo = mongodb_adapter
        self.llm_provider = llm_provider
        self.collection = collection_name
        self.rerank_results = rerank_results
        self.rerank_top_k = rerank_top_k

        # --- Initialize Semantic Splitter ---
        if not hasattr(self.llm_provider, 'embed_text'):
            raise ValueError(
                "LLMProvider must have an 'embed_text' method for semantic chunking.")

        class LLMProviderEmbeddingAdapter(BaseEmbedding):
            def __init__(self, provider: LLMProvider, model_name: Optional[str] = None, dimensions: Optional[int] = None):
                self._provider = provider
                super().__init__(model_name=model_name or "default_embed_model")
                self._dimensions = dimensions

            # Text Embedding Methods (for documents/nodes)
            async def _aget_text_embedding(self, text: str) -> List[float]:
                return await self._provider.embed_text(text, model=self.model_name, dimensions=self._dimensions)

            async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                tasks = [self._aget_text_embedding(text) for text in texts]
                return await asyncio.gather(*tasks)

            def _get_text_embedding(self, text: str) -> List[float]:
                try:
                    loop = asyncio.get_running_loop()
                    future = asyncio.run_coroutine_threadsafe(
                        self._aget_text_embedding(text), loop)
                    return future.result()
                except RuntimeError:
                    return asyncio.run(self._aget_text_embedding(text))

            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                try:
                    loop = asyncio.get_running_loop()
                    future = asyncio.run_coroutine_threadsafe(
                        self._aget_text_embeddings(texts), loop)
                    return future.result()
                except RuntimeError:
                    return asyncio.run(self._aget_text_embeddings(texts))

            # FIX: Implement Query Embedding Methods
            async def _aget_query_embedding(self, query: str) -> List[float]:
                # Often the same as text embedding, but could differ based on provider
                return await self._provider.embed_text(query, model=self.model_name, dimensions=self._dimensions)

            def _get_query_embedding(self, query: str) -> List[float]:
                # Sync wrapper for query embedding
                try:
                    loop = asyncio.get_running_loop()
                    future = asyncio.run_coroutine_threadsafe(
                        self._aget_query_embedding(query), loop)
                    return future.result()
                except RuntimeError:
                    return asyncio.run(self._aget_query_embedding(query))

            # Optional: Add embed_batch_size if needed/available
            # @property
            # def embed_batch_size(self) -> int:
            #     return 10 # Or get from provider

        embed_dimensions = self.pinecone.embedding_dimensions
        llama_embed_model = LLMProviderEmbeddingAdapter(
            self.llm_provider,
            dimensions=embed_dimensions
        )

        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=splitter_buffer_size,
            breakpoint_percentile_threshold=splitter_breakpoint_percentile,
            embed_model=llama_embed_model
        )
        # --- End Semantic Splitter Init ---

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Set up MongoDB collection with appropriate indexes."""
        if not self.mongo.collection_exists(self.collection):
            self.mongo.create_collection(self.collection)

        # Indexes for retrieval and filtering
        self.mongo.create_index(
            self.collection, [("document_id", 1)], unique=True)
        # Index for finding chunks related to a parent PDF
        self.mongo.create_index(self.collection, [("parent_document_id", 1)])
        self.mongo.create_index(self.collection, [("source", 1)])
        self.mongo.create_index(self.collection, [("created_at", -1)])
        self.mongo.create_index(self.collection, [("tags", 1)])
        # Index to differentiate between full docs and chunks
        self.mongo.create_index(self.collection, [("is_chunk", 1)])

    async def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> str:
        """
        Add a plain text document to the knowledge base.

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
            "content": text,  # Store full text
            "is_chunk": False,  # Mark as not a chunk
            "parent_document_id": None,  # No parent for full text doc
            **metadata,
            "created_at": metadata.get("created_at", dt.now()),
            "updated_at": dt.now()
        }
        self.mongo.insert_one(self.collection, mongo_doc)

        # Prepare minimal metadata for Pinecone vector
        pinecone_metadata = {
            "document_id": doc_id,  # Links back to MongoDB full doc
            "is_chunk": False,
            "source": metadata.get("source", "unknown"),
            "tags": metadata.get("tags", [])
        }
        # Add text itself if reranking is used
        if self.pinecone.use_reranking:
            pinecone_metadata[self.pinecone.rerank_text_field] = text

        # Add embedding to Pinecone
        await self.pinecone.upsert_text(
            texts=[text],
            ids=[doc_id],
            metadatas=[pinecone_metadata],
            namespace=namespace
        )
        return doc_id

    async def add_pdf_document(
        self,
        pdf_data: Union[bytes, str],  # PDF bytes or file path
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        chunk_batch_size: int = 50
    ) -> str:
        """
        Add a PDF document, performs semantic chunking, stores full PDF in Mongo,
        and chunks in Pinecone.

        Args:
            pdf_data: PDF content as bytes or a path to the PDF file.
            metadata: Document metadata (applies to the parent PDF).
            document_id: Optional parent document ID.
            namespace: Optional Pinecone namespace for chunks.
            chunk_batch_size: Batch size for upserting chunks to Pinecone.

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
                raise ValueError(
                    "pdf_data must be bytes or a file path string.")

            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            extracted_text = "".join(
                page.extract_text() or "" for page in reader.pages)
            if not extracted_text.strip():
                print(f"Warning: No text extracted from PDF {parent_doc_id}.")
                # Optionally raise error or return early
                # return parent_doc_id # Or raise

        except Exception as e:
            print(
                f"Error reading or extracting text from PDF {parent_doc_id}: {e}")
            raise  # Re-raise the error

        # --- 2. Store Full PDF and Metadata in MongoDB ---
        mongo_parent_doc = {
            "document_id": parent_doc_id,
            "content": extracted_text,
            # Store raw bytes directly - PyMongo handles BSON conversion
            "pdf_data": pdf_bytes,
            "is_chunk": False,
            "parent_document_id": None,
            **metadata,
            "created_at": metadata.get("created_at", dt.now()),
            "updated_at": dt.now()
        }
        self.mongo.insert_one(self.collection, mongo_parent_doc)
        print(f"Stored full PDF {parent_doc_id} in MongoDB.")

        # --- 3. Semantic Chunking ---
        if not extracted_text.strip():
            print(
                f"Skipping chunking for PDF {parent_doc_id} due to no extracted text.")
            return parent_doc_id

        try:
            # Create a LlamaIndex Document object
            llama_doc = LlamaDocument(text=extracted_text)
            # Get semantic chunks (nodes)
            nodes = self.semantic_splitter.get_nodes_from_documents([
                                                                    llama_doc])
            print(
                f"Generated {len(nodes)} semantic chunks for PDF {parent_doc_id}.")
        except Exception as e:
            print(
                f"Error during semantic chunking for PDF {parent_doc_id}: {e}")
            # Decide how to handle: maybe store PDF but no chunks?
            raise  # Re-raise for now

        # --- 4. Prepare and Batch Upsert Chunks to Pinecone ---
        if not nodes:
            return parent_doc_id  # No chunks generated

        pinecone_chunk_texts = []
        pinecone_chunk_ids = []
        pinecone_chunk_metadatas = []

        for i, node in enumerate(nodes):
            chunk_text = node.get_content()
            chunk_id = f"{parent_doc_id}_chunk_{i}"

            pinecone_chunk_texts.append(chunk_text)
            pinecone_chunk_ids.append(chunk_id)

            # Metadata for the chunk vector
            chunk_metadata = {
                "document_id": chunk_id,  # Unique ID for the chunk itself
                "parent_document_id": parent_doc_id,  # Link back to the PDF
                "chunk_index": i,
                "is_chunk": True,
                "source": metadata.get("source", "unknown"),  # Inherit source
                "tags": metadata.get("tags", [])  # Inherit tags
            }
            # Add chunk text itself if reranking is used
            if self.pinecone.use_reranking:
                chunk_metadata[self.pinecone.rerank_text_field] = chunk_text

            pinecone_chunk_metadatas.append(chunk_metadata)

        # Upsert chunks in batches
        print(
            f"Upserting {len(pinecone_chunk_ids)} chunks to Pinecone in batches of {chunk_batch_size}...")
        for i in range(0, len(pinecone_chunk_ids), chunk_batch_size):
            batch_texts = pinecone_chunk_texts[i:i+chunk_batch_size]
            batch_ids = pinecone_chunk_ids[i:i+chunk_batch_size]
            batch_metadatas = pinecone_chunk_metadatas[i:i+chunk_batch_size]

            try:
                await self.pinecone.upsert_text(
                    texts=batch_texts,
                    ids=batch_ids,
                    metadatas=batch_metadatas,
                    namespace=namespace
                )
                print(f"Upserted batch {i//chunk_batch_size + 1}...")
            except Exception as e:
                print(
                    f"Error upserting chunk batch {i//chunk_batch_size + 1} to Pinecone: {e}")
                # Decide on error handling: continue? stop? log?
                # For now, just print and continue
            # Optional delay between batches
            if i + chunk_batch_size < len(pinecone_chunk_ids):
                await asyncio.sleep(0.2)

        print(f"Finished processing PDF {parent_doc_id}.")
        return parent_doc_id

    async def query(
        self,
        query_text: str,
        filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        namespace: Optional[str] = None,
        include_content: bool = True,  # Now refers to chunk content if it's a chunk
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base using semantic search.

        Args:
            query_text: The query text.
            filter: Optional Pinecone metadata filter.
            top_k: Number of results to return.
            namespace: Optional Pinecone namespace.
            include_content: Whether to include document/chunk content in results.
            include_metadata: Whether to include document/chunk metadata in results.

        Returns:
            List of result dictionaries, each containing score, document_id,
            and optionally content and metadata.
        """
        effective_top_k = self.rerank_top_k if self.rerank_results else top_k

        # --- Query Pinecone ---
        try:
            pinecone_results = await self.pinecone.query_text(
                query_text=query_text,
                filter=filter,
                top_k=effective_top_k,
                namespace=namespace,
                include_values=False,  # Don't need embeddings in the result
                include_metadata=True  # Need metadata for linking and info
            )
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

        if not pinecone_results:
            return []

        # Extract IDs, scores, and metadata from Pinecone results
        result_ids = [res['id'] for res in pinecone_results]
        scores = {res['id']: res['score'] for res in pinecone_results}
        pinecone_metadatas = {res['id']: res.get(
            'metadata', {}) for res in pinecone_results}

        # --- Fetch corresponding data from MongoDB ---
        # We need the content (text or chunk text) and potentially more metadata
        # Fetch based on the IDs returned by Pinecone
        mongo_docs_map = {}  # FIX: Initialize as empty
        try:  # FIX: Add try-except block for find
            mongo_docs = self.mongo.find(
                self.collection,
                {"document_id": {"$in": result_ids}}
            )
            mongo_docs_map = {doc["document_id"]: doc for doc in mongo_docs}
        except Exception as e:
            print(f"Error fetching documents from MongoDB: {e}")
            # Proceed with empty mongo_docs_map, results will lack Mongo data

        # --- Combine Results ---
        results = []
        for res_id in result_ids:  # Iterate in Pinecone's score order
            pinecone_meta = pinecone_metadatas.get(res_id, {})
            # Use get() for safety, although find should have handled errors
            mongo_doc = mongo_docs_map.get(res_id)

            # Basic result structure
            result = {
                "document_id": res_id,  # This is the chunk_id if it's a chunk
                "score": scores.get(res_id, 0.0),
                "is_chunk": pinecone_meta.get("is_chunk", False),
                # Important for chunks
                "parent_document_id": pinecone_meta.get("parent_document_id"),
            }

            # Include content (chunk text or full doc text)
            if include_content:
                content = None
                # Priority 1: Reranking field in Pinecone metadata
                if self.pinecone.use_reranking and self.pinecone.rerank_text_field in pinecone_meta:
                    content = pinecone_meta[self.pinecone.rerank_text_field]
                # Priority 2: If it's a chunk, get parent content from Mongo
                elif result["is_chunk"]:  # FIX: Check if it's a chunk first
                    parent_doc_id_for_content = pinecone_meta.get(
                        "parent_document_id")
                    if parent_doc_id_for_content:
                        # Fetch parent doc (could optimize caching later)
                        parent_doc_for_content = None
                        try:  # FIX: Add try-except for find_one for content
                            parent_doc_for_content = self.mongo.find_one(
                                self.collection, {"document_id": parent_doc_id_for_content})
                        except Exception as e:
                            print(
                                f"Error fetching parent document {parent_doc_id_for_content} for chunk content: {e}")
                        if parent_doc_for_content:
                            content = parent_doc_for_content.get("content")
                # Priority 3: If it's a non-chunk, get content from its own Mongo doc
                elif mongo_doc:
                    content = mongo_doc.get("content")

                # Use empty string if no content found
                result["content"] = content or ""

            # Include metadata
            if include_metadata:
                # Combine metadata from Pinecone (chunk info) and Mongo (original meta)
                combined_meta = {}
                # Get original metadata from parent if it's a chunk and parent exists
                parent_doc_id = pinecone_meta.get("parent_document_id")
                parent_mongo_doc = None
                if parent_doc_id:
                    # Avoid hitting DB repeatedly if multiple chunks from same parent
                    # This simple approach fetches each time, could optimize later
                    try:  # FIX: Add try-except for find_one for metadata
                        parent_mongo_doc = self.mongo.find_one(
                            self.collection, {"document_id": parent_doc_id})
                    except Exception as e:
                        print(
                            f"Error fetching parent document {parent_doc_id} for metadata: {e}")
                        parent_mongo_doc = None  # Ensure it's None on error

                if parent_mongo_doc:  # Use parent's original metadata
                    combined_meta = {
                        k: v for k, v in parent_mongo_doc.items()
                        if k not in ["_id", "document_id", "content", "pdf_data", "is_chunk", "parent_document_id"]
                    }
                # Use non-chunk doc's metadata
                elif mongo_doc and not result["is_chunk"]:
                    combined_meta = {
                        k: v for k, v in mongo_doc.items()
                        if k not in ["_id", "document_id", "content", "pdf_data", "is_chunk", "parent_document_id"]
                    }

                # Add chunk-specific info from Pinecone meta
                combined_meta.update({
                    k: v for k, v in pinecone_meta.items()
                    # Avoid redundancy
                    if k not in ["document_id", self.pinecone.rerank_text_field]
                })
                result["metadata"] = combined_meta

            results.append(result)

        return results

    async def delete_document(
        self,
        document_id: str,  # Can be a parent PDF ID or a plain text doc ID
        namespace: Optional[str] = None
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
            f"Attempting to delete document and associated chunks for ID: {document_id}")
        mongo_deleted_count = 0
        pinecone_deleted = False

        # --- 1. Find all associated items in MongoDB ---
        # This includes the parent doc and potentially chunk metadata if we stored it
        docs_to_delete_mongo = list(self.mongo.find(
            self.collection,
            {"$or": [{"document_id": document_id}, {
                "parent_document_id": document_id}]}
        ))
        mongo_ids_to_delete = [doc["document_id"]
                               for doc in docs_to_delete_mongo]

        # --- 2. Find all associated vectors in Pinecone ---
        # These are the parent doc vector (if text) and all chunk vectors
        # We can use the mongo_ids_to_delete list
        pinecone_ids_to_delete = mongo_ids_to_delete  # IDs are the same

        # --- 3. Delete from Pinecone ---
        if pinecone_ids_to_delete:
            try:
                # Delete in batches if necessary, though Pinecone delete handles lists
                await self.pinecone.delete(ids=pinecone_ids_to_delete, namespace=namespace)
                print(
                    f"Deleted {len(pinecone_ids_to_delete)} vectors from Pinecone for parent {document_id}.")
                pinecone_deleted = True
            except Exception as e:
                print(
                    f"Error deleting vectors from Pinecone for {document_id}: {e}")
                # Continue to delete from Mongo even if Pinecone fails

        # --- 4. Delete from MongoDB ---
        if mongo_ids_to_delete:
            try:
                delete_result = self.mongo.delete_many(
                    self.collection,
                    {"document_id": {"$in": mongo_ids_to_delete}}
                )
                mongo_deleted_count = delete_result.deleted_count
                print(
                    f"Deleted {mongo_deleted_count} documents from MongoDB for parent {document_id}.")
            except Exception as e:
                print(
                    f"Error deleting documents from MongoDB for {document_id}: {e}")

        # Return True if at least something was deleted
        return pinecone_deleted or mongo_deleted_count > 0

    async def update_document(
        self,
        document_id: str,
        text: Optional[str] = None,  # For plain text docs
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
        # pdf_data: Union[bytes, str] = None # Updating PDFs requires re-chunking, complex.
    ) -> bool:
        """
        Update an existing plain text document or metadata.
        Updating PDF content requires deleting and re-adding.

        Args:
            document_id: ID of document to update.
            text: Optional new text content (for plain text docs only).
            metadata: Optional metadata to update.
            namespace: Optional Pinecone namespace.

        Returns:
            True if successful.
        """
        # Fetch current document
        current_doc = self.mongo.find_one(
            self.collection, {"document_id": document_id})
        if not current_doc:
            print(f"Document {document_id} not found.")
            return False

        # Disallow updating content of chunks or full PDFs via this method
        if current_doc.get("is_chunk"):
            print(
                f"Cannot update chunk {document_id} directly. Update the parent PDF.")
            return False
        if current_doc.get("pdf_data") and text is not None:
            print(
                f"Cannot update PDF content via this method. Delete and re-add the PDF.")
            return False

        update_text = text is not None and not current_doc.get("pdf_data")
        text_content = text if update_text else current_doc.get("content", "")

        # --- Update MongoDB ---
        mongo_update = {}
        if metadata:
            # Update only specified metadata fields
            mongo_update.update(metadata)
        if update_text:
            mongo_update["content"] = text_content
        mongo_update["updated_at"] = dt.now()

        update_result = self.mongo.update_one(
            self.collection,
            {"document_id": document_id},
            {"$set": mongo_update}
        )
        mongo_updated = update_result.modified_count > 0

        # --- Update Pinecone (only if text changed) ---
        pinecone_updated = False
        if update_text:
            # Prepare metadata for Pinecone vector
            # Use potentially updated metadata if provided, else current
            final_metadata = {**current_doc, **(metadata or {})}
            pinecone_metadata = {
                "document_id": document_id,
                "is_chunk": False,
                # "source": final_metadata.get("source", "unknown"), # FIX: Remove hardcoding
                # "tags": final_metadata.get("tags", []) # FIX: Remove hardcoding
            }
            # FIX: Add all relevant metadata fields from the merged dict
            for key, value in final_metadata.items():
                # Exclude internal/large fields not suitable for Pinecone metadata
                if key not in ["_id", "content", "pdf_data", "created_at", "updated_at", "document_id", "is_chunk", "parent_document_id"]:
                    pinecone_metadata[key] = value

            if self.pinecone.use_reranking:
                pinecone_metadata[self.pinecone.rerank_text_field] = text_content

            try:
                await self.pinecone.upsert_text(
                    texts=[text_content],
                    ids=[document_id],
                    metadatas=[pinecone_metadata],
                    namespace=namespace
                )
                pinecone_updated = True
            except Exception as e:
                print(
                    f"Error updating vector in Pinecone for {document_id}: {e}")
                # Mongo update might have succeeded, but Pinecone failed

        # Return True if either Mongo or Pinecone (if applicable) was updated
        return mongo_updated or pinecone_updated

    async def add_documents_batch(
        self,
        # Expects {'text': ..., 'metadata': ...}
        documents: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 50
    ) -> List[str]:
        """
        Add multiple plain text documents in batches.
        Use add_pdf_document for PDFs.

        Args:
            documents: List of documents, each with 'text' and 'metadata'.
            namespace: Optional Pinecone namespace.
            batch_size: Number of documents per batch.

        Returns:
            List of added document IDs.
        """
        all_doc_ids = []
        for i in range(0, len(documents), batch_size):
            batch_docs_input = documents[i:i+batch_size]
            batch_texts = [doc['text'] for doc in batch_docs_input]
            batch_metadatas = [doc['metadata'] for doc in batch_docs_input]
            batch_doc_ids = [doc['metadata'].get(
                'document_id', str(uuid.uuid4())) for doc in batch_docs_input]
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
                    "created_at": metadata.get("created_at", dt.now()),
                    "updated_at": dt.now()
                }
                mongo_batch.append(mongo_doc)

            # Insert into MongoDB
            if mongo_batch:
                self.mongo.insert_many(self.collection, mongo_batch)

            # Prepare Pinecone metadata
            pinecone_batch_metadatas = []
            for idx, doc_id in enumerate(batch_doc_ids):
                metadata = batch_metadatas[idx]
                pinecone_meta = {
                    "document_id": doc_id,
                    "is_chunk": False,
                    "source": metadata.get("source", "unknown"),
                    "tags": metadata.get("tags", [])
                }
                if self.pinecone.use_reranking:
                    pinecone_meta[self.pinecone.rerank_text_field] = batch_texts[idx]
                pinecone_batch_metadatas.append(pinecone_meta)

            # Upsert to Pinecone
            try:
                await self.pinecone.upsert_text(
                    texts=batch_texts,
                    ids=batch_doc_ids,
                    metadatas=pinecone_batch_metadatas,
                    namespace=namespace
                )
            except Exception as e:
                print(
                    f"Error upserting text batch {i//batch_size + 1} to Pinecone: {e}")

            if i + batch_size < len(documents):
                await asyncio.sleep(0.2)

        return all_doc_ids

    async def get_full_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the full document entry (including PDF data if applicable) from MongoDB.

        Args:
            document_id: The ID of the document (parent ID if it was a PDF).

        Returns:
            The document dictionary from MongoDB, or None if not found.
        """
        return self.mongo.find_one(self.collection, {"document_id": document_id})

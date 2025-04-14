from typing import List, Dict, Any, Optional, Literal, Union
from pinecone import PineconeAsyncio, ServerlessSpec
from pinecone.exceptions import PineconeApiException

from solana_agent.interfaces.providers.vector_storage import VectorStorageProvider
from solana_agent.interfaces.providers.llm import LLMProvider
import os
import asyncio

# Define types for Pinecone models
PineconeEmbeddingModel = Literal[
    "multilingual-e5-large",
    "llama-text-embed-v2",
]
PineconeRerankModel = Literal[
    "cohere-rerank-3.5",
    "bge-reranker-v2-m3",
    "pinecone-rerank-v0",
]
InputType = Literal["query", "passage"]
TruncateType = Literal["END", "NONE"]


class PineconeAdapter(VectorStorageProvider):
    """
    Adapter for interacting with Pinecone vector database using PineconeAsyncio.
    Supports Pinecone native embeddings and reranking.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        llm_provider: Optional[LLMProvider] = None,
        embedding_dimensions: int = 3072,  # Default for OpenAI text-embedding-3-large
        cloud_provider: str = "aws",
        region: str = "us-east-1",
        metric: str = "cosine",
        create_index_if_not_exists: bool = True,
        # Pinecone Native Embedding Config
        use_pinecone_embeddings: bool = False,
        pinecone_embedding_model: Optional[PineconeEmbeddingModel] = None,
        pinecone_embedding_dimension_override: Optional[int] = None,
        # Reranking Config
        use_reranking: bool = False,
        rerank_model: Optional[PineconeRerankModel] = None,
        rerank_top_k: int = 3,  # Final number of results after reranking
        initial_query_top_k_multiplier: int = 5,
        rerank_text_field: str = "text",  # Metadata field containing text for reranking
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.llm_provider = llm_provider
        self.embedding_dimensions = embedding_dimensions
        self.cloud_provider = cloud_provider
        self.region = region
        self.metric = metric
        self.create_index_if_not_exists = create_index_if_not_exists

        # Pinecone Native Embedding Config
        self.use_pinecone_embeddings = use_pinecone_embeddings
        self.pinecone_embedding_model = pinecone_embedding_model
        self.pinecone_embedding_dimension_override = pinecone_embedding_dimension_override

        # Reranking Config
        self.use_reranking = use_reranking
        self.rerank_model = rerank_model
        self.rerank_top_k = rerank_top_k
        self.initial_query_top_k = rerank_top_k * \
            initial_query_top_k_multiplier if use_reranking else rerank_top_k
        self.rerank_text_field = rerank_text_field

        self.pinecone: Optional[PineconeAsyncio] = None
        self.index: Optional[Any] = None

        if not self.api_key:
            raise ValueError("Pinecone API key is required.")
        if not self.index_name:
            raise ValueError("Pinecone index name is required.")
        if not self.use_pinecone_embeddings and not self.llm_provider:
            raise ValueError(
                "Either Pinecone embeddings must be enabled or an LLMProvider must be provided.")
        if self.use_pinecone_embeddings and not self.pinecone_embedding_model:
            raise ValueError(
                "pinecone_embedding_model must be specified when use_pinecone_embeddings is True.")
        if self.use_reranking and not self.rerank_model:
            raise ValueError(
                "rerank_model must be specified when use_reranking is True.")

        # Determine embedding dimension based on config
        if self.use_pinecone_embeddings:
            if self.pinecone_embedding_model == "multilingual-e5-large":
                self.embedding_dimensions = 1024
            elif self.pinecone_embedding_model == "llama-text-embed-v2":
                self.embedding_dimensions = self.pinecone_embedding_dimension_override or 1024
            print(
                f"Using Pinecone embedding model '{self.pinecone_embedding_model}' with dimension {self.embedding_dimensions}.")
        else:
            print(
                f"Using external LLM provider for embeddings with dimension {self.embedding_dimensions}.")

        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _initialize_async(self):
        """Asynchronously initialize the Pinecone client and index."""
        async with self._init_lock:
            if self._initialized:
                return

            try:
                print("Initializing PineconeAsyncio client...")
                self.pinecone = PineconeAsyncio(api_key=self.api_key)

                if self.create_index_if_not_exists:
                    await self._create_index_if_not_exists_async()

                print(f"Connecting to Pinecone index '{self.index_name}'...")
                self.index = await self.pinecone.Index(self.index_name)
                print(f"Connected to Pinecone index '{self.index_name}'.")

                # Verify index dimensions
                stats = await self.describe_index_stats()
                index_dimension = stats.get("dimension", 0)
                if index_dimension != 0 and index_dimension != self.embedding_dimensions:
                    print(
                        f"⚠️ Warning: Index dimension ({index_dimension}) does not match configured embedding dimension ({self.embedding_dimensions})")
                total_vector_count = stats.get("total_vector_count", 0)
                print(f"Current index contains {total_vector_count} vectors.")

                self._initialized = True

            except PineconeApiException as e:
                print(f"Pinecone API error during async initialization: {e}")
                self.pinecone = None
                self.index = None
                raise
            except Exception as e:
                print(
                    f"Failed to initialize Pinecone async index '{self.index_name}': {e}")
                self.pinecone = None
                self.index = None
                raise

    async def _create_index_if_not_exists_async(self) -> None:
        """Create the Pinecone index asynchronously if it doesn't already exist."""
        if not self.pinecone:
            raise RuntimeError(
                "Pinecone client not initialized before creating index.")
        try:
            indexes_response = await self.pinecone.list_indexes()
            existing_indexes = indexes_response.get('indexes', [])
            existing_names = [idx.get('name') for idx in existing_indexes]

            if self.index_name not in existing_names:
                print(
                    f"Creating Pinecone index '{self.index_name}' with dimension {self.embedding_dimensions}...")

                spec_data = {
                    "cloud": self.cloud_provider,
                    "region": self.region
                }

                create_params = {
                    "name": self.index_name,
                    "dimension": self.embedding_dimensions,
                    "metric": self.metric,
                    "spec": ServerlessSpec(**spec_data)
                }

                await self.pinecone.create_index(**create_params)
                print(
                    f"✅ Successfully created Pinecone index '{self.index_name}'. Waiting for it to be ready...")
                await asyncio.sleep(20)  # Wait time for the index to be ready
            else:
                print(f"Using existing Pinecone index '{self.index_name}'")
        except Exception as e:
            print(f"Error creating Pinecone index asynchronously: {e}")
            raise

    async def _ensure_initialized(self):
        """Ensure the async client is initialized before use."""
        if not self._initialized:
            await self._initialize_async()
        if not self._initialized or self.index is None:
            raise RuntimeError(
                "Pinecone async client failed to initialize or connect to index.")

    async def _get_embedding(self, text: str, input_type: InputType) -> List[float]:
        """Get embedding using either Pinecone native or external LLM provider."""
        if self.use_pinecone_embeddings:
            if not self.pinecone:
                raise RuntimeError(
                    "Pinecone client not initialized for embedding.")

            try:
                # Prepare embedding parameters
                embed_params = {
                    "texts": [text],
                    "model": self.pinecone_embedding_model,
                }

                # Add parameters dictionary
                parameters = {
                    "input_type": input_type,
                    "truncate": "END",  # Default truncation
                }

                # Add dimension parameter for llama model if specified
                if (self.pinecone_embedding_model == "llama-text-embed-v2" and
                        self.pinecone_embedding_dimension_override):
                    parameters["dimension"] = self.pinecone_embedding_dimension_override

                embed_params["parameters"] = parameters

                # Call the async embed endpoint
                response = await self.pinecone.embed(**embed_params)

                # Extract the embeddings from the response
                if response and "embeddings" in response and response["embeddings"]:
                    return response["embeddings"][0]
                else:
                    raise ValueError(
                        "Pinecone embedding response did not contain expected data.")

            except Exception as e:
                print(
                    f"Error getting Pinecone embedding for '{text[:50]}...': {e}")
                raise
        else:
            if not self.llm_provider:
                raise RuntimeError("LLMProvider not available for embedding.")
            # Use external LLM provider for embedding
            return await self.llm_provider.embed_text(text, dimensions=self.embedding_dimensions)

    async def upsert_text(
        self,
        texts: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None
    ) -> None:  # pragma: no cover
        """Embeds texts and upserts them into Pinecone."""
        await self._ensure_initialized()

        if len(texts) != len(ids):
            raise ValueError("Number of texts must match number of IDs.")
        if metadatas and len(texts) != len(metadatas):
            raise ValueError(
                "Number of texts must match number of metadatas if provided.")

        vectors_to_upsert = []
        for i, text in enumerate(texts):
            try:
                # Use passage type for documents being stored
                embedding = await self._get_embedding(text, input_type="passage")
                vector_data = {
                    "id": ids[i],
                    "values": embedding,
                }

                # Ensure the text itself is in metadata if reranking is used
                current_metadata = metadatas[i] if metadatas and i < len(
                    metadatas) else {}
                if self.use_reranking and self.rerank_text_field not in current_metadata:
                    # Add text if missing
                    current_metadata[self.rerank_text_field] = text

                if current_metadata:
                    vector_data["metadata"] = current_metadata

                vectors_to_upsert.append(vector_data)
            except Exception as e:
                print(f"Failed to embed text with ID {ids[i]}: {e}")

        if vectors_to_upsert:
            await self.upsert(vectors_to_upsert, namespace=namespace)

    async def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None
    ) -> None:  # pragma: no cover
        """Upsert pre-embedded vectors into Pinecone asynchronously."""
        await self._ensure_initialized()
        try:
            upsert_params = {"vectors": vectors}
            if namespace:
                upsert_params["namespace"] = namespace

            await self.index.upsert(**upsert_params)
            print(
                f"Successfully upserted {len(vectors)} vectors into namespace '{namespace or 'default'}'.")
        except PineconeApiException as e:
            print(f"Pinecone API error during async upsert: {e}")
            raise
        except Exception as e:
            print(f"Error during async upsert: {e}")
            raise

    async def query_text(
        self,
        query_text: str,
        # Use rerank_top_k if reranking, else default
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:  # pragma: no cover
        """Embeds query text and queries Pinecone, optionally reranking results."""
        await self._ensure_initialized()

        final_top_k = top_k if top_k is not None else (
            self.rerank_top_k if self.use_reranking else 5)
        query_top_k = self.initial_query_top_k if self.use_reranking else final_top_k

        try:
            # Use query type for the search query
            query_vector = await self._get_embedding(query_text, input_type="query")

            # Perform initial vector search
            initial_results = await self.query(
                vector=query_vector,
                top_k=query_top_k,  # Fetch more candidates if reranking
                namespace=namespace,
                filter=filter,
                include_values=include_values,  # Values not needed for rerank
                # Metadata (esp. text field) is needed for rerank
                include_metadata=True
            )

            if not self.use_reranking or not initial_results:
                # Return initial results if not reranking or if no results found
                # Ensure we only return the requested final_top_k
                return initial_results[:final_top_k]

            # --- Reranking Step ---
            if not self.pinecone:
                raise RuntimeError(
                    "Pinecone client not initialized for reranking.")

            # Prepare documents for reranking
            documents_to_rerank = []
            original_results_map = {}
            for match in initial_results:
                doc_text = match.get("metadata", {}).get(
                    self.rerank_text_field)
                if doc_text:
                    documents_to_rerank.append(doc_text)
                    # Store original match by text
                    original_results_map[doc_text] = match

            if not documents_to_rerank:
                print(
                    "⚠️ Reranking skipped: No documents found with the specified text field.")
                return initial_results[:final_top_k]

            try:
                print(
                    f"Reranking {len(documents_to_rerank)} results using {self.rerank_model}...")

                # Define rerank parameters based on model
                rerank_params = {}
                if self.rerank_model == "cohere-rerank-3.5":
                    # Example, adjust as needed
                    rerank_params = {"max_chunks_per_doc": 3072}
                elif self.rerank_model in ["bge-reranker-v2-m3", "pinecone-rerank-v0"]:
                    rerank_params = {"truncate": "END"}  # Default for these

                # Prepare the rerank request
                rerank_request = {
                    "query": query_text,
                    "documents": documents_to_rerank,
                    "model": self.rerank_model,
                    "top_n": final_top_k,
                    "parameters": rerank_params
                }

                # Call the rerank API
                rerank_response = await self.pinecone.rerank(**rerank_request)

                # Process rerank results - structure will depend on specific Pinecone API version
                # This assumes a structure with 'results' containing document/score pairs
                reranked_results = []

                # Parse the response to extract document text and scores
                if "results" in rerank_response:
                    for result in rerank_response["results"]:
                        doc_text = result.get("document", {}).get("text", "")
                        score = result.get("relevance_score", 0)

                        # Find the original match based on document text
                        original_match = original_results_map.get(doc_text)
                        if original_match:
                            # Update score with rerank score, keep other metadata
                            updated_match = dict(original_match)
                            updated_match["score"] = score
                            reranked_results.append(updated_match)

                if reranked_results:
                    print(
                        f"Reranking complete. Returning {len(reranked_results)} results.")
                    return reranked_results
                else:
                    print(
                        "No matches after reranking, falling back to vector search results.")
                    return initial_results[:final_top_k]

            except Exception as rerank_error:
                print(
                    f"Error during reranking with {self.rerank_model}: {rerank_error}. Returning initial results.")
                # Fallback to initial results
                return initial_results[:final_top_k]

        except Exception as e:
            print(f"Failed to embed or query text '{query_text[:50]}...': {e}")
            return []

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:  # pragma: no cover
        """Query Pinecone for similar vectors asynchronously."""
        await self._ensure_initialized()
        try:
            # Build query parameters dictionary
            query_params = {
                "vector": vector,
                "top_k": top_k,
                "include_values": include_values,
                "include_metadata": include_metadata
            }

            if namespace:
                query_params["namespace"] = namespace
            if filter:
                query_params["filter"] = filter

            # Execute the query
            query_response = await self.index.query(**query_params)

            # Extract matches from response
            matches = query_response.get("matches", [])
            return matches

        except PineconeApiException as e:
            print(f"Pinecone API error during async query: {e}")
            raise
        except Exception as e:
            print(f"Error during async query: {e}")
            return []  # Return empty list on general error

    async def delete(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> None:  # pragma: no cover
        """Delete vectors by IDs from Pinecone asynchronously."""
        await self._ensure_initialized()
        try:
            delete_params = {"ids": ids}
            if namespace:
                delete_params["namespace"] = namespace

            await self.index.delete(**delete_params)
            print(
                f"Attempted to delete {len(ids)} vectors from namespace '{namespace or 'default'}'.")
        except PineconeApiException as e:
            print(f"Pinecone API error during async delete: {e}")
            raise
        except Exception as e:
            print(f"Error during async delete: {e}")
            raise

    async def describe_index_stats(self) -> Dict[str, Any]:  # pragma: no cover
        """Get statistics about the index asynchronously."""
        await self._ensure_initialized()
        try:
            stats = await self.index.describe_index_stats()
            # Convert response to dict if not already
            return stats if isinstance(stats, dict) else dict(stats)
        except Exception as e:
            print(f"Error describing index stats asynchronously: {e}")
            return {}

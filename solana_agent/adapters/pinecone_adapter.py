from typing import List, Dict, Any, Optional, Literal, Union
from pinecone import PineconeAsyncio, ServerlessSpec
from pinecone.exceptions import PineconeApiException

from solana_agent.interfaces.providers.vector_storage import VectorStorageProvider
from solana_agent.interfaces.providers.llm import LLMProvider
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
    Follows context management patterns for Pinecone client v3+.
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
        # Metadata field containing text for reranking
        rerank_text_field: str = "text",
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
        # FIX: Store index host instead of index object
        self.index_host: Optional[str] = None

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
            # FIX: Update dimension logic based on actual Pinecone model dimensions if needed
            if self.pinecone_embedding_model == "multilingual-e5-large":
                self.embedding_dimensions = 1024
            # Add other models as needed
            # elif self.pinecone_embedding_model == "some-other-model":
            #     self.embedding_dimensions = XXXX
            else:  # Default or fallback if model name doesn't match known ones
                # Use override if provided, otherwise keep the default/passed value
                self.embedding_dimensions = self.pinecone_embedding_dimension_override or self.embedding_dimensions

            print(
                f"Using Pinecone embedding model '{self.pinecone_embedding_model}' with dimension {self.embedding_dimensions}.")
        else:
            print(
                f"Using external LLM provider for embeddings with dimension {self.embedding_dimensions}.")

        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _initialize_async(self):
        """Asynchronously initialize the Pinecone client and get index host."""
        async with self._init_lock:
            if self._initialized:
                return

            try:
                print("Initializing PineconeAsyncio client...")
                self.pinecone = PineconeAsyncio(api_key=self.api_key)

                if self.create_index_if_not_exists:
                    await self._create_index_if_not_exists_async()

                print(
                    f"Describing Pinecone index '{self.index_name}' to get host...")
                index_description: IndexDescription = await self.pinecone.describe_index(self.index_name)
                self.index_host = index_description.host
                if not self.index_host:
                    raise RuntimeError(
                        f"Could not obtain host for index '{self.index_name}'.")
                print(f"Obtained index host: {self.index_host}")

                index_dimension = index_description.dimension
                if index_dimension != 0 and index_dimension != self.embedding_dimensions:
                    print(
                        f"⚠️ Warning: Index dimension ({index_dimension}) does not match configured embedding dimension ({self.embedding_dimensions})")

                # --- Add Logging around describe_index_stats ---
                print("Attempting to get index stats...")
                stats = await self.describe_index_stats()
                print(f"Successfully retrieved index stats: {stats}")
                # --- End Logging ---

                total_vector_count = stats.get("total_vector_count", 0)
                print(f"Current index contains {total_vector_count} vectors.")

                self._initialized = True
                # Add final confirmation
                print("Pinecone adapter initialization complete.")

            except PineconeApiException as e:
                print(f"Pinecone API error during async initialization: {e}")
                self.pinecone = None
                self.index_host = None
                raise
            except Exception as e:
                print(
                    f"Failed to initialize Pinecone async adapter for index '{self.index_name}': {e}")
                self.pinecone = None
                self.index_host = None
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
                # Wait time might need adjustment based on index size/type
                await asyncio.sleep(20)
            else:
                print(f"Using existing Pinecone index '{self.index_name}'")
        except Exception as e:
            print(f"Error creating Pinecone index asynchronously: {e}")
            raise

    async def _ensure_initialized(self):
        """Ensure the async client is initialized before use."""
        if not self._initialized:
            await self._initialize_async()
        # FIX: Check for pinecone client and index_host
        if not self._initialized or not self.pinecone or not self.index_host:
            raise RuntimeError(
                "Pinecone async client failed to initialize or get index host.")

    async def _get_embedding(self, text: str, input_type: InputType) -> List[float]:
        """Get embedding using either Pinecone native or external LLM provider."""
        if self.use_pinecone_embeddings:
            if not self.pinecone:
                raise RuntimeError(
                    "Pinecone client not initialized for embedding.")

            try:
                embed_params = {
                    "texts": [text],
                    "model": self.pinecone_embedding_model,
                }
                parameters = {
                    "input_type": input_type,
                    "truncate": "END",
                }
                # FIX: Adjust dimension parameter logic if needed for specific models
                if self.pinecone_embedding_dimension_override:
                    parameters["dimension"] = self.pinecone_embedding_dimension_override

                embed_params["parameters"] = parameters

                response = await self.pinecone.embed(**embed_params)

                if response and response.embeddings:
                    return response.embeddings[0]
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
            return await self.llm_provider.embed_text(text, dimensions=self.embedding_dimensions)

    async def upsert_text(
        self,
        texts: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None
    ) -> None:
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
                embedding = await self._get_embedding(text, input_type="passage")
                vector_data = {
                    "id": ids[i],
                    "values": embedding,
                }
                current_metadata = metadatas[i] if metadatas and i < len(
                    metadatas) else {}
                if self.use_reranking and self.rerank_text_field not in current_metadata:
                    current_metadata[self.rerank_text_field] = text
                if current_metadata:
                    vector_data["metadata"] = current_metadata
                vectors_to_upsert.append(vector_data)
            except Exception as e:
                print(f"Failed to embed text with ID {ids[i]}: {e}")
                # Decide if one failure should stop the whole batch

        if vectors_to_upsert:
            await self.upsert(vectors_to_upsert, namespace=namespace)

    async def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None
    ) -> None:
        """Upsert pre-embedded vectors into Pinecone asynchronously."""
        await self._ensure_initialized()
        try:
            # FIX: Use async with for the index instance
            async with self.pinecone.IndexAsyncio(host=self.index_host) as index_instance:
                upsert_params = {"vectors": vectors}
                if namespace:
                    upsert_params["namespace"] = namespace
                await index_instance.upsert(**upsert_params)
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
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """Embeds query text and queries Pinecone, optionally reranking results."""
        await self._ensure_initialized()

        final_top_k = top_k if top_k is not None else (
            self.rerank_top_k if self.use_reranking else 5)
        query_top_k = self.initial_query_top_k if self.use_reranking else final_top_k

        try:
            query_vector = await self._get_embedding(query_text, input_type="query")

            initial_results = await self.query(
                vector=query_vector,
                top_k=query_top_k,
                namespace=namespace,
                filter=filter,
                include_values=include_values,
                include_metadata=True  # Always needed for potential reranking
            )

            if not self.use_reranking or not initial_results:
                return initial_results[:final_top_k]

            # --- Reranking Step ---
            if not self.pinecone:
                raise RuntimeError(
                    "Pinecone client not initialized for reranking.")

            documents_to_rerank = []
            original_results_map = {}
            for match in initial_results:
                doc_text = match.get("metadata", {}).get(
                    self.rerank_text_field)
                if doc_text:
                    documents_to_rerank.append(doc_text)
                    original_results_map[doc_text] = match

            if not documents_to_rerank:
                print(
                    "⚠️ Reranking skipped: No documents found with the specified text field.")
                return initial_results[:final_top_k]

            try:
                print(
                    f"Reranking {len(documents_to_rerank)} results using {self.rerank_model}...")
                rerank_params = {}
                # Add model-specific params if needed
                # if self.rerank_model == "cohere-rerank-3.5":
                #     rerank_params = {"max_chunks_per_doc": 3072}
                # elif self.rerank_model in ["bge-reranker-v2-m3", "pinecone-rerank-v0"]:
                #     rerank_params = {"truncate": "END"}

                rerank_request = {
                    "query": query_text,
                    "documents": documents_to_rerank,
                    "model": self.rerank_model,
                    "top_n": final_top_k,
                    "parameters": rerank_params
                }

                rerank_response = await self.pinecone.rerank(**rerank_request)

                reranked_results = []
                if rerank_response and rerank_response.results:
                    for result in rerank_response.results:
                        # Adjust based on actual rerank response structure
                        doc_text = result.document.text if result.document else ""
                        score = result.relevance_score
                        original_match = original_results_map.get(doc_text)
                        if original_match:
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
    ) -> List[Dict[str, Any]]:
        """Query Pinecone for similar vectors asynchronously."""
        await self._ensure_initialized()
        try:
            # FIX: Use async with for the index instance
            async with self.pinecone.IndexAsyncio(host=self.index_host) as index_instance:
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
                query_response = await index_instance.query(**query_params)

            matches = query_response.get("matches", [])
            return matches

        except PineconeApiException as e:
            print(f"Pinecone API error during async query: {e}")
            raise
        except Exception as e:
            print(f"Error during async query: {e}")
            return []

    async def delete(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> None:
        """Delete vectors by IDs from Pinecone asynchronously."""
        await self._ensure_initialized()
        try:
            # FIX: Use async with for the index instance
            async with self.pinecone.IndexAsyncio(host=self.index_host) as index_instance:
                delete_params = {"ids": ids}
                if namespace:
                    delete_params["namespace"] = namespace
                await index_instance.delete(**delete_params)
            print(
                f"Attempted to delete {len(ids)} vectors from namespace '{namespace or 'default'}'.")
        except PineconeApiException as e:
            print(f"Pinecone API error during async delete: {e}")
            raise
        except Exception as e:
            print(f"Error during async delete: {e}")
            raise

    async def describe_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index asynchronously."""
        # Add entry log
        print(f"describe_index_stats: Entering for host {self.index_host}")
        try:
            print(
                f"describe_index_stats: Getting IndexAsyncio context for host {self.index_host}...")
            async with self.pinecone.IndexAsyncio(host=self.index_host) as index_instance:
                print(
                    f"describe_index_stats: Context acquired. Calling describe_index_stats on index instance...")
                stats_response = await index_instance.describe_index_stats()
                print(
                    f"describe_index_stats: Call completed. Response: {stats_response}")
            # Convert response to dict if necessary
            result_dict = stats_response.to_dict() if hasattr(
                stats_response, 'to_dict') else dict(stats_response)
            print(f"describe_index_stats: Returning stats dict: {result_dict}")
            return result_dict
        except Exception as e:
            print(f"Error describing index stats asynchronously: {e}")
            return {}

import logging  # Import logging
from typing import List, Dict, Any, Optional, Literal
from pinecone import PineconeAsyncio, ServerlessSpec
from pinecone.exceptions import PineconeApiException
import asyncio

from solana_agent.interfaces.providers.vector_storage import VectorStorageProvider
# LLMProvider is no longer needed here

# Setup logger for this module
logger = logging.getLogger(__name__)

# Type definitions remain useful
PineconeRerankModel = Literal[
    "cohere-rerank-3.5",
    "bge-reranker-v2-m3",
    "pinecone-rerank-v0",
]
# Kept for potential future use, but not used internally now
InputType = Literal["query", "passage"]
TruncateType = Literal["END", "NONE"]  # Kept for potential future use


class PineconeAdapter(VectorStorageProvider):
    """
    Adapter for interacting with Pinecone vector database using PineconeAsyncio.
    Assumes embeddings are generated externally (e.g., via OpenAI).
    Supports Pinecone native reranking.
    Follows context management patterns for Pinecone client v3+.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        # Default for OpenAI text-embedding-3-large, MUST match external embedder
        embedding_dimensions: int = 3072,
        cloud_provider: str = "aws",
        region: str = "us-east-1",
        metric: str = "cosine",
        create_index_if_not_exists: bool = True,
        # Reranking Config
        use_reranking: bool = False,
        rerank_model: Optional[PineconeRerankModel] = None,
        rerank_top_k: int = 3,  # Final number of results after reranking
        # Multiplier for initial fetch before rerank
        initial_query_top_k_multiplier: int = 5,
        # Metadata field containing text for reranking
        rerank_text_field: str = "text",
    ):
        """
        Initialize the Pinecone Adapter.

        Args:
            api_key: Pinecone API key.
            index_name: Name of the Pinecone index.
            embedding_dimensions: Dimension of the embeddings generated externally. MUST match the index dimension.
            cloud_provider: Cloud provider for the index (e.g., 'aws', 'gcp').
            region: Region for the index.
            metric: Distance metric for the index (e.g., 'cosine', 'dotproduct', 'euclidean').
            create_index_if_not_exists: Attempt to create the index if it doesn't exist.
            use_reranking: Enable Pinecone native reranking.
            rerank_model: The reranking model to use (required if use_reranking is True).
            rerank_top_k: Final number of results to return after reranking.
            initial_query_top_k_multiplier: Fetch top_k * multiplier results initially for reranking.
            rerank_text_field: The key in vector metadata containing the text content for reranking.
        """
        self.api_key = api_key
        self.index_name = index_name
        # Crucial: Must match external embedder and index
        self.embedding_dimensions = embedding_dimensions
        self.cloud_provider = cloud_provider
        self.region = region
        self.metric = metric
        self.create_index_if_not_exists = create_index_if_not_exists

        # Reranking Config
        self.use_reranking = use_reranking
        self.rerank_model = rerank_model
        self.rerank_top_k = rerank_top_k
        # Calculate how many results to fetch initially if reranking
        self.initial_query_top_k_multiplier = initial_query_top_k_multiplier

        self.rerank_text_field = rerank_text_field

        self.pinecone: Optional[PineconeAsyncio] = None
        # Store index host for connections
        self.index_host: Optional[str] = None

        # --- Validation ---
        if not self.api_key:
            raise ValueError("Pinecone API key is required.")
        if not self.index_name:
            raise ValueError("Pinecone index name is required.")
        if self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be a positive integer.")
        if self.use_reranking and not self.rerank_model:
            raise ValueError(
                "rerank_model must be specified when use_reranking is True."
            )

        logger.info(
            f"PineconeAdapter configured for index '{self.index_name}' using external embeddings with dimension {self.embedding_dimensions}."
        )
        if self.use_reranking:
            logger.info(f"Reranking enabled using model '{self.rerank_model}'.")

        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _initialize_async(self):
        """Asynchronously initialize the Pinecone client and get index host."""
        async with self._init_lock:
            if self._initialized:
                return

            try:
                logger.info("Initializing PineconeAsyncio client...")
                self.pinecone = PineconeAsyncio(api_key=self.api_key)

                if self.create_index_if_not_exists:
                    await self._create_index_if_not_exists_async()

                logger.info(
                    f"Describing Pinecone index '{self.index_name}' to get host..."
                )
                index_description = await self.pinecone.describe_index(self.index_name)
                self.index_host = index_description.host
                if not self.index_host:
                    raise RuntimeError(
                        f"Could not obtain host for index '{self.index_name}'."
                    )
                logger.info(f"Obtained index host: {self.index_host}")

                # Validate index dimension matches configured dimension
                index_dimension = index_description.dimension
                if (
                    index_dimension != 0
                    and index_dimension != self.embedding_dimensions
                ):
                    # This is a critical mismatch
                    raise ValueError(
                        f"CRITICAL MISMATCH: Pinecone index dimension ({index_dimension}) "
                        f"does not match configured embedding dimension ({self.embedding_dimensions}). "
                        f"Ensure the index was created with the correct dimension or update the adapter configuration."
                    )
                elif index_dimension == 0:
                    logger.warning(
                        f"Pinecone index dimension reported as 0. Cannot verify match with configured dimension ({self.embedding_dimensions})."
                    )

                logger.info("Attempting to get index stats...")
                stats = await self.describe_index_stats()
                logger.info(f"Successfully retrieved index stats: {stats}")

                total_vector_count = stats.get("total_vector_count", 0)
                logger.info(
                    f"Current index '{self.index_name}' contains {total_vector_count} vectors."
                )

                self._initialized = True
                logger.info("Pinecone adapter initialization complete.")

            except PineconeApiException as e:
                logger.error(
                    f"Pinecone API error during async initialization: {e}",
                    exc_info=True,
                )
                self.pinecone = None
                self.index_host = None
                raise
            except Exception as e:
                logger.exception(
                    f"Failed to initialize Pinecone async adapter for index '{self.index_name}': {e}"
                )
                self.pinecone = None
                self.index_host = None
                raise

    async def _create_index_if_not_exists_async(self) -> None:
        """Create the Pinecone index asynchronously if it doesn't already exist."""
        if not self.pinecone:
            raise RuntimeError("Pinecone client not initialized before creating index.")
        try:
            indexes_response = await self.pinecone.list_indexes()
            existing_indexes = indexes_response.get("indexes", [])
            existing_names = [idx.get("name") for idx in existing_indexes]

            if self.index_name not in existing_names:
                logger.info(
                    f"Creating Pinecone index '{self.index_name}' with dimension {self.embedding_dimensions}..."
                )

                spec_data = {"cloud": self.cloud_provider, "region": self.region}

                create_params = {
                    "name": self.index_name,
                    "dimension": self.embedding_dimensions,  # Use configured dimension
                    "metric": self.metric,
                    # Assuming serverless, adjust if needed
                    "spec": ServerlessSpec(**spec_data),
                }

                await self.pinecone.create_index(**create_params)
                logger.info(
                    f"Successfully initiated creation of Pinecone index '{self.index_name}'. Waiting for it to be ready..."
                )
                # Wait time might need adjustment based on index size/type and cloud provider
                await asyncio.sleep(30)  # Increased wait time
            else:
                logger.info(f"Using existing Pinecone index '{self.index_name}'")
        except Exception as e:
            logger.exception(
                f"Error checking or creating Pinecone index asynchronously: {e}"
            )
            raise

    async def _ensure_initialized(self):
        """Ensure the async client is initialized before use."""
        if not self._initialized:
            await self._initialize_async()
        if not self._initialized or not self.pinecone or not self.index_host:
            raise RuntimeError(
                "Pinecone async client failed to initialize or get index host."
            )

    # _get_embedding method is removed as embeddings are handled externally

    async def upsert_text(self, *args, **kwargs):  # pragma: no cover
        """Deprecated: Embeddings should be generated externally."""
        raise NotImplementedError(
            "upsert_text is deprecated. Use the generic upsert method with pre-computed vectors."
        )

    async def upsert(
        self,
        # Expects {"id": str, "values": List[float], "metadata": Optional[Dict]}
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> None:  # pragma: no cover
        """Upsert pre-embedded vectors into Pinecone asynchronously."""
        await self._ensure_initialized()
        if not vectors:
            logger.info("Upsert skipped: No vectors provided.")
            return
        try:
            async with self.pinecone.IndexAsyncio(
                host=self.index_host
            ) as index_instance:
                upsert_params = {"vectors": vectors}
                if namespace:
                    upsert_params["namespace"] = namespace
                await index_instance.upsert(**upsert_params)
            logger.info(
                f"Successfully upserted {len(vectors)} vectors into namespace '{namespace or 'default'}'."
            )
        except PineconeApiException as e:
            logger.error(f"Pinecone API error during async upsert: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.exception(f"Error during async upsert: {e}")
            raise

    async def query_text(self, *args, **kwargs):  # pragma: no cover
        """Deprecated: Use query() for simple vector search or query_and_rerank() for reranking."""
        raise NotImplementedError(
            "query_text is deprecated. Use query() or query_and_rerank() with a pre-computed vector."
        )

    async def query_and_rerank(
        self,
        vector: List[float],
        query_text_for_rerank: str,  # The original query text is needed for the reranker
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:  # pragma: no cover
        """
        Queries Pinecone with a vector and reranks the results using Pinecone's reranker.
        Requires 'use_reranking' to be True and 'rerank_model' to be set during init.

        Args:
            vector: The query vector.
            query_text_for_rerank: The original text query used for the reranking model.
            top_k: The final number of results desired after reranking.
            namespace: Optional Pinecone namespace.
            filter: Optional metadata filter for the initial query.
            include_values: Whether to include vector values in the results.
            include_metadata: Whether to include metadata in the results.

        Returns:
            A list of reranked result dictionaries.
        """
        await self._ensure_initialized()

        if not self.use_reranking:
            logger.warning(
                "query_and_rerank called but use_reranking is False. Performing standard query."
            )
            return await self.query(
                vector, top_k, namespace, filter, include_values, include_metadata
            )

        if not self.rerank_model:
            raise ValueError(
                "Cannot rerank: rerank_model was not specified during initialization."
            )

        # Determine how many results to fetch initially for reranking
        initial_k = top_k * self.initial_query_top_k_multiplier

        try:
            # 1. Initial Vector Search
            initial_results = await self.query(
                vector=vector,
                top_k=initial_k,
                namespace=namespace,
                filter=filter,
                include_values=include_values,  # Include values if requested in final output
                include_metadata=True,  # Always need metadata for reranking text field
            )

            if not initial_results:
                return []  # No results from initial query

            # 2. Prepare for Reranking
            documents_to_rerank = []
            original_results_map = {}
            for match in initial_results:
                # Ensure metadata exists and contains the rerank text field
                doc_metadata = match.get("metadata")
                if isinstance(doc_metadata, dict):
                    doc_text = doc_metadata.get(self.rerank_text_field)
                    if doc_text and isinstance(doc_text, str):
                        documents_to_rerank.append(doc_text)
                        # Store original match keyed by the text for easy lookup after reranking
                        original_results_map[doc_text] = match
                    else:
                        logger.warning(
                            f"Skipping result ID {match.get('id')} for reranking - missing or invalid text in field '{self.rerank_text_field}'."
                        )
                else:
                    logger.warning(
                        f"Skipping result ID {match.get('id')} for reranking - metadata is missing or not a dictionary."
                    )

            if not documents_to_rerank:
                logger.warning(
                    f"Reranking skipped: No documents found with text in the specified field ('{self.rerank_text_field}'). Returning top {top_k} initial results."
                )
                # Return the originally requested top_k
                return initial_results[:top_k]

            # 3. Perform Reranking Call
            if not self.pinecone:
                raise RuntimeError("Pinecone client not initialized for reranking.")

            try:
                logger.info(
                    f"Reranking {len(documents_to_rerank)} results using {self.rerank_model} for query: '{query_text_for_rerank[:50]}...'"
                )
                rerank_params = {}  # Add model-specific params if needed

                rerank_request = {
                    "query": query_text_for_rerank,
                    "documents": documents_to_rerank,
                    "model": self.rerank_model,
                    "top_n": top_k,  # Request the final desired number
                    "parameters": rerank_params,
                }

                rerank_response = await self.pinecone.rerank(**rerank_request)

                # 4. Process Reranked Results
                reranked_results = []
                if rerank_response and rerank_response.results:
                    for result in rerank_response.results:
                        # Adjust based on actual rerank response structure (assuming v3+)
                        doc_text = result.document.text if result.document else ""
                        score = result.relevance_score
                        original_match = original_results_map.get(doc_text)
                        if original_match:
                            # Create a new dict to avoid modifying the original map values
                            updated_match = dict(original_match)
                            # Update score with relevance score
                            updated_match["score"] = score
                            reranked_results.append(updated_match)
                        else:
                            logger.warning(
                                f"Reranked document text not found in original results map: '{doc_text[:50]}...'"
                            )

                if reranked_results:
                    logger.info(
                        f"Reranking complete. Returning {len(reranked_results)} results."
                    )
                    return reranked_results
                else:
                    # Should not happen if rerank_response.results existed, but handle defensively
                    logger.warning(
                        "No matches found after processing reranking response. Falling back to initial vector search results."
                    )
                    return initial_results[:top_k]

            except Exception as rerank_error:
                logger.error(
                    f"Error during reranking with {self.rerank_model}: {rerank_error}. Returning initial results.",
                    exc_info=True,
                )
                # Fallback to top_k initial results
                return initial_results[:top_k]

        except Exception as e:
            logger.exception(f"Failed to query or rerank: {e}")
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
        """
        Query Pinecone for similar vectors asynchronously (no reranking).

        Args:
            vector: The query vector.
            top_k: The number of results to return.
            namespace: Optional Pinecone namespace.
            filter: Optional metadata filter.
            include_values: Whether to include vector values in the results.
            include_metadata: Whether to include metadata in the results.

        Returns:
            A list of result dictionaries.
        """
        await self._ensure_initialized()
        try:
            async with self.pinecone.IndexAsyncio(
                host=self.index_host
            ) as index_instance:
                query_params = {
                    "vector": vector,
                    "top_k": top_k,
                    "include_values": include_values,
                    "include_metadata": include_metadata,
                }
                if namespace:
                    query_params["namespace"] = namespace
                if filter:
                    query_params["filter"] = filter
                query_response = await index_instance.query(**query_params)

            # Ensure response structure is handled safely
            matches = query_response.get("matches", []) if query_response else []
            return matches

        except PineconeApiException as e:
            logger.error(f"Pinecone API error during async query: {e}", exc_info=True)
            raise  # Re-raise API errors
        except Exception as e:
            logger.exception(f"Error during async query: {e}")
            return []  # Return empty list for general errors

    async def delete(
        self, ids: List[str], namespace: Optional[str] = None
    ) -> None:  # pragma: no cover
        """Delete vectors by IDs from Pinecone asynchronously."""
        await self._ensure_initialized()
        if not ids:
            logger.info("Delete skipped: No IDs provided.")
            return
        try:
            async with self.pinecone.IndexAsyncio(
                host=self.index_host
            ) as index_instance:
                delete_params = {"ids": ids}
                if namespace:
                    delete_params["namespace"] = namespace
                await index_instance.delete(**delete_params)
            logger.info(
                f"Attempted to delete {len(ids)} vectors from namespace '{namespace or 'default'}'."
            )
        except PineconeApiException as e:
            logger.error(f"Pinecone API error during async delete: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.exception(f"Error during async delete: {e}")
            raise

    async def describe_index_stats(self) -> Dict[str, Any]:  # pragma: no cover
        """Get statistics about the index asynchronously."""
        logger.debug(
            f"describe_index_stats: Entering for host {self.index_host}"
        )  # Changed to debug
        try:
            logger.debug(
                f"describe_index_stats: Getting IndexAsyncio context for host {self.index_host}..."
            )  # Changed to debug
            async with self.pinecone.IndexAsyncio(
                host=self.index_host
            ) as index_instance:
                logger.debug(
                    "describe_index_stats: Context acquired. Calling describe_index_stats on index instance..."
                )  # Changed to debug
                stats_response = await index_instance.describe_index_stats()
                logger.debug(
                    f"describe_index_stats: Call completed. Response: {stats_response}"
                )  # Changed to debug

            # Convert response to dict if necessary (handle potential None or different types)
            if hasattr(stats_response, "to_dict"):
                result_dict = stats_response.to_dict()
            elif isinstance(stats_response, dict):
                result_dict = stats_response
            else:
                # Attempt basic conversion or return empty
                try:
                    result_dict = dict(stats_response)
                except (TypeError, ValueError):
                    logger.warning(
                        f"Could not convert stats_response to dict: {stats_response}"
                    )
                    result_dict = {}

            logger.debug(
                f"describe_index_stats: Returning stats dict: {result_dict}"
            )  # Changed to debug
            return result_dict
        except PineconeApiException as e:
            logger.error(
                f"Pinecone API error describing index stats asynchronously: {e}",
                exc_info=True,
            )
            raise  # Re-raise API errors
        except Exception as e:
            logger.exception(f"Error describing index stats asynchronously: {e}")
            return {}  # Return empty dict for general errors

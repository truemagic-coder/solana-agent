from typing import List, Dict, Any, Optional


class VectorStorageProvider:
    """Interface for Vector Storage Providers."""

    async def upsert(
        self, vectors: List[Dict[str, Any]], namespace: Optional[str] = None
    ) -> None:
        """Upsert vectors into the storage."""
        pass

    async def upsert_text(
        self,
        texts: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Embeds texts and upserts them into the storage."""
        pass

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """Query for similar vectors."""
        pass

    async def query_text(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """Embeds query text and queries the storage."""
        pass

    async def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        """Delete vectors by IDs."""
        pass

    async def describe_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index/collection."""
        pass

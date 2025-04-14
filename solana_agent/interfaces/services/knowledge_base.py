from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union


class KnowledgeService(ABC):
    """
    Interface for a Knowledge Base service.
    """

    @abstractmethod
    async def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> str:
        """
        Add a document to the knowledge base.
        """
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
    async def delete_document(
        self,
        document_id: str,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Delete a document from the knowledge base.
        """
        pass

    @abstractmethod
    async def update_document(
        self,
        document_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Update an existing document in the knowledge base.
        """
        pass

    @abstractmethod
    async def add_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 50
    ) -> List[str]:
        """
        Add multiple documents in batches.
        """
        pass

    @abstractmethod
    async def add_pdf_document(
        self,
        pdf_data: Union[bytes, str],  # PDF bytes or file path
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        chunk_batch_size: int = 50
    ) -> str:
        """
        Add a PDF document to the knowledge base.
        """
        pass

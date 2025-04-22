from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Literal, Optional, Union
from solana_agent.interfaces.plugins.plugins import Tool
from solana_agent.interfaces.services.routing import RoutingService as RoutingInterface


class SolanaAgent(ABC):
    """Interface for the Solana Agent client."""

    @abstractmethod
    async def process(
        self,
        user_id: str,
        message: Union[str, bytes],
        prompt: Optional[str] = None,
        output_format: Literal["text", "audio"] = "text",
        audio_voice: Literal[
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "fable",
            "onyx",
            "nova",
            "sage",
            "shimmer",
        ] = "nova",
        audio_instructions: str = "You speak in a friendly and helpful manner.",
        audio_output_format: Literal[
            "mp3", "opus", "aac", "flac", "wav", "pcm"
        ] = "aac",
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
        router: Optional[RoutingInterface] = None,
        images: Optional[List[Union[str, bytes]]] = None,
    ) -> AsyncGenerator[Union[str, bytes], None]:
        """Process a user message and return the response stream."""
        pass

    @abstractmethod
    async def delete_user_history(self, user_id: str) -> None:
        """Delete the conversation history for a user."""
        pass

    @abstractmethod
    async def get_user_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """Get paginated message history for a user."""
        pass

    @abstractmethod
    def register_tool(self, agent_name: str, tool: Tool) -> bool:
        """Register a tool with the agent system."""
        pass

    @abstractmethod
    async def kb_add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """Add a document to the knowledge base."""
        pass

    @abstractmethod
    async def kb_query(
        self,
        query_text: str,
        filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        namespace: Optional[str] = None,
        include_content: bool = True,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """Query the knowledge base."""
        pass

    @abstractmethod
    async def kb_delete_document(
        self, document_id: str, namespace: Optional[str] = None
    ) -> bool:
        """Delete a document from the knowledge base."""
        pass

    @abstractmethod
    async def kb_update_document(
        self,
        document_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> bool:
        """Update an existing document in the knowledge base."""
        pass

    @abstractmethod
    async def kb_add_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 50,
    ) -> List[str]:
        """Add multiple documents to the knowledge base in batches."""
        pass

    @abstractmethod
    async def kb_add_pdf_document(
        self,
        pdf_data: Union[bytes, str],
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        chunk_batch_size: int = 50,
    ) -> str:
        """Add a PDF document to the knowledge base."""
        pass

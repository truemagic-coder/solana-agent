"""
Simplified client interface for interacting with the Solana Agent system.

This module provides a clean API for end users to interact with
the agent system without dealing with internal implementation details.
"""

import json
import importlib.util
from typing import AsyncGenerator, Dict, Any, List, Literal, Optional, Type, Union

from pydantic import BaseModel

from solana_agent.factories.agent_factory import SolanaAgentFactory
from solana_agent.interfaces.client.client import SolanaAgent as SolanaAgentInterface
from solana_agent.interfaces.plugins.plugins import Tool
from solana_agent.services.knowledge_base import KnowledgeBaseService
from solana_agent.interfaces.services.routing import RoutingService as RoutingInterface


class SolanaAgent(SolanaAgentInterface):
    """Simplified client interface for interacting with the agent system."""

    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """Initialize the agent system from config file or dictionary.

        Args:
            config_path: Path to configuration file (JSON or Python)
            config: Configuration dictionary
        """
        if not config and not config_path:
            raise ValueError("Either config or config_path must be provided")

        if config_path:
            with open(config_path, "r") as f:
                if config_path.endswith(".json"):
                    config = json.load(f)
                else:
                    # Assume it's a Python file
                    spec = importlib.util.spec_from_file_location("config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    config = config_module.config

        self.query_service = SolanaAgentFactory.create_from_config(config)

    async def process(
        self,
        user_id: str,
        message: Union[str, bytes],
        prompt: Optional[str] = None,
        capture_schema: Optional[Dict[str, Any]] = None,
        capture_name: Optional[str] = None,
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
        audio_output_format: Literal[
            "mp3", "opus", "aac", "flac", "wav", "pcm"
        ] = "aac",
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
        router: Optional[RoutingInterface] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        output_model: Optional[Type[BaseModel]] = None,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:  # pragma: no cover
        """Process a user message (text or audio) and optional images, returning the response stream.

        Args:
            user_id: User ID
            message: Text message or audio bytes
            prompt: Optional prompt for the agent
            output_format: Response format ("text" or "audio")
            capture_schema: Optional Pydantic schema for structured output
            capture_name: Optional name for structured output capture
            audio_voice: Voice to use for audio output
            audio_output_format: Audio output format
            audio_input_format: Audio input format
            router: Optional routing service for processing
            images: Optional list of image URLs (str) or image bytes.
            output_model: Optional Pydantic model for structured output

        Returns:
            Async generator yielding response chunks (text strings or audio bytes)
        """
        async for chunk in self.query_service.process(
            user_id=user_id,
            query=message,
            images=images,
            output_format=output_format,
            audio_voice=audio_voice,
            audio_output_format=audio_output_format,
            audio_input_format=audio_input_format,
            prompt=prompt,
            router=router,
            output_model=output_model,
            capture_schema=capture_schema,
            capture_name=capture_name,
        ):
            yield chunk

    async def delete_user_history(self, user_id: str) -> None:
        """
        Delete the conversation history for a user.

        Args:
            user_id: User ID
        """
        await self.query_service.delete_user_history(user_id)

    async def get_user_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "desc",  # "asc" for oldest-first, "desc" for newest-first
    ) -> Dict[str, Any]:  # pragma: no cover
        """
        Get paginated message history for a user.

        Args:
            user_id: User ID
            page_num: Page number (starting from 1)
            page_size: Number of messages per page
            sort_order: Sort order ("asc" or "desc")

        Returns:
            Dictionary with paginated results and metadata
        """
        return await self.query_service.get_user_history(
            user_id, page_num, page_size, sort_order
        )

    def register_tool(self, agent_name: str, tool: Tool) -> bool:
        """
        Register a tool with the agent system.

        Args:
            agent_name: Name of the agent to register the tool with
            tool: Tool instance to register

        Returns:
            True if successful, False
        """
        success = self.query_service.agent_service.tool_registry.register_tool(tool)
        if success:
            self.query_service.agent_service.assign_tool_for_agent(
                agent_name, tool.name
            )
        return success

    def _ensure_kb(self) -> KnowledgeBaseService:
        """Checks if the knowledge base service is available and returns it."""
        if (
            hasattr(self.query_service, "knowledge_base")
            and self.query_service.knowledge_base
        ):
            return self.query_service.knowledge_base
        raise ValueError("Knowledge base not configured")

    async def kb_add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """
        Add a document to the knowledge base.

        Args:
            text: Document text
            metadata: Document metadata
            document_id: Optional document ID
            namespace: Optional namespace

        Returns:
            Document ID
        """
        kb = self._ensure_kb()
        return await kb.add_document(
            text=text,
            metadata=metadata,
            document_id=document_id,
            namespace=namespace,
        )

    async def kb_query(
        self,
        query_text: str,
        filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        namespace: Optional[str] = None,
        include_content: bool = True,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base.

        Args:
            query_text: Query text
            filter: Optional metadata filter
            top_k: Number of results
            namespace: Optional namespace
            include_content: Whether to include content
            include_metadata: Whether to include metadata

        Returns:
            List of matching documents
        """
        kb = self._ensure_kb()
        return await kb.query(
            query_text=query_text,
            filter=filter,
            top_k=top_k,
            namespace=namespace,
            include_content=include_content,
            include_metadata=include_metadata,
        )

    async def kb_delete_document(
        self, document_id: str, namespace: Optional[str] = None
    ) -> bool:
        """
        Delete a document from the knowledge base.

        Args:
            document_id: Document ID
            namespace: Optional namespace

        Returns:
            True if successful
        """
        kb = self._ensure_kb()
        return await kb.delete_document(
            document_id=document_id,
            namespace=namespace,
        )

    async def kb_add_pdf_document(
        self,
        pdf_data: Union[bytes, str],
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        chunk_batch_size: int = 50,
    ) -> str:
        """
        Add a PDF document to the knowledge base.

        Args:
            pdf_data: PDF bytes or file path
            metadata: Document metadata
            document_id: Optional document ID
            namespace: Optional namespace
            chunk_batch_size: Batch size for chunking

        Returns:
            Document ID
        """
        kb = self._ensure_kb()
        return await kb.add_pdf_document(
            pdf_data=pdf_data,
            metadata=metadata,
            document_id=document_id,
            namespace=namespace,
            chunk_batch_size=chunk_batch_size,
        )

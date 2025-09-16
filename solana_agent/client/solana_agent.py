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
from solana_agent.interfaces.providers.realtime import RealtimeChunk


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
        audio_preset: Literal["default", "expo_pcm16"] = "default",
        # Realtime (WebSocket) options — used when realtime=True
        realtime: bool = False,
        vad: Optional[bool] = False,
        rt_encode_input: bool = False,
        rt_encode_output: bool = False,
        rt_output_modalities: Optional[List[Literal["audio", "text"]]] = None,
        rt_voice: Literal[
            "alloy",
            "ash",
            "ballad",
            "cedar",
            "coral",
            "echo",
            "marin",
            "sage",
            "shimmer",
            "verse",
        ] = "marin",
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
    ) -> AsyncGenerator[
        Union[str, bytes, BaseModel, RealtimeChunk], None
    ]:  # pragma: no cover
        """Process a user message (text or audio) and optional images, returning the response stream.

        Args:
            user_id: User ID
            message: Text message or audio bytes
            prompt: Optional prompt for the agent
            output_format: Response format ("text" or "audio")
            capture_schema: Optional Pydantic schema for structured output
            capture_name: Optional name for structured output capture
            realtime: Whether to use realtime (WebSocket) processing
            vad: Whether to use voice activity detection (for audio input)
            rt_encode_input: Whether to re-encode input audio for compatibility
            rt_encode_output: Whether to re-encode output audio for compatibility
            rt_output_modalities: Modalities to return in realtime (default both if None)
            rt_voice: Voice to use for realtime audio output
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
            realtime=realtime,
            vad=vad,
            rt_encode_input=rt_encode_input,
            rt_encode_output=rt_encode_output,
            rt_output_modalities=rt_output_modalities,
            rt_voice=rt_voice,
            audio_voice=audio_voice,
            audio_output_format=audio_output_format,
            audio_input_format=audio_input_format,
            audio_preset=audio_preset,
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
        else:
            raise AttributeError("Knowledge base service not configured or available.")

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
            text: Document text content.
            metadata: Document metadata.
            document_id: Optional document ID.
            namespace: Optional Pinecone namespace.

        Returns:
            The document ID.
        """
        kb = self._ensure_kb()
        return await kb.add_document(text, metadata, document_id, namespace)

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
            query_text: Search query text.
            filter: Optional filter criteria.
            top_k: Maximum number of results.
            namespace: Optional Pinecone namespace.
            include_content: Include document content in results.
            include_metadata: Include document metadata in results.

        Returns:
            List of matching documents.
        """
        kb = self._ensure_kb()
        return await kb.query(
            query_text, filter, top_k, namespace, include_content, include_metadata
        )

    async def kb_delete_document(
        self, document_id: str, namespace: Optional[str] = None
    ) -> bool:
        """
        Delete a document from the knowledge base.

        Args:
            document_id: ID of document to delete.
            namespace: Optional Pinecone namespace.

        Returns:
            True if successful.
        """
        kb = self._ensure_kb()
        return await kb.delete_document(document_id, namespace)

    async def kb_add_pdf_document(
        self,
        pdf_data: Union[bytes, str],
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        chunk_batch_size: int = 50,
    ) -> str:
        """
        Add a PDF document to the knowledge base via the client.

        Args:
            pdf_data: PDF content as bytes or a path to the PDF file.
            metadata: Document metadata.
            document_id: Optional parent document ID.
            namespace: Optional Pinecone namespace for chunks.
            chunk_batch_size: Batch size for upserting chunks.

        Returns:
            The parent document ID.
        """
        kb = self._ensure_kb()
        # Type check added for clarity, though handled in service
        if not isinstance(pdf_data, (bytes, str)):
            raise TypeError("pdf_data must be bytes or a file path string.")
        return await kb.add_pdf_document(
            pdf_data, metadata, document_id, namespace, chunk_batch_size
        )

    async def create_realtime_session(
        self,
        user_id: str,
        vad: bool = True,
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm", "pcm"
        ] = "pcm",
        audio_output_format: Literal[
            "mp3", "opus", "aac", "flac", "wav", "pcm"
        ] = "pcm",
        rt_output_modalities: Optional[List[Literal["audio", "text"]]] = None,
        rt_voice: Literal[
            "alloy",
            "ash",
            "ballad",
            "cedar",
            "coral",
            "echo",
            "marin",
            "sage",
            "shimmer",
            "verse",
        ] = "marin",
        agent_name: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Create a persistent realtime WebSocket session for reuse across requests.

        This method creates a WebSocket session that can be reused for multiple
        realtime audio/text interactions, avoiding the overhead of creating new
        connections for each request.

        Args:
            user_id: User ID for the session
            vad: Whether to use voice activity detection (semantic_vad)
            audio_input_format: Input audio format
            audio_output_format: Output audio format
            rt_output_modalities: Modalities to return (default both if None)
            rt_voice: Voice to use for audio output
            agent_name: Optional agent name to use for session
            prompt: Optional prompt for the session

        Returns:
            Session ID for use with realtime_send()
        """
        return await self.query_service.create_realtime_session(
            user_id=user_id,
            vad=vad,
            audio_input_format=audio_input_format,
            audio_output_format=audio_output_format,
            rt_output_modalities=rt_output_modalities,
            rt_voice=rt_voice,
            agent_name=agent_name,
            prompt=prompt,
        )

    async def realtime_send(
        self,
        session_id: str,
        query: Union[str, bytes],
        vad: Optional[bool] = None,
        output_format: Literal["text", "audio"] = "audio",
        audio_preset: Literal["default", "expo_pcm16"] = "default",
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm", "pcm"
        ] = "pcm",
        audio_output_format: Literal[
            "mp3", "opus", "aac", "flac", "wav", "pcm"
        ] = "pcm",
    ) -> AsyncGenerator[Union[str, bytes], None]:
        """
        Send audio/text to an existing realtime session and get response stream.

        This method reuses an existing WebSocket session created with
        create_realtime_session(), providing much better performance than
        creating new sessions for each request.

        Supports the same audio encoding/decoding patterns as the main process()
        method, including expo_pcm16 preset for 16kHz input/output with FFmpeg
        transcoding.

        Args:
            session_id: Session ID from create_realtime_session()
            query: Text message or audio bytes to send
            vad: Override VAD setting for this request (optional)
            output_format: Response format ("text" or "audio")
            audio_preset: Audio processing preset ("default" or "expo_pcm16")
            audio_input_format: Input audio format for encoding detection
            audio_output_format: Output audio format for encoding

        Yields:
            Response chunks (text strings or audio bytes)
        """
        # Check if we need expo mode processing
        expo_mode = audio_preset == "expo_pcm16" and output_format == "audio"
        is_audio_bytes = isinstance(query, (bytes, bytearray))

        # Handle input audio encoding for expo mode
        processed_query = query
        if expo_mode and is_audio_bytes:
            # Parse WAV header for mono PCM16 to detect 16k input (same logic as process method)
            def _parse_wav_header(b: bytes) -> int:
                try:
                    if len(b) < 44:
                        return 0
                    if b[0:4] != b"RIFF" or b[8:12] != b"WAVE":
                        return 0
                    if int.from_bytes(b[20:22], "little") != 1:  # audio format PCM
                        return 0
                    if int.from_bytes(b[22:24], "little") != 1:  # mono
                        return 0
                    return int.from_bytes(b[24:28], "little")
                except Exception:
                    return 0

            bq = bytes(query)
            if bq:
                in_rate = 0
                fmt_lower = audio_input_format.lower()

                if fmt_lower == "wav":
                    in_rate = _parse_wav_header(bq)
                elif fmt_lower == "pcm":
                    # Assume 16k for raw PCM in expo mode
                    in_rate = 16000

                if in_rate == 16000:
                    try:
                        if fmt_lower == "wav":
                            pcm16_16k = bq[44:]
                        else:  # raw pcm
                            pcm16_16k = bq

                        from solana_agent.adapters.ffmpeg_transcoder import (
                            FFmpegTranscoder,
                        )

                        # Upsample from 16k to 24k for realtime API
                        pcm16_24k = await FFmpegTranscoder().resample_pcm16(
                            pcm16_16k,
                            16000,
                            24000,
                            output_container="raw",
                        )
                        processed_query = pcm16_24k

                    except Exception:
                        # Fall back to original if upsampling fails
                        processed_query = bq

        # Handle output processing based on preset and format
        if output_format == "text":
            # Text output - no audio encoding needed
            async for chunk in self.query_service.realtime_send(
                session_id=session_id,
                query=processed_query,
                vad=vad,
                output_format=output_format,
            ):
                yield chunk
        elif expo_mode:
            # Expo mode: force PCM output, then downsample to 16k WAV
            from solana_agent.adapters.ffmpeg_transcoder import FFmpegTranscoder

            async def _pcm24_iter():
                async for chunk in self.query_service.realtime_send(
                    session_id=session_id,
                    query=processed_query,
                    vad=vad,
                    output_format=output_format,
                ):
                    yield chunk

            # Downsample from 24k to 16k and wrap in WAV
            async for resampled_chunk in FFmpegTranscoder().stream_resample_pcm16(
                _pcm24_iter(), 24000, 16000, output_container="wav"
            ):
                yield resampled_chunk
        elif output_format == "audio" and audio_output_format.lower() != "pcm":
            # Need to encode from PCM to the requested format
            from solana_agent.adapters.ffmpeg_transcoder import FFmpegTranscoder

            async def _pcm_iter():
                async for chunk in self.query_service.realtime_send(
                    session_id=session_id,
                    query=processed_query,
                    vad=vad,
                    output_format=output_format,
                ):
                    yield chunk

            # Encode PCM to requested format
            async for encoded_chunk in FFmpegTranscoder().stream_from_pcm16(
                _pcm_iter(), output_container=audio_output_format.lower()
            ):
                yield encoded_chunk
        else:
            # Standard processing - PCM output, no encoding needed
            async for chunk in self.query_service.realtime_send(
                session_id=session_id,
                query=processed_query,
                vad=vad,
                output_format=output_format,
            ):
                yield chunk

    async def close_realtime_session(self, session_id: str) -> bool:
        """
        Close and clean up a persistent realtime session.

        Args:
            session_id: Session ID from create_realtime_session()

        Returns:
            True if session was found and closed, False if not found
        """
        return await self.query_service.close_realtime_session(session_id)

    def list_realtime_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List active realtime sessions for a user.

        Args:
            user_id: User ID to list sessions for

        Returns:
            List of session info dictionaries with metadata
        """
        return self.query_service.list_realtime_sessions(user_id)

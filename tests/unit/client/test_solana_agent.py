"""
Tests for the SolanaAgent client interface.

This module provides comprehensive test coverage for the SolanaAgent client
including initialization, message processing, history management, and tool registration.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, AsyncGenerator, Union

from solana_agent.client.solana_agent import SolanaAgent
from solana_agent.interfaces.plugins.plugins import Tool


@pytest.fixture
def config_dict():
    """Fixture providing test configuration."""
    return {
        "mongo": {
            "connection_string": "mongodb://localhost:27017",
            "database": "test_db"
        },
        "openai": {"api_key": "test_key"},
        "agents": [{
            "name": "test_agent",
            "instructions": "Test agent instructions",
            "specialization": "Testing"
        }],
        "knowledge_base": {
            "pinecone": {"api_key": "fake", "environment": "fake"},
            "mongodb": {"connection_string": "mongodb://localhost:27017", "database": "test_db"}
        }
    }


@pytest.fixture
def mock_query_service():
    """Create a mock query service with all required methods."""
    mock = AsyncMock()

    async def mock_process(*args, **kwargs):
        yield "Test response"

    mock.process.side_effect = mock_process
    mock.delete_user_history = AsyncMock()
    mock.get_user_history = AsyncMock(
        return_value={"messages": [], "total": 0})

    # Configure agent service
    mock.agent_service = MagicMock()
    mock.agent_service.tool_registry = MagicMock()
    mock.agent_service.get_all_ai_agents = MagicMock(
        return_value=["test_agent"])
    mock.agent_service.assign_tool_for_agent = MagicMock()  # Ensure this is mockable

    # Add mock knowledge_base service
    mock.knowledge_base = AsyncMock()

    return mock


class TestSolanaAgent:
    """Test suite for SolanaAgent client."""

    @patch("solana_agent.client.solana_agent.SolanaAgentFactory")
    def test_init_with_config(self, mock_factory, config_dict, mock_query_service):
        """Test initialization with configuration dictionary."""
        mock_factory.create_from_config.return_value = mock_query_service
        agent = SolanaAgent(config=config_dict)
        mock_factory.create_from_config.assert_called_once_with(config_dict)
        assert agent.query_service == mock_query_service

    # Test init with config_path would require mocking open/json/importlib

    def test_init_without_config(self):
        """Test initialization fails without configuration."""
        with pytest.raises(ValueError, match="Either config or config_path must be provided"):
            SolanaAgent()

    @pytest.mark.asyncio
    async def test_get_user_history(self, config_dict, mock_query_service):
        """Test retrieving user message history."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            expected = {"messages": [], "total": 0}
            mock_query_service.get_user_history.return_value = expected

            result = await agent.get_user_history(
                user_id="test_user",
                page_num=1,
                page_size=20,
                sort_order="desc"
            )

            assert result == expected
            mock_query_service.get_user_history.assert_called_once_with(
                "test_user", 1, 20, "desc"
            )

    @pytest.mark.asyncio
    async def test_delete_user_history(self, config_dict, mock_query_service):
        """Test deleting user message history."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            await agent.delete_user_history("test_user")
            mock_query_service.delete_user_history.assert_called_once_with(
                "test_user")

    def test_register_tool_success(self, config_dict, mock_query_service):
        """Test successful tool registration."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            # Setup mock tool and registry
            mock_tool = MagicMock(spec=Tool)
            mock_tool.name = "test_tool"
            mock_query_service.agent_service.tool_registry.register_tool.return_value = True

            # Test registration
            result = agent.register_tool("test_agent", mock_tool)

            # Verify results
            assert result is True
            mock_query_service.agent_service.tool_registry.register_tool.assert_called_once_with(
                mock_tool)
            mock_query_service.agent_service.assign_tool_for_agent.assert_called_once_with(
                "test_agent", "test_tool")

    def test_register_tool_failure(self, config_dict, mock_query_service):
        """Test failed tool registration."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            # Setup mock tool and registry
            mock_tool = MagicMock(spec=Tool)
            mock_tool.name = "test_tool"
            mock_query_service.agent_service.tool_registry.register_tool.return_value = False

            # Test registration
            result = agent.register_tool("test_agent", mock_tool)

            # Verify results
            assert result is False
            mock_query_service.agent_service.tool_registry.register_tool.assert_called_once_with(
                mock_tool)
            # Verify assign_tool_for_agent was not called
            mock_query_service.agent_service.assign_tool_for_agent.assert_not_called()

    # --- Knowledge Base Method Tests ---

    @pytest.mark.asyncio
    async def test_kb_add_document(self, config_dict, mock_query_service):
        """Test adding a document via the KB client method."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            doc_text = "Test document content."
            doc_meta = {"source": "test"}
            doc_id = "test-doc-kb-1"
            namespace = "kb_ns_1"

            # Mock the underlying service call
            mock_query_service.knowledge_base.add_document.return_value = doc_id

            result = await agent.kb_add_document(
                text=doc_text,
                metadata=doc_meta,
                document_id=doc_id,
                namespace=namespace
            )

            assert result == doc_id
            mock_query_service.knowledge_base.add_document.assert_called_once_with(
                doc_text, doc_meta, doc_id, namespace
            )

    @pytest.mark.asyncio
    async def test_kb_query(self, config_dict, mock_query_service):
        """Test querying the KB via the client method."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            query = "What is Solana?"
            kb_filter = {"tags": "blockchain"}
            top_k = 3
            namespace = "kb_ns_query"
            expected_results = [{"document_id": "doc1", "score": 0.9}]

            # Mock the underlying service call
            mock_query_service.knowledge_base.query.return_value = expected_results

            results = await agent.kb_query(
                query_text=query,
                filter=kb_filter,
                top_k=top_k,
                namespace=namespace,
                include_content=False,
                include_metadata=True
            )

            assert results == expected_results
            mock_query_service.knowledge_base.query.assert_called_once_with(
                query, kb_filter, top_k, namespace, False, True
            )

    @pytest.mark.asyncio
    async def test_kb_delete_document(self, config_dict, mock_query_service):
        """Test deleting a document via the KB client method."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            doc_id = "doc-to-delete-kb"
            namespace = "kb_ns_delete"

            # Mock the underlying service call
            mock_query_service.knowledge_base.delete_document.return_value = True

            result = await agent.kb_delete_document(
                document_id=doc_id,
                namespace=namespace
            )

            assert result is True
            mock_query_service.knowledge_base.delete_document.assert_called_once_with(
                doc_id, namespace
            )

    @pytest.mark.asyncio
    async def test_kb_update_document(self, config_dict, mock_query_service):
        """Test updating a document via the KB client method."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            doc_id = "doc-to-update-kb"
            new_text = "Updated text."
            new_meta = {"source": "updated"}
            namespace = "kb_ns_update"

            # Mock the underlying service call
            mock_query_service.knowledge_base.update_document.return_value = True

            result = await agent.kb_update_document(
                document_id=doc_id,
                text=new_text,
                metadata=new_meta,
                namespace=namespace
            )

            assert result is True
            mock_query_service.knowledge_base.update_document.assert_called_once_with(
                doc_id, new_text, new_meta, namespace
            )

    @pytest.mark.asyncio
    async def test_kb_add_documents_batch(self, config_dict, mock_query_service):
        """Test adding documents in batch via the KB client method."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            docs = [{"text": "batch1", "metadata": {"id": "b1"}},
                    {"text": "batch2", "metadata": {"id": "b2"}}]
            namespace = "kb_ns_batch"
            batch_size = 10
            expected_ids = ["b1", "b2"]

            # Mock the underlying service call
            mock_query_service.knowledge_base.add_documents_batch.return_value = expected_ids

            result_ids = await agent.kb_add_documents_batch(
                documents=docs,
                namespace=namespace,
                batch_size=batch_size
            )

            assert result_ids == expected_ids
            mock_query_service.knowledge_base.add_documents_batch.assert_called_once_with(
                docs, namespace, batch_size
            )

    @pytest.mark.asyncio
    async def test_kb_add_pdf_document_bytes(self, config_dict, mock_query_service):
        """Test adding a PDF document from bytes via the KB client method."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            pdf_bytes = b"%PDF-1.4..."  # Dummy PDF bytes
            pdf_meta = {"title": "Test PDF"}
            pdf_id = "pdf-kb-1"
            namespace = "kb_ns_pdf"
            chunk_size = 25

            # Mock the underlying service call
            mock_query_service.knowledge_base.add_pdf_document.return_value = pdf_id

            result_id = await agent.kb_add_pdf_document(
                pdf_data=pdf_bytes,
                metadata=pdf_meta,
                document_id=pdf_id,
                namespace=namespace,
                chunk_batch_size=chunk_size
            )

            assert result_id == pdf_id
            mock_query_service.knowledge_base.add_pdf_document.assert_called_once_with(
                pdf_bytes, pdf_meta, pdf_id, namespace, chunk_size
            )

    @pytest.mark.asyncio
    async def test_kb_add_pdf_document_path(self, config_dict, mock_query_service):
        """Test adding a PDF document from path via the KB client method."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            pdf_path = "/fake/path/doc.pdf"
            pdf_meta = {"title": "Test PDF Path"}
            pdf_id = "pdf-kb-path-1"
            namespace = "kb_ns_pdf_path"
            chunk_size = 30

            # Mock the underlying service call
            mock_query_service.knowledge_base.add_pdf_document.return_value = pdf_id

            result_id = await agent.kb_add_pdf_document(
                pdf_data=pdf_path,
                metadata=pdf_meta,
                document_id=pdf_id,
                namespace=namespace,
                chunk_batch_size=chunk_size
            )

            assert result_id == pdf_id
            mock_query_service.knowledge_base.add_pdf_document.assert_called_once_with(
                pdf_path, pdf_meta, pdf_id, namespace, chunk_size
            )

    @pytest.mark.asyncio
    async def test_kb_add_pdf_document_invalid_type(self, config_dict, mock_query_service):
        """Test adding a PDF document with invalid data type raises TypeError."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            with pytest.raises(TypeError, match="pdf_data must be bytes or a file path string."):
                await agent.kb_add_pdf_document(
                    pdf_data=12345,  # Invalid type
                    metadata={"title": "Invalid PDF"}
                )
            mock_query_service.knowledge_base.add_pdf_document.assert_not_called()

    # --- Test _ensure_kb ---
    def test_ensure_kb_missing(self, config_dict, mock_query_service):
        """Test _ensure_kb raises AttributeError if knowledge_base service is missing."""
        # Remove the knowledge_base mock for this test
        del mock_query_service.knowledge_base

        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            with pytest.raises(AttributeError, match="Knowledge base service not configured or available."):
                agent._ensure_kb()  # Call the internal method directly for testing

    def test_ensure_kb_present(self, config_dict, mock_query_service):
        """Test _ensure_kb returns the knowledge_base service if present."""
        with patch("solana_agent.client.solana_agent.SolanaAgentFactory") as mock_factory:
            mock_factory.create_from_config.return_value = mock_query_service
            agent = SolanaAgent(config=config_dict)

            kb_service = agent._ensure_kb()
            assert kb_service == mock_query_service.knowledge_base

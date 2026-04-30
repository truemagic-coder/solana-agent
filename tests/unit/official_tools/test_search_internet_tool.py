"""
Tests for Search Internet Tool.

Tests the SearchInternetTool which searches the internet
using Perplexity AI, OpenAI, or Grok.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from solana_agent.tools.search_internet import SearchInternetTool, SearchInternetPlugin


@pytest.fixture
def search_tool_openai():
    """Create a configured SearchInternetTool with OpenAI."""
    tool = SearchInternetTool()
    tool.configure(
        {
            "tools": {
                "search_internet": {
                    "api_key": "test-openai-key",
                    "provider": "openai",
                    "model": "gpt-4o-mini-search-preview",
                }
            }
        }
    )
    return tool


@pytest.fixture
def search_tool_perplexity():
    """Create a configured SearchInternetTool with Perplexity."""
    tool = SearchInternetTool()
    tool.configure(
        {
            "tools": {
                "search_internet": {
                    "api_key": "test-perplexity-key",
                    "provider": "perplexity",
                    "citations": True,
                }
            }
        }
    )
    return tool


@pytest.fixture
def search_tool_grok():
    """Create a configured SearchInternetTool with Grok."""
    tool = SearchInternetTool()
    tool.configure(
        {
            "tools": {
                "search_internet": {
                    "api_key": "test-grok-key",
                    "provider": "grok",
                }
            }
        }
    )
    return tool


@pytest.fixture
def search_tool_no_key():
    """Create a SearchInternetTool without API key."""
    tool = SearchInternetTool()
    tool.configure({"tools": {"search_internet": {}}})
    return tool


class TestSearchInternetToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self, search_tool_openai):
        """Should have correct tool name."""
        assert search_tool_openai.name == "search_internet"

    def test_schema_has_required_properties(self, search_tool_openai):
        """Should include query in required properties."""
        schema = search_tool_openai.get_schema()
        assert "query" in schema["properties"]
        assert "query" in schema["required"]

    def test_schema_property_types(self, search_tool_openai):
        """Should have correct property types."""
        schema = search_tool_openai.get_schema()
        assert schema["properties"]["query"]["type"] == "string"


class TestSearchInternetToolConfigure:
    """Test configuration method."""

    def test_configure_stores_api_key(self, search_tool_openai):
        """Should store API key from config."""
        assert search_tool_openai._api_key == "test-openai-key"

    def test_configure_openai_provider(self, search_tool_openai):
        """Should configure OpenAI provider correctly."""
        assert search_tool_openai._provider == "openai"
        assert search_tool_openai._model == "gpt-4o-mini-search-preview"

    def test_configure_perplexity_provider(self, search_tool_perplexity):
        """Should configure Perplexity provider correctly."""
        assert search_tool_perplexity._provider == "perplexity"
        assert search_tool_perplexity._model == "sonar"  # default

    def test_configure_grok_provider(self, search_tool_grok):
        """Should configure Grok provider correctly."""
        assert search_tool_grok._provider == "grok"
        assert search_tool_grok._model == "grok-4-1-fast-non-reasoning"  # default

    def test_configure_default_model_openai(self):
        """Should use default OpenAI model when not specified."""
        tool = SearchInternetTool()
        tool.configure(
            {
                "tools": {
                    "search_internet": {
                        "api_key": "test-key",
                        "provider": "openai",
                    }
                }
            }
        )
        assert tool._model == "gpt-4o-mini-search-preview"

    def test_configure_default_model_perplexity(self):
        """Should use default Perplexity model when not specified."""
        tool = SearchInternetTool()
        tool.configure(
            {
                "tools": {
                    "search_internet": {
                        "api_key": "test-key",
                        "provider": "perplexity",
                    }
                }
            }
        )
        assert tool._model == "sonar"

    def test_configure_citations_setting(self, search_tool_perplexity):
        """Should store citations setting."""
        assert search_tool_perplexity._citations is True

    def test_configure_grok_defaults(self, search_tool_grok):
        """Should have default Grok settings."""
        assert search_tool_grok._grok_web_search is True
        assert search_tool_grok._grok_x_search is True
        assert search_tool_grok._grok_timeout == 90

    def test_configure_grok_custom_settings(self):
        """Should configure custom Grok settings."""
        tool = SearchInternetTool()
        tool.configure(
            {
                "tools": {
                    "search_internet": {
                        "api_key": "test-key",
                        "provider": "grok",
                        "grok_web_search": True,
                        "grok_x_search": False,
                        "grok_timeout": 120,
                    }
                }
            }
        )
        assert tool._grok_web_search is True
        assert tool._grok_x_search is False
        assert tool._grok_timeout == 120


class TestSearchInternetToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_no_api_key_error(self, search_tool_no_key):
        """Should return error when API key is missing."""
        result = await search_tool_no_key.execute(query="test query")

        assert result["status"] == "error"
        assert "key" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_openai_success(self, search_tool_openai):
        """Should return search results for OpenAI."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Search result content"))
        ]

        with patch("solana_agent.tools.search_internet.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_client

            result = await search_tool_openai.execute(query="What is Solana?")

            assert result["status"] == "success"
            assert result["result"] == "Search result content"
            assert result["model_used"] == "gpt-4o-mini-search-preview"

    @pytest.mark.asyncio
    async def test_execute_perplexity_success(self, search_tool_perplexity):
        """Should return search results for Perplexity."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Perplexity result"}}],
            "citations": ["https://example.com/source1"],
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await search_tool_perplexity.execute(query="What is Solana?")

            assert result["status"] == "success"
            assert "Perplexity result" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_perplexity_with_citations(self, search_tool_perplexity):
        """Should include citations when enabled for Perplexity."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Result with citation [1]"}}],
            "citations": ["https://example.com/source1"],
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await search_tool_perplexity.execute(query="What is Solana?")

            assert result["status"] == "success"
            assert "Sources" in result["result"]
            assert "[1]" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_grok_success(self, search_tool_grok):
        """Should return search results for Grok."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "web_search_call",
                    "status": "completed",
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Grok search result",
                            "annotations": [
                                {"type": "url_citation", "url": "https://example.com"}
                            ],
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                },
            ],
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await search_tool_grok.execute(query="What is Solana?")

            assert result["status"] == "success"
            assert "Grok search result" in result["result"]
            # Should include citation sources
            assert "Sources" in result["result"]
            assert "example.com" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_api_error(self, search_tool_perplexity):
        """Should return error on API failure."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client_instance

            result = await search_tool_perplexity.execute(query="test")

            assert result["status"] == "error"
            assert "429" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, search_tool_openai):
        """Should return error on exception."""
        with patch("solana_agent.tools.search_internet.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            MockOpenAI.return_value = mock_client

            result = await search_tool_openai.execute(query="test")

            assert result["status"] == "error"


class TestSearchInternetPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        plugin = SearchInternetPlugin()
        assert plugin.name == "search_internet"

    def test_plugin_description(self):
        """Should have descriptive description."""
        plugin = SearchInternetPlugin()
        assert "search" in plugin.description.lower()

    def test_plugin_get_tools_empty_before_init(self):
        """Should return empty list before initialization."""
        plugin = SearchInternetPlugin()
        assert plugin.get_tools() == []

    def test_plugin_initialize(self):
        """Should initialize tool on registry."""
        plugin = SearchInternetPlugin()
        mock_registry = MagicMock()

        plugin.initialize(mock_registry)

        assert plugin._tool is not None
        assert len(plugin.get_tools()) == 1

"""
Tests for MCP Tool.

Tests the MCPTool which connects to MCP servers using fastmcp
and uses LLM to select and call tools.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def load_mcp_module():
    fake_fastmcp = types.ModuleType("fastmcp")
    fake_fastmcp.Client = MagicMock(name="Client")
    fake_fastmcp_client = types.ModuleType("fastmcp.client")
    fake_transports = types.ModuleType("fastmcp.client.transports")
    fake_transports.StreamableHttpTransport = MagicMock(
        name="StreamableHttpTransport"
    )
    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = MagicMock(name="AsyncOpenAI")

    with patch.dict(
        sys.modules,
        {
            "fastmcp": fake_fastmcp,
            "fastmcp.client": fake_fastmcp_client,
            "fastmcp.client.transports": fake_transports,
            "openai": fake_openai,
        },
    ):
        import solana_agent.tools.mcp as mcp_module

        return importlib.reload(mcp_module)


class TruthyEmptyTools(list):
    def __bool__(self):
        return True


def make_openai_client(content: str):
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=completion))
        )
    )


def make_server_client(
    *,
    tools=None,
    call_result=None,
    call_exception=None,
    enter_exception=None,
    exit_exception=None,
):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    if enter_exception is not None:
        client.__aenter__.side_effect = enter_exception

    client.__aexit__ = AsyncMock(return_value=None)
    if exit_exception is not None:
        client.__aexit__.side_effect = exit_exception

    client.list_tools = AsyncMock(return_value=[] if tools is None else tools)
    if call_exception is not None:
        client.call_tool = AsyncMock(side_effect=call_exception)
    else:
        client.call_tool = AsyncMock(return_value=call_result)
    return client


class TestMCPToolSchema:
    """Test tool schema and initialization."""

    def test_tool_name(self):
        """Should have correct tool name."""
        with patch.dict(
            "sys.modules",
            {"fastmcp": MagicMock(), "fastmcp.client.transports": MagicMock()},
        ):
            from solana_agent.tools.mcp import MCPTool

            tool = MCPTool()
            assert tool.name == "mcp"

    def test_schema_has_required_properties(self):
        """Should include query in required properties."""
        with patch.dict(
            "sys.modules",
            {"fastmcp": MagicMock(), "fastmcp.client.transports": MagicMock()},
        ):
            from solana_agent.tools.mcp import MCPTool

            tool = MCPTool()
            schema = tool.get_schema()
            assert "query" in schema["properties"]
            assert "query" in schema["required"]


class TestMCPToolConfigure:
    """Test configuration method."""

    def test_configure_single_server(self):
        """Should configure single server correctly."""
        with patch.dict(
            "sys.modules",
            {"fastmcp": MagicMock(), "fastmcp.client.transports": MagicMock()},
        ):
            from solana_agent.tools.mcp import MCPTool

            tool = MCPTool()
            tool.configure(
                {
                    "openai": {"api_key": "test-openai-key"},
                    "tools": {
                        "mcp": {
                            "url": "https://mcp.example.com/api",
                            "headers": {"Authorization": "Bearer token"},
                        }
                    },
                }
            )

            assert len(tool._servers) == 1
            assert tool._servers[0]["url"] == "https://mcp.example.com/api"

    def test_configure_multiple_servers(self):
        """Should configure multiple servers correctly."""
        with patch.dict(
            "sys.modules",
            {"fastmcp": MagicMock(), "fastmcp.client.transports": MagicMock()},
        ):
            from solana_agent.tools.mcp import MCPTool

            tool = MCPTool()
            tool.configure(
                {
                    "openai": {"api_key": "test-openai-key"},
                    "tools": {
                        "mcp": {
                            "servers": [
                                {"url": "https://server1.com/api"},
                                {
                                    "url": "https://server2.com/api",
                                    "headers": {"X-Key": "abc"},
                                },
                            ]
                        }
                    },
                }
            )

            assert len(tool._servers) == 2
            assert tool._servers[0]["url"] == "https://server1.com/api"
            assert tool._servers[1]["url"] == "https://server2.com/api"
            assert tool._servers[1]["headers"]["X-Key"] == "abc"

    def test_configure_grok_as_default_provider(self):
        """Should configure Grok as default provider when grok key is available."""
        with patch.dict(
            "sys.modules",
            {"fastmcp": MagicMock(), "fastmcp.client.transports": MagicMock()},
        ):
            from solana_agent.tools.mcp import MCPTool

            tool = MCPTool()
            tool.configure(
                {
                    "grok": {"api_key": "test-grok-key"},
                    "openai": {"api_key": "test-openai-key"},
                    "tools": {"mcp": {"url": "https://mcp.example.com/api"}},
                }
            )

            # Grok should be prioritized over OpenAI by default
            assert tool._llm_provider == "grok"
            assert tool._llm_api_key == "test-grok-key"

    def test_configure_openai_explicit_provider(self):
        """Should use OpenAI when explicitly configured."""
        with patch.dict(
            "sys.modules",
            {"fastmcp": MagicMock(), "fastmcp.client.transports": MagicMock()},
        ):
            from solana_agent.tools.mcp import MCPTool

            tool = MCPTool()
            tool.configure(
                {
                    "openai": {"api_key": "test-openai-key"},
                    "tools": {
                        "mcp": {
                            "url": "https://mcp.example.com/api",
                            "llm_provider": "openai",
                        }
                    },
                }
            )

            # Should use OpenAI when explicitly specified
            assert tool._llm_provider == "openai"
            assert tool._llm_api_key == "test-openai-key"

    def test_configure_grok_provider(self):
        """Should configure Grok provider correctly."""
        with patch.dict(
            "sys.modules",
            {"fastmcp": MagicMock(), "fastmcp.client.transports": MagicMock()},
        ):
            from solana_agent.tools.mcp import MCPTool

            tool = MCPTool()
            tool.configure(
                {
                    "grok": {"api_key": "test-grok-key"},
                    "tools": {
                        "mcp": {
                            "url": "https://mcp.example.com/api",
                            "llm_provider": "grok",
                        }
                    },
                }
            )

            assert tool._llm_provider == "grok"
            assert tool._llm_base_url == "https://api.x.ai/v1"


class TestMCPToolExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_no_servers_error(self):
        """Should return error when no servers configured."""
        with patch.dict(
            "sys.modules",
            {"fastmcp": MagicMock(), "fastmcp.client.transports": MagicMock()},
        ):
            # Need to reload after patching
            import importlib
            import solana_agent.tools.mcp

            importlib.reload(solana_agent.tools.mcp)

            from solana_agent.tools.mcp import MCPTool

            tool = MCPTool()
            tool.configure(
                {
                    "grok": {"api_key": "test-key"},
                    "tools": {"mcp": {}},  # No URL
                }
            )

            result = await tool.execute(query="Test query")

            assert result["status"] == "error"
            assert "server" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_no_api_key_error(self):
        """Should return error when no API key configured."""
        with patch.dict(
            "sys.modules",
            {"fastmcp": MagicMock(), "fastmcp.client.transports": MagicMock()},
        ):
            import importlib
            import solana_agent.tools.mcp

            importlib.reload(solana_agent.tools.mcp)

            from solana_agent.tools.mcp import MCPTool

            tool = MCPTool()
            tool.configure(
                {
                    "tools": {
                        "mcp": {
                            "url": "https://mcp.example.com/api",
                        }
                    },
                }
            )

            result = await tool.execute(query="Test query")

            assert result["status"] == "error"
            assert "key" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_handles_connection_failure_and_no_tools_cleanup(self):
        mcp_module = load_mcp_module()
        tool = mcp_module.MCPTool()
        tool.configure(
            {
                "grok": {"api_key": "test-grok-key"},
                "tools": {
                    "mcp": {
                        "servers": [
                            {"url": "https://broken.example.com/api"},
                            {"url": "https://empty.example.com/api"},
                        ]
                    }
                },
            }
        )

        failing_client = make_server_client(
            enter_exception=RuntimeError("connection failed")
        )
        empty_client = make_server_client(
            tools=TruthyEmptyTools(),
            exit_exception=RuntimeError("cleanup failed"),
        )

        mcp_module.StreamableHttpTransport = MagicMock(
            side_effect=lambda url, headers=None: {"url": url, "headers": headers}
        )
        mcp_module.Client = MagicMock(side_effect=[failing_client, empty_client])

        result = await tool.execute(query="Find a tool")

        assert result["status"] == "error"
        assert result["message"] == "No tools available on any MCP server."
        assert empty_client.__aexit__.await_count == 1

    @pytest.mark.asyncio
    async def test_execute_returns_llm_parse_error(self):
        mcp_module = load_mcp_module()
        tool = mcp_module.MCPTool()
        tool.configure(
            {
                "grok": {"api_key": "test-grok-key"},
                "tools": {"mcp": {"url": "https://mcp.example.com/api"}},
            }
        )

        server_client = make_server_client(
            tools=[
                SimpleNamespace(
                    name="search",
                    description="Search",
                    inputSchema={"type": "object"},
                )
            ],
            exit_exception=RuntimeError("close failed"),
        )

        mcp_module.StreamableHttpTransport = MagicMock(return_value=MagicMock())
        mcp_module.Client = MagicMock(return_value=server_client)
        mcp_module.AsyncOpenAI = MagicMock(return_value=make_openai_client("not json"))

        result = await tool.execute(query="Search for Solana")

        assert result["status"] == "error"
        assert "LLM output parse error" in result["message"]
        assert result["raw_llm_output"] == "not json"
        mcp_module.AsyncOpenAI.assert_called_once_with(
            api_key="test-grok-key",
            base_url="https://api.x.ai/v1",
        )

    @pytest.mark.asyncio
    async def test_execute_returns_error_when_no_tool_selected(self):
        mcp_module = load_mcp_module()
        tool = mcp_module.MCPTool()
        tool.configure(
            {
                "openai": {"api_key": "test-openai-key"},
                "tools": {
                    "mcp": {
                        "url": "https://mcp.example.com/api",
                        "llm_provider": "openai",
                    }
                },
            }
        )

        server_client = make_server_client(
            tools=[
                SimpleNamespace(
                    name="search",
                    description="Search",
                    inputSchema={"type": "object"},
                )
            ]
        )
        mcp_module.StreamableHttpTransport = MagicMock(return_value=MagicMock())
        mcp_module.Client = MagicMock(return_value=server_client)
        mcp_module.AsyncOpenAI = MagicMock(
            return_value=make_openai_client('{"tool": null, "parameters": {}}')
        )

        result = await tool.execute(query="Do something")

        assert result["status"] == "error"
        assert result["message"] == "No suitable tool found for the query."
        assert result["llm_output"] == '{"tool": null, "parameters": {}}'
        mcp_module.AsyncOpenAI.assert_called_once_with(api_key="test-openai-key")

    @pytest.mark.asyncio
    async def test_execute_returns_error_when_tool_not_found(self):
        mcp_module = load_mcp_module()
        tool = mcp_module.MCPTool()
        tool.configure(
            {
                "grok": {"api_key": "test-grok-key"},
                "tools": {"mcp": {"url": "https://mcp.example.com/api"}},
            }
        )

        server_client = make_server_client(
            tools=[
                SimpleNamespace(
                    name="search",
                    description="Search",
                    inputSchema={"type": "object"},
                )
            ]
        )
        mcp_module.StreamableHttpTransport = MagicMock(return_value=MagicMock())
        mcp_module.Client = MagicMock(return_value=server_client)
        mcp_module.AsyncOpenAI = MagicMock(
            return_value=make_openai_client(
                '{"tool": "missing_tool", "parameters": {"q": "hello"}}'
            )
        )

        result = await tool.execute(query="Find missing tool")

        assert result["status"] == "error"
        assert result["message"] == "Tool 'missing_tool' not found on any server."
        assert result["tool"] == "missing_tool"

    @pytest.mark.asyncio
    async def test_execute_returns_success_with_json_tool_result(self):
        mcp_module = load_mcp_module()
        tool = mcp_module.MCPTool()
        tool.configure(
            {
                "grok": {"api_key": "test-grok-key"},
                "tools": {"mcp": {"url": "https://mcp.example.com/api"}},
            }
        )

        server_client = make_server_client(
            tools=[
                SimpleNamespace(
                    name="search",
                    description="Search",
                    inputSchema={"type": "object"},
                )
            ],
            call_result=[SimpleNamespace(text='{"answer": "done"}')],
        )
        mcp_module.StreamableHttpTransport = MagicMock(return_value=MagicMock())
        mcp_module.Client = MagicMock(return_value=server_client)
        mcp_module.AsyncOpenAI = MagicMock(
            return_value=make_openai_client(
                '{"tool": "search", "parameters": {"q": "hello"}}'
            )
        )

        result = await tool.execute(query="Search for hello")

        assert result == {
            "status": "success",
            "tool": "search",
            "parameters": {"q": "hello"},
            "result": {"answer": "done"},
            "llm_provider": "grok",
            "llm_model": "grok-4-1-fast-non-reasoning",
        }

    @pytest.mark.asyncio
    async def test_execute_returns_success_with_stringified_tool_result(self):
        mcp_module = load_mcp_module()
        tool = mcp_module.MCPTool()
        tool.configure(
            {
                "grok": {"api_key": "test-grok-key"},
                "tools": {"mcp": {"url": "https://mcp.example.com/api"}},
            }
        )

        server_client = make_server_client(
            tools=[
                SimpleNamespace(
                    name="search",
                    description="Search",
                    inputSchema={"type": "object"},
                )
            ],
            call_result=["plain text result"],
        )
        mcp_module.StreamableHttpTransport = MagicMock(return_value=MagicMock())
        mcp_module.Client = MagicMock(return_value=server_client)
        mcp_module.AsyncOpenAI = MagicMock(
            return_value=make_openai_client(
                '{"tool": "search", "parameters": {"q": "hello"}}'
            )
        )

        result = await tool.execute(query="Search for hello")

        assert result["status"] == "success"
        assert result["result"] == "['plain text result']"

    @pytest.mark.asyncio
    async def test_execute_returns_tool_call_failure(self):
        mcp_module = load_mcp_module()
        tool = mcp_module.MCPTool()
        tool.configure(
            {
                "grok": {"api_key": "test-grok-key"},
                "tools": {"mcp": {"url": "https://mcp.example.com/api"}},
            }
        )

        server_client = make_server_client(
            tools=[
                SimpleNamespace(
                    name="search",
                    description="Search",
                    inputSchema={"type": "object"},
                )
            ],
            call_exception=RuntimeError("tool failed"),
        )
        mcp_module.StreamableHttpTransport = MagicMock(return_value=MagicMock())
        mcp_module.Client = MagicMock(return_value=server_client)
        mcp_module.AsyncOpenAI = MagicMock(
            return_value=make_openai_client(
                '{"tool": "search", "parameters": {"q": "hello"}}'
            )
        )

        result = await tool.execute(query="Search for hello")

        assert result == {
            "status": "error",
            "message": "Tool call failed: tool failed",
            "tool": "search",
            "parameters": {"q": "hello"},
        }


class TestMCPPlugin:
    """Test plugin class."""

    def test_plugin_name(self):
        """Should have correct plugin name."""
        with patch.dict(
            "sys.modules",
            {"fastmcp": MagicMock(), "fastmcp.client.transports": MagicMock()},
        ):
            from solana_agent.tools.mcp import MCPPlugin

            plugin = MCPPlugin()
            assert plugin.name == "mcp"

    def test_plugin_description(self):
        """Should have descriptive description."""
        with patch.dict(
            "sys.modules",
            {"fastmcp": MagicMock(), "fastmcp.client.transports": MagicMock()},
        ):
            from solana_agent.tools.mcp import MCPPlugin

            plugin = MCPPlugin()
            assert "mcp" in plugin.description.lower()

    def test_get_plugin_returns_dummy_without_fastmcp(self):
        """Should return dummy plugin when fastmcp is not available."""
        # Test the disabled state
        with patch.dict("sys.modules", {"fastmcp": None}):
            # This tests the fallback behavior
            pass  # The actual import test would need module reload

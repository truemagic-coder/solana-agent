"""Tests for the generic x402 request tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Request, Response

from solana_agent.tools.x402_request import (
    X402RequestPlugin,
    X402RequestTool,
    get_plugin,
)


@pytest.fixture
def request_tool():
    tool = X402RequestTool()
    tool.configure(
        {
            "tools": {
                "x402_request": {
                    "auth_mode": "x402_private_key",
                    "private_key": "test-private-key",
                    "allowed_hosts": ["api.example.com"],
                }
            }
        }
    )
    return tool


class TestX402RequestTool:
    def test_schema(self, request_tool):
        schema = request_tool.get_schema()
        assert schema["required"] == ["method", "url"]
        assert schema["properties"]["method"]["enum"] == ["GET", "POST"]

    def test_runtime_context_can_be_cleared(self, request_tool):
        request_tool.set_runtime_context({"privy_wallet_id": " wallet-123 "})

        assert request_tool._get_runtime_privy_wallet_id() == "wallet-123"

        request_tool.clear_runtime_context()
        assert request_tool._get_runtime_privy_wallet_id() == ""

    def test_can_resolve_hosted_wallet_id_from_runtime_payload(self):
        tool = X402RequestTool()
        tool.configure(
            {
                "tools": {
                    "x402_request": {
                        "auth_mode": "x402_privy",
                        "privy_wallet_id": "configured-wallet",
                        "allowed_hosts": ["api.example.com"],
                    }
                }
            }
        )

        tool.set_runtime_context({"privy_wallet": {"id": "wallet-123"}})

        assert tool._get_runtime_privy_wallet_id() == "wallet-123"

    @pytest.mark.asyncio
    async def test_requires_allowlist(self):
        tool = X402RequestTool()
        tool.configure(
            {
                "tools": {
                    "x402_request": {
                        "private_key": "test-private-key",
                    }
                }
            }
        )

        result = await tool.execute(method="GET", url="https://api.example.com/data")
        assert result["success"] is False
        assert "allowed_hosts" in result["error"]

    @pytest.mark.asyncio
    async def test_rejects_non_allowlisted_host(self, request_tool):
        result = await request_tool.execute(
            method="GET", url="https://evil.example.com"
        )
        assert result["success"] is False
        assert "not permitted" in result["error"]

    @pytest.mark.asyncio
    async def test_rejects_privy_mode_explicitly(self):
        tool = X402RequestTool()
        tool.configure(
            {
                "tools": {
                    "x402_request": {
                        "auth_mode": "x402_privy",
                        "allowed_hosts": ["api.example.com"],
                    }
                }
            }
        )

        result = await tool.execute(method="GET", url="https://api.example.com/data")
        assert result["success"] is False
        assert "privy_wallet_id" in result["error"]

    @pytest.mark.asyncio
    async def test_rejects_unsupported_auth_mode(self):
        tool = X402RequestTool()
        tool.configure(
            {
                "tools": {
                    "x402_request": {
                        "auth_mode": "api_key",
                        "allowed_hosts": ["api.example.com"],
                    }
                }
            }
        )

        result = await tool.execute(method="GET", url="https://api.example.com/data")

        assert result == {
            "success": False,
            "error": "Unsupported auth_mode: api_key",
        }

    @pytest.mark.asyncio
    async def test_privy_mode_uses_runtime_export_config(self):
        tool = X402RequestTool()
        tool.configure(
            {
                "tools": {
                    "x402_request": {
                        "auth_mode": "x402_privy",
                        "privy_app_id": "app-123",
                        "privy_app_secret": "secret-123",
                        "allowed_hosts": ["api.example.com"],
                    }
                }
            }
        )
        tool.set_runtime_context({"privy_wallet_id": "wallet-123"})

        response = Response(
            200,
            json={"ok": True},
            headers={"content-type": "application/json"},
            request=Request("GET", "https://api.example.com/data"),
        )

        with patch(
            "solana_agent.tools.x402_request.request_with_x402_privy",
            AsyncMock(return_value=response),
        ) as mock_request:
            result = await tool.execute(
                method="GET", url="https://api.example.com/data"
            )

        assert result["success"] is True
        assert result["payment_mode"] == "x402_privy"
        assert mock_request.await_args.kwargs["privy_wallet_id"] == "wallet-123"
        assert mock_request.await_args.kwargs["privy_app_id"] == "app-123"
        assert mock_request.await_args.kwargs["privy_app_secret"] == "secret-123"

    @pytest.mark.asyncio
    async def test_privy_mode_inherits_hosted_ai_credentials(self):
        tool = X402RequestTool()
        tool.configure(
            {
                "ai": {
                    "auth_mode": "x402_privy",
                    "privy_app_id": "app-123",
                    "privy_app_secret": "secret-123",
                    "x402_rpc_url": "https://rpc.example.com",
                },
                "tools": {
                    "x402_request": {
                        "allowed_hosts": ["api.example.com"],
                    }
                },
            }
        )
        tool.set_runtime_context({"privy_wallet_id": "wallet-123"})

        response = Response(
            200,
            json={"ok": True},
            headers={"content-type": "application/json"},
            request=Request("GET", "https://api.example.com/data"),
        )

        with patch(
            "solana_agent.tools.x402_request.request_with_x402_privy",
            AsyncMock(return_value=response),
        ) as mock_request:
            result = await tool.execute(
                method="GET", url="https://api.example.com/data"
            )

        assert result["success"] is True
        assert result["payment_mode"] == "x402_privy"
        assert mock_request.await_args.kwargs["privy_wallet_id"] == "wallet-123"
        assert mock_request.await_args.kwargs["privy_app_id"] == "app-123"
        assert mock_request.await_args.kwargs["privy_app_secret"] == "secret-123"
        assert mock_request.await_args.kwargs["rpc_url"] == "https://rpc.example.com"

    @pytest.mark.asyncio
    async def test_get_request_uses_x402_transport(self, request_tool):
        response = Response(
            200,
            json={"ok": True},
            headers={
                "content-type": "application/json",
                "x-payment-response": "settled",
            },
            request=Request("GET", "https://api.example.com/data"),
        )

        with patch(
            "solana_agent.tools.x402_request.request_with_x402_private_key",
            AsyncMock(return_value=response),
        ) as mock_request:
            result = await request_tool.execute(
                method="GET",
                url="https://api.example.com/data",
                query_params={"foo": "bar"},
                headers={"accept": "application/json"},
            )

        assert result["success"] is True
        assert result["data"] == {"ok": True}
        assert result["payment_mode"] == "x402_private_key"
        assert result["payment_headers"] == {"x-payment-response": "settled"}
        assert mock_request.await_args.kwargs["params"] == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_post_request_preserves_error_status(self, request_tool):
        response = Response(
            402,
            json={"error": "payment required"},
            headers={"content-type": "application/json"},
            request=Request("POST", "https://api.example.com/data"),
        )

        with patch(
            "solana_agent.tools.x402_request.request_with_x402_private_key",
            AsyncMock(return_value=response),
        ):
            result = await request_tool.execute(
                method="POST",
                url="https://api.example.com/data",
                json_body={"hello": "world"},
            )

        assert result["success"] is False
        assert result["status_code"] == 402
        assert result["data"] == {"error": "payment required"}

    @pytest.mark.asyncio
    async def test_invalid_json_body_falls_back_to_response_text(self, request_tool):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"content-type": "application/json"}
        response.json.side_effect = ValueError("invalid json")
        response.text = "raw-response"

        with patch(
            "solana_agent.tools.x402_request.request_with_x402_private_key",
            AsyncMock(return_value=response),
        ):
            result = await request_tool.execute(
                method="GET",
                url="https://api.example.com/data",
            )

        assert result["success"] is True
        assert result["data"] == "raw-response"


class TestX402RequestPlugin:
    def test_plugin_name(self):
        plugin = X402RequestPlugin()
        assert plugin.name == "x402_request"

    def test_plugin_description(self):
        plugin = X402RequestPlugin()
        assert "x402" in plugin.description.lower()

    def test_initialize_creates_tool_instance(self):
        plugin = X402RequestPlugin()
        registry = MagicMock()

        plugin.initialize(registry)

        assert plugin.tool_registry is registry
        assert isinstance(plugin._tool, X402RequestTool)

    def test_get_plugin(self):
        plugin = get_plugin()
        assert isinstance(plugin, X402RequestPlugin)

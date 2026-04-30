"""Focused tests for OpenAIAdapter auth-mode wiring."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from solana_agent.adapters.openai_adapter import OpenAIAdapter


class TestOpenAIAdapter:
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    @patch("solana_agent.adapters.openai_adapter.create_x402_httpx_client")
    def test_x402_private_key_uses_x402_http_client(
        self,
        mock_create_x402_httpx_client,
        mock_async_openai,
    ):
        """x402 private-key auth should swap in the paid transport client."""
        mock_http_client = MagicMock()
        mock_create_x402_httpx_client.return_value = mock_http_client

        OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="http://127.0.0.1:8000/v1",
            auth_mode="x402_private_key",
            private_key="test-private-key",
        )

        mock_create_x402_httpx_client.assert_called_once()
        x402_config = mock_create_x402_httpx_client.call_args.args[0]
        assert x402_config.private_key == "test-private-key"
        assert x402_config.rpc_url is None

        mock_async_openai.assert_called_once_with(
            api_key="x402",
            base_url="http://127.0.0.1:8000/v1",
            http_client=mock_http_client,
        )

    def test_x402_private_key_requires_signing_key(self):
        """x402 private-key auth should reject missing signing keys."""
        with pytest.raises(
            ValueError,
            match="x402_private_key requires a configured Solana signing key",
        ):
            OpenAIAdapter(
                api_key="x402",
                base_url="http://127.0.0.1:8000/v1",
                auth_mode="x402_private_key",
            )

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    @patch(
        "solana_agent.adapters.openai_adapter.create_x402_httpx_client_for_auth",
        new_callable=AsyncMock,
    )
    async def test_x402_privy_creates_runtime_wallet_client(
        self,
        mock_create_x402_httpx_client_for_auth,
        mock_async_openai,
    ):
        """x402 Privy auth should create a wallet-scoped client from runtime context."""
        mock_http_client = MagicMock()
        mock_create_x402_httpx_client_for_auth.return_value = mock_http_client

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="http://127.0.0.1:8000/v1",
            auth_mode="x402_privy",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
        )

        await adapter._get_client({"privy_wallet_id": "wallet-123"})

        mock_create_x402_httpx_client_for_auth.assert_awaited_once_with(
            auth_mode="x402_privy",
            private_key=None,
            privy_wallet_id="wallet-123",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
            privy_authorization_signature=None,
            privy_request_expiry=None,
            privy_api_url=None,
            rpc_url=None,
        )
        mock_async_openai.assert_called_once_with(
            api_key="x402",
            base_url="http://127.0.0.1:8000/v1",
            http_client=mock_http_client,
        )

    @pytest.mark.asyncio
    async def test_x402_privy_requires_runtime_wallet_id(self):
        """x402 Privy auth should require a runtime wallet identifier."""
        adapter = OpenAIAdapter(
            api_key="x402",
            base_url="http://127.0.0.1:8000/v1",
            auth_mode="x402_privy",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
        )

        with pytest.raises(
            ValueError,
            match="x402_privy requires runtime_context.privy_wallet_id for each request",
        ):
            await adapter._get_client({})

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_non_openai_chat_completions_send_budgeted_max_tokens(
        self,
        mock_async_openai,
    ):
        """Non-Responses chat requests should include a context-safe max_tokens."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )
        )
        mock_async_openai.return_value = mock_client

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            context_window_tokens=10,
            max_output_tokens=32,
        )

        result = await adapter.generate_text("hello")

        assert result is not None
        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert kwargs["max_tokens"] == 3
        assert kwargs["extra_headers"]["Idempotency-Key"]

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_hosted_chat_completions_forward_memory_extensions(
        self,
        mock_async_openai,
    ):
        """Hosted memory calls should forward conversation isolation hints."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )
        )
        mock_async_openai.return_value = mock_client

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            context_window_tokens=64,
            max_output_tokens=32,
        )

        result = await adapter.generate_text(
            "hello",
            runtime_context={
                "conversation_id": "conv-123",
                "memory_ttl_tier": "project",
            },
        )

        assert result is not None
        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert kwargs["extra_body"]["conversation_id"] == "conv-123"
        assert kwargs["extra_body"]["memory_ttl_tier"] == "project"

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_hosted_chat_completion_stream_forwards_memory_extensions(
        self,
        mock_async_openai,
    ):
        """Hosted streaming calls should send memory extensions in extra_body."""
        async def mock_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="ok", tool_calls=None),
                        finish_reason="stop",
                    )
                ]
            )

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_async_openai.return_value = mock_client

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            context_window_tokens=64,
            max_output_tokens=32,
        )

        responses = [
            event
            async for event in adapter.chat_stream(
                [{"role": "user", "content": "hello"}],
                runtime_context={
                    "conversation_id": "conv-123",
                    "memory_ttl_tier": "project",
                },
            )
        ]

        assert responses[-1]["type"] == "message_end"
        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert kwargs["extra_body"]["conversation_id"] == "conv-123"
        assert kwargs["extra_body"]["memory_ttl_tier"] == "project"

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_non_openai_chat_completions_fail_fast_when_prompt_fills_context(
        self,
        mock_async_openai,
    ):
        """The adapter should stop locally when no output budget remains."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        mock_async_openai.return_value = mock_client

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            context_window_tokens=7,
            max_output_tokens=32,
        )

        result = await adapter.generate_text("hello")

        assert result is None
        mock_client.chat.completions.create.assert_not_awaited()
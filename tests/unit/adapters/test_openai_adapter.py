"""Focused tests for OpenAIAdapter auth-mode wiring."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import based58
import httpx
import pytest
from openai import APIConnectionError
from solders.keypair import Keypair

from solana_agent.adapters.openai_adapter import OpenAIAdapter


class TestOpenAIAdapter:
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    @patch(
        "solana_agent.adapters.openai_adapter.create_hosted_managed_x402_httpx_client"
    )
    def test_hosted_managed_uses_local_x402_signer_when_configured(
        self,
        mock_create_hosted_managed_x402_httpx_client,
        mock_async_openai,
    ):
        """Hosted-managed auth should auto-enable x402 settlement from a local signer."""
        mock_http_client = MagicMock()
        mock_create_hosted_managed_x402_httpx_client.return_value = mock_http_client

        OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="hosted_managed",
            private_key="base58-private-key",
            x402_rpc_url="https://rpc.example",
        )

        config = mock_create_hosted_managed_x402_httpx_client.call_args.args[0]
        assert config.signing_key == "base58-private-key"
        assert config.rpc_url == "https://rpc.example"
        assert config.timeout == 180.0
        mock_async_openai.assert_called_once_with(
            api_key="x402",
            base_url="https://ai.solana-agent.com/v1",
            max_retries=0,
            http_client=mock_http_client,
        )

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    @patch(
        "solana_agent.adapters.openai_adapter.create_hosted_managed_x402_httpx_client"
    )
    async def test_hosted_managed_local_x402_settlement_keeps_idempotency_header(
        self,
        mock_create_hosted_managed_x402_httpx_client,
        mock_async_openai,
    ):
        """Local x402 settlement should still send the required idempotency header."""
        mock_create_hosted_managed_x402_httpx_client.return_value = MagicMock()
        mock_async_openai.return_value = MagicMock()
        private_key = based58.b58encode(bytes(Keypair())).decode("ascii")

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="hosted_managed",
            private_key=private_key,
        )

        options = await adapter._hosted_chat_completion_request_options()

        assert options["extra_headers"]["Idempotency-Key"]

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    @patch(
        "solana_agent.adapters.openai_adapter.create_hosted_managed_x402_httpx_client"
    )
    async def test_hosted_managed_uses_exported_hosted_wallet_signer_when_available(
        self,
        mock_create_hosted_managed_x402_httpx_client,
        mock_async_openai,
    ):
        mock_http_client = MagicMock()
        mock_create_hosted_managed_x402_httpx_client.return_value = mock_http_client
        wallet_client = MagicMock()
        mock_async_openai.return_value = wallet_client

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="hosted_managed",
            privy_user_id="did:privy:user123",
            x402_rpc_url="https://rpc.example",
        )
        adapter.export_wallet_private_key = AsyncMock(
            return_value={"private_key": "wallet-export-key"}
        )

        resolved_client = await adapter._get_client(
            {"hosted_privy_wallet_id": "wallet-123"}
        )

        assert resolved_client is wallet_client
        adapter.export_wallet_private_key.assert_awaited_once_with(
            privy_user_id="did:privy:user123",
            wallet_id="wallet-123",
        )
        config = mock_create_hosted_managed_x402_httpx_client.call_args.args[0]
        assert config.signing_key == "wallet-export-key"
        assert config.rpc_url == "https://rpc.example"
        assert config.timeout == 180.0
        mock_async_openai.assert_called_once_with(
            api_key="x402",
            base_url="https://ai.solana-agent.com/v1",
            max_retries=0,
            http_client=mock_http_client,
        )

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    @patch(
        "solana_agent.adapters.openai_adapter.create_hosted_managed_x402_httpx_client"
    )
    async def test_hosted_managed_reuses_wallet_scoped_x402_client(
        self,
        mock_create_hosted_managed_x402_httpx_client,
        mock_async_openai,
    ):
        mock_create_hosted_managed_x402_httpx_client.return_value = MagicMock()
        wallet_client = MagicMock()
        mock_async_openai.return_value = wallet_client

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="hosted_managed",
            privy_user_id="did:privy:user123",
        )
        adapter.export_wallet_private_key = AsyncMock(
            return_value={"private_key": "wallet-export-key"}
        )

        resolved_first = await adapter._get_client({"privy_wallet_id": "wallet-123"})
        resolved_second = await adapter._get_client(
            {"hosted_privy_wallet_id": "wallet-123"}
        )

        assert resolved_first is wallet_client
        assert resolved_second is wallet_client
        adapter.export_wallet_private_key.assert_awaited_once_with(
            privy_user_id="did:privy:user123",
            wallet_id="wallet-123",
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
            match="x402_privy requires a runtime wallet id",
        ):
            await adapter._get_client({})

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    @patch(
        "solana_agent.adapters.openai_adapter.create_x402_httpx_client_for_auth",
        new_callable=AsyncMock,
    )
    async def test_x402_privy_accepts_hosted_wallet_alias(
        self,
        mock_create_x402_httpx_client_for_auth,
        mock_async_openai,
    ):
        """x402 Privy auth should accept the hosted wallet alias from runtime context."""
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

        await adapter._get_client({"hosted_privy_wallet_id": "wallet-123"})

        mock_create_x402_httpx_client_for_auth.assert_awaited_once_with(
            auth_mode="x402_privy",
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
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_non_privy_client_recreates_when_event_loop_changes(
        self,
        mock_async_openai,
    ):
        """Cached async clients should be recreated when reused from a different loop."""
        first_client = MagicMock()
        first_client.close = AsyncMock(return_value=None)
        second_client = MagicMock()
        second_client.close = AsyncMock(return_value=None)
        mock_async_openai.side_effect = [first_client, second_client]

        adapter = OpenAIAdapter(api_key="test-key")

        loop_one = object()
        loop_two = object()
        with patch.object(
            adapter,
            "_current_running_loop",
            side_effect=[loop_one, loop_two],
        ):
            resolved_first = await adapter._get_client()
            resolved_second = await adapter._get_client()

        assert resolved_first is first_client
        assert resolved_second is second_client
        first_client.close.assert_awaited_once_with()
        assert mock_async_openai.call_count == 2

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
    async def test_hosted_chat_completions_forward_runtime_preferred_asset(
        self,
        mock_async_openai,
    ):
        """Hosted calls should forward request-scoped preferred settlement asset."""
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
            runtime_context={"x402_preferred_asset": "usdc"},
        )

        assert result is not None
        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert kwargs["extra_body"]["x402_preferred_asset"] == "USDC"

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_hosted_chat_completions_forward_search_enabled(
        self,
        mock_async_openai,
    ):
        """Hosted calls should forward the search add-on flag in extra_body."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )
        )
        mock_async_openai.return_value = mock_client

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
            context_window_tokens=64,
            max_output_tokens=32,
        )

        result = await adapter.generate_text(
            "hello",
            runtime_context={"search_enabled": True},
        )

        assert result is not None
        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert kwargs["extra_body"]["search_enabled"] is True

    def test_hosted_extensions_normalize_runtime_preferred_asset(self):
        """Runtime preferred asset values should normalize to the USDC-only symbol."""
        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            x402_preferred_asset="USDC",
        )

        extensions = adapter._hosted_chat_completion_extensions(
            {"x402_preferred_asset": "usdc"}
        )

        assert extensions == {"extra_body": {"x402_preferred_asset": "USDC"}}

    def test_hosted_extensions_use_config_default_preferred_asset(self):
        """Hosted calls should use the configured default asset when runtime context omits it."""
        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            x402_preferred_asset="USDC",
        )

        extensions = adapter._hosted_chat_completion_extensions()

        assert extensions == {"extra_body": {"x402_preferred_asset": "USDC"}}

    def test_hosted_extensions_reject_invalid_preferred_asset(self):
        """Preferred asset validation should fail fast for unsupported values."""
        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
        )

        with pytest.raises(
            ValueError,
            match="x402_preferred_asset must be one of: USDC",
        ):
            adapter._hosted_chat_completion_extensions({"x402_preferred_asset": "bonk"})

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_hosted_chat_completion_forwards_memory_extensions_without_streaming(
        self,
        mock_async_openai,
    ):
        """Hosted chat completions should send memory extensions without stream=True."""

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=None),
                        finish_reason="stop",
                    )
                ]
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

        responses = [
            event
            async for event in adapter.chat_stream(
                [{"role": "user", "content": "hello"}],
                runtime_context={
                    "conversation_id": "conv-123",
                    "memory_ttl_tier": "project",
                    "max_tool_iterations": 7,
                    "request_timeout_seconds": 33,
                    "_raise_stream_errors": True,
                },
            )
        ]

        assert responses == [
            {"type": "content", "delta": "ok"},
            {"type": "message_end", "finish_reason": "stop"},
        ]
        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert "stream" not in kwargs
        assert kwargs["extra_body"]["conversation_id"] == "conv-123"
        assert kwargs["extra_body"]["memory_ttl_tier"] == "project"
        assert kwargs["extra_body"]["max_tool_iterations"] == 7
        assert kwargs["extra_body"]["request_timeout_seconds"] == 33.0
        assert kwargs["extra_body"]["raise_stream_errors"] is True

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_hosted_chat_stream_search_enabled_uses_non_streaming_completion(
        self,
        mock_async_openai,
    ):
        """Hosted search-enabled requests should downgrade to a single completion."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="search result", tool_calls=None
                        ),
                        finish_reason="stop",
                    )
                ]
            )
        )
        mock_async_openai.return_value = mock_client

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
            context_window_tokens=64,
            max_output_tokens=32,
        )

        events = [
            event
            async for event in adapter.chat_stream(
                [{"role": "user", "content": "hello"}],
                runtime_context={"search_enabled": True},
            )
        ]

        assert events == [
            {"type": "content", "delta": "search result"},
            {"type": "message_end", "finish_reason": "stop"},
        ]
        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert "stream" not in kwargs
        assert kwargs["extra_body"]["search_enabled"] is True

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_hosted_chat_stream_retries_connection_error_with_fresh_client(
        self,
        mock_async_openai,
    ):
        """Hosted chat should retry once with a fresh client after a connection error."""
        request = httpx.Request(
            "POST",
            "https://ai.solana-agent.com/v1/chat/completions",
        )
        first_client = MagicMock()
        first_client.close = AsyncMock(return_value=None)
        first_client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=request)
        )
        second_client = MagicMock()
        second_client.close = AsyncMock(return_value=None)
        second_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="retry ok", tool_calls=None),
                        finish_reason="stop",
                    )
                ]
            )
        )
        mock_async_openai.side_effect = [first_client, second_client]

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
            context_window_tokens=64,
            max_output_tokens=32,
        )

        events = [
            event
            async for event in adapter.chat_stream(
                [{"role": "user", "content": "hello"}],
                runtime_context={"search_enabled": True},
            )
        ]

        assert events == [
            {"type": "content", "delta": "retry ok"},
            {"type": "message_end", "finish_reason": "stop"},
        ]
        first_client.close.assert_awaited_once_with()
        kwargs = second_client.chat.completions.create.await_args.kwargs
        assert kwargs["model"] == "solana-agent-chat"
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
        assert kwargs["max_tokens"] == 32
        assert kwargs["extra_body"]["search_enabled"] is True
        assert kwargs["extra_headers"]["Idempotency-Key"]
        assert mock_async_openai.call_count == 2

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_hosted_chat_stream_search_enabled_clamps_max_tokens(
        self,
        mock_async_openai,
    ):
        """Hosted search requests should respect the server-side output ceiling."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="search result", tool_calls=None
                        ),
                        finish_reason="stop",
                    )
                ]
            )
        )
        mock_async_openai.return_value = mock_client

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
            context_window_tokens=16384,
            max_output_tokens=4096,
        )

        _ = [
            event
            async for event in adapter.chat_stream(
                [{"role": "user", "content": "hello"}],
                runtime_context={"search_enabled": True},
            )
        ]

        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert kwargs["max_tokens"] == 4000

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    @patch(
        "solana_agent.adapters.openai_adapter.httpx.AsyncClient",
    )
    @patch(
        "solana_agent.adapters.openai_adapter.resolve_x402_signing_key",
        new_callable=AsyncMock,
    )
    async def test_get_account_summary_uses_wallet_challenge_headers(
        self,
        mock_resolve_x402_signing_key,
        mock_async_client,
        mock_async_openai,
    ):
        """Account summary should fetch a wallet challenge and send signed auth headers."""
        keypair = Keypair()
        private_key = based58.b58encode(bytes(keypair)).decode("ascii")
        wallet = str(keypair.pubkey())
        challenge = {
            "challenge_id": "challenge-123",
            "wallet": wallet,
            "message": "sign me",
            "expires_at": "2099-05-01T12:05:00+00:00",
        }
        summary_response = MagicMock()
        summary_response.raise_for_status = MagicMock()
        summary_response.json.return_value = {
            "spend": {"today": 1.23},
            "tooling": {"lifetime": {"totals": {"requests": 5}}},
        }
        challenge_response = MagicMock()
        challenge_response.raise_for_status = MagicMock()
        challenge_response.json.return_value = challenge

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=challenge_response)
        mock_http_client.get = AsyncMock(return_value=summary_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_http_client
        )
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_resolve_x402_signing_key.return_value = private_key
        mock_async_openai.return_value = MagicMock()

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="x402_privy",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
        )

        result = await adapter.get_account_summary(
            runtime_context={"privy_wallet_id": "wallet-123"}
        )

        assert result == {
            "spend": {"today": 1.23},
            "tooling": {"lifetime": {"totals": {"requests": 5}}},
        }
        assert result["tooling"]["lifetime"]["totals"]["requests"] == 5
        mock_resolve_x402_signing_key.assert_awaited_once_with(
            auth_mode="x402_privy",
            privy_wallet_id="wallet-123",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
            privy_authorization_signature=None,
            privy_request_expiry=None,
            privy_api_url=None,
            timeout=30.0,
            rpc_url=None,
        )
        mock_http_client.post.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/auth/challenge",
            json={"wallet": wallet},
        )
        mock_http_client.get.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/summary",
            headers={
                "X-Wallet-Address": wallet,
                "X-Account-Challenge-Id": "challenge-123",
                "X-Account-Signature": str(
                    keypair.sign_message(challenge["message"].encode("utf-8"))
                ),
            },
            params=None,
        )
        summary_response.raise_for_status.assert_called_once_with()

    @pytest.mark.asyncio
    @patch(
        "solana_agent.adapters.openai_adapter.httpx.AsyncClient",
    )
    async def test_get_account_summary_uses_hosted_identity_params(
        self,
        mock_async_client,
    ):
        """Hosted-managed account summary should use runtime identity params instead of local signing."""
        summary_response = MagicMock()
        summary_response.raise_for_status = MagicMock()
        summary_response.json.return_value = {
            "spend": {"today": 1.23},
            "tooling": {"lifetime": {"totals": {"requests": 5}}},
        }

        mock_http_client = MagicMock()
        mock_http_client.get = AsyncMock(return_value=summary_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_http_client
        )
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="hosted_managed",
            privy_user_id="did:privy:user123",
        )

        result = await adapter.get_account_summary()

        assert result["tooling"]["lifetime"]["totals"]["requests"] == 5
        mock_http_client.get.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/summary",
            params={
                "privy_user_id": "did:privy:user123",
            },
        )

    @pytest.mark.asyncio
    async def test_get_account_summary_requires_identity_for_hosted_managed_auth(self):
        """Hosted-managed account summary should require runtime identity."""
        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="hosted_managed",
        )

        with pytest.raises(
            ValueError,
            match="Hosted account reporting requires config.ai.privy_user_id",
        ):
            await adapter.get_account_summary()

    @pytest.mark.asyncio
    @patch(
        "solana_agent.adapters.openai_adapter.httpx.AsyncClient",
    )
    @patch(
        "solana_agent.adapters.openai_adapter.resolve_x402_signing_key",
        new_callable=AsyncMock,
    )
    async def test_get_usage_report_uses_wallet_challenge_headers_with_privy_export(
        self,
        mock_resolve_x402_signing_key,
        mock_async_client,
    ):
        """Usage reporting should resolve a Privy wallet signer and send challenge headers."""
        keypair = Keypair()
        private_key = based58.b58encode(bytes(keypair)).decode("ascii")
        wallet = str(keypair.pubkey())
        challenge = {
            "challenge_id": "challenge-456",
            "wallet": wallet,
            "message": "sign usage",
            "expires_at": "2099-05-01T12:05:00+00:00",
        }
        usage_response = MagicMock()
        usage_response.raise_for_status = MagicMock()
        usage_response.json.return_value = {
            "buckets": [
                {
                    "totals": {
                        "tooling": {
                            "totals": {
                                "requests": 7,
                                "signature_count": 2,
                            }
                        }
                    }
                }
            ]
        }
        challenge_response = MagicMock()
        challenge_response.raise_for_status = MagicMock()
        challenge_response.json.return_value = challenge

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=challenge_response)
        mock_http_client.get = AsyncMock(return_value=usage_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_http_client
        )
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_resolve_x402_signing_key.return_value = private_key

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="x402_privy",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
        )

        result = await adapter.get_usage_report(
            "month",
            from_date="2026-05-01",
            to_date="2026-05-31",
            group_by="conversation",
            runtime_context={"privy_wallet_id": "wallet-123"},
        )

        assert result == {
            "buckets": [
                {
                    "totals": {
                        "tooling": {
                            "totals": {
                                "requests": 7,
                                "signature_count": 2,
                            }
                        }
                    }
                }
            ]
        }
        assert result["buckets"][0]["totals"]["tooling"]["totals"]["requests"] == 7
        mock_resolve_x402_signing_key.assert_awaited_once_with(
            auth_mode="x402_privy",
            privy_wallet_id="wallet-123",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
            privy_authorization_signature=None,
            privy_request_expiry=None,
            privy_api_url=None,
            timeout=30.0,
            rpc_url=None,
        )
        mock_http_client.post.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/auth/challenge",
            json={"wallet": wallet},
        )
        mock_http_client.get.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/usage",
            headers={
                "X-Wallet-Address": wallet,
                "X-Account-Challenge-Id": "challenge-456",
                "X-Account-Signature": str(
                    keypair.sign_message(challenge["message"].encode("utf-8"))
                ),
            },
            params={
                "granularity": "month",
                "from": "2026-05-01",
                "to": "2026-05-31",
                "group_by": "conversation",
            },
        )
        usage_response.raise_for_status.assert_called_once_with()

    @pytest.mark.asyncio
    @patch(
        "solana_agent.adapters.openai_adapter.httpx.AsyncClient",
    )
    async def test_get_usage_report_uses_hosted_identity_params(
        self,
        mock_async_client,
    ):
        """Hosted-managed usage reporting should send runtime identity in the request params."""
        usage_response = MagicMock()
        usage_response.raise_for_status = MagicMock()
        usage_response.json.return_value = {
            "buckets": [{"totals": {"tooling": {"totals": {"requests": 7}}}}]
        }

        mock_http_client = MagicMock()
        mock_http_client.get = AsyncMock(return_value=usage_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_http_client
        )
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="hosted_managed",
            privy_user_id="did:privy:user123",
        )

        result = await adapter.get_usage_report(
            "month",
            from_date="2026-05-01",
            to_date="2026-05-31",
            group_by="conversation",
        )

        assert result["buckets"][0]["totals"]["tooling"]["totals"]["requests"] == 7
        mock_http_client.get.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/usage",
            params={
                "granularity": "month",
                "from": "2026-05-01",
                "to": "2026-05-31",
                "group_by": "conversation",
                "privy_user_id": "did:privy:user123",
            },
        )

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    @patch(
        "solana_agent.adapters.openai_adapter.create_x402_httpx_client_for_auth",
        new_callable=AsyncMock,
    )
    @patch(
        "solana_agent.adapters.openai_adapter.OpenAIAdapter._build_account_auth_headers",
        new_callable=AsyncMock,
    )
    async def test_hosted_chat_completions_forward_wallet_auth_headers(
        self,
        mock_build_account_auth_headers,
        mock_create_x402_httpx_client_for_auth,
        mock_async_openai,
    ):
        """Hosted chat requests should attach wallet-auth headers for account-aware quotes."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )
        )
        mock_async_openai.return_value = mock_client
        mock_create_x402_httpx_client_for_auth.return_value = MagicMock()
        mock_build_account_auth_headers.return_value = {
            "X-Wallet-Address": "wallet-123",
            "X-Account-Challenge-Id": "challenge-123",
            "X-Account-Signature": "signature-123",
        }

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="x402_privy",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
            context_window_tokens=64,
            max_output_tokens=32,
        )

        result = await adapter.generate_text(
            "hello",
            runtime_context={"privy_wallet_id": "wallet-123"},
        )

        assert result is not None
        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert kwargs["extra_headers"]["Idempotency-Key"]
        assert kwargs["extra_headers"]["X-Wallet-Address"] == "wallet-123"
        assert kwargs["extra_headers"]["X-Account-Challenge-Id"] == "challenge-123"
        assert kwargs["extra_headers"]["X-Account-Signature"] == "signature-123"

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    @patch(
        "solana_agent.adapters.openai_adapter.create_hosted_managed_x402_httpx_client"
    )
    @patch(
        "solana_agent.adapters.openai_adapter.OpenAIAdapter._build_account_auth_headers",
        new_callable=AsyncMock,
    )
    async def test_hosted_managed_chat_completions_forward_wallet_auth_headers(
        self,
        mock_build_account_auth_headers,
        mock_create_hosted_managed_x402_httpx_client,
        mock_async_openai,
    ):
        """Hosted-managed chat should attach wallet-auth headers for hosted wallet billing."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )
        )
        mock_async_openai.return_value = mock_client
        mock_create_hosted_managed_x402_httpx_client.return_value = MagicMock()
        mock_build_account_auth_headers.return_value = {
            "X-Wallet-Address": "wallet-123",
            "X-Account-Challenge-Id": "challenge-123",
            "X-Account-Signature": "signature-123",
        }

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="hosted_managed",
            privy_user_id="did:privy:user123",
            private_key="base58-private-key",
            context_window_tokens=64,
            max_output_tokens=32,
        )

        result = await adapter.generate_text("hello")

        assert result is not None
        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert kwargs["extra_headers"]["Idempotency-Key"]
        assert kwargs["extra_headers"]["X-Wallet-Address"] == "wallet-123"
        assert kwargs["extra_headers"]["X-Account-Challenge-Id"] == "challenge-123"
        assert kwargs["extra_headers"]["X-Account-Signature"] == "signature-123"

    @pytest.mark.asyncio
    @patch(
        "solana_agent.adapters.openai_adapter.OpenAIAdapter._create_account_auth_challenge",
        new_callable=AsyncMock,
    )
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_hosted_managed_account_auth_prefers_runtime_wallet_signer(
        self,
        mock_async_openai,
        mock_create_account_auth_challenge,
    ):
        """Hosted-managed wallet auth should sign as the runtime hosted wallet, not a configured local key."""
        local_keypair = Keypair()
        hosted_keypair = Keypair()
        local_private_key = based58.b58encode(bytes(local_keypair)).decode("ascii")
        hosted_private_key = based58.b58encode(bytes(hosted_keypair)).decode("ascii")
        hosted_wallet = str(hosted_keypair.pubkey())
        mock_async_openai.return_value = MagicMock()
        mock_create_account_auth_challenge.return_value = {
            "challenge_id": "challenge-123",
            "wallet": hosted_wallet,
            "message": "sign me",
            "expires_at": "2099-05-01T12:05:00+00:00",
        }

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="hosted_managed",
            privy_user_id="did:privy:user123",
            private_key=local_private_key,
        )
        adapter.export_wallet_private_key = AsyncMock(
            return_value={"private_key": hosted_private_key}
        )

        headers = await adapter._build_account_auth_headers(
            {"hosted_privy_wallet_id": "wallet-123"}
        )

        adapter.export_wallet_private_key.assert_awaited_once_with(
            privy_user_id="did:privy:user123",
            wallet_id="wallet-123",
        )
        mock_create_account_auth_challenge.assert_awaited_once_with(hosted_wallet)
        assert headers["X-Wallet-Address"] == hosted_wallet
        assert headers["X-Account-Challenge-Id"] == "challenge-123"
        assert headers["X-Account-Signature"] == str(
            hosted_keypair.sign_message(b"sign me")
        )

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_hosted_chat_completions_forward_privy_context(
        self,
        mock_async_openai,
    ):
        """Hosted chat requests should serialize Privy metadata into the body."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )
        )
        mock_async_openai.return_value = mock_client

        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
            privy_user_id="did:privy:user123",
            context_window_tokens=64,
            max_output_tokens=32,
        )

        result = await adapter.generate_text(
            "hello",
            runtime_context={
                "privy_wallet_id": "wallet-123",
                "privy_wallet_address": "WalletPubkey123",
                "service_tier": "priority",
            },
        )

        assert result is not None
        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert kwargs["extra_body"]["privy_user_id"] == "did:privy:user123"
        assert kwargs["extra_body"]["privy_wallet_id"] == "wallet-123"
        assert kwargs["extra_body"]["privy_wallet_address"] == "WalletPubkey123"
        assert kwargs["extra_body"]["service_tier"] == "priority"

    @pytest.mark.asyncio
    @patch(
        "solana_agent.adapters.openai_adapter.httpx.AsyncClient",
    )
    async def test_create_privy_user_posts_to_hosted_user_endpoint(
        self,
        mock_async_client,
    ):
        """Hosted user creation should call the explicit user endpoint."""
        user_response = MagicMock()
        user_response.raise_for_status = MagicMock()
        user_response.json.return_value = {
            "privy_user_id": "did:privy:user123",
            "created": True,
        }

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=user_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_http_client
        )
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
        )

        result = await adapter.create_privy_user()

        assert result["privy_user_id"] == "did:privy:user123"
        mock_http_client.post.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/user",
            json={},
        )

    @pytest.mark.asyncio
    async def test_create_privy_user_posts_empty_payload(self):
        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
        )

        with patch(
            "solana_agent.adapters.openai_adapter.httpx.AsyncClient",
        ) as mock_async_client:
            user_response = MagicMock()
            user_response.raise_for_status = MagicMock()
            user_response.json.return_value = {"privy_user_id": "did:privy:user123"}

            mock_http_client = MagicMock()
            mock_http_client.post = AsyncMock(return_value=user_response)
            mock_async_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_http_client
            )
            mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

            await adapter.create_privy_user()

        mock_http_client.post.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/user",
            json={},
        )

    @pytest.mark.asyncio
    @patch(
        "solana_agent.adapters.openai_adapter.httpx.AsyncClient",
    )
    async def test_create_wallet_posts_to_hosted_wallet_endpoint(
        self,
        mock_async_client,
    ):
        """Hosted wallet creation should call the explicit wallet endpoint."""
        wallet_response = MagicMock()
        wallet_response.raise_for_status = MagicMock()
        wallet_response.json.return_value = {
            "privy_user_id": "did:privy:user123",
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
            "old_wallets": [],
        }

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=wallet_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_http_client
        )
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
        )

        result = await adapter.create_wallet(privy_user_id="did:privy:user123")

        assert result["wallet_id"] == "wallet-123"
        assert result["privy_user_id"] == "did:privy:user123"
        mock_http_client.post.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/wallet",
            json={
                "privy_user_id": "did:privy:user123",
                "chain_type": "solana",
            },
        )

    @pytest.mark.asyncio
    async def test_create_wallet_requires_privy_user_id(self):
        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
        )

        with pytest.raises(ValueError, match="privy_user_id is required"):
            await adapter.create_wallet(privy_user_id="")

    @pytest.mark.asyncio
    async def test_create_wallet_rejects_invalid_chain_type(self):
        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
        )

        with pytest.raises(
            ValueError, match="chain_type must be one of: ethereum, solana"
        ):
            await adapter.create_wallet(
                privy_user_id="did:privy:user123",
                chain_type="bitcoin",
            )

    @pytest.mark.asyncio
    @patch(
        "solana_agent.adapters.openai_adapter.httpx.AsyncClient",
    )
    async def test_rotate_wallet_posts_to_hosted_rotate_endpoint(
        self,
        mock_async_client,
    ):
        wallet_response = MagicMock()
        wallet_response.raise_for_status = MagicMock()
        wallet_response.json.return_value = {
            "privy_user_id": "did:privy:user123",
            "wallet_id": "wallet-new",
            "address": "WalletPubkeyNew",
            "old_wallets": [{"wallet_id": "wallet-old", "address": "WalletOld"}],
        }

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=wallet_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_http_client
        )
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
        )

        result = await adapter.rotate_wallet(privy_user_id="did:privy:user123")

        assert result["wallet_id"] == "wallet-new"
        assert result["old_wallets"][0]["wallet_id"] == "wallet-old"
        mock_http_client.post.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/wallet/rotate",
            json={
                "privy_user_id": "did:privy:user123",
                "chain_type": "solana",
            },
        )

    @pytest.mark.asyncio
    async def test_rotate_wallet_validates_privy_user_id_and_chain_type(self):
        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
        )

        with pytest.raises(ValueError, match="privy_user_id is required"):
            await adapter.rotate_wallet(privy_user_id="")

        with pytest.raises(
            ValueError, match="chain_type must be one of: ethereum, solana"
        ):
            await adapter.rotate_wallet(
                privy_user_id="did:privy:user123",
                chain_type="bitcoin",
            )

    @pytest.mark.asyncio
    @patch(
        "solana_agent.adapters.openai_adapter.httpx.AsyncClient",
    )
    async def test_export_wallet_private_key_posts_to_hosted_export_endpoint(
        self,
        mock_async_client,
    ):
        export_response = MagicMock()
        export_response.raise_for_status = MagicMock()
        export_response.json.return_value = {
            "privy_user_id": "did:privy:user123",
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
            "private_key": "base58-private-key",
        }

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=export_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_http_client
        )
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
        )

        result = await adapter.export_wallet_private_key(
            privy_user_id="did:privy:user123",
            wallet_id="wallet-123",
        )

        assert result["private_key"] == "base58-private-key"
        mock_http_client.post.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/wallet/export",
            json={
                "privy_user_id": "did:privy:user123",
                "chain_type": "solana",
                "confirm_export": True,
                "wallet_id": "wallet-123",
            },
        )

    @pytest.mark.asyncio
    async def test_export_wallet_private_key_validates_privy_user_id_and_chain_type(
        self,
    ):
        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
        )

        with pytest.raises(ValueError, match="privy_user_id is required"):
            await adapter.export_wallet_private_key(privy_user_id="")

        with pytest.raises(
            ValueError, match="chain_type must be one of: ethereum, solana"
        ):
            await adapter.export_wallet_private_key(
                privy_user_id="did:privy:user123",
                chain_type="bitcoin",
            )

    @pytest.mark.asyncio
    @patch(
        "solana_agent.adapters.openai_adapter.httpx.AsyncClient",
    )
    async def test_get_wallet_address_calls_hosted_wallet_endpoint(
        self,
        mock_async_client,
    ):
        """Hosted wallet address lookups should call the explicit address endpoint."""
        wallet_response = MagicMock()
        wallet_response.raise_for_status = MagicMock()
        wallet_response.json.return_value = {
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
        }

        mock_http_client = MagicMock()
        mock_http_client.get = AsyncMock(return_value=wallet_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_http_client
        )
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
        )

        result = await adapter.get_wallet_address(wallet_id="wallet-123")

        assert result["address"] == "WalletPubkey123"
        assert result["wallet_id"] == "wallet-123"
        mock_http_client.get.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/wallet/address",
            params={"wallet_id": "wallet-123"},
        )

    @pytest.mark.asyncio
    async def test_get_wallet_address_requires_wallet_id(self):
        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-chat",
            base_url="https://ai.solana-agent.com/v1",
        )

        with pytest.raises(ValueError, match="wallet_id is required"):
            await adapter.get_wallet_address(wallet_id="")

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_get_usage_report_rejects_invalid_granularity(
        self,
        mock_async_openai,
    ):
        """Usage reporting should validate granularity locally."""
        mock_async_openai.return_value = MagicMock()
        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="x402_privy",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
        )

        with pytest.raises(
            ValueError, match="granularity must be one of: day, month, year"
        ):
            await adapter.get_usage_report("week")

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    @patch(
        "solana_agent.adapters.openai_adapter.httpx.AsyncClient",
    )
    @patch(
        "solana_agent.adapters.openai_adapter.resolve_x402_signing_key",
        new_callable=AsyncMock,
    )
    async def test_get_usage_forecast_forwards_window_days(
        self,
        mock_resolve_x402_signing_key,
        mock_async_client,
        mock_async_openai,
    ):
        """Forecast requests should send the requested window length."""
        keypair = Keypair()
        private_key = based58.b58encode(bytes(keypair)).decode("ascii")
        wallet = str(keypair.pubkey())
        challenge = {
            "challenge_id": "challenge-789",
            "wallet": wallet,
            "message": "sign forecast",
            "expires_at": "2099-05-01T12:05:00+00:00",
        }
        forecast_response = MagicMock()
        forecast_response.raise_for_status = MagicMock()
        forecast_response.json.return_value = {
            "forecast": {"projected_spend": 7.5},
            "tooling": {
                "current_month": {"totals": {"requests_to_date": 2}},
                "projected_month_end": {"totals": {"projected_requests": 4}},
            },
        }
        challenge_response = MagicMock()
        challenge_response.raise_for_status = MagicMock()
        challenge_response.json.return_value = challenge

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=challenge_response)
        mock_http_client.get = AsyncMock(return_value=forecast_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_http_client
        )
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_resolve_x402_signing_key.return_value = private_key
        mock_async_openai.return_value = MagicMock()

        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="x402_privy",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
        )

        result = await adapter.get_usage_forecast(
            window_days=45,
            runtime_context={"privy_wallet_id": "wallet-123"},
        )

        assert result == {
            "forecast": {"projected_spend": 7.5},
            "tooling": {
                "current_month": {"totals": {"requests_to_date": 2}},
                "projected_month_end": {"totals": {"projected_requests": 4}},
            },
        }
        assert (
            result["tooling"]["projected_month_end"]["totals"]["projected_requests"]
            == 4
        )
        mock_resolve_x402_signing_key.assert_awaited_once_with(
            auth_mode="x402_privy",
            privy_wallet_id="wallet-123",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
            privy_authorization_signature=None,
            privy_request_expiry=None,
            privy_api_url=None,
            timeout=30.0,
            rpc_url=None,
        )
        mock_http_client.get.assert_awaited_once_with(
            "https://ai.solana-agent.com/v1/account/forecast",
            headers={
                "X-Wallet-Address": wallet,
                "X-Account-Challenge-Id": "challenge-789",
                "X-Account-Signature": str(
                    keypair.sign_message(challenge["message"].encode("utf-8"))
                ),
            },
            params={"window_days": 45},
        )
        forecast_response.raise_for_status.assert_called_once_with()

    @pytest.mark.asyncio
    @patch("solana_agent.adapters.openai_adapter.AsyncOpenAI")
    async def test_get_usage_forecast_rejects_non_positive_window_days(
        self,
        mock_async_openai,
    ):
        """Forecast requests should validate window_days locally."""
        mock_async_openai.return_value = MagicMock()
        adapter = OpenAIAdapter(
            api_key="x402",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
            auth_mode="x402_privy",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
        )

        with pytest.raises(ValueError, match="window_days must be a positive integer"):
            await adapter.get_usage_forecast(0)

    @pytest.mark.asyncio
    async def test_account_reporting_requires_supported_hosted_auth_mode(self):
        """Account reporting should be unavailable for auth modes outside the hosted SDK paths."""
        adapter = OpenAIAdapter(
            api_key="test-api-key",
            model="solana-agent-memory",
            base_url="https://ai.solana-agent.com/v1",
        )

        with pytest.raises(
            NotImplementedError,
            match="Account reporting requires hosted_managed or x402_privy auth",
        ):
            await adapter.get_pricing_info()

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

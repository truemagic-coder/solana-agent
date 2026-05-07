import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from x402.extensions.payment_identifier import declare_payment_identifier_extension

from solana_agent.tools.utils.x402 import (
    HostedManagedX402AsyncTransport,
    X402SigningKeyConfig,
    X402PrivyWalletExportConfig,
    create_x402_httpx_client,
    create_x402_httpx_client_for_auth,
    export_privy_wallet_private_key,
    has_x402_auth_config,
    request_with_x402_signing_key,
    request_with_x402_privy,
    resolve_x402_privy_config,
    resolve_x402_signing_key,
)


def make_async_context_manager(inner):
    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=inner)
    context_manager.__aexit__ = AsyncMock(return_value=None)
    return context_manager


@pytest.fixture
def privy_export_config():
    return X402PrivyWalletExportConfig(
        wallet_id="wallet-123",
        app_id="app-123",
        app_secret="secret-123",
        authorization_signature="sig-123",
        request_expiry="12345",
        api_url="https://api.privy.io",
        timeout=15.0,
        rpc_url="https://rpc.example.com",
    )


def test_resolve_x402_privy_config_returns_none_for_non_privy_mode():
    assert (
        resolve_x402_privy_config(
            auth_mode="api_key",
            privy_wallet_id="wallet-123",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
        )
        is None
    )


def test_resolve_x402_privy_config_returns_runtime_export_config():
    config = resolve_x402_privy_config(
        auth_mode="x402_privy",
        privy_wallet_id="wallet-123",
        privy_app_id="app-123",
        privy_app_secret="secret-123",
        privy_authorization_signature="sig-123",
        privy_request_expiry="12345",
        privy_api_url="https://api.privy.io/",
        timeout=15.0,
        rpc_url="https://rpc.example.com",
    )

    assert config == X402PrivyWalletExportConfig(
        wallet_id="wallet-123",
        app_id="app-123",
        app_secret="secret-123",
        authorization_signature="sig-123",
        request_expiry="12345",
        api_url="https://api.privy.io",
        timeout=15.0,
        rpc_url="https://rpc.example.com",
    )


def test_has_x402_auth_config_accepts_runtime_privy_export_config():
    assert has_x402_auth_config(
        auth_mode="x402_privy",
        privy_wallet_id="wallet-123",
        privy_app_id="app-123",
        privy_app_secret="secret-123",
    )


def test_has_x402_auth_config_rejects_unknown_auth_mode():
    assert not has_x402_auth_config(auth_mode="api_key")


@pytest.mark.asyncio
async def test_export_privy_wallet_private_key_success(privy_export_config):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "encryption_type": "HPKE",
        "encapsulated_key": base64.b64encode(b"encapsulated").decode("ascii"),
        "ciphertext": base64.b64encode(b"ciphertext").decode("ascii"),
    }
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=response)
    recipient_key_pair = SimpleNamespace(
        public_key=SimpleNamespace(
            to_public_bytes=MagicMock(return_value=b"public-key")
        ),
        private_key="recipient-private-key",
    )
    fake_suite = SimpleNamespace(
        kem=SimpleNamespace(derive_key_pair=MagicMock(return_value=recipient_key_pair)),
        open=MagicMock(return_value=b"decrypted-private-key"),
    )

    with (
        patch("solana_agent.tools.utils.x402.os.urandom", return_value=b"seed"),
        patch(
            "solana_agent.tools.utils.x402.PRIVY_HPKE_CIPHER_SUITE",
            fake_suite,
        ),
        patch(
            "solana_agent.tools.utils.x402.httpx.AsyncClient",
            return_value=make_async_context_manager(http_client),
        ) as mock_async_client,
        patch(
            "solana_agent.tools.utils.x402._normalize_privy_exported_private_key",
            return_value="normalized-key",
        ) as mock_normalize,
    ):
        resolved_key = await export_privy_wallet_private_key(privy_export_config)

    assert resolved_key == "normalized-key"
    mock_async_client.assert_called_once_with(timeout=15.0)
    http_client.post.assert_awaited_once()
    assert (
        http_client.post.await_args.args[0]
        == "https://api.privy.io/v1/wallets/wallet-123/export"
    )
    assert http_client.post.await_args.kwargs["headers"] == {
        "privy-app-id": "app-123",
        "Content-Type": "application/json",
        "privy-authorization-signature": "sig-123",
        "privy-request-expiry": "12345",
    }
    assert http_client.post.await_args.kwargs["auth"] == ("app-123", "secret-123")
    assert http_client.post.await_args.kwargs["json"]["encryption_type"] == "HPKE"
    fake_suite.open.assert_called_once_with(
        enc=b"encapsulated",
        skr="recipient-private-key",
        ct=b"ciphertext",
    )
    mock_normalize.assert_called_once_with(b"decrypted-private-key")


@pytest.mark.asyncio
async def test_export_privy_wallet_private_key_rejects_unsupported_encryption_type(
    privy_export_config,
):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"encryption_type": "RSA"}
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=response)
    fake_suite = SimpleNamespace(
        kem=SimpleNamespace(
            derive_key_pair=MagicMock(
                return_value=SimpleNamespace(
                    public_key=SimpleNamespace(
                        to_public_bytes=MagicMock(return_value=b"public-key")
                    ),
                    private_key="recipient-private-key",
                )
            )
        ),
        open=MagicMock(),
    )

    with (
        patch("solana_agent.tools.utils.x402.os.urandom", return_value=b"seed"),
        patch(
            "solana_agent.tools.utils.x402.PRIVY_HPKE_CIPHER_SUITE",
            fake_suite,
        ),
        patch(
            "solana_agent.tools.utils.x402.httpx.AsyncClient",
            return_value=make_async_context_manager(http_client),
        ),
    ):
        with pytest.raises(ValueError, match="unsupported encryption_type"):
            await export_privy_wallet_private_key(privy_export_config)


@pytest.mark.asyncio
async def test_export_privy_wallet_private_key_requires_encrypted_payload_fields(
    privy_export_config,
):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "encryption_type": "HPKE",
        "encapsulated_key": base64.b64encode(b"encapsulated").decode("ascii"),
    }
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=response)
    fake_suite = SimpleNamespace(
        kem=SimpleNamespace(
            derive_key_pair=MagicMock(
                return_value=SimpleNamespace(
                    public_key=SimpleNamespace(
                        to_public_bytes=MagicMock(return_value=b"public-key")
                    ),
                    private_key="recipient-private-key",
                )
            )
        ),
        open=MagicMock(),
    )

    with (
        patch("solana_agent.tools.utils.x402.os.urandom", return_value=b"seed"),
        patch(
            "solana_agent.tools.utils.x402.PRIVY_HPKE_CIPHER_SUITE",
            fake_suite,
        ),
        patch(
            "solana_agent.tools.utils.x402.httpx.AsyncClient",
            return_value=make_async_context_manager(http_client),
        ),
    ):
        with pytest.raises(ValueError, match="missing encapsulated_key or ciphertext"):
            await export_privy_wallet_private_key(privy_export_config)


@pytest.mark.asyncio
async def test_resolve_x402_signing_key_returns_direct_private_key():
    assert await resolve_x402_signing_key(auth_mode="api_key") is None


@pytest.mark.asyncio
async def test_resolve_x402_signing_key_uses_runtime_privy_export():
    with patch(
        "solana_agent.tools.utils.x402.export_privy_wallet_private_key",
        AsyncMock(return_value="resolved-base58-key"),
    ) as mock_export:
        resolved = await resolve_x402_signing_key(
            auth_mode="x402_privy",
            privy_wallet_id="wallet-123",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
        )

    assert resolved == "resolved-base58-key"
    exported_config = mock_export.await_args.args[0]
    assert exported_config.wallet_id == "wallet-123"
    assert exported_config.app_id == "app-123"
    assert exported_config.app_secret == "secret-123"


@pytest.mark.asyncio
async def test_resolve_x402_signing_key_returns_none_without_config():
    assert await resolve_x402_signing_key(auth_mode="x402_privy") is None


@pytest.mark.asyncio
async def test_create_x402_httpx_client_for_auth_builds_client_from_resolved_key():
    with (
        patch(
            "solana_agent.tools.utils.x402.resolve_x402_signing_key",
            AsyncMock(return_value="resolved-key"),
        ),
        patch(
            "solana_agent.tools.utils.x402.create_x402_httpx_client",
            return_value="client-object",
        ) as mock_create_client,
    ):
        client = await create_x402_httpx_client_for_auth(
            auth_mode="x402_privy",
            privy_wallet_id="wallet-123",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
            timeout=12.0,
            rpc_url="https://rpc.example.com",
        )

    assert client == "client-object"
    config = mock_create_client.call_args.args[0]
    assert config == X402SigningKeyConfig(
        signing_key="resolved-key",
        timeout=12.0,
        rpc_url="https://rpc.example.com",
    )


@pytest.mark.asyncio
async def test_create_x402_httpx_client_for_auth_raises_without_credentials():
    with patch(
        "solana_agent.tools.utils.x402.resolve_x402_signing_key",
        AsyncMock(return_value=None),
    ):
        with pytest.raises(
            ValueError,
            match="x402_privy requires configured x402 signing credentials",
        ):
            await create_x402_httpx_client_for_auth(auth_mode="x402_privy")


def test_create_x402_httpx_client_registers_signer():
    x402_client = MagicMock()
    x402_client.on_after_payment_creation.return_value = x402_client

    with (
        patch("solana_agent.tools.utils.x402.x402Client", return_value=x402_client),
        patch(
            "solana_agent.tools.utils.x402.KeypairSigner.from_base58",
            return_value="signer",
        ) as mock_signer,
        patch(
            "solana_agent.tools.utils.x402.register_exact_svm_client"
        ) as mock_register,
        patch(
            "solana_agent.tools.utils.x402.x402HttpxClient",
            return_value="http-client",
        ) as mock_http_client,
    ):
        client = create_x402_httpx_client(
            X402SigningKeyConfig(
                signing_key="base58-key",
                timeout=9.0,
                rpc_url="https://rpc.example.com",
            )
        )

    assert client == "http-client"
    mock_signer.assert_called_once_with("base58-key")
    mock_register.assert_called_once_with(
        x402_client,
        signer="signer",
        rpc_url="https://rpc.example.com",
    )
    mock_http_client.assert_called_once_with(x402_client, timeout=9.0)


def test_create_x402_httpx_client_appends_required_payment_identifier():
    client = MagicMock()
    client.on_after_payment_creation.return_value = client

    with (
        patch("solana_agent.tools.utils.x402.x402Client", return_value=client),
        patch(
            "solana_agent.tools.utils.x402.KeypairSigner.from_base58",
            return_value="signer",
        ),
        patch("solana_agent.tools.utils.x402.register_exact_svm_client"),
        patch(
            "solana_agent.tools.utils.x402.x402HttpxClient",
            return_value="http-client",
        ),
    ):
        create_x402_httpx_client(X402SigningKeyConfig(signing_key="base58-key"))

    hook = client.on_after_payment_creation.call_args.args[0]
    payment_payload = SimpleNamespace(
        x402_version=2,
        extensions={
            "payment-identifier": declare_payment_identifier_extension(required=True)
        },
    )

    hook(SimpleNamespace(payment_payload=payment_payload))

    assert payment_payload.extensions["payment-identifier"]["info"]["id"]


def test_create_hosted_managed_x402_httpx_client_appends_required_payment_identifier():
    client = MagicMock()
    client.on_after_payment_creation.return_value = client

    with (
        patch("solana_agent.tools.utils.x402.x402Client", return_value=client),
        patch(
            "solana_agent.tools.utils.x402.KeypairSigner.from_base58",
            return_value="signer",
        ),
        patch("solana_agent.tools.utils.x402.register_exact_svm_client"),
        patch(
            "solana_agent.tools.utils.x402.HostedManagedX402HttpxClient",
            return_value="http-client",
        ),
    ):
        from solana_agent.tools.utils.x402 import (
            create_hosted_managed_x402_httpx_client,
        )

        create_hosted_managed_x402_httpx_client(
            X402SigningKeyConfig(signing_key="base58-key")
        )

    hook = client.on_after_payment_creation.call_args.args[0]
    payment_payload = SimpleNamespace(
        x402_version=2,
        extensions={
            "payment-identifier": declare_payment_identifier_extension(required=True)
        },
    )

    hook(SimpleNamespace(payment_payload=payment_payload))

    assert payment_payload.extensions["payment-identifier"]["info"]["id"]


@pytest.mark.asyncio
async def test_request_with_x402_privy_requires_runtime_export_config():
    with patch(
        "solana_agent.tools.utils.x402.resolve_x402_signing_key",
        AsyncMock(return_value=None),
    ):
        with pytest.raises(ValueError, match="Privy x402 requires runtime Privy"):
            await request_with_x402_privy(
                "GET",
                "https://api.example.com/data",
            )


@pytest.mark.asyncio
async def test_request_with_x402_privy_delegates_to_private_key_request():
    response = object()
    with (
        patch(
            "solana_agent.tools.utils.x402.resolve_x402_signing_key",
            AsyncMock(return_value="resolved-key"),
        ),
        patch(
            "solana_agent.tools.utils.x402.request_with_x402_signing_key",
            AsyncMock(return_value=response),
        ) as mock_request,
    ):
        result = await request_with_x402_privy(
            "post",
            "https://api.example.com/data",
            privy_wallet_id="wallet-123",
            privy_app_id="app-123",
            privy_app_secret="secret-123",
            headers={"accept": "application/json"},
            params={"foo": "bar"},
            json_data={"hello": "world"},
            timeout=4.0,
            rpc_url="https://rpc.example.com",
        )

    assert result is response
    mock_request.assert_awaited_once_with(
        method="post",
        url="https://api.example.com/data",
        signing_key="resolved-key",
        headers={"accept": "application/json"},
        params={"foo": "bar"},
        json_data={"hello": "world"},
        timeout=4.0,
        rpc_url="https://rpc.example.com",
    )


@pytest.mark.asyncio
async def test_request_with_x402_signing_key_uses_configured_client():
    response = object()
    http_client = MagicMock()
    http_client.request = AsyncMock(return_value=response)

    with patch(
        "solana_agent.tools.utils.x402.create_x402_httpx_client",
        return_value=make_async_context_manager(http_client),
    ) as mock_create_client:
        result = await request_with_x402_signing_key(
            "post",
            "https://api.example.com/data",
            signing_key="base58-key",
            headers={"accept": "application/json"},
            params={"foo": "bar"},
            json_data={"hello": "world"},
            timeout=6.0,
            rpc_url="https://rpc.example.com",
        )

    assert result is response
    config = mock_create_client.call_args.args[0]
    assert config == X402SigningKeyConfig(
        signing_key="base58-key",
        timeout=6.0,
        rpc_url="https://rpc.example.com",
    )
    http_client.request.assert_awaited_once_with(
        method="POST",
        url="https://api.example.com/data",
        headers={"accept": "application/json"},
        params={"foo": "bar"},
        json={"hello": "world"},
    )


@pytest.mark.asyncio
async def test_hosted_managed_transport_rotates_idempotency_key_on_retry():
    class StubTransport:
        def __init__(self, responses):
            self._responses = list(responses)
            self.requests = []

        async def handle_async_request(self, request):
            self.requests.append(request)
            return self._responses[len(self.requests) - 1]

        async def aclose(self):
            return None

    payment_helper = MagicMock()
    payment_helper.get_payment_required_response.return_value = "requirements"
    payment_helper.encode_payment_signature_header.return_value = {"x-payment": "paid"}
    x402_client = MagicMock()
    x402_client.create_payment_payload = AsyncMock(return_value="payload")
    original_request = __import__("httpx").Request(
        "POST",
        "https://example.com/v1/chat/completions",
        headers={"Idempotency-Key": "original-key"},
        content=b"{}",
    )
    stub_transport = StubTransport(
        [
            __import__("httpx").Response(
                402,
                json={"error": "Payment required"},
                request=original_request,
            ),
            __import__("httpx").Response(
                200,
                json={"ok": True},
                request=original_request,
            ),
        ]
    )

    with patch(
        "solana_agent.tools.utils.x402.x402HTTPClient",
        return_value=payment_helper,
    ):
        transport = HostedManagedX402AsyncTransport(
            x402_client,
            transport=stub_transport,
        )
        response = await transport.handle_async_request(original_request)

    assert response.status_code == 200
    assert len(stub_transport.requests) == 2
    retry_request = stub_transport.requests[1]
    assert retry_request.headers["x-payment"] == "paid"
    assert retry_request.headers["Idempotency-Key"] != "original-key"

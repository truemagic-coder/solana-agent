"""Shared x402 HTTP helpers for OpenAI-compatible paid endpoints."""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import based58
import httpx
from pyhpke import AEADId, CipherSuite, KDFId, KEMId
from x402 import x402Client
from x402.extensions.payment_identifier import append_payment_identifier_to_extensions
from x402.http.clients import x402HttpxClient
from x402.http.x402_http_client import x402HTTPClient
from x402.mechanisms.svm import KeypairSigner
from x402.mechanisms.svm.exact.register import register_exact_svm_client


logger = logging.getLogger(__name__)

PRIVY_API_URL = "https://api.privy.io"
PRIVY_HPKE_CIPHER_SUITE = CipherSuite.new(
    KEMId.DHKEM_P256_HKDF_SHA256,
    KDFId.HKDF_SHA256,
    AEADId.CHACHA20_POLY1305,
)


@dataclass
class X402SigningKeyConfig:
    """Configuration for a resolved Solana signing key used for x402 payments."""

    signing_key: str
    timeout: float = 30.0
    rpc_url: Optional[str] = None


@dataclass
class X402PrivyWalletExportConfig:
    """Configuration for runtime Privy wallet export used for x402 signing."""

    wallet_id: str
    app_id: str
    app_secret: str
    authorization_signature: Optional[str] = None
    request_expiry: Optional[str] = None
    api_url: str = PRIVY_API_URL
    timeout: float = 30.0
    rpc_url: Optional[str] = None


class HostedManagedX402AsyncTransport(httpx.AsyncBaseTransport):
    """Retry x402 challenges with a fresh idempotency key for hosted chat flows."""

    RETRY_KEY = "_x402_is_retry"

    def __init__(
        self,
        client: x402Client,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._http_client = x402HTTPClient(client)
        self._client = client
        self._transport = transport or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._transport.handle_async_request(request)
        if response.status_code != 402:
            return response
        if request.extensions.get(self.RETRY_KEY):
            return response

        await response.aread()

        def get_header(name: str) -> str | None:
            return response.headers.get(name)

        body = None
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = None

        payment_required = self._http_client.get_payment_required_response(
            get_header,
            body,
        )
        payment_payload = await self._client.create_payment_payload(payment_required)
        payment_headers = self._http_client.encode_payment_signature_header(
            payment_payload
        )

        retry_headers = dict(request.headers)
        retry_headers.update(payment_headers)
        retry_headers["Access-Control-Expose-Headers"] = (
            "PAYMENT-RESPONSE,X-PAYMENT-RESPONSE"
        )
        saw_idempotency_key = False
        for header_name in list(retry_headers):
            if header_name.lower() == "idempotency-key":
                saw_idempotency_key = True
                del retry_headers[header_name]
        if saw_idempotency_key:
            retry_headers["Idempotency-Key"] = uuid.uuid4().hex

        retry_extensions = dict(request.extensions)
        retry_extensions[self.RETRY_KEY] = True

        retry_request = httpx.Request(
            method=request.method,
            url=request.url,
            headers=retry_headers,
            content=request.content,
            extensions=retry_extensions,
        )
        return await self._transport.handle_async_request(retry_request)

    async def aclose(self) -> None:
        await self._transport.aclose()


class HostedManagedX402HttpxClient(httpx.AsyncClient):
    """Async httpx client that rotates idempotency keys on x402 retries."""

    def __init__(
        self,
        x402_client: x402Client,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            transport=HostedManagedX402AsyncTransport(x402_client),
            **kwargs,
        )


def _configure_x402_client(client: x402Client) -> x402Client:
    def _append_payment_identifier(payment_context: Any) -> None:
        payment_payload = getattr(payment_context, "payment_payload", None)
        if isinstance(payment_payload, dict):
            extensions = payment_payload.get("extensions")
        else:
            extensions = getattr(payment_payload, "extensions", None)
        if isinstance(extensions, dict):
            append_payment_identifier_to_extensions(extensions)

    client.on_after_payment_creation(_append_payment_identifier)
    return client


def resolve_x402_privy_config(
    *,
    auth_mode: str,
    privy_wallet_id: Optional[str] = None,
    privy_app_id: Optional[str] = None,
    privy_app_secret: Optional[str] = None,
    privy_authorization_signature: Optional[str] = None,
    privy_request_expiry: Optional[str] = None,
    privy_api_url: Optional[str] = None,
    timeout: float = 30.0,
    rpc_url: Optional[str] = None,
) -> Optional[X402PrivyWalletExportConfig]:
    """Resolve runtime Privy wallet export configuration for x402."""

    if auth_mode != "x402_privy":
        return None
    if not (privy_wallet_id and privy_app_id and privy_app_secret):
        return None

    return X402PrivyWalletExportConfig(
        wallet_id=privy_wallet_id,
        app_id=privy_app_id,
        app_secret=privy_app_secret,
        authorization_signature=privy_authorization_signature,
        request_expiry=privy_request_expiry,
        api_url=(privy_api_url or PRIVY_API_URL).rstrip("/"),
        timeout=timeout,
        rpc_url=rpc_url,
    )


def has_x402_auth_config(
    *,
    auth_mode: str,
    privy_wallet_id: Optional[str] = None,
    privy_app_id: Optional[str] = None,
    privy_app_secret: Optional[str] = None,
    privy_authorization_signature: Optional[str] = None,
    privy_request_expiry: Optional[str] = None,
    privy_api_url: Optional[str] = None,
    timeout: float = 30.0,
    rpc_url: Optional[str] = None,
) -> bool:
    """Return whether the auth mode has enough config to resolve a signer."""

    if auth_mode != "x402_privy":
        return False
    return bool(
        resolve_x402_privy_config(
            auth_mode=auth_mode,
            privy_wallet_id=privy_wallet_id,
            privy_app_id=privy_app_id,
            privy_app_secret=privy_app_secret,
            privy_authorization_signature=privy_authorization_signature,
            privy_request_expiry=privy_request_expiry,
            privy_api_url=privy_api_url,
            timeout=timeout,
            rpc_url=rpc_url,
        )
    )


def _normalize_privy_exported_private_key(private_key_bytes: bytes) -> str:
    """Normalize decrypted Privy key material into the base58 form x402 expects."""

    raw_bytes = private_key_bytes.strip()
    if not raw_bytes:
        raise ValueError("Privy wallet export returned empty private key material")

    try:
        candidate = raw_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        candidate = ""

    if candidate:
        try:
            KeypairSigner.from_base58(candidate)
            return candidate
        except Exception:
            logger.debug("Privy export was not a base58 string; encoding raw bytes")

    candidate = based58.b58encode(raw_bytes).decode("ascii")
    KeypairSigner.from_base58(candidate)
    return candidate


async def export_privy_wallet_private_key(config: X402PrivyWalletExportConfig) -> str:
    """Export a Privy wallet private key at runtime and return it as base58."""

    recipient_key_pair = PRIVY_HPKE_CIPHER_SUITE.kem.derive_key_pair(os.urandom(32))
    recipient_public_key = base64.b64encode(
        recipient_key_pair.public_key.to_public_bytes()
    ).decode("ascii")

    headers = {
        "privy-app-id": config.app_id,
        "Content-Type": "application/json",
    }
    if config.authorization_signature:
        headers["privy-authorization-signature"] = config.authorization_signature
    if config.request_expiry:
        headers["privy-request-expiry"] = config.request_expiry

    url = f"{config.api_url}/v1/wallets/{config.wallet_id}/export"
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        response = await client.post(
            url,
            headers=headers,
            auth=(config.app_id, config.app_secret),
            json={
                "encryption_type": "HPKE",
                "recipient_public_key": recipient_public_key,
            },
        )
        response.raise_for_status()
        payload = response.json()

    if payload.get("encryption_type") != "HPKE":
        raise ValueError("Privy wallet export returned an unsupported encryption_type")

    encapsulated_key = payload.get("encapsulated_key")
    ciphertext = payload.get("ciphertext")
    if not encapsulated_key or not ciphertext:
        raise ValueError(
            "Privy wallet export response is missing encapsulated_key or ciphertext"
        )

    decrypted_private_key = PRIVY_HPKE_CIPHER_SUITE.open(
        enc=base64.b64decode(encapsulated_key),
        skr=recipient_key_pair.private_key,
        ct=base64.b64decode(ciphertext),
    )
    return _normalize_privy_exported_private_key(decrypted_private_key)


async def resolve_x402_signing_key(
    *,
    auth_mode: str,
    privy_wallet_id: Optional[str] = None,
    privy_app_id: Optional[str] = None,
    privy_app_secret: Optional[str] = None,
    privy_authorization_signature: Optional[str] = None,
    privy_request_expiry: Optional[str] = None,
    privy_api_url: Optional[str] = None,
    timeout: float = 30.0,
    rpc_url: Optional[str] = None,
) -> Optional[str]:
    """Resolve an x402 signing key from hosted Privy runtime export."""

    if auth_mode != "x402_privy":
        return None

    privy_config = resolve_x402_privy_config(
        auth_mode=auth_mode,
        privy_wallet_id=privy_wallet_id,
        privy_app_id=privy_app_id,
        privy_app_secret=privy_app_secret,
        privy_authorization_signature=privy_authorization_signature,
        privy_request_expiry=privy_request_expiry,
        privy_api_url=privy_api_url,
        timeout=timeout,
        rpc_url=rpc_url,
    )
    if privy_config:
        return await export_privy_wallet_private_key(privy_config)

    return None


async def create_x402_httpx_client_for_auth(
    *,
    auth_mode: str,
    privy_wallet_id: Optional[str] = None,
    privy_app_id: Optional[str] = None,
    privy_app_secret: Optional[str] = None,
    privy_authorization_signature: Optional[str] = None,
    privy_request_expiry: Optional[str] = None,
    privy_api_url: Optional[str] = None,
    timeout: float = 30.0,
    rpc_url: Optional[str] = None,
) -> x402HttpxClient:
    """Create an x402 httpx client for the configured hosted auth flow."""

    resolved_signing_key = await resolve_x402_signing_key(
        auth_mode=auth_mode,
        privy_wallet_id=privy_wallet_id,
        privy_app_id=privy_app_id,
        privy_app_secret=privy_app_secret,
        privy_authorization_signature=privy_authorization_signature,
        privy_request_expiry=privy_request_expiry,
        privy_api_url=privy_api_url,
        timeout=timeout,
        rpc_url=rpc_url,
    )
    if not resolved_signing_key:
        raise ValueError(f"{auth_mode} requires configured x402 signing credentials")

    return create_x402_httpx_client(
        X402SigningKeyConfig(
            signing_key=resolved_signing_key,
            timeout=timeout,
            rpc_url=rpc_url,
        )
    )


def create_x402_httpx_client(config: X402SigningKeyConfig) -> x402HttpxClient:
    """Create an async httpx client with automatic x402 payment handling."""

    client = _configure_x402_client(x402Client())
    signer = KeypairSigner.from_base58(config.signing_key)
    register_exact_svm_client(client, signer=signer, rpc_url=config.rpc_url)
    return x402HttpxClient(client, timeout=config.timeout)


def create_hosted_managed_x402_httpx_client(
    config: X402SigningKeyConfig,
) -> httpx.AsyncClient:
    """Create an x402 client that refreshes idempotency keys on paid retries."""

    client = _configure_x402_client(x402Client())
    signer = KeypairSigner.from_base58(config.signing_key)
    register_exact_svm_client(client, signer=signer, rpc_url=config.rpc_url)
    return HostedManagedX402HttpxClient(client, timeout=config.timeout)


async def request_with_x402_privy(
    method: str,
    url: str,
    *,
    privy_wallet_id: Optional[str] = None,
    privy_app_id: Optional[str] = None,
    privy_app_secret: Optional[str] = None,
    privy_authorization_signature: Optional[str] = None,
    privy_request_expiry: Optional[str] = None,
    privy_api_url: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, Any]] = None,
    json_data: Optional[dict[str, Any]] = None,
    timeout: float = 30.0,
    rpc_url: Optional[str] = None,
) -> Any:
    """Make an x402 request using Privy-backed runtime wallet export."""

    resolved_signing_key = await resolve_x402_signing_key(
        auth_mode="x402_privy",
        privy_wallet_id=privy_wallet_id,
        privy_app_id=privy_app_id,
        privy_app_secret=privy_app_secret,
        privy_authorization_signature=privy_authorization_signature,
        privy_request_expiry=privy_request_expiry,
        privy_api_url=privy_api_url,
        timeout=timeout,
        rpc_url=rpc_url,
    )
    if not resolved_signing_key:
        raise ValueError(
            "Privy x402 requires runtime Privy wallet export config. "
            "Provide privy_wallet_id at runtime and set privy_app_id and privy_app_secret in config."
        )

    return await request_with_x402_signing_key(
        method=method,
        url=url,
        signing_key=resolved_signing_key,
        headers=headers,
        params=params,
        json_data=json_data,
        timeout=timeout,
        rpc_url=rpc_url,
    )


async def request_with_x402_signing_key(
    method: str,
    url: str,
    *,
    signing_key: str,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, Any]] = None,
    json_data: Optional[dict[str, Any]] = None,
    timeout: float = 30.0,
    rpc_url: Optional[str] = None,
) -> Any:
    """Make an HTTP request that automatically settles x402 payment challenges."""

    async with create_x402_httpx_client(
        X402SigningKeyConfig(
            signing_key=signing_key,
            timeout=timeout,
            rpc_url=rpc_url,
        )
    ) as client:
        response = await client.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            json=json_data,
        )
        return response

"""
LLM provider adapters for the Solana Agent system.

These adapters implement the LLMProvider interface for different LLM services.
"""

import asyncio
import base64
import inspect
import io
import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import (
    AsyncGenerator,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Dict,
    Any,
    Union,
)
import httpx
from PIL import Image
import logfire
from openai import APIConnectionError, AsyncOpenAI, OpenAIError
from pydantic import BaseModel
from solders.keypair import Keypair
import tiktoken

from solana_agent.interfaces.providers.llm import LLMProvider
from solana_agent.tools.utils.x402 import (
    X402SigningKeyConfig,
    create_hosted_managed_x402_httpx_client,
    create_x402_httpx_client_for_auth,
    resolve_x402_signing_key,
)

# Setup logger for this module
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_CHAT_MODEL = "gpt-5.4"
DEFAULT_VISION_MODEL = "gpt-5.4"
DEFAULT_PARSE_MODEL = "gpt-5.4"
DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_TTS_MODEL = "tts-1"

# Image constants
SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}
MAX_IMAGE_SIZE_MB = 20
MAX_TOTAL_IMAGE_SIZE_MB = 50
MAX_IMAGE_COUNT = 500
GPT41_PATCH_SIZE = 32
GPT41_MAX_PATCHES = 1536
GPT41_MINI_MULTIPLIER = 1.62
GPT41_NANO_MULTIPLIER = 2.46
DEFAULT_TOKENIZER_MODEL = "gpt-oss-120b"
DEFAULT_CONTEXT_WINDOW_TOKENS = 131000
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_HOSTED_SEARCH_MAX_OUTPUT_TOKENS = 4000
DEFAULT_HOSTED_MANAGED_X402_TIMEOUT_SECONDS = 180.0
DEFAULT_FALLBACK_ENCODING = "o200k_base"
GPT_OSS_FALLBACK_ENCODING = "o200k_harmony"
TOKENS_PER_MESSAGE = 4
TOKENS_PER_NAME = 1
TOKENS_PER_REPLY_PRIMER = 2
SUPPORTED_X402_PREFERRED_ASSETS = frozenset({"USDC"})
SUPPORTED_USAGE_GRANULARITIES = frozenset({"day", "month", "year"})
ACCOUNT_AUTH_TIMEOUT_SECONDS = 30.0
ACCOUNT_AUTH_CACHE_REFRESH_SKEW_SECONDS = 5.0


def _timing_trace_enabled() -> bool:
    return str(os.getenv("SOLANA_AGENT_TIMING_TRACE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tokenizer_candidates(model_name: str | None) -> tuple[str, ...]:
    raw_name = str(model_name or "").strip()
    candidates: list[str] = []
    if raw_name:
        candidates.append(raw_name)
        if "/" in raw_name:
            suffix = raw_name.rsplit("/", 1)[-1].strip()
            if suffix and suffix not in candidates:
                candidates.append(suffix)
    if DEFAULT_TOKENIZER_MODEL not in candidates:
        candidates.append(DEFAULT_TOKENIZER_MODEL)
    return tuple(candidates)


def _fallback_encoding_name(model_name: str | None) -> str:
    normalized = str(model_name or "").strip().lower()
    if "gpt-oss" in normalized:
        return GPT_OSS_FALLBACK_ENCODING
    return DEFAULT_FALLBACK_ENCODING


@lru_cache(maxsize=32)
def _encoding_for_model(model_name: str | None):
    for candidate in _tokenizer_candidates(model_name):
        try:
            return tiktoken.encoding_for_model(candidate)
        except KeyError:
            continue
    return tiktoken.get_encoding(_fallback_encoding_name(model_name))


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                part_type = str(item.get("type") or "").strip().lower()
                if part_type == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        parts.append(text)
                    continue
                parts.append(json.dumps(item, sort_keys=True, separators=(",", ":")))
                continue
            if item is not None:
                parts.append(str(item).strip())
        return "\n".join(part for part in parts if part).strip()
    if content is None:
        return ""
    if isinstance(content, dict):
        return json.dumps(content, sort_keys=True, separators=(",", ":"))
    return str(content).strip()


class OpenAIAdapter(LLMProvider):
    """OpenAI implementation of LLMProvider using the Responses API."""

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        logfire_api_key: Optional[str] = None,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        auth_mode: str = "api_key",
        private_key: Optional[str] = None,
        privy_user_id: Optional[str] = None,
        privy_app_id: Optional[str] = None,
        privy_app_secret: Optional[str] = None,
        privy_authorization_signature: Optional[str] = None,
        privy_request_expiry: Optional[str] = None,
        privy_api_url: Optional[str] = None,
        x402_rpc_url: Optional[str] = None,
        x402_preferred_asset: Optional[str] = None,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        tokenizer_model: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.auth_mode = auth_mode
        self.private_key = private_key
        self.privy_user_id = str(privy_user_id or "").strip()
        self.privy_app_id = privy_app_id
        self.privy_app_secret = privy_app_secret
        self.privy_authorization_signature = privy_authorization_signature
        self.privy_request_expiry = privy_request_expiry
        self.privy_api_url = privy_api_url
        self.x402_rpc_url = x402_rpc_url
        self.x402_preferred_asset = self._normalize_x402_preferred_asset(
            x402_preferred_asset
        )
        self.context_window_tokens = max(1, int(context_window_tokens))
        self.max_output_tokens = max(1, int(max_output_tokens))
        resolved_tokenizer_model = str(
            tokenizer_model or model or DEFAULT_TOKENIZER_MODEL
        ).strip()
        self.tokenizer_model = resolved_tokenizer_model or DEFAULT_TOKENIZER_MODEL
        self._is_openai_endpoint = base_url is None or "api.openai.com" in base_url
        self._client_lock = asyncio.Lock()
        self._privy_clients: Dict[str, AsyncOpenAI] = {}
        self._hosted_managed_clients: Dict[str, AsyncOpenAI] = {}
        self._hosted_managed_signing_key_cache: Dict[str, str] = {}
        self._account_auth_headers_cache: Dict[
            str, tuple[datetime, Dict[str, str]]
        ] = {}
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self.client: Optional[AsyncOpenAI] = None
        if self.auth_mode != "x402_privy" and (
            self.auth_mode != "hosted_managed"
            or self._uses_hosted_managed_local_x402_settlement(base_url)
        ):
            self.client = self._create_client(api_key=api_key, base_url=base_url)

        self.logfire = False
        if logfire_api_key:
            try:
                if self._is_openai_endpoint:
                    logfire.configure(token=logfire_api_key)
                    self.logfire = True
                    # Instrument the main client immediately after configuring logfire
                    if self.client is not None:
                        logfire.instrument_openai(self.client)
                        logger.info(
                            "Logfire configured and OpenAI client instrumented successfully."
                        )
                else:
                    logger.warning(
                        "Logfire OpenAI instrumentation disabled for non-OpenAI base_url."
                    )
            except Exception as e:
                logger.error(f"Failed to configure Logfire: {e}")
                self.logfire = False

        # Use provided model or defaults
        if model:
            self.parse_model = model
            self.text_model = model
            self.vision_model = model
        else:
            # Use OpenAI defaults
            self.parse_model = DEFAULT_PARSE_MODEL
            self.text_model = DEFAULT_CHAT_MODEL
            self.vision_model = DEFAULT_VISION_MODEL

        # OpenAI-specific models
        self.transcription_model = DEFAULT_TRANSCRIPTION_MODEL
        self.tts_model = DEFAULT_TTS_MODEL

    def _create_client(
        self,
        api_key: str,
        base_url: Optional[str] = None,
    ) -> AsyncOpenAI:
        """Create an OpenAI-compatible client for the configured auth mode."""

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        if self._uses_hosted_managed_local_x402_settlement(base_url):
            client_kwargs["api_key"] = api_key or "x402"
            client_kwargs["max_retries"] = 0
            client_kwargs["http_client"] = create_hosted_managed_x402_httpx_client(
                X402SigningKeyConfig(
                    signing_key=str(self.private_key or "").strip(),
                    timeout=DEFAULT_HOSTED_MANAGED_X402_TIMEOUT_SECONDS,
                    rpc_url=self.x402_rpc_url,
                )
            )
            return AsyncOpenAI(**client_kwargs)

        return AsyncOpenAI(**client_kwargs)

    def _uses_hosted_managed_local_x402_settlement(
        self,
        base_url: Optional[str] = None,
    ) -> bool:
        resolved_base_url = base_url if base_url is not None else self.base_url
        return bool(
            self.auth_mode == "hosted_managed"
            and str(self.private_key or "").strip()
            and resolved_base_url
            and not self._is_openai_endpoint
        )

    def _runtime_privy_wallet_id(
        self, runtime_context: Optional[Dict[str, Any]] = None
    ) -> str:
        context = dict(runtime_context or {})
        for context_key in ("privy_wallet_id", "hosted_privy_wallet_id"):
            wallet_id = str(context.get(context_key) or "").strip()
            if wallet_id:
                return wallet_id
        wallet_payload = context.get("privy_wallet")
        if isinstance(wallet_payload, dict):
            return str(
                wallet_payload.get("wallet_id") or wallet_payload.get("id") or ""
            ).strip()
        return ""

    def _uses_hosted_managed_wallet_x402_settlement(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
    ) -> bool:
        resolved_base_url = base_url if base_url is not None else self.base_url
        return bool(
            self.auth_mode == "hosted_managed"
            and not str(self.private_key or "").strip()
            and resolved_base_url
            and not self._is_openai_endpoint
            and self._runtime_privy_wallet_id(runtime_context)
            and str(self.privy_user_id or "").strip()
        )

    def _resolve_runtime_privy_wallet_id(
        self, runtime_context: Optional[Dict[str, Any]] = None
    ) -> str:
        wallet_id = self._runtime_privy_wallet_id(runtime_context)
        if not wallet_id:
            raise ValueError(
                "x402_privy requires a runtime wallet id via runtime_context.privy_wallet_id, "
                "runtime_context.hosted_privy_wallet_id, or runtime_context.privy_wallet.id"
            )
        return wallet_id

    @staticmethod
    def _current_running_loop() -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    async def _close_cached_client(self, client: Optional[AsyncOpenAI]) -> None:
        if client is None:
            return

        close_method = getattr(client, "close", None)
        if callable(close_method):
            try:
                result = close_method()
                if inspect.isawaitable(result):
                    await result
                return
            except Exception:
                logger.debug(
                    "Failed to close cached OpenAI client cleanly.", exc_info=True
                )
                return

        http_client = getattr(client, "_client", None)
        aclose_method = getattr(http_client, "aclose", None)
        if callable(aclose_method):
            try:
                await aclose_method()
            except Exception:
                logger.debug(
                    "Failed to close cached underlying http client.", exc_info=True
                )

    async def _resolve_hosted_managed_signing_key(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
        *,
        prefer_runtime_wallet: bool = False,
    ) -> str:
        wallet_id = self._runtime_privy_wallet_id(runtime_context)
        if prefer_runtime_wallet and wallet_id:
            cached_private_key = self._hosted_managed_signing_key_cache.get(wallet_id)
            if cached_private_key:
                return cached_private_key
            privy_user_id = str(self.privy_user_id or "").strip()
            if not privy_user_id:
                raise ValueError(
                    "hosted_managed x402 settlement requires config.ai.privy_user_id"
                )

            exported = await self.export_wallet_private_key(
                privy_user_id=privy_user_id,
                wallet_id=wallet_id,
            )
            resolved_private_key = str(exported.get("private_key") or "").strip()
            if not resolved_private_key:
                raise ValueError(
                    "Hosted wallet export response is missing a private_key"
                )
            self._hosted_managed_signing_key_cache[wallet_id] = resolved_private_key
            return resolved_private_key

        configured_private_key = str(self.private_key or "").strip()
        if configured_private_key:
            return configured_private_key

        privy_user_id = str(self.privy_user_id or "").strip()
        if not privy_user_id:
            raise ValueError(
                "hosted_managed x402 settlement requires config.ai.privy_user_id"
            )

        if not wallet_id:
            raise ValueError(
                "hosted_managed x402 settlement requires runtime_context.privy_wallet_id or hosted_privy_wallet_id"
            )

        cached_private_key = self._hosted_managed_signing_key_cache.get(wallet_id)
        if cached_private_key:
            return cached_private_key

        exported = await self.export_wallet_private_key(
            privy_user_id=privy_user_id,
            wallet_id=wallet_id,
        )
        resolved_private_key = str(exported.get("private_key") or "").strip()
        if not resolved_private_key:
            raise ValueError("Hosted wallet export response is missing a private_key")
        self._hosted_managed_signing_key_cache[wallet_id] = resolved_private_key
        return resolved_private_key

    def _get_cached_account_auth_headers(
        self,
        wallet: str,
    ) -> Optional[Dict[str, str]]:
        cache_entry = self._account_auth_headers_cache.get(wallet)
        if cache_entry is None:
            return None
        expires_at, headers = cache_entry
        refresh_deadline = datetime.now(timezone.utc) + timedelta(
            seconds=ACCOUNT_AUTH_CACHE_REFRESH_SKEW_SECONDS
        )
        if expires_at <= refresh_deadline:
            self._account_auth_headers_cache.pop(wallet, None)
            return None
        return dict(headers)

    def _cache_account_auth_headers(
        self,
        wallet: str,
        headers: Dict[str, str],
        *,
        expires_at: datetime,
    ) -> None:
        self._account_auth_headers_cache[wallet] = (expires_at, dict(headers))

    async def _get_client(
        self, runtime_context: Optional[Dict[str, Any]] = None
    ) -> AsyncOpenAI:
        """Return the configured client, creating Privy-backed clients lazily."""

        current_loop = self._current_running_loop()

        if self._uses_hosted_managed_wallet_x402_settlement(
            runtime_context,
            self.base_url,
        ):
            wallet_id = self._runtime_privy_wallet_id(runtime_context)
            cached_client = self._hosted_managed_clients.get(wallet_id)
            if cached_client is not None:
                return cached_client

        if (
            self.auth_mode != "x402_privy"
            and not self._uses_hosted_managed_wallet_x402_settlement(
                runtime_context,
                self.base_url,
            )
        ):
            if self.client is not None:
                if self._client_loop is None:
                    self._client_loop = current_loop
                    return self.client
                if current_loop is self._client_loop:
                    return self.client

                stale_client = self.client
                self.client = self._create_client(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                self._client_loop = current_loop
                await self._close_cached_client(stale_client)
                return self.client

            self.client = self._create_client(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self._client_loop = current_loop
            return self.client

        if self.auth_mode == "x402_privy":
            wallet_id = self._resolve_runtime_privy_wallet_id(runtime_context)
            cached_client = self._privy_clients.get(wallet_id)
            if cached_client is not None:
                return cached_client

        async with self._client_lock:
            if self._uses_hosted_managed_wallet_x402_settlement(
                runtime_context,
                self.base_url,
            ):
                wallet_id = self._runtime_privy_wallet_id(runtime_context)
                cached_client = self._hosted_managed_clients.get(wallet_id)
                if cached_client is not None:
                    return cached_client
                client = await self._create_async_client(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    runtime_context=runtime_context,
                )
                self._hosted_managed_clients[wallet_id] = client
                return client
            if self.auth_mode != "x402_privy" and self.client is not None:
                return self.client
            if self.auth_mode == "x402_privy":
                wallet_id = self._resolve_runtime_privy_wallet_id(runtime_context)
                cached_client = self._privy_clients.get(wallet_id)
                if cached_client is not None:
                    return cached_client
                client = await self._create_async_client(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    runtime_context=runtime_context,
                )
                self._privy_clients[wallet_id] = client
                return client

            self.client = await self._create_async_client(
                api_key=self.api_key,
                base_url=self.base_url,
                runtime_context=runtime_context,
            )
            return self.client

    async def _create_async_client(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> AsyncOpenAI:
        """Create an OpenAI-compatible client for async-only auth flows."""

        if self._uses_hosted_managed_wallet_x402_settlement(
            runtime_context,
            base_url,
        ):
            if not base_url:
                raise ValueError(f"{self.auth_mode} requires a configured base_url")

            signing_key = await self._resolve_hosted_managed_signing_key(
                runtime_context
            )
            http_client = create_hosted_managed_x402_httpx_client(
                X402SigningKeyConfig(
                    signing_key=signing_key,
                    timeout=DEFAULT_HOSTED_MANAGED_X402_TIMEOUT_SECONDS,
                    rpc_url=self.x402_rpc_url,
                )
            )
            return AsyncOpenAI(
                api_key=api_key or "x402",
                base_url=base_url,
                max_retries=0,
                http_client=http_client,
            )

        if self.auth_mode != "x402_privy":
            return self._create_client(api_key=api_key, base_url=base_url)

        if not base_url:
            raise ValueError(f"{self.auth_mode} requires a configured base_url")

        http_client = await create_x402_httpx_client_for_auth(
            auth_mode=self.auth_mode,
            privy_wallet_id=self._resolve_runtime_privy_wallet_id(runtime_context),
            privy_app_id=self.privy_app_id,
            privy_app_secret=self.privy_app_secret,
            privy_authorization_signature=self.privy_authorization_signature,
            privy_request_expiry=self.privy_request_expiry,
            privy_api_url=self.privy_api_url,
            rpc_url=self.x402_rpc_url,
        )
        return AsyncOpenAI(
            api_key=api_key or "x402",
            base_url=base_url,
            http_client=http_client,
        )

    async def _refresh_cached_client(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> AsyncOpenAI:
        if self._uses_hosted_managed_wallet_x402_settlement(
            runtime_context,
            self.base_url,
        ):
            wallet_id = self._runtime_privy_wallet_id(runtime_context)
            stale_client = self._hosted_managed_clients.pop(wallet_id, None)
            await self._close_cached_client(stale_client)
            client = await self._create_async_client(
                api_key=self.api_key,
                base_url=self.base_url,
                runtime_context=runtime_context,
            )
            self._hosted_managed_clients[wallet_id] = client
            return client

        if self.auth_mode == "x402_privy":
            wallet_id = self._resolve_runtime_privy_wallet_id(runtime_context)
            stale_client = self._privy_clients.pop(wallet_id, None)
            await self._close_cached_client(stale_client)
            client = await self._create_async_client(
                api_key=self.api_key,
                base_url=self.base_url,
                runtime_context=runtime_context,
            )
            self._privy_clients[wallet_id] = client
            return client

        stale_client = self.client
        self.client = self._create_client(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self._client_loop = self._current_running_loop()
        await self._close_cached_client(stale_client)
        return self.client

    async def _chat_completions_create(
        self,
        *,
        runtime_context: Optional[Dict[str, Any]] = None,
        **cc_params: Any,
    ) -> Any:
        last_error: APIConnectionError | None = None
        for attempt in range(2):
            started_at = time.perf_counter()
            client = await self._get_client(runtime_context)
            self._instrument_client(client)
            try:
                response = await client.chat.completions.create(**cc_params)
                if _timing_trace_enabled():
                    logger.warning(
                        "SDK timing: hosted_completion seconds=%.3f model=%s stream=%s attempt=%d",
                        time.perf_counter() - started_at,
                        cc_params.get("model"),
                        bool(cc_params.get("stream")),
                        attempt + 1,
                    )
                return response
            except APIConnectionError as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning(
                        "Retrying chat completions request after connection error with a fresh client."
                    )
                    await self._refresh_cached_client(runtime_context)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("chat completions retry exhausted without an error")

    def _resolve_tokenizer_model(self, model: Optional[str] = None) -> str:
        candidate = str(
            model or self.tokenizer_model or DEFAULT_TOKENIZER_MODEL
        ).strip()
        return candidate or DEFAULT_TOKENIZER_MODEL

    def _estimate_text_tokens(self, text: str, model: Optional[str] = None) -> int:
        raw_text = str(text or "")
        if not raw_text:
            return 0
        return len(
            _encoding_for_model(self._resolve_tokenizer_model(model)).encode(raw_text)
        )

    def _estimate_chat_completion_input_tokens(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> int:
        total = 0
        for message in messages:
            total += TOKENS_PER_MESSAGE
            total += self._estimate_text_tokens(
                _flatten_message_content(message.get("content")),
                model,
            )
            name = str(message.get("name") or "").strip()
            if name:
                total += TOKENS_PER_NAME
                total += self._estimate_text_tokens(name, model)
        return max(1, total + TOKENS_PER_REPLY_PRIMER)

    def _resolve_chat_completion_max_tokens(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> int:
        prompt_tokens = self._estimate_chat_completion_input_tokens(messages, model)
        remaining_tokens = self.context_window_tokens - prompt_tokens
        if remaining_tokens <= 0:
            raise ValueError(
                "Prompt exceeds the configured context window before output tokens are reserved. "
                f"prompt_tokens={prompt_tokens}, context_window_tokens={self.context_window_tokens}."
            )
        max_tokens = min(self.max_output_tokens, remaining_tokens)
        if max_tokens < self.max_output_tokens:
            logger.debug(
                "Clamped chat completion max_tokens from %s to %s for model %s.",
                self.max_output_tokens,
                max_tokens,
                model or self.text_model,
            )
        if self.base_url and self._hosted_search_enabled(runtime_context):
            max_tokens = min(max_tokens, DEFAULT_HOSTED_SEARCH_MAX_OUTPUT_TOKENS)
        return max_tokens

    def _chat_completion_request_options(self) -> Dict[str, Any]:
        return {"extra_headers": {"Idempotency-Key": uuid.uuid4().hex}}

    def _normalize_x402_preferred_asset(
        self, preferred_asset: Optional[str]
    ) -> Optional[str]:
        normalized = str(preferred_asset or "").strip().upper()
        if not normalized:
            return None
        if normalized not in SUPPORTED_X402_PREFERRED_ASSETS:
            supported_assets = ", ".join(sorted(SUPPORTED_X402_PREFERRED_ASSETS))
            raise ValueError(f"x402_preferred_asset must be one of: {supported_assets}")
        return normalized

    def _resolve_x402_preferred_asset(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        context = dict(runtime_context or {})
        if "x402_preferred_asset" in context:
            return self._normalize_x402_preferred_asset(
                context.get("x402_preferred_asset")
            )
        return self.x402_preferred_asset

    def _require_account_reporting_auth(self) -> None:
        if self.auth_mode not in {"x402_privy", "hosted_managed"}:
            raise NotImplementedError(
                "Account reporting requires hosted_managed or x402_privy auth"
            )
        if not self.base_url:
            raise ValueError("Account reporting requires a configured base_url")

    def _require_hosted_base_url(self, capability: str) -> None:
        if not self.base_url:
            raise ValueError(f"{capability} requires a configured base_url")

    def _account_endpoint_url(self, endpoint: str) -> str:
        self._require_account_reporting_auth()
        normalized_base_url = str(self.base_url or "").rstrip("/")
        normalized_endpoint = str(endpoint or "").strip().lstrip("/")
        return f"{normalized_base_url}/account/{normalized_endpoint}"

    def _hosted_endpoint_url(self, endpoint: str, *, capability: str) -> str:
        self._require_hosted_base_url(capability)
        normalized_base_url = str(self.base_url or "").rstrip("/")
        normalized_endpoint = str(endpoint or "").strip().lstrip("/")
        return f"{normalized_base_url}/{normalized_endpoint}"

    async def _resolve_account_auth_keypair(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Keypair:
        self._require_account_reporting_auth()
        if self.auth_mode == "hosted_managed":
            resolved_private_key = await self._resolve_hosted_managed_signing_key(
                runtime_context,
                prefer_runtime_wallet=True,
            )
        else:
            privy_wallet_id = self._resolve_runtime_privy_wallet_id(runtime_context)
            resolved_private_key = await resolve_x402_signing_key(
                auth_mode=self.auth_mode,
                privy_wallet_id=privy_wallet_id,
                privy_app_id=self.privy_app_id,
                privy_app_secret=self.privy_app_secret,
                privy_authorization_signature=self.privy_authorization_signature,
                privy_request_expiry=self.privy_request_expiry,
                privy_api_url=self.privy_api_url,
                timeout=ACCOUNT_AUTH_TIMEOUT_SECONDS,
                rpc_url=self.x402_rpc_url,
            )
        if not resolved_private_key:
            raise ValueError(
                f"{self.auth_mode} requires configured x402 signing credentials"
            )
        return Keypair.from_base58_string(resolved_private_key)

    async def _create_account_auth_challenge(
        self,
        wallet: str,
    ) -> Dict[str, Any]:
        challenge_url = self._account_endpoint_url("auth/challenge")
        async with httpx.AsyncClient(timeout=ACCOUNT_AUTH_TIMEOUT_SECONDS) as client:
            response = await client.post(challenge_url, json={"wallet": wallet})
            response.raise_for_status()
            return response.json()

    async def _build_account_auth_headers(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        keypair = await self._resolve_account_auth_keypair(runtime_context)
        wallet = str(keypair.pubkey())
        cached_headers = self._get_cached_account_auth_headers(wallet)
        if cached_headers is not None:
            return cached_headers
        challenge = await self._create_account_auth_challenge(wallet)

        challenge_id = str(challenge.get("challenge_id") or "").strip()
        message = str(challenge.get("message") or "")
        if not challenge_id or not message:
            raise ValueError(
                "Account auth challenge response is missing challenge_id or message"
            )

        challenge_wallet = str(challenge.get("wallet") or wallet).strip()
        if challenge_wallet and challenge_wallet != wallet:
            raise ValueError("Account auth challenge wallet mismatch")

        expires_at_raw = str(challenge.get("expires_at") or "").strip()
        expires_at: datetime | None = None
        if expires_at_raw:
            normalized_expires_at = expires_at_raw.replace("Z", "+00:00")
            expires_at = datetime.fromisoformat(normalized_expires_at)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= datetime.now(timezone.utc):
                raise ValueError("Account auth challenge already expired")

        signature = str(keypair.sign_message(message.encode("utf-8")))
        headers = {
            "X-Wallet-Address": wallet,
            "X-Account-Challenge-Id": challenge_id,
            "X-Account-Signature": signature,
        }
        if expires_at is not None:
            self._cache_account_auth_headers(wallet, headers, expires_at=expires_at)
        return headers

    def _managed_account_identity_params(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = dict(runtime_context or {})
        params: Dict[str, Any] = {}
        if "privy_user_id" in context:
            raise ValueError(
                "privy_user_id belongs in config.ai.privy_user_id, not runtime_context"
            )

        privy_user_id = str(self.privy_user_id or "").strip()
        if privy_user_id:
            params["privy_user_id"] = privy_user_id

        if params:
            return params

        raise ValueError(
            "Hosted account reporting requires config.ai.privy_user_id or runtime_context with privy_user_id."
        )

    async def _hosted_chat_completion_request_options(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        options = dict(self._hosted_chat_completion_extensions(runtime_context))
        headers = dict(
            self._chat_completion_request_options().get("extra_headers") or {}
        )
        if self.auth_mode in {"x402_privy", "hosted_managed"} and self.base_url:
            headers.update(await self._build_account_auth_headers(runtime_context))
        if headers:
            options["extra_headers"] = headers
        return options

    async def _request_account_json(
        self,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self._account_endpoint_url(endpoint)
        request_params = dict(params or {})
        headers: Optional[Dict[str, str]] = None
        if self.auth_mode == "x402_privy":
            headers = await self._build_account_auth_headers(runtime_context)
        else:
            request_params.update(
                self._managed_account_identity_params(runtime_context)
            )
        async with httpx.AsyncClient(timeout=ACCOUNT_AUTH_TIMEOUT_SECONDS) as client:
            if headers is None:
                response = await client.get(url, params=request_params or None)
            else:
                response = await client.get(
                    url,
                    headers=headers,
                    params=request_params or None,
                )
            response.raise_for_status()
            return response.json()

    async def _request_hosted_json(
        self,
        endpoint: str,
        *,
        method: str = "get",
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        capability: str,
    ) -> Dict[str, Any]:
        url = self._hosted_endpoint_url(endpoint, capability=capability)
        async with httpx.AsyncClient(timeout=ACCOUNT_AUTH_TIMEOUT_SECONDS) as client:
            if method == "get":
                response = await client.get(url, params=params)
            elif method == "post":
                response = await client.post(url, json=json_body)
            else:  # pragma: no cover
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            return response.json()

    def _hosted_search_enabled(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.text_model.startswith("solana-agent-"):
            return False

        value = (runtime_context or {}).get("search_enabled")
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", ""}:
                return False
        return bool(value)

    def _hosted_chat_completions_require_non_streaming(
        self,
        model: Optional[str] = None,
    ) -> bool:
        if not self.base_url:
            return False

        requested_model = str(model or self.text_model or "").strip().lower()
        if requested_model.startswith("solana-agent-"):
            return True

        return self.auth_mode == "x402_privy"

    def _chat_completion_response_events(self, response: Any) -> list[Dict[str, Any]]:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return [{"type": "message_end", "finish_reason": "stop"}]

        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        events: list[Dict[str, Any]] = []

        if isinstance(content, str) and content:
            events.append({"type": "content", "delta": content})

        tool_calls = getattr(message, "tool_calls", None) or []
        for index, tool_call in enumerate(tool_calls):
            function = getattr(tool_call, "function", None)
            events.append(
                {
                    "type": "tool_call_delta",
                    "id": getattr(tool_call, "id", None),
                    "index": index,
                    "name": getattr(function, "name", None) if function else None,
                    "arguments_delta": getattr(function, "arguments", "")
                    if function
                    else "",
                }
            )

        events.append(
            {
                "type": "message_end",
                "finish_reason": getattr(choice, "finish_reason", None) or "stop",
            }
        )
        return events

    def _hosted_chat_completion_extensions(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if (
            not self.text_model.startswith("solana-agent-")
            and self.auth_mode != "x402_privy"
        ):
            return {}

        context = dict(runtime_context or {})
        extensions: Dict[str, Any] = {}
        if "privy_user_id" in context:
            raise ValueError(
                "privy_user_id belongs in config.ai.privy_user_id, not runtime_context"
            )

        conversation_id = str(context.get("conversation_id") or "").strip()
        if conversation_id:
            extensions["conversation_id"] = conversation_id

        for key in (
            "privy_wallet_id",
            "privy_wallet_public_key",
            "privy_wallet_address",
        ):
            value = str(context.get(key) or "").strip()
            if value:
                extensions[key] = value

        configured_privy_user_id = str(self.privy_user_id or "").strip()
        if configured_privy_user_id:
            extensions["privy_user_id"] = configured_privy_user_id

        memory_ttl_tier = str(context.get("memory_ttl_tier") or "").strip()
        if memory_ttl_tier:
            extensions["memory_ttl_tier"] = memory_ttl_tier

        service_tier = str(context.get("service_tier") or "").strip().lower()
        if service_tier:
            if service_tier not in {"standard", "priority"}:
                raise ValueError("service_tier must be one of: standard, priority")
            extensions["service_tier"] = service_tier

        preferred_asset = self._resolve_x402_preferred_asset(context)
        if preferred_asset:
            extensions["x402_preferred_asset"] = preferred_asset

        if self._hosted_search_enabled(context):
            extensions["search_enabled"] = True

        max_tool_iterations = context.get("max_tool_iterations")
        if max_tool_iterations is not None:
            try:
                extensions["max_tool_iterations"] = int(max_tool_iterations)
            except (TypeError, ValueError):
                pass

        request_timeout_seconds = context.get("request_timeout_seconds")
        if request_timeout_seconds is not None:
            try:
                extensions["request_timeout_seconds"] = float(request_timeout_seconds)
            except (TypeError, ValueError):
                pass

        if bool(context.get("_raise_stream_errors")):
            extensions["raise_stream_errors"] = True

        if not extensions:
            return {}

        return {"extra_body": extensions}

    async def get_account_summary(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the hosted account summary for the authenticated wallet."""
        return await self._request_account_json(
            "summary",
            runtime_context=runtime_context,
        )

    async def create_privy_user(self) -> Dict[str, Any]:
        """Create a hosted Privy user and return its DID."""
        return await self._request_hosted_json(
            "account/user",
            method="post",
            json_body={},
            capability="Hosted wallet management",
        )

    async def create_wallet(
        self,
        *,
        privy_user_id: str,
        chain_type: str = "solana",
    ) -> Dict[str, Any]:
        """Create or return the hosted Privy wallet for a user."""
        normalized_privy_user_id = str(privy_user_id or "").strip()
        if not normalized_privy_user_id:
            raise ValueError("privy_user_id is required")

        normalized_chain_type = str(chain_type or "solana").strip().lower() or "solana"
        if normalized_chain_type not in {"ethereum", "solana"}:
            raise ValueError("chain_type must be one of: ethereum, solana")

        return await self._request_hosted_json(
            "account/wallet",
            method="post",
            json_body={
                "privy_user_id": normalized_privy_user_id,
                "chain_type": normalized_chain_type,
            },
            capability="Hosted wallet management",
        )

    async def rotate_wallet(
        self,
        *,
        privy_user_id: str,
        chain_type: str = "solana",
    ) -> Dict[str, Any]:
        """Rotate the hosted Privy wallet for a user."""
        normalized_privy_user_id = str(privy_user_id or "").strip()
        if not normalized_privy_user_id:
            raise ValueError("privy_user_id is required")

        normalized_chain_type = str(chain_type or "solana").strip().lower() or "solana"
        if normalized_chain_type not in {"ethereum", "solana"}:
            raise ValueError("chain_type must be one of: ethereum, solana")

        return await self._request_hosted_json(
            "account/wallet/rotate",
            method="post",
            json_body={
                "privy_user_id": normalized_privy_user_id,
                "chain_type": normalized_chain_type,
            },
            capability="Hosted wallet management",
        )

    async def export_wallet_private_key(
        self,
        *,
        privy_user_id: str,
        wallet_id: Optional[str] = None,
        chain_type: str = "solana",
    ) -> Dict[str, Any]:
        """Export a hosted Privy wallet private key for self-custody."""
        normalized_privy_user_id = str(privy_user_id or "").strip()
        if not normalized_privy_user_id:
            raise ValueError("privy_user_id is required")

        normalized_chain_type = str(chain_type or "solana").strip().lower() or "solana"
        if normalized_chain_type not in {"ethereum", "solana"}:
            raise ValueError("chain_type must be one of: ethereum, solana")

        json_body: Dict[str, Any] = {
            "privy_user_id": normalized_privy_user_id,
            "chain_type": normalized_chain_type,
            "confirm_export": True,
        }
        normalized_wallet_id = str(wallet_id or "").strip()
        if normalized_wallet_id:
            json_body["wallet_id"] = normalized_wallet_id

        return await self._request_hosted_json(
            "account/wallet/export",
            method="post",
            json_body=json_body,
            capability="Hosted wallet management",
        )

    async def get_wallet_address(self, *, wallet_id: str) -> Dict[str, Any]:
        """Return the hosted Privy wallet address payload for a wallet id."""
        normalized_wallet_id = str(wallet_id or "").strip()
        if not normalized_wallet_id:
            raise ValueError("wallet_id is required")

        return await self._request_hosted_json(
            "account/wallet/address",
            params={"wallet_id": normalized_wallet_id},
            capability="Hosted wallet management",
        )

    async def get_usage_report(
        self,
        granularity: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        group_by: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return hosted usage buckets for the authenticated wallet."""
        normalized_granularity = str(granularity or "").strip().lower()
        if normalized_granularity not in SUPPORTED_USAGE_GRANULARITIES:
            supported_granularities = ", ".join(sorted(SUPPORTED_USAGE_GRANULARITIES))
            raise ValueError(f"granularity must be one of: {supported_granularities}")

        params: Dict[str, Any] = {"granularity": normalized_granularity}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if group_by:
            params["group_by"] = group_by

        return await self._request_account_json(
            "usage",
            params=params,
            runtime_context=runtime_context,
        )

    async def get_usage_forecast(
        self,
        window_days: int = 30,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the hosted usage forecast for the authenticated wallet."""
        normalized_window_days = int(window_days)
        if normalized_window_days <= 0:
            raise ValueError("window_days must be a positive integer")

        return await self._request_account_json(
            "forecast",
            params={"window_days": normalized_window_days},
            runtime_context=runtime_context,
        )

    async def get_pricing_info(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the hosted pricing view for the authenticated wallet."""
        return await self._request_account_json(
            "pricing",
            runtime_context=runtime_context,
        )

    def _instrument_client(self, client: AsyncOpenAI) -> None:
        """Instrument a lazily created client if logfire is enabled."""

        if self.logfire:
            logfire.instrument_openai(client)

    def _supports_responses_api(self) -> bool:
        """Check if the configured endpoint supports the OpenAI Responses API.

        OpenAI and Groq support the Responses API. Cerebras and other
        providers only support the Chat Completions API.
        """
        if self._is_openai_endpoint:
            return True
        if self.base_url and "api.groq.com" in self.base_url:
            return True
        return False

    def get_api_key(self) -> Optional[str]:  # pragma: no cover
        """Return the API key used to configure the OpenAI client."""
        return getattr(self, "api_key", None)

    async def tts(
        self,
        text: str,
        instructions: str = "You speak in a friendly and helpful manner.",
        voice: Literal[
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
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "aac",
    ) -> AsyncGenerator[bytes, None]:  # pragma: no cover
        """Stream text-to-speech audio from OpenAI models.

        Args:
            text: Text to convert to speech
            instructions: Not used in this implementation
            voice: Voice to use for synthesis
            response_format: Audio format

        Yields:
            Audio bytes as they become available
        """
        try:
            client = await self._get_client()
            if self.logfire:  # Instrument only if logfire is enabled
                logfire.instrument_openai(client)
            async with client.audio.speech.with_streaming_response.create(
                model=self.tts_model,
                voice=voice,
                input=text,
                response_format=response_format,
            ) as stream:
                # Stream the bytes in 16KB chunks
                async for chunk in stream.iter_bytes(chunk_size=1024 * 16):
                    yield chunk

        except Exception as e:
            # Log the exception with traceback
            logger.exception(f"Error in text_to_speech: {e}")
            yield b""  # Return empty bytes on error

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
    ) -> AsyncGenerator[str, None]:  # pragma: no cover
        """Stream transcription of an audio file.

        Args:
            audio_bytes: Audio file bytes
            input_format: Format of the input audio file

        Yields:
            Transcript text chunks as they become available
        """
        try:
            client = await self._get_client()
            if self.logfire:  # Instrument only if logfire is enabled
                logfire.instrument_openai(client)
            async with client.audio.transcriptions.with_streaming_response.create(
                model=self.transcription_model,
                file=(f"file.{input_format}", audio_bytes),
                response_format="text",
            ) as stream:
                # Stream the text in 16KB chunks
                async for chunk in stream.iter_text(chunk_size=1024 * 16):
                    yield chunk

        except Exception as e:
            # Log the exception with traceback
            logger.exception(f"Error in transcribe_audio: {e}")
            yield f"I apologize, but I encountered an error transcribing the audio: {str(e)}"

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Any:  # pragma: no cover
        """Generate text using Responses API or Chat Completions API."""
        try:
            if self._supports_responses_api():
                client = await self._get_client(runtime_context)
                self._instrument_client(client)
                request_params: Dict[str, Any] = {
                    "model": model or self.text_model,
                    "input": prompt,
                }

                if self.reasoning_effort and not self.base_url:
                    request_params["reasoning"] = {"effort": self.reasoning_effort}

                if system_prompt:
                    request_params["instructions"] = system_prompt

                if tools:
                    responses_tools = []
                    for tool in tools:
                        if tool.get("type") == "function":
                            func = tool.get("function", {})
                            responses_tools.append(
                                {
                                    "type": "function",
                                    "name": func.get("name"),
                                    "description": func.get("description", ""),
                                    "parameters": func.get("parameters", {}),
                                }
                            )
                        else:
                            responses_tools.append(tool)
                    request_params["tools"] = responses_tools

                response = await client.responses.create(**request_params)
                return response

            # Chat Completions API path (e.g., Cerebras)
            messages: List[Dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            cc_params: Dict[str, Any] = {
                "model": model or self.text_model,
                "messages": messages,
                "max_tokens": self._resolve_chat_completion_max_tokens(
                    messages,
                    model or self.text_model,
                    runtime_context,
                ),
                **await self._hosted_chat_completion_request_options(runtime_context),
            }
            if tools:
                cc_params["tools"] = tools

            response = await self._chat_completions_create(
                runtime_context=runtime_context,
                **cc_params,
            )
            return response
        except OpenAIError as e:
            logger.error(f"OpenAI API error during text generation: {e}")
            return None
        except Exception as e:
            logger.exception(f"Error in generate_text: {e}")
            return None

    def _calculate_gpt41_image_cost(self, width: int, height: int, model: str) -> int:
        """Calculates the token cost for an image with GPT-4.1 models."""
        patches_wide = math.ceil(width / GPT41_PATCH_SIZE)
        patches_high = math.ceil(height / GPT41_PATCH_SIZE)
        total_patches_needed = patches_wide * patches_high

        if total_patches_needed > GPT41_MAX_PATCHES:
            scale_factor = math.sqrt(GPT41_MAX_PATCHES / total_patches_needed)
            new_width = math.floor(width * scale_factor)
            new_height = math.floor(height * scale_factor)

            final_patches_wide_scaled = math.ceil(new_width / GPT41_PATCH_SIZE)
            final_patches_high_scaled = math.ceil(new_height / GPT41_PATCH_SIZE)
            image_tokens = final_patches_wide_scaled * final_patches_high_scaled

            # Ensure it doesn't exceed the cap due to ceiling operations after scaling
            image_tokens = min(image_tokens, GPT41_MAX_PATCHES)

            logger.debug(
                f"Image scaled down. Original patches: {total_patches_needed}, New dims: ~{new_width}x{new_height}, Final patches: {image_tokens}"
            )

        else:
            image_tokens = total_patches_needed
            logger.debug(f"Image fits within patch limit. Patches: {image_tokens}")

        # Apply model-specific multiplier
        if "mini" in model:
            total_tokens = math.ceil(image_tokens * GPT41_MINI_MULTIPLIER)
        elif "nano" in model:
            total_tokens = math.ceil(image_tokens * GPT41_NANO_MULTIPLIER)
        else:  # Assume base gpt-4.1
            total_tokens = image_tokens

        logger.info(
            f"Calculated token cost for image ({width}x{height}) with model '{model}': {total_tokens} tokens (base image tokens: {image_tokens})"
        )
        return total_tokens

    async def generate_text_with_images(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        system_prompt: str = "",
        detail: Literal["low", "high", "auto"] = "auto",
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:  # pragma: no cover
        """Generate text from OpenAI models using text and image inputs."""
        if not self._supports_responses_api():
            logger.warning(
                "Vision requests require the Responses API. "
                "Falling back to text-only Chat Completions."
            )
            try:
                fallback_messages: List[Dict[str, Any]] = []
                if system_prompt:
                    fallback_messages.append(
                        {"role": "system", "content": system_prompt}
                    )
                fallback_messages.append({"role": "user", "content": prompt})
                completion = await self._chat_completions_create(
                    runtime_context=runtime_context,
                    model=self.text_model,
                    messages=fallback_messages,
                    max_tokens=self._resolve_chat_completion_max_tokens(
                        fallback_messages,
                        self.text_model,
                        runtime_context,
                    ),
                    **await self._hosted_chat_completion_request_options(
                        runtime_context
                    ),
                )
                return completion.choices[0].message.content or ""
            except Exception as e:
                logger.exception(f"Error in text-only fallback for images: {e}")
                return ""

        if not images:
            logger.warning(
                "generate_text_with_images called with no images. Falling back to generate_text."
            )
            return await self.generate_text(
                prompt,
                system_prompt,
                runtime_context=runtime_context,
            )

        client = await self._get_client(runtime_context)
        self._instrument_client(client)

        target_model = self.vision_model
        if "gpt-4.1" not in target_model:  # Basic check for vision model
            logger.warning(
                f"Model '{target_model}' might not support vision. Using it anyway."
            )

        content_list: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        total_image_bytes = 0
        total_image_tokens = 0

        if len(images) > MAX_IMAGE_COUNT:
            logger.error(
                f"Too many images provided ({len(images)}). Maximum is {MAX_IMAGE_COUNT}."
            )
            return f"Error: Too many images provided ({len(images)}). Maximum is {MAX_IMAGE_COUNT}."

        for i, image_input in enumerate(images):
            image_url_data: Dict[str, Any] = {"detail": detail}
            image_bytes: Optional[bytes] = None
            image_format: Optional[str] = None
            width: Optional[int] = None
            height: Optional[int] = None

            try:
                if isinstance(image_input, str):  # It's a URL
                    logger.debug(f"Processing image URL: {image_input[:50]}...")
                    image_url_data["url"] = image_input
                    # Cannot easily validate size/format/dimensions or calculate cost for URLs
                    logger.warning(
                        "Cannot validate size/format or calculate token cost for image URLs."
                    )

                elif isinstance(image_input, bytes):  # It's image bytes
                    logger.debug(
                        f"Processing image bytes (size: {len(image_input)})..."
                    )
                    image_bytes = image_input
                    size_mb = len(image_bytes) / (1024 * 1024)
                    if size_mb > MAX_IMAGE_SIZE_MB:
                        logger.error(
                            f"Image {i + 1} size ({size_mb:.2f}MB) exceeds limit ({MAX_IMAGE_SIZE_MB}MB)."
                        )
                        return f"Error: Image {i + 1} size ({size_mb:.2f}MB) exceeds limit ({MAX_IMAGE_SIZE_MB}MB)."
                    total_image_bytes += len(image_bytes)

                    # Use Pillow to validate format and get dimensions
                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        image_format = img.format
                        width, height = img.size
                        img.verify()  # Verify integrity
                        # Re-open after verify
                        img = Image.open(io.BytesIO(image_bytes))
                        width, height = img.size  # Get dimensions again

                        if image_format not in SUPPORTED_IMAGE_FORMATS:
                            logger.error(
                                f"Unsupported image format '{image_format}' for image {i + 1}."
                            )
                            return f"Error: Unsupported image format '{image_format}'. Supported formats: {SUPPORTED_IMAGE_FORMATS}."

                        logger.debug(
                            f"Image {i + 1}: Format={image_format}, Dimensions={width}x{height}"
                        )

                        # Calculate cost only if dimensions are available
                        if width and height and "gpt-4.1" in target_model:
                            total_image_tokens += self._calculate_gpt41_image_cost(
                                width, height, target_model
                            )

                    except (IOError, SyntaxError) as img_err:
                        logger.error(
                            f"Invalid or corrupted image data for image {i + 1}: {img_err}"
                        )
                        return f"Error: Invalid or corrupted image data provided for image {i + 1}."
                    except Exception as pillow_err:
                        logger.error(
                            f"Pillow error processing image {i + 1}: {pillow_err}"
                        )
                        return f"Error: Could not process image data for image {i + 1}."

                    # Encode to Base64 Data URL
                    mime_type = Image.MIME.get(image_format)
                    if not mime_type:
                        logger.warning(
                            f"Could not determine MIME type for format {image_format}. Defaulting to image/jpeg."
                        )
                        mime_type = "image/jpeg"
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    image_url_data["url"] = f"data:{mime_type};base64,{base64_image}"

                else:
                    logger.error(
                        f"Invalid image input type for image {i + 1}: {type(image_input)}"
                    )
                    return f"Error: Invalid image input type for image {i + 1}. Must be URL (str) or bytes."

                content_list.append({"type": "image_url", "image_url": image_url_data})

            except Exception as proc_err:
                logger.error(
                    f"Error processing image {i + 1}: {proc_err}", exc_info=True
                )
                return f"Error: Failed to process image {i + 1}."

        total_size_mb = total_image_bytes / (1024 * 1024)
        if total_size_mb > MAX_TOTAL_IMAGE_SIZE_MB:
            logger.error(
                f"Total image size ({total_size_mb:.2f}MB) exceeds limit ({MAX_TOTAL_IMAGE_SIZE_MB}MB)."
            )
            return f"Error: Total image size ({total_size_mb:.2f}MB) exceeds limit ({MAX_TOTAL_IMAGE_SIZE_MB}MB)."

        # Build input for Responses API
        input_content = content_list

        request_params: Dict[str, Any] = {
            "model": target_model,
            "input": [{"role": "user", "content": input_content}],
        }

        if self.reasoning_effort and not self.base_url:
            request_params["reasoning"] = {"effort": self.reasoning_effort}

        if system_prompt:
            request_params["instructions"] = system_prompt

        logger.info(
            f"Sending request to '{target_model}' with {len(images)} images. Total calculated image tokens (approx): {total_image_tokens}"
        )

        try:
            response = await client.responses.create(**request_params)
            # Extract text from Responses API response
            if hasattr(response, "output_text") and response.output_text:
                # Log actual usage if available
                if hasattr(response, "usage") and response.usage:
                    logger.info(
                        f"OpenAI API Usage: Input={response.usage.input_tokens}, Output={response.usage.output_tokens}, Total={response.usage.total_tokens}"
                    )
                return response.output_text
            else:
                logger.warning("Received vision response with no content.")
                return ""
        except OpenAIError as e:  # Catch specific OpenAI errors
            logger.error(f"OpenAI API error during vision request: {e}")
            return f"I apologize, but I encountered an API error: {e}"
        except Exception as e:
            logger.exception(f"Error in generate_text_with_images: {e}")
            return f"I apologize, but I encountered an unexpected error: {e}"

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:  # pragma: no cover
        """Stream responses with optional tool calls."""
        try:
            if self._hosted_chat_completions_require_non_streaming(model):
                cc_params: Dict[str, Any] = {
                    "model": model or self.text_model,
                    "messages": messages,
                    "max_tokens": self._resolve_chat_completion_max_tokens(
                        messages,
                        model or self.text_model,
                        runtime_context,
                    ),
                    **await self._hosted_chat_completion_request_options(
                        runtime_context
                    ),
                }
                if tools:
                    cc_params["tools"] = tools

                response = await self._chat_completions_create(
                    runtime_context=runtime_context,
                    **cc_params,
                )
                for event in self._chat_completion_response_events(response):
                    yield event
                return

            client = await self._get_client(runtime_context)
            self._instrument_client(client)

            if self._supports_responses_api():
                # Responses API path (OpenAI, Groq)
                instructions = None
                input_items = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content")

                    if role == "system":
                        instructions = content
                    elif role == "user":
                        input_items.append({"role": "user", "content": content or ""})
                    elif role == "assistant":
                        tool_calls = msg.get("tool_calls")
                        if tool_calls:
                            for tc in tool_calls:
                                func = tc.get("function", {})
                                input_items.append(
                                    {
                                        "type": "function_call",
                                        "call_id": tc.get("id", ""),
                                        "name": func.get("name", ""),
                                        "arguments": func.get("arguments", "{}"),
                                    }
                                )
                        elif content:
                            input_items.append(
                                {"role": "assistant", "content": content}
                            )
                    elif role == "tool":
                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": msg.get("tool_call_id", ""),
                                "output": content or "",
                            }
                        )

                request_params: Dict[str, Any] = {
                    "model": model or self.text_model,
                    "input": input_items,
                    "stream": True,
                }

                if self.reasoning_effort and not self.base_url:
                    request_params["reasoning"] = {"effort": self.reasoning_effort}

                if instructions:
                    request_params["instructions"] = instructions

                if tools:
                    responses_tools = []
                    for tool in tools:
                        if tool.get("type") == "function":
                            func = tool.get("function", {})
                            responses_tools.append(
                                {
                                    "type": "function",
                                    "name": func.get("name"),
                                    "description": func.get("description", ""),
                                    "parameters": func.get("parameters", {}),
                                }
                            )
                        else:
                            responses_tools.append(tool)
                    request_params["tools"] = responses_tools

                stream = await client.responses.create(**request_params)
                async for event in stream:
                    try:
                        event_type = getattr(event, "type", None)

                        if event_type == "response.output_text.delta":
                            delta = getattr(event, "delta", "")
                            if delta:
                                yield {"type": "content", "delta": delta}

                        elif event_type == "response.function_call_arguments.delta":
                            yield {
                                "type": "tool_call_delta",
                                "id": getattr(event, "call_id", None),
                                "index": getattr(event, "output_index", 0),
                                "name": getattr(event, "name", None),
                                "arguments_delta": getattr(event, "delta", ""),
                            }

                        elif event_type == "response.output_item.added":
                            item = getattr(event, "item", None)
                            if item and getattr(item, "type", None) == "function_call":
                                yield {
                                    "type": "tool_call_delta",
                                    "id": getattr(item, "call_id", None),
                                    "index": getattr(event, "output_index", 0),
                                    "name": getattr(item, "name", None),
                                    "arguments_delta": "",
                                }

                        elif event_type == "response.completed":
                            yield {"type": "message_end", "finish_reason": "stop"}

                    except Exception as parse_err:
                        logger.debug(f"Error parsing stream event: {parse_err}")
                        continue

                yield {"type": "message_end", "finish_reason": "end_of_stream"}
            else:
                # Chat Completions API path (e.g., Cerebras)
                cc_params: Dict[str, Any] = {
                    "model": model or self.text_model,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": self._resolve_chat_completion_max_tokens(
                        messages,
                        model or self.text_model,
                        runtime_context,
                    ),
                    **await self._hosted_chat_completion_request_options(
                        runtime_context
                    ),
                }
                if tools:
                    cc_params["tools"] = tools

                stream = await self._chat_completions_create(
                    runtime_context=runtime_context,
                    **cc_params,
                )
                async for event in stream:
                    try:
                        if not getattr(event, "choices", None):
                            continue
                        choice = event.choices[0]
                        delta = getattr(choice, "delta", None)
                        if delta:
                            content = getattr(delta, "content", None)
                            if content:
                                yield {"type": "content", "delta": content}

                            tc_list = getattr(delta, "tool_calls", None)
                            if tc_list:
                                for tc in tc_list:
                                    func = getattr(tc, "function", None)
                                    yield {
                                        "type": "tool_call_delta",
                                        "id": getattr(tc, "id", None),
                                        "index": getattr(tc, "index", 0),
                                        "name": getattr(func, "name", None)
                                        if func
                                        else None,
                                        "arguments_delta": getattr(
                                            func, "arguments", ""
                                        )
                                        if func
                                        else "",
                                    }

                        finish_reason = getattr(choice, "finish_reason", None)
                        if finish_reason:
                            yield {
                                "type": "message_end",
                                "finish_reason": finish_reason,
                            }
                    except Exception as parse_err:
                        logger.debug(f"Error parsing chat stream event: {parse_err}")
                        continue

                yield {"type": "message_end", "finish_reason": "end_of_stream"}
        except Exception as e:
            logger.exception(f"Error in chat_stream: {e}")
            yield {"type": "error", "error": str(e)}

    async def parse_structured_output(
        self,
        prompt: str,
        system_prompt: str,
        model_class: Type[T],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> T:  # pragma: no cover
        """Generate structured output using Responses API or Chat Completions fallback."""
        current_parse_model = model or self.parse_model
        client = await self._get_client(runtime_context)
        self._instrument_client(client)

        try:
            if not self._supports_responses_api():
                raise RuntimeError(
                    "Responses API not supported; using Chat Completions fallback."
                )

            # Use Responses API with text.format for structured output
            response = await client.responses.create(
                model=current_parse_model,
                instructions=system_prompt,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": model_class.__name__,
                        "strict": True,
                        "schema": model_class.model_json_schema(),
                    }
                },
            )

            # Extract the JSON from response and parse with Pydantic
            json_str = response.output_text
            return model_class.model_validate_json(json_str)

        except Exception as e:
            logger.warning(f"Responses API structured output failed: {e}")

            try:
                # Fallback: Use chat completions with response_format
                logger.info("Falling back to chat completions with JSON schema.")
                fallback_system_prompt = f"""
{system_prompt}

You must respond with valid JSON that matches this schema:
{model_class.model_json_schema()}

Respond with ONLY the JSON object.
"""

                completion = await self._chat_completions_create(
                    runtime_context=runtime_context,
                    model=current_parse_model,
                    messages=[
                        {"role": "system", "content": fallback_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=self._resolve_chat_completion_max_tokens(
                        [
                            {"role": "system", "content": fallback_system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        current_parse_model,
                        runtime_context,
                    ),
                    **await self._hosted_chat_completion_request_options(
                        runtime_context
                    ),
                )

                json_str = completion.choices[0].message.content
                return model_class.model_validate_json(json_str)

            except Exception as fallback_error:
                logger.exception(
                    f"All structured output methods failed: {fallback_error}"
                )
                raise ValueError(f"Failed to generate structured output: {e}") from e

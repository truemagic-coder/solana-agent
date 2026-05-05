"""CLI smoke harness for a local AGI OpenAI-compatible x402 endpoint.

This is a nested-repo copy of the AGI smoke harness so Solana Agent can run the
same local-first validation contract before cutover.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import secrets
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solana_agent import SolanaAgent  # noqa: E402
from solana_agent.tools.utils.x402 import request_with_x402_private_key  # noqa: E402


@dataclass
class SmokeResult:
    name: str
    ok: bool
    elapsed_ms: float
    detail: str


def _load_env_file() -> None:
    dotenv_path = (
        os.getenv("X402_SMOKE_DOTENV_PATH")
        or os.getenv("OPENAI_API_DOTENV_PATH")
        or ".env"
    )
    load_dotenv(dotenv_path=dotenv_path, override=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("X402_SMOKE_BASE_URL", "http://127.0.0.1:8000"),
        help="OpenAI-compatible service base URL",
    )
    parser.add_argument(
        "--scenario",
        choices=[
            "health",
            "stateless",
            "memory",
            "memory-stream",
            "duplicate",
            "stream",
            "sdk-stateless",
            "sdk-memory",
            "sdk-success",
            "error-bad-json",
            "error-missing-idempotency",
            "error-invalid-model",
            "error-payment-required",
            "error-ops-unauthorized",
            "error-idempotency-conflict",
            "error-quota",
            "error-provider-circuit",
            "error-memory-unavailable",
            "error-upstream-failure",
            "error-runtime-unavailable",
            "error-settlement-failure",
            "success",
            "errors",
            "internal-errors",
            "all",
        ],
        default=os.getenv("X402_SMOKE_SCENARIO", "all"),
        help="Smoke scenario to run",
    )
    parser.add_argument(
        "--private-key",
        default=os.getenv("X402_PRIVATE_KEY") or os.getenv("SOLANA_PRIVATE_KEY"),
        help="Base58 Solana private key for x402 signing",
    )
    parser.add_argument(
        "--rpc-url",
        default=os.getenv("X402_SMOKE_RPC_URL") or os.getenv("SOLANA_RPC_URL"),
        help="Optional Solana RPC URL for x402 settlement",
    )
    parser.add_argument(
        "--ops-token",
        default=os.getenv("OPENAI_API_OPS_ACCESS_TOKEN"),
        help="Optional bearer token used to verify /ops/metrics during duplicate replay",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("X402_SMOKE_TIMEOUT", "60.0")),
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("X402_SMOKE_MAX_TOKENS", "64")),
        help="max_tokens sent to chat completions",
    )
    parser.add_argument(
        "--conversation-id",
        default=os.getenv("X402_SMOKE_CONVERSATION_ID"),
        help="Optional conversation_id to reuse for the memory smoke",
    )
    parser.add_argument(
        "--memory-ttl-tier",
        choices=["work", "project"],
        default=os.getenv("X402_SMOKE_MEMORY_TTL_TIER", "work"),
        help="Retention tier used for the memory smoke",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("X402_SMOKE_USER", "x402-smoke"),
        help="Optional OpenAI-compatible user field",
    )
    parser.add_argument(
        "--search-enabled",
        action="store_true",
        default=os.getenv("X402_SMOKE_SEARCH_ENABLED", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Enable the hosted search add-on for non-streaming smoke checks.",
    )
    return parser.parse_args()


def _resolve_private_key(cli_value: str | None) -> str:
    candidate = (
        cli_value or os.getenv("X402_PRIVATE_KEY") or os.getenv("SOLANA_PRIVATE_KEY")
    )
    if candidate:
        return candidate.strip()

    try:
        pasted = getpass.getpass("Paste base58 Solana private key for x402 signing: ")
    except EOFError as exc:
        raise SystemExit(
            "ERROR: no private key provided; use --private-key or set X402_PRIVATE_KEY"
        ) from exc

    pasted = pasted.strip()
    if not pasted:
        raise SystemExit("ERROR: private key is required")
    return pasted


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _compact_text(value: str, *, limit: int = 160) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _contains_token(text: str, token: str) -> bool:
    normalized_text = str(text or "").upper()
    normalized_token = str(token or "").upper()
    return bool(normalized_token) and normalized_token in normalized_text


def _extract_json(response: httpx.Response) -> dict[str, Any] | None:
    try:
        payload = response.json()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _assistant_content(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    return ""


def _stream_content(body_text: str) -> str:
    parts: list[str] = []
    for raw_line in body_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices") or []
        if not choices:
            continue
        first = choices[0] if isinstance(choices[0], dict) else {}
        delta = first.get("delta") if isinstance(first, dict) else {}
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "".join(parts).strip()


def _chat_body(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    user: str,
    stream: bool = False,
    conversation_id: str | None = None,
    memory_ttl_tier: str | None = None,
    search_enabled: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Follow the user's requested output format exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "user": user,
    }
    if conversation_id:
        body["conversation_id"] = conversation_id
    if memory_ttl_tier:
        body["memory_ttl_tier"] = memory_ttl_tier
    if search_enabled:
        body["search_enabled"] = True
    return body


def _sdk_base_url(base_url: str) -> str:
    return f"{_normalize_base_url(base_url)}/v1"


def _sdk_config(
    *,
    base_url: str,
    private_key: str,
    model: str,
    max_output_tokens: int,
    rpc_url: str | None,
) -> dict[str, Any]:
    openai_config: dict[str, Any] = {
        "auth_mode": "x402_private_key",
        "private_key": private_key,
        "base_url": _sdk_base_url(base_url),
        "model": model,
        "max_output_tokens": max_output_tokens,
    }
    if rpc_url:
        openai_config["x402_rpc_url"] = rpc_url

    return {
        "ai": openai_config,
        "agents": [
            {
                "name": "default",
                "instructions": (
                    "You are a helpful Solana AI assistant. "
                    "Follow the user's requested output format exactly and do not add extra commentary."
                ),
                "specialization": "general",
            }
        ],
    }


async def _collect_agent_text_response(
    agent: SolanaAgent,
    *,
    user_id: str,
    message: str,
    runtime_context: dict[str, Any] | None = None,
    search_enabled: bool = False,
) -> tuple[str, float]:
    started = time.perf_counter()
    chunks: list[str] = []
    async for chunk in agent.process(
        user_id,
        message,
        runtime_context=runtime_context,
        search_enabled=search_enabled,
    ):
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode("utf-8", errors="replace"))
        else:
            chunks.append(str(chunk))
    return "".join(chunks).strip(), (time.perf_counter() - started) * 1000


async def _health_check(base_url: str, timeout: float) -> SmokeResult:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/healthz")
        payload = _extract_json(response) or {}
        quota_profile = (
            payload.get("quota_profile") if isinstance(payload, dict) else {}
        )
        source = ""
        fault_injection_enabled = False
        if isinstance(quota_profile, dict):
            source = str(quota_profile.get("source") or "")
        if isinstance(payload, dict):
            fault_injection_enabled = bool(payload.get("fault_injection_enabled"))
        detail = f"status={response.status_code}"
        if source:
            detail = f"{detail}; quota_profile.source={source}"
        if fault_injection_enabled:
            detail = f"{detail}; fault_injection_enabled=true"
        return SmokeResult(
            name="healthz",
            ok=response.status_code == 200,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            detail=detail,
        )
    except Exception as exc:
        return SmokeResult(
            name="healthz",
            ok=False,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            detail=str(exc),
        )


async def _paid_chat(
    *,
    base_url: str,
    private_key: str,
    rpc_url: str | None,
    timeout: float,
    idempotency_key: str,
    body: dict[str, Any],
    extra_headers: dict[str, str] | None = None,
) -> tuple[httpx.Response, float]:
    started = time.perf_counter()
    headers = {"Idempotency-Key": idempotency_key}
    if extra_headers:
        headers.update(extra_headers)
    response = await request_with_x402_private_key(
        method="POST",
        url=f"{base_url}/v1/chat/completions",
        private_key=private_key,
        headers=headers,
        json_data=body,
        timeout=timeout,
        rpc_url=rpc_url,
    )
    return response, (time.perf_counter() - started) * 1000


async def _ops_metrics(
    base_url: str,
    timeout: float,
    ops_token: str,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(
            f"{base_url}/ops/metrics",
            headers={"Authorization": f"Bearer {ops_token}"},
        )
    payload = _extract_json(response)
    if response.status_code != 200 or payload is None:
        raise RuntimeError(
            f"ops metrics request failed with status={response.status_code}: {_compact_text(response.text)}"
        )
    return payload


def _settlement_success_count(payload: dict[str, Any]) -> int:
    activity = payload.get("activity") if isinstance(payload, dict) else {}
    payments = activity.get("payments") if isinstance(activity, dict) else {}
    return int(payments.get("settlement_success") or 0)


def _replay_count(payload: dict[str, Any]) -> int:
    idempotency = payload.get("idempotency") if isinstance(payload, dict) else {}
    decisions = idempotency.get("decisions") if isinstance(idempotency, dict) else {}
    return int(decisions.get("replay") or 0)


async def _public_chat(
    *,
    base_url: str,
    timeout: float,
    body: Any,
    headers: dict[str, str] | None = None,
) -> tuple[httpx.Response, float]:
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        if isinstance(body, (bytes, str)):
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                content=body,
                headers=headers,
            )
        else:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
    return response, (time.perf_counter() - started) * 1000


def _error_message(response: httpx.Response) -> str:
    payload = _extract_json(response)
    if payload and isinstance(payload.get("error"), dict):
        message = payload["error"].get("message")
        if isinstance(message, str):
            return message
    return response.text


async def _server_fault_injection_status(
    base_url: str,
    timeout: float,
) -> tuple[bool | None, str | None]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/healthz")
        payload = _extract_json(response)
        if response.status_code != 200 or payload is None:
            return None, f"healthz returned status={response.status_code}"
        return bool(payload.get("fault_injection_enabled")), None
    except Exception as exc:
        return None, str(exc)


async def _require_fault_injection(
    *,
    name: str,
    base_url: str,
    timeout: float,
) -> str | None:
    """Require the running service, not the local shell, to advertise fault injection."""

    enabled, error = await _server_fault_injection_status(base_url, timeout)
    if error is not None:
        return f"{name} could not verify /healthz fault injection status: {error}"
    if enabled:
        return None
    return (
        f"{name} requires the running service to start with OPENAI_API_ENABLE_FAULT_INJECTION=1; "
        f"restart the server with that env var and rerun this scenario"
    )


async def _expect_public_error(
    *,
    name: str,
    base_url: str,
    timeout: float,
    body: Any,
    headers: dict[str, str] | None,
    expected_status: int,
    expected_text: str,
) -> SmokeResult:
    try:
        response, elapsed_ms = await _public_chat(
            base_url=base_url,
            timeout=timeout,
            body=body,
            headers=headers,
        )
        message = _error_message(response)
        ok = response.status_code == expected_status and expected_text in message
        detail = f"status={response.status_code}; message={_compact_text(message)}"
        return SmokeResult(name, ok, elapsed_ms, detail)
    except Exception as exc:
        return SmokeResult(name, False, 0.0, str(exc))


async def _expect_paid_error(
    *,
    name: str,
    args: argparse.Namespace,
    private_key: str,
    base_url: str,
    body: dict[str, Any],
    expected_status: int,
    expected_text: str,
    extra_headers: dict[str, str] | None = None,
) -> SmokeResult:
    try:
        response, elapsed_ms = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=f"smoke-{name}-{uuid.uuid4().hex}",
            body=body,
            extra_headers=extra_headers,
        )
        message = _error_message(response)
        ok = response.status_code == expected_status and expected_text in message
        detail = f"status={response.status_code}; message={_compact_text(message)}"
        return SmokeResult(name, ok, elapsed_ms, detail)
    except Exception as exc:
        return SmokeResult(name, False, 0.0, str(exc))


async def _stateless_check(
    args: argparse.Namespace, private_key: str, base_url: str
) -> SmokeResult:
    token = f"STATELESS-{secrets.token_hex(4).upper()}"
    body = _chat_body(
        model="solana-agent-chat",
        prompt=f"Reply with ONLY this token: {token}",
        max_tokens=args.max_tokens,
        user=args.user,
        search_enabled=args.search_enabled,
    )
    try:
        response, elapsed_ms = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=f"smoke-stateless-{uuid.uuid4().hex}",
            body=body,
        )
        payload = _extract_json(response)
        content = _assistant_content(payload)
        ok = response.status_code == 200 and _contains_token(content, token)
        detail = f"status={response.status_code}; reply={_compact_text(content or response.text)}"
        return SmokeResult("paid-stateless", ok, elapsed_ms, detail)
    except Exception as exc:
        return SmokeResult("paid-stateless", False, 0.0, str(exc))


async def _memory_check(
    args: argparse.Namespace, private_key: str, base_url: str
) -> SmokeResult:
    token = f"MEMORY-{secrets.token_hex(4).upper()}"
    conversation_id = args.conversation_id or f"smoke-{token.lower()}"
    store_body = _chat_body(
        model="solana-agent-memory",
        prompt=f"Remember this exact token for later: {token}. Reply with ONLY: STORED {token}",
        max_tokens=args.max_tokens,
        user=args.user,
        conversation_id=conversation_id,
        memory_ttl_tier=args.memory_ttl_tier,
        search_enabled=args.search_enabled,
    )
    recall_body = _chat_body(
        model="solana-agent-memory",
        prompt=("What token did I ask you to remember? Reply with ONLY the token."),
        max_tokens=args.max_tokens,
        user=args.user,
        conversation_id=conversation_id,
        memory_ttl_tier=args.memory_ttl_tier,
        search_enabled=args.search_enabled,
    )
    started = time.perf_counter()
    try:
        first, first_ms = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=f"smoke-memory-store-{uuid.uuid4().hex}",
            body=store_body,
        )
        second, second_ms = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=f"smoke-memory-recall-{uuid.uuid4().hex}",
            body=recall_body,
        )
        second_content = _assistant_content(_extract_json(second))
        ok = (
            first.status_code == 200
            and second.status_code == 200
            and _contains_token(second_content, token)
        )
        detail = (
            f"store={first.status_code} ({first_ms:.1f} ms); "
            f"recall={second.status_code} ({second_ms:.1f} ms); "
            f"recalled={_compact_text(second_content or second.text)}"
        )
        return SmokeResult(
            "paid-memory-recall",
            ok,
            (time.perf_counter() - started) * 1000,
            detail,
        )
    except Exception as exc:
        return SmokeResult(
            "paid-memory-recall",
            False,
            (time.perf_counter() - started) * 1000,
            str(exc),
        )


async def _memory_stream_check(
    args: argparse.Namespace, private_key: str, base_url: str
) -> SmokeResult:
    token = f"MEMSTREAM-{secrets.token_hex(4).upper()}"
    conversation_id = args.conversation_id or f"smoke-{token.lower()}"
    store_body = _chat_body(
        model="solana-agent-memory",
        prompt=f"Remember this exact token for later: {token}. Reply with ONLY: STORED {token}",
        max_tokens=args.max_tokens,
        user=args.user,
        conversation_id=conversation_id,
        memory_ttl_tier=args.memory_ttl_tier,
    )
    stream_body = _chat_body(
        model="solana-agent-memory",
        prompt=("What token did I ask you to remember? Reply with ONLY the token."),
        max_tokens=args.max_tokens,
        user=args.user,
        conversation_id=conversation_id,
        memory_ttl_tier=args.memory_ttl_tier,
        stream=True,
    )
    started = time.perf_counter()
    try:
        first, first_ms = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=f"smoke-memory-stream-store-{uuid.uuid4().hex}",
            body=store_body,
        )
        second, second_ms = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=f"smoke-memory-stream-{uuid.uuid4().hex}",
            body=stream_body,
        )
        response_text = second.text
        streamed = _stream_content(response_text)
        ok = (
            first.status_code == 200
            and second.status_code == 200
            and "data: [DONE]" in response_text
            and _contains_token(streamed, token)
        )
        detail = (
            f"store={first.status_code} ({first_ms:.1f} ms); "
            f"stream={second.status_code} ({second_ms:.1f} ms); "
            f"streamed={_compact_text(streamed or response_text)}"
        )
        return SmokeResult(
            "paid-memory-stream",
            ok,
            (time.perf_counter() - started) * 1000,
            detail,
        )
    except Exception as exc:
        return SmokeResult(
            "paid-memory-stream",
            False,
            (time.perf_counter() - started) * 1000,
            str(exc),
        )


async def _duplicate_check(
    args: argparse.Namespace, private_key: str, base_url: str
) -> SmokeResult:
    token = f"DUPLICATE-{secrets.token_hex(4).upper()}"
    body = _chat_body(
        model="solana-agent-chat",
        prompt=f"Reply with ONLY this token: {token}",
        max_tokens=args.max_tokens,
        user=args.user,
        search_enabled=args.search_enabled,
    )
    idempotency_key = f"smoke-duplicate-{uuid.uuid4().hex}"
    started = time.perf_counter()
    try:
        before_metrics = None
        after_metrics = None
        metrics_detail = "add --ops-token to verify settlement counters"
        if args.ops_token:
            try:
                before_metrics = await _ops_metrics(
                    base_url, args.timeout, args.ops_token
                )
            except Exception as exc:
                metrics_detail = f"ops metrics unavailable: {_compact_text(str(exc))}"

        first, _ = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=idempotency_key,
            body=body,
        )
        second, _ = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=idempotency_key,
            body=body,
        )
        if args.ops_token and before_metrics is not None:
            try:
                after_metrics = await _ops_metrics(
                    base_url, args.timeout, args.ops_token
                )
            except Exception as exc:
                metrics_detail = f"ops metrics unavailable: {_compact_text(str(exc))}"

        first_payload = _extract_json(first)
        second_payload = _extract_json(second)
        ok = (
            first.status_code == 200
            and second.status_code == 200
            and first_payload == second_payload
        )
        detail = "identical replay response"
        if before_metrics is not None and after_metrics is not None:
            settlement_delta = _settlement_success_count(
                after_metrics
            ) - _settlement_success_count(before_metrics)
            replay_delta = _replay_count(after_metrics) - _replay_count(before_metrics)
            ok = ok and settlement_delta == 1 and replay_delta >= 1
            detail = (
                f"identical replay response; settlement_delta={settlement_delta}; "
                f"replay_delta={replay_delta}"
            )
        else:
            detail = f"{detail}; {metrics_detail}"

        return SmokeResult(
            "duplicate-replay",
            ok,
            (time.perf_counter() - started) * 1000,
            detail,
        )
    except Exception as exc:
        return SmokeResult(
            "duplicate-replay",
            False,
            (time.perf_counter() - started) * 1000,
            str(exc),
        )


async def _stream_check(
    args: argparse.Namespace, private_key: str, base_url: str
) -> SmokeResult:
    token = f"STREAM-{secrets.token_hex(4).upper()}"
    body = _chat_body(
        model="solana-agent-chat",
        prompt=f"Reply with ONLY this token: {token}",
        max_tokens=args.max_tokens,
        user=args.user,
        stream=True,
    )
    started = time.perf_counter()
    try:
        response, _ = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=f"smoke-stream-{uuid.uuid4().hex}",
            body=body,
        )
        response_text = response.text
        streamed = _stream_content(response_text)
        ok = (
            response.status_code == 200
            and "data: [DONE]" in response_text
            and _contains_token(streamed, token)
        )
        detail = f"status={response.status_code}; streamed={_compact_text(streamed or response_text)}"
        return SmokeResult(
            "streaming",
            ok,
            (time.perf_counter() - started) * 1000,
            detail,
        )
    except Exception as exc:
        return SmokeResult(
            "streaming",
            False,
            (time.perf_counter() - started) * 1000,
            str(exc),
        )


async def _sdk_stateless_check(
    args: argparse.Namespace,
    private_key: str,
    base_url: str,
) -> SmokeResult:
    token = f"SDK-STATELESS-{secrets.token_hex(4).upper()}"
    try:
        agent = SolanaAgent(
            config=_sdk_config(
                base_url=base_url,
                private_key=private_key,
                model="stateless",
                max_output_tokens=args.max_tokens,
                rpc_url=args.rpc_url,
            )
        )
        content, elapsed_ms = await _collect_agent_text_response(
            agent,
            user_id=args.user,
            message=f"Reply with ONLY this token: {token}",
            search_enabled=args.search_enabled,
        )
        ok = _contains_token(content, token)
        detail = f"reply={_compact_text(content)}"
        return SmokeResult("sdk-stateless", ok, elapsed_ms, detail)
    except Exception as exc:
        return SmokeResult("sdk-stateless", False, 0.0, str(exc))


async def _sdk_memory_check(
    args: argparse.Namespace,
    private_key: str,
    base_url: str,
) -> SmokeResult:
    token = f"SDK-MEMORY-{secrets.token_hex(4).upper()}"
    conversation_id = args.conversation_id or f"sdk-smoke-{token.lower()}"
    started = time.perf_counter()
    try:
        agent = SolanaAgent(
            config=_sdk_config(
                base_url=base_url,
                private_key=private_key,
                model="memory",
                max_output_tokens=args.max_tokens,
                rpc_url=args.rpc_url,
            )
        )
        runtime_context = {
            "conversation_id": conversation_id,
            "memory_ttl_tier": args.memory_ttl_tier,
        }
        first_content, first_ms = await _collect_agent_text_response(
            agent,
            user_id=args.user,
            message=(
                f"Remember this exact token for later: {token}. "
                f"Reply with ONLY: STORED {token}"
            ),
            runtime_context=runtime_context,
            search_enabled=args.search_enabled,
        )
        second_content, second_ms = await _collect_agent_text_response(
            agent,
            user_id=args.user,
            message=(
                "What token did I ask you to remember? Reply with ONLY the token."
            ),
            runtime_context=runtime_context,
            search_enabled=args.search_enabled,
        )
        ok = _contains_token(first_content, token) and _contains_token(
            second_content,
            token,
        )
        detail = (
            f"store={first_ms:.1f} ms; recall={second_ms:.1f} ms; "
            f"reply={_compact_text(second_content)}"
        )
        return SmokeResult(
            "sdk-memory",
            ok,
            (time.perf_counter() - started) * 1000,
            detail,
        )
    except Exception as exc:
        return SmokeResult(
            "sdk-memory",
            False,
            (time.perf_counter() - started) * 1000,
            str(exc),
        )


async def _bad_json_error(base_url: str, timeout: float) -> SmokeResult:
    return await _expect_public_error(
        name="error-bad-json",
        base_url=base_url,
        timeout=timeout,
        body="{bad json",
        headers={"Content-Type": "application/json"},
        expected_status=400,
        expected_text="Request body must be valid JSON",
    )


async def _missing_idempotency_error(base_url: str, timeout: float) -> SmokeResult:
    return await _expect_public_error(
        name="error-missing-idempotency",
        base_url=base_url,
        timeout=timeout,
        body={
            "model": "solana-agent-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
        },
        headers=None,
        expected_status=400,
        expected_text="Idempotency-Key header is required",
    )


async def _invalid_model_error(base_url: str, timeout: float) -> SmokeResult:
    return await _expect_public_error(
        name="error-invalid-model",
        base_url=base_url,
        timeout=timeout,
        body={
            "model": "not-a-real-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
        },
        headers={"Idempotency-Key": f"smoke-invalid-model-{uuid.uuid4().hex}"},
        expected_status=400,
        expected_text="Unsupported model",
    )


async def _payment_required_error(base_url: str, timeout: float) -> SmokeResult:
    started = time.perf_counter()
    try:
        response, elapsed_ms = await _public_chat(
            base_url=base_url,
            timeout=timeout,
            body={
                "model": "solana-agent-chat",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
            headers={"Idempotency-Key": f"smoke-payment-required-{uuid.uuid4().hex}"},
        )
        ok = response.status_code == 402 and "payment-required" in {
            key.lower() for key in response.headers.keys()
        }
        detail = f"status={response.status_code}; payment-required={response.headers.get('payment-required', '')}"
        return SmokeResult("error-payment-required", ok, elapsed_ms, detail)
    except Exception as exc:
        return SmokeResult(
            "error-payment-required",
            False,
            (time.perf_counter() - started) * 1000,
            str(exc),
        )


async def _ops_unauthorized_error(
    args: argparse.Namespace, base_url: str
) -> SmokeResult:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            response = await client.get(f"{base_url}/ops/metrics")
        ok = response.status_code == 401
        detail = (
            f"status={response.status_code}; message={_compact_text(response.text)}"
        )
        return SmokeResult(
            "error-ops-unauthorized", ok, (time.perf_counter() - started) * 1000, detail
        )
    except Exception as exc:
        return SmokeResult(
            "error-ops-unauthorized",
            False,
            (time.perf_counter() - started) * 1000,
            str(exc),
        )


async def _idempotency_conflict_error(
    args: argparse.Namespace,
    private_key: str,
    base_url: str,
) -> SmokeResult:
    idempotency_key = f"smoke-conflict-{uuid.uuid4().hex}"
    first_body = _chat_body(
        model="solana-agent-chat",
        prompt="Reply with ONLY: first",
        max_tokens=args.max_tokens,
        user=args.user,
    )
    second_body = _chat_body(
        model="solana-agent-chat",
        prompt="Reply with ONLY: second",
        max_tokens=args.max_tokens,
        user=args.user,
    )
    started = time.perf_counter()
    try:
        first, _ = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=idempotency_key,
            body=first_body,
        )
        second, _ = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=idempotency_key,
            body=second_body,
        )
        message = _error_message(second)
        ok = (
            first.status_code == 200
            and second.status_code == 409
            and "different request body" in message
        )
        detail = f"first={first.status_code}; second={second.status_code}; message={_compact_text(message)}"
        return SmokeResult(
            "error-idempotency-conflict",
            ok,
            (time.perf_counter() - started) * 1000,
            detail,
        )
    except Exception as exc:
        return SmokeResult(
            "error-idempotency-conflict",
            False,
            (time.perf_counter() - started) * 1000,
            str(exc),
        )


async def _quota_error(
    args: argparse.Namespace, private_key: str, base_url: str
) -> SmokeResult:
    requirement = await _require_fault_injection(
        name="error-quota",
        base_url=base_url,
        timeout=args.timeout,
    )
    if requirement is not None:
        return SmokeResult("error-quota", False, 0.0, requirement)
    return await _expect_paid_error(
        name="error-quota",
        args=args,
        private_key=private_key,
        base_url=base_url,
        body=_chat_body(
            model="solana-agent-chat",
            prompt="Reply with ONLY: quota",
            max_tokens=args.max_tokens,
            user=args.user,
        ),
        expected_status=429,
        expected_text="fault_injection:quota_generation",
        extra_headers={"X-OpenAI-API-Fault": "quota_generation"},
    )


async def _provider_circuit_error(
    args: argparse.Namespace, private_key: str, base_url: str
) -> SmokeResult:
    requirement = await _require_fault_injection(
        name="error-provider-circuit",
        base_url=base_url,
        timeout=args.timeout,
    )
    if requirement is not None:
        return SmokeResult("error-provider-circuit", False, 0.0, requirement)
    return await _expect_paid_error(
        name="error-provider-circuit",
        args=args,
        private_key=private_key,
        base_url=base_url,
        body=_chat_body(
            model="solana-agent-chat",
            prompt="Reply with ONLY: circuit",
            max_tokens=args.max_tokens,
            user=args.user,
        ),
        expected_status=503,
        expected_text="Upstream provider temporarily unavailable",
        extra_headers={"X-OpenAI-API-Fault": "provider_circuit_open"},
    )


async def _memory_unavailable_error(
    args: argparse.Namespace, private_key: str, base_url: str
) -> SmokeResult:
    requirement = await _require_fault_injection(
        name="error-memory-unavailable",
        base_url=base_url,
        timeout=args.timeout,
    )
    if requirement is not None:
        return SmokeResult("error-memory-unavailable", False, 0.0, requirement)
    return await _expect_paid_error(
        name="error-memory-unavailable",
        args=args,
        private_key=private_key,
        base_url=base_url,
        body=_chat_body(
            model="solana-agent-memory",
            prompt="Remember this",
            max_tokens=args.max_tokens,
            user=args.user,
            conversation_id=args.conversation_id
            or f"smoke-memory-error-{uuid.uuid4().hex}",
            memory_ttl_tier=args.memory_ttl_tier,
        ),
        expected_status=503,
        expected_text="Memory-backed model is not available",
        extra_headers={"X-OpenAI-API-Fault": "memory_unavailable"},
    )


async def _upstream_failure_error(
    args: argparse.Namespace, private_key: str, base_url: str
) -> SmokeResult:
    requirement = await _require_fault_injection(
        name="error-upstream-failure",
        base_url=base_url,
        timeout=args.timeout,
    )
    if requirement is not None:
        return SmokeResult("error-upstream-failure", False, 0.0, requirement)
    return await _expect_paid_error(
        name="error-upstream-failure",
        args=args,
        private_key=private_key,
        base_url=base_url,
        body=_chat_body(
            model="solana-agent-chat",
            prompt="Reply with ONLY: upstream",
            max_tokens=args.max_tokens,
            user=args.user,
        ),
        expected_status=502,
        expected_text="Upstream generation failed: injected upstream failure",
        extra_headers={"X-OpenAI-API-Fault": "upstream_generation"},
    )


async def _runtime_unavailable_error(
    args: argparse.Namespace, private_key: str, base_url: str
) -> SmokeResult:
    requirement = await _require_fault_injection(
        name="error-runtime-unavailable",
        base_url=base_url,
        timeout=args.timeout,
    )
    if requirement is not None:
        return SmokeResult("error-runtime-unavailable", False, 0.0, requirement)
    return await _expect_paid_error(
        name="error-runtime-unavailable",
        args=args,
        private_key=private_key,
        base_url=base_url,
        body=_chat_body(
            model="solana-agent-chat",
            prompt="Reply with ONLY: runtime",
            max_tokens=args.max_tokens,
            user=args.user,
        ),
        expected_status=503,
        expected_text="API runtime is unavailable",
        extra_headers={"X-OpenAI-API-Fault": "runtime_unavailable"},
    )


async def _settlement_failure_error(
    args: argparse.Namespace, private_key: str, base_url: str
) -> SmokeResult:
    requirement = await _require_fault_injection(
        name="error-settlement-failure",
        base_url=base_url,
        timeout=args.timeout,
    )
    if requirement is not None:
        return SmokeResult("error-settlement-failure", False, 0.0, requirement)
    started = time.perf_counter()
    try:
        response, _ = await _paid_chat(
            base_url=base_url,
            private_key=private_key,
            rpc_url=args.rpc_url,
            timeout=args.timeout,
            idempotency_key=f"smoke-settlement-failure-{uuid.uuid4().hex}",
            body=_chat_body(
                model="solana-agent-chat",
                prompt="Reply with ONLY: settlement",
                max_tokens=args.max_tokens,
                user=args.user,
            ),
            extra_headers={"X-OpenAI-API-Fault": "settlement_failure"},
        )
        ok = (
            response.status_code == 402
            and response.headers.get("payment-required") == "retry"
        )
        detail = (
            f"status={response.status_code}; payment-required={response.headers.get('payment-required', '')}; "
            f"body={_compact_text(response.text)}"
        )
        return SmokeResult(
            "error-settlement-failure",
            ok,
            (time.perf_counter() - started) * 1000,
            detail,
        )
    except Exception as exc:
        return SmokeResult(
            "error-settlement-failure",
            False,
            (time.perf_counter() - started) * 1000,
            str(exc),
        )


def _print_result(result: SmokeResult) -> None:
    marker = "PASS" if result.ok else "FAIL"
    print(f"[{marker}] {result.name:20s} {result.elapsed_ms:8.1f} ms  {result.detail}")


async def _run(args: argparse.Namespace) -> int:
    base_url = _normalize_base_url(args.base_url)
    private_key = ""
    public_only = {
        "health",
        "error-bad-json",
        "error-missing-idempotency",
        "error-invalid-model",
        "error-payment-required",
        "error-ops-unauthorized",
    }
    if args.scenario not in public_only:
        private_key = _resolve_private_key(args.private_key)

    scenario_order = {
        "health": ["health"],
        "stateless": ["health", "stateless"],
        "memory": ["health", "memory"],
        "memory-stream": ["health", "memory-stream"],
        "duplicate": ["health", "duplicate"],
        "stream": ["health", "stream"],
        "sdk-stateless": ["health", "sdk-stateless"],
        "sdk-memory": ["health", "sdk-memory"],
        "sdk-success": ["health", "sdk-stateless", "sdk-memory"],
        "error-bad-json": ["health", "error-bad-json"],
        "error-missing-idempotency": ["health", "error-missing-idempotency"],
        "error-invalid-model": ["health", "error-invalid-model"],
        "error-payment-required": ["health", "error-payment-required"],
        "error-ops-unauthorized": ["health", "error-ops-unauthorized"],
        "error-idempotency-conflict": ["health", "error-idempotency-conflict"],
        "error-quota": ["health", "error-quota"],
        "error-provider-circuit": ["health", "error-provider-circuit"],
        "error-memory-unavailable": ["health", "error-memory-unavailable"],
        "error-upstream-failure": ["health", "error-upstream-failure"],
        "error-runtime-unavailable": ["health", "error-runtime-unavailable"],
        "error-settlement-failure": ["health", "error-settlement-failure"],
        "success": ["health", "stateless", "memory", "duplicate", "stream"],
        "errors": [
            "health",
            "error-bad-json",
            "error-missing-idempotency",
            "error-invalid-model",
            "error-payment-required",
            "error-ops-unauthorized",
            "error-idempotency-conflict",
        ],
        "internal-errors": [
            "health",
            "error-quota",
            "error-provider-circuit",
            "error-memory-unavailable",
            "error-upstream-failure",
            "error-runtime-unavailable",
            "error-settlement-failure",
        ],
        "all": [
            "health",
            "stateless",
            "memory",
            "memory-stream",
            "duplicate",
            "stream",
            "error-bad-json",
            "error-missing-idempotency",
            "error-invalid-model",
            "error-payment-required",
            "error-ops-unauthorized",
            "error-idempotency-conflict",
            "error-quota",
            "error-provider-circuit",
            "error-memory-unavailable",
            "error-upstream-failure",
            "error-runtime-unavailable",
            "error-settlement-failure",
        ],
    }

    results: list[SmokeResult] = []
    for name in scenario_order[args.scenario]:
        if name == "health":
            results.append(await _health_check(base_url, args.timeout))
        elif name == "stateless":
            results.append(await _stateless_check(args, private_key, base_url))
        elif name == "memory":
            results.append(await _memory_check(args, private_key, base_url))
        elif name == "memory-stream":
            results.append(await _memory_stream_check(args, private_key, base_url))
        elif name == "duplicate":
            results.append(await _duplicate_check(args, private_key, base_url))
        elif name == "stream":
            results.append(await _stream_check(args, private_key, base_url))
        elif name == "sdk-stateless":
            results.append(await _sdk_stateless_check(args, private_key, base_url))
        elif name == "sdk-memory":
            results.append(await _sdk_memory_check(args, private_key, base_url))
        elif name == "error-bad-json":
            results.append(await _bad_json_error(base_url, args.timeout))
        elif name == "error-missing-idempotency":
            results.append(await _missing_idempotency_error(base_url, args.timeout))
        elif name == "error-invalid-model":
            results.append(await _invalid_model_error(base_url, args.timeout))
        elif name == "error-payment-required":
            results.append(await _payment_required_error(base_url, args.timeout))
        elif name == "error-ops-unauthorized":
            results.append(await _ops_unauthorized_error(args, base_url))
        elif name == "error-idempotency-conflict":
            results.append(
                await _idempotency_conflict_error(args, private_key, base_url)
            )
        elif name == "error-quota":
            results.append(await _quota_error(args, private_key, base_url))
        elif name == "error-provider-circuit":
            results.append(await _provider_circuit_error(args, private_key, base_url))
        elif name == "error-memory-unavailable":
            results.append(await _memory_unavailable_error(args, private_key, base_url))
        elif name == "error-upstream-failure":
            results.append(await _upstream_failure_error(args, private_key, base_url))
        elif name == "error-runtime-unavailable":
            results.append(
                await _runtime_unavailable_error(args, private_key, base_url)
            )
        elif name == "error-settlement-failure":
            results.append(await _settlement_failure_error(args, private_key, base_url))

    for result in results:
        _print_result(result)

    passed = sum(1 for result in results if result.ok)
    print(f"Summary: {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


def main() -> int:
    _load_env_file()
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

"""Run one hosted SDK smoke step from a source checkout."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


SDK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SDK_ROOT.parent
DEFAULT_AGI_ROOT = WORKSPACE_ROOT / "solana-agent-agi"
DEFAULT_DOTENV_PATH = DEFAULT_AGI_ROOT / ".env"
DEFAULT_SDK_CONFIG_PATH = SDK_ROOT / "config.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TIMEOUT_SECONDS = 900

if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from solana_agent.client.solana_agent import SolanaAgent  # noqa: E402
from solana_agent.smoke import (  # noqa: E402
    DEFAULT_EARN_AMOUNT_USDC,
    DEFAULT_SWAP_AMOUNT_USDC,
    DEFAULT_TRIGGER_AMOUNT_USDC,
    EARN_SMOKE_SENTINEL,
    MEMORY_PRIORITY_PROJECT_SMOKE_SENTINEL,
    MEMORY_PRIORITY_PROJECT_SMOKE_TOKEN,
    MEMORY_PRIORITY_WORK_SMOKE_SENTINEL,
    MEMORY_PRIORITY_WORK_SMOKE_TOKEN,
    MEMORY_PROJECT_SMOKE_SENTINEL,
    MEMORY_PROJECT_SMOKE_TOKEN,
    MEMORY_WORK_SMOKE_SENTINEL,
    MEMORY_WORK_SMOKE_TOKEN,
    PROJECT_MEMORY_TTL_DAYS,
    SOLANA_USDC_MINT,
    SWAP_SMOKE_SENTINEL,
    TECHNICAL_ANALYSIS_SMOKE_SENTINEL,
    TOKEN_MATH_SMOKE_SENTINEL,
    TRIGGER_SMOKE_SENTINEL,
    WORK_MEMORY_TTL_DAYS,
    WRAPPED_SOL_MINT,
    WRITE_SMOKE_FAILURE_TERMS,
    _bootstrap_local_hosted_x402_signer,
    _earn_smoke_message,
    _memory_recall_smoke_message,
    _memory_store_smoke_message,
    _run_memory_smoke_step,
    _run_message_smoke_step,
    _swap_smoke_message,
    _technical_analysis_smoke_message,
    _token_math_smoke_message,
    _trigger_smoke_message,
    build_public_sdk_smoke_preview,
)


STEP_CONFIGS: dict[str, dict[str, Any]] = {
    "token_math": {
        "step_name": "token_math",
        "prompt_text": _token_math_smoke_message(),
        "expected_sentinel": TOKEN_MATH_SMOKE_SENTINEL,
        "error_message": f"Token-math smoke response did not include {TOKEN_MATH_SMOKE_SENTINEL}",
        "conversation_prefix": "sdk-smoke-token-math",
        "chain_type": "solana",
        "required_response_terms": ("100000", "0.1"),
    },
    "technical_analysis": {
        "step_name": "technical_analysis",
        "prompt_text": _technical_analysis_smoke_message(),
        "expected_sentinel": TECHNICAL_ANALYSIS_SMOKE_SENTINEL,
        "error_message": f"Technical-analysis smoke response did not include {TECHNICAL_ANALYSIS_SMOKE_SENTINEL}",
        "conversation_prefix": "sdk-smoke-technical-analysis",
        "chain_type": "solana",
    },
    "swap_live": {
        "step_name": "swap_live",
        "prompt_text": _swap_smoke_message(amount_usdc=DEFAULT_SWAP_AMOUNT_USDC),
        "expected_sentinel": SWAP_SMOKE_SENTINEL,
        "error_message": f"Swap smoke response did not include {SWAP_SMOKE_SENTINEL}",
        "conversation_prefix": "sdk-smoke-swap",
        "chain_type": "solana",
        "step_payload": {
            "amount_usdc": str(DEFAULT_SWAP_AMOUNT_USDC),
            "input_mint": SOLANA_USDC_MINT,
            "output_mint": WRAPPED_SOL_MINT,
        },
        "failure_response_terms": WRITE_SMOKE_FAILURE_TERMS,
    },
    "trigger_limit_order": {
        "step_name": "trigger_limit_order",
        "prompt_text": _trigger_smoke_message(amount_usdc=DEFAULT_TRIGGER_AMOUNT_USDC),
        "expected_sentinel": TRIGGER_SMOKE_SENTINEL,
        "error_message": f"Trigger smoke response did not include {TRIGGER_SMOKE_SENTINEL}",
        "conversation_prefix": "sdk-smoke-trigger",
        "chain_type": "solana",
        "step_payload": {
            "amount_usdc": str(DEFAULT_TRIGGER_AMOUNT_USDC),
            "input_mint": SOLANA_USDC_MINT,
            "output_mint": WRAPPED_SOL_MINT,
        },
        "failure_response_terms": WRITE_SMOKE_FAILURE_TERMS,
    },
    "earn_reversible": {
        "step_name": "earn_reversible",
        "prompt_text": _earn_smoke_message(amount_usdc=DEFAULT_EARN_AMOUNT_USDC),
        "expected_sentinel": EARN_SMOKE_SENTINEL,
        "error_message": f"Earn smoke response did not include {EARN_SMOKE_SENTINEL}",
        "conversation_prefix": "sdk-smoke-earn",
        "chain_type": "solana",
        "step_payload": {
            "amount_usdc": str(DEFAULT_EARN_AMOUNT_USDC),
            "asset": SOLANA_USDC_MINT,
        },
        "required_response_terms": ("deposit", "withdraw"),
        "failure_response_terms": WRITE_SMOKE_FAILURE_TERMS,
    },
    "memory_recall_work_7d": {
        "runner": "memory",
        "step_name": "memory_recall_work_7d",
        "expected_sentinel": MEMORY_WORK_SMOKE_SENTINEL,
        "error_message": f"Memory smoke response did not include {MEMORY_WORK_SMOKE_SENTINEL} {MEMORY_WORK_SMOKE_TOKEN}",
        "conversation_prefix": "sdk-smoke-memory-work",
        "chain_type": "solana",
        "memory_ttl_tier": "work",
        "retention_days": WORK_MEMORY_TTL_DAYS,
        "remember_token": MEMORY_WORK_SMOKE_TOKEN,
    },
    "memory_recall_work_7d_priority_tier": {
        "runner": "memory",
        "step_name": "memory_recall_work_7d_priority_tier",
        "expected_sentinel": MEMORY_PRIORITY_WORK_SMOKE_SENTINEL,
        "error_message": f"Priority memory smoke response did not include {MEMORY_PRIORITY_WORK_SMOKE_SENTINEL} {MEMORY_PRIORITY_WORK_SMOKE_TOKEN}",
        "conversation_prefix": "sdk-smoke-memory-work-priority",
        "chain_type": "solana",
        "memory_ttl_tier": "work",
        "retention_days": WORK_MEMORY_TTL_DAYS,
        "remember_token": MEMORY_PRIORITY_WORK_SMOKE_TOKEN,
        "service_tier": "priority",
    },
    "memory_recall_project_30d": {
        "runner": "memory",
        "step_name": "memory_recall_project_30d",
        "expected_sentinel": MEMORY_PROJECT_SMOKE_SENTINEL,
        "error_message": f"Memory smoke response did not include {MEMORY_PROJECT_SMOKE_SENTINEL} {MEMORY_PROJECT_SMOKE_TOKEN}",
        "conversation_prefix": "sdk-smoke-memory-project",
        "chain_type": "solana",
        "memory_ttl_tier": "project",
        "retention_days": PROJECT_MEMORY_TTL_DAYS,
        "remember_token": MEMORY_PROJECT_SMOKE_TOKEN,
    },
    "memory_recall_project_30d_priority_tier": {
        "runner": "memory",
        "step_name": "memory_recall_project_30d_priority_tier",
        "expected_sentinel": MEMORY_PRIORITY_PROJECT_SMOKE_SENTINEL,
        "error_message": f"Priority memory smoke response did not include {MEMORY_PRIORITY_PROJECT_SMOKE_SENTINEL} {MEMORY_PRIORITY_PROJECT_SMOKE_TOKEN}",
        "conversation_prefix": "sdk-smoke-memory-project-priority",
        "chain_type": "solana",
        "memory_ttl_tier": "project",
        "retention_days": PROJECT_MEMORY_TTL_DAYS,
        "remember_token": MEMORY_PRIORITY_PROJECT_SMOKE_TOKEN,
        "service_tier": "priority",
    },
}


def _load_dotenv(dotenv_path: Path | None) -> None:
    if dotenv_path is None:
        return
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=True)


def _resolve_privy_user_id(cli_value: str | None) -> str:
    privy_user_id = str(
        cli_value
        or os.getenv("SOLANA_AGENT_SMOKE_PRIVY_USER_ID")
        or os.getenv("PRIVY_USER_ID")
        or ""
    ).strip()
    if not privy_user_id:
        raise ValueError(
            "Set --privy-user-id or SOLANA_AGENT_SMOKE_PRIVY_USER_ID to run live smoke steps."
        )
    return privy_user_id


async def _bootstrap_agent(
    args: argparse.Namespace,
) -> tuple[SolanaAgent, dict[str, Any]]:
    _load_dotenv(args.dotenv_path)
    privy_user_id = _resolve_privy_user_id(args.privy_user_id)
    agent = SolanaAgent(
        config_path=str(args.sdk_config_path),
        base_url=args.base_url,
        privy_user_id=privy_user_id,
    )
    preview = await build_public_sdk_smoke_preview(
        agent,
        include_search=False,
        include_priority=False,
        include_memory=False,
        include_jupiter=False,
        include_birdeye=False,
        include_swap=False,
        include_trigger=False,
        include_earn=False,
        include_technical_analysis=False,
        include_token_math=False,
    )
    wallet = dict(preview.get("wallet") or {})
    await _bootstrap_local_hosted_x402_signer(
        agent,
        wallet_id=str(wallet.get("wallet_id") or ""),
        privy_user_id=str(preview.get("privy_user_id") or privy_user_id),
        chain_type="solana",
    )
    return agent, wallet


async def _run_full_response(args: argparse.Namespace) -> None:
    agent, wallet = await _bootstrap_agent(args)
    config = STEP_CONFIGS[args.step]
    if config.get("runner") == "memory":
        context = await agent.context(
            conversation_id=f"{config['conversation_prefix']}-full-response",
            model="memory",
            memory_ttl_tier=config["memory_ttl_tier"],
            service_tier=config.get("service_tier", "standard"),
            search_enabled=False,
            chain_type=config["chain_type"],
        )
        context["_raise_stream_errors"] = True
        context["max_tool_iterations"] = args.max_tool_iterations
        context["request_timeout_seconds"] = args.server_timeout_seconds
        store_response = await asyncio.wait_for(
            agent.message(
                _memory_store_smoke_message(
                    remember_token=config["remember_token"],
                    retention_days=config["retention_days"],
                    memory_ttl_tier=config["memory_ttl_tier"],
                ),
                **context,
            ),
            timeout=args.timeout_seconds,
        )
        recall_response = await asyncio.wait_for(
            agent.message(
                _memory_recall_smoke_message(
                    remember_token=config["remember_token"],
                    expected_sentinel=config["expected_sentinel"],
                    retention_days=config["retention_days"],
                    memory_ttl_tier=config["memory_ttl_tier"],
                ),
                **context,
            ),
            timeout=args.timeout_seconds,
        )
        print(
            json.dumps(
                {
                    "wallet_address": wallet.get("address"),
                    "wallet_id": wallet.get("wallet_id"),
                    "llm_wallet_creation_used": False,
                    "step": args.step,
                    "responses": {
                        "store": store_response,
                        "recall": recall_response,
                    },
                },
                indent=2,
            )
        )
        return
    context = await agent.context(
        conversation_id=f"{config['conversation_prefix']}-full-response",
        model=config.get("model", "chat"),
        memory_ttl_tier=config.get("memory_ttl_tier", "work"),
        service_tier=config.get("service_tier", "standard"),
        search_enabled=config.get("search_enabled", False),
        chain_type="solana",
    )
    context["_raise_stream_errors"] = True
    context["max_tool_iterations"] = args.max_tool_iterations
    context["request_timeout_seconds"] = args.server_timeout_seconds
    response = await asyncio.wait_for(
        agent.message(config["prompt_text"], **context),
        timeout=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "wallet_address": wallet.get("address"),
                "wallet_id": wallet.get("wallet_id"),
                "llm_wallet_creation_used": False,
                "step": args.step,
                "response": response,
            },
            indent=2,
        )
    )


async def _run_smoke_step(args: argparse.Namespace) -> None:
    agent, wallet = await _bootstrap_agent(args)
    config = dict(STEP_CONFIGS[args.step])
    runner = str(config.pop("runner", "message"))
    config["max_tool_iterations"] = args.max_tool_iterations
    config["request_timeout_seconds"] = args.server_timeout_seconds
    result = await asyncio.wait_for(
        (
            _run_memory_smoke_step(agent, **config)
            if runner == "memory"
            else _run_message_smoke_step(agent, **config)
        ),
        timeout=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "wallet_address": wallet.get("address"),
                "wallet_id": wallet.get("wallet_id"),
                "llm_wallet_creation_used": False,
                "step": args.step,
                "result": result,
            },
            indent=2,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one live SDK smoke step against a hosted-compatible API."
    )
    parser.add_argument("step", choices=sorted(STEP_CONFIGS))
    parser.add_argument(
        "--mode",
        choices=("smoke-step", "full-response"),
        default="smoke-step",
    )
    parser.add_argument(
        "--base-url", default=os.getenv("SOLANA_AGENT_SMOKE_BASE_URL", DEFAULT_BASE_URL)
    )
    parser.add_argument("--privy-user-id", default=None)
    parser.add_argument(
        "--dotenv-path",
        type=Path,
        default=Path(os.getenv("OPENAI_API_DOTENV_PATH") or DEFAULT_DOTENV_PATH),
    )
    parser.add_argument(
        "--sdk-config-path",
        type=Path,
        default=Path(
            os.getenv("SOLANA_AGENT_SMOKE_SDK_CONFIG") or DEFAULT_SDK_CONFIG_PATH
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(
            os.getenv("SOLANA_AGENT_SMOKE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        ),
    )
    parser.add_argument(
        "--max-tool-iterations",
        type=int,
        default=int(os.getenv("SOLANA_AGENT_SMOKE_MAX_TOOL_ITERATIONS", "8")),
    )
    parser.add_argument(
        "--server-timeout-seconds",
        type=int,
        default=int(os.getenv("SOLANA_AGENT_SMOKE_SERVER_TIMEOUT_SECONDS", "120")),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.mode == "full-response":
        asyncio.run(_run_full_response(args))
        return
    asyncio.run(_run_smoke_step(args))


if __name__ == "__main__":
    main()

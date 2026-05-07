# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![Python 3.13-3.14](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/solana-agent)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/ci.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/ci.yml)
[![Ruff Style](https://img.shields.io/badge/style-ruff-41B5BE)](https://github.com/astral-sh/ruff)

Thin public SDK for the hosted [Solana Agent](https://solana-agent.com) platform.

The public package is intentionally small:

- One agent per client instance.
- Hosted chat and hosted wallet flows.
- Hosted account and pricing APIs.
- MCP plugin support for external tools.

## Python Support

We support the current and previous CPython minor versions.

Today that means:

- Python 3.14
- Python 3.13

## Install

```bash
pip install solana-agent
```

## Quick Start

Use the Solana Agent CLI to get a new `privy_user_id` from the interactive menu:

```bash
uvx solana-agent wallet menu
```

The generated `privy_user_id` is saved automatically in the user-local Solana Agent app data directory on Windows, macOS, and Linux. Passing `privy_user_id=` or `config.ai.privy_user_id` still overrides the saved value.

```python
from solana_agent import SolanaAgent

agent = SolanaAgent(
    instructions="You are a Solana trading bot.",
    privy_user_id="my-privy-user-id",
)

context = await agent.context(
    conversation_id="my-conversation-id",
    model="chat",
)

response = await agent.message(
    message="What is the price of SOL?",
    **context
)

print(response)
```

## Hosted Search

For live web and X results in hosted chat, enable the hosted search add-on:

```bash
uvx solana-agent chat --search-enabled
```

When you start hosted chat without a local `config.json`, the CLI now asks once at startup whether to enable hosted search for that session.

## Hosted Memory

```python
context = await agent.context(
    conversation_id="my-conversation-id",
    model="memory",
    memory_ttl_tier="project",
)
```

## Priority Service Tier

```python
context = await agent.context(
    service_tier="priority",
)
```

## MCP Tools

Connect Streamable HTTP MCP servers by enabling the `mcp` tool and adding server config:

```python
import os
from solana_agent import SolanaAgent

agent = SolanaAgent(
    config={
        "ai": {
            "instructions": "Use connected MCP tools when they help the user.",
            "privy_user_id": os.environ["PRIVY_USER_ID"],
            "tools": ["mcp"],
        },
        "tools": {
            "mcp": {
                "servers": [
                    {
                        "url": os.environ["MCP_SERVER_URL"],
                        "headers": {
                            "Authorization": f"Bearer {os.environ['MCP_SERVER_TOKEN']}",
                        },
                    }
                ],
                "llm_provider": "openai",
                "api_key": os.environ["OPENAI_API_KEY"],
                "llm_model": "gpt-4.1-mini",
            }
        },
    }
)

context = await agent.context(
    conversation_id="mcp-demo",
    model="chat",
)

response = await agent.message(
    "Use the connected MCP tools to summarize my latest CRM tasks.",
    **context,
)

print(response)
```

## Wallet and Billing Helpers

```python

wallet_address = await agent.get_wallet_address()

summary = await agent.get_account_summary()

report = await agent.get_usage_report(
    "month"
)

forecast = await agent.get_usage_forecast(
    window_days=30,
)

pricing = await agent.get_pricing_info()

tooling_totals = summary.get("tooling", {}).get("lifetime", {}).get("totals", {})
tooling_projection = (
    forecast.get("tooling", {})
    .get("projected_month_end", {})
    .get("totals", {})
)

print(tooling_totals)
print(tooling_projection)
```

## Private Key Export

Export the hosted wallet private key when you want self-custody:

```bash
uvx solana-agent wallet export --yes
```

Pass `--wallet-id` to export an older rotated wallet from `old_wallets`.

## Rotating Wallets

If a wallet needs to be retired, rotate it from the interactive menu:

```bash
uxv solana-agent wallet menu
```

## Smoke Testing
For a live hosted smoke run from the CLI, use the dev-gated wallet smoke command or the dev-only wallet menu item:

```bash
uvx solana-agent wallet smoke --dev
uvx solana-agent wallet menu --dev
```

The smoke preview prints a funding estimate in USDC before the live chat checks run. Search-enabled checks are included by default; wallet rotation and private-key export checks are opt-in.
Use `--include-memory` to add hosted memory recall checks for both the 7-day work tier and the 30-day project tier.
Use `--include-priority` to add priority validation for chat and any enabled memory checks.
Use `--big` to add the full expanded profile: memory 7-day and 30-day checks, priority chat and memory, Jupiter and Birdeye read checks, along with token math, technical analysis, swap, trigger, and earn checks.
Use `--include-transfer --transfer-recipient <wallet> [--transfer-amount-usdc 0.10]` when you want the smoke run to execute a live USDC transfer.
Add `--json` to `wallet smoke` when you want machine-readable output instead of tables.

For source-checkout debugging against a local hosted API, run one permanent step at a time with `scripts/smoke_step_runner.py`:

```bash
SOLANA_AGENT_SMOKE_PRIVY_USER_ID=did:privy:your-funded-user \
python scripts/smoke_step_runner.py trigger_limit_order --mode full-response
```

The runner loads `../solana-agent-agi/.env` by default, targets `http://127.0.0.1:8000/v1`, and prints JSON including the wallet id, wallet address, step name, and response.


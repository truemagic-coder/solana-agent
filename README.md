# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![Python 3.13-3.14](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/solana-agent)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/ci.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/ci.yml)
[![Ruff Style](https://img.shields.io/badge/style-ruff-41B5BE)](https://github.com/astral-sh/ruff)

Thin public SDK for the hosted Solana Agent platform.

The public package is intentionally small:

- One agent per client instance.
- Hosted chat and hosted wallet flows.
- Hosted account and pricing APIs.
- MCP and raw `x402_request` plugin support.

The public SDK does not ship local memory stores, local history APIs, guardrails, or multi-agent routing.

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

### x402 private key

```python
import os
from solana_agent import SolanaAgent

config = {
    "ai": {
        "auth_mode": "x402_private_key",
        "private_key": os.environ["X402_PRIVATE_KEY"],
    },
    "agents": [
        {
            "name": "assistant",
            "instructions": "You are a concise Solana operations assistant.",
            "specialization": "general",
            "tools": ["x402_request"],
        }
    ],
}

agent = SolanaAgent(config=config)

async for chunk in agent.process(
    user_id="user-123",
    message="Summarize my account usage and suggest the next wallet action.",
    runtime_context={"conversation_id": "acct-session-1"},
):
    print(chunk, end="")
```

### Privy wallet

```python
import os
from solana_agent import SolanaAgent

config = {
    "ai": {
        "auth_mode": "x402_privy",
        "privy_app_id": os.environ["PRIVY_APP_ID"],
        "privy_app_secret": os.environ["PRIVY_APP_SECRET"],
    },
    "agents": [
        {
            "name": "assistant",
            "instructions": "You help users operate safely with hosted Solana wallets.",
            "specialization": "wallets",
            "tools": ["x402_request"],
        }
    ],
}

agent = SolanaAgent(config=config)

runtime_context = await agent.prepare_x402_runtime_context(
    user_id="did:privy:user-123",
    runtime_context={"conversation_id": "wallet-session-1"},
)

async for chunk in agent.process(
    user_id="did:privy:user-123",
    message="Check my wallet status and explain the next step.",
    runtime_context=runtime_context,
):
    print(chunk, end="")
```

## Public SDK Boundary

Included:

- Single-agent request processing.
- Hosted Privy wallet creation and wallet address lookup.
- Hosted account summary, usage report, usage forecast, and pricing helpers.
- Plugin loading for `mcp` and `x402_request`.
- Runtime context forwarding for hosted features such as `conversation_id`, `service_tier`, and wallet identifiers.

## Hosted Memory

If you want memory, use the hosted platform memory model. The SDK itself does not persist anything locally.

You can still select hosted memory explicitly through model and runtime context:

```python
config = {
    "ai": {
        "auth_mode": "x402_private_key",
        "private_key": os.environ["X402_PRIVATE_KEY"],
        "model": "memory",
    },
    "agents": [
        {
            "name": "assistant",
            "instructions": "You are a concise Solana assistant.",
            "specialization": "general",
        }
    ],
}

runtime_context = {
    "conversation_id": "portfolio-session-7",
    "memory_ttl_tier": "project",
}
```

That memory lives in the hosted service, not in the local SDK process.

## Wallet and Billing Helpers

```python
summary = await agent.get_account_summary(
    runtime_context={"privy_wallet_id": "wallet-123"}
)

report = await agent.get_usage_report(
    "month",
    runtime_context={"privy_wallet_id": "wallet-123"},
)

pricing = await agent.get_pricing_info(
    runtime_context={"privy_wallet_id": "wallet-123"},
)
```

## Development

```bash
python -m pytest tests -q
ruff check .
ruff format --check .
tokei solana_agent tests
```

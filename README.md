# Solana Agent

AI agents for Solana on a local-first AGI x402 runtime.

## Runtime Contract

The v34 runtime path in this repo uses one OpenAI-compatible transport surface and one AI backend target:

- `config["openai"]` is the only supported LLM config section.
- `auth_mode: "x402_private_key"` and `auth_mode: "x402_privy"` are the funded AGI paths.
- `base_url` points at a local AGI-compatible service during development and can be promoted to `https://agi.solana-agent.com/v1` after local validation.
- `solana-agent-memory` is the default remote-memory model.
- `solana-agent-chat` is the explicit stateless override.
- Remote AGI memory is the only supported memory mode in the runtime contract.
- Local Mongo and Zep are not part of the runtime path.
- The official first-party tool surface now ships in this package and registers through `solana_agent.plugins`.

## Installation

Install the package:

```bash
pip install solana-agent
```

For local development:

```bash
poetry install
cp .env.example .env
```

Set at least these environment variables for local funded calls:

- `X402_PRIVATE_KEY`
- `SOLANA_RPC_URL` when your x402 settlement flow needs an explicit RPC endpoint

If you use Privy-backed signing, also configure the matching Privy app credentials.

## Quick Start

```python
from solana_agent import SolanaAgent

config = {
    "openai": {
        "auth_mode": "x402_private_key",
        "private_key": "your-base58-solana-private-key",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "memory",
        "stateless_model": "solana-agent-chat",
    },
    "agents": [
        {
            "name": "default_agent",
            "instructions": "You are a helpful Solana AI assistant.",
            "specialization": "general",
        }
    ],
}

solana_agent = SolanaAgent(config=config)

async for response in solana_agent.process("user123", "Summarize the latest Solana ecosystem changes."):
    print(response, end="")
```

## Model Selection

In AGI x402 mode:

- Omit `model`, or set it to `"memory"`, to use `solana-agent-memory`.
- Set `model` to `"stateless"` to resolve through `stateless_model`.
- Set `model` directly to `"solana-agent-chat"` for the explicit stateless SKU.

## Privy Payer Example

Use `auth_mode: "x402_privy"` when the signer must be exported from Privy at request time:

```python
from solana_agent import SolanaAgent

config = {
    "openai": {
        "auth_mode": "x402_privy",
        "privy_app_id": "your-privy-app-id",
        "privy_app_secret": "your-privy-app-secret",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "memory",
        "stateless_model": "solana-agent-chat",
    },
    "agents": [
        {
            "name": "default_agent",
            "instructions": "You are a helpful Solana AI assistant.",
            "specialization": "general",
        }
    ],
}

solana_agent = SolanaAgent(config=config)

runtime_context = {"privy_wallet_id": "wallet-id-from-privy"}

async for response in solana_agent.process(
    "user123",
    "Check my wallet activity.",
    runtime_context=runtime_context,
):
    print(response, end="")
```

## Bundled Tool Surface

The official first-party tools are bundled directly in `solana_agent.tools`. Do not install Solana Agent Kit as a separate runtime dependency for v34.

The bundled surface includes:

- Birdeye
- Jupiter holdings, earn, recurring, shield, token search, and trigger flows
- Kamino
- MCP
- Privy account and transaction tools
- Solana transfer, Ultra, and dFlow tools
- Rugcheck
- Search, technical analysis, token math, image generation, and Vybe
- x402 request helpers

Inline tools are still supported for app-specific integrations.

## Local Smoke Validation

Run the fast local success bundle:

```bash
make x402-smoke
```

Run the full local matrix:

```bash
make x402-smoke SCENARIO=all
```

Run one targeted scenario directly:

```bash
poetry run python scripts/openai_x402_smoke.py --scenario memory-stream
```

The harness covers:

- `health`
- `stateless`
- `memory`
- `memory-stream`
- `duplicate`
- `stream`
- public error scenarios
- internal fault-injection scenarios when the local AGI service enables them

After local validation is green, point `base_url` at `https://agi.solana-agent.com/v1` and rerun the same scenarios before production use.

## Migration Notes

When moving from the v33 contract to the v34 runtime path:

- Replace direct provider sections such as `groq`, `cerebras`, and `grok` with a single `openai` transport section.
- Replace upstream model-provider API keys with x402 payer configuration.
- Replace local-memory assumptions with the AGI remote-memory default.
- Use `model: "stateless"` or `model: "solana-agent-chat"` when you need the stateless SKU.
- Use the bundled first-party tools in this package rather than installing a separate Solana Agent Kit runtime.
- Treat conversation history and memory as remote-only runtime behavior.

The broader migration plan is tracked in `V34_AGI_X402_CHECKLIST.md`.

## Development

Useful local commands:

```bash
poetry run pytest tests/unit -q -W error
poetry run python scripts/openai_x402_smoke.py --scenario all
make livehtml
```

Documentation: https://docs.solana-agent.com

## License

This project is licensed under the MIT License. See `LICENSE` for details.

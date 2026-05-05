# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/solana-agent)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/ci.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/ci.yml)
[![Ruff Style](https://img.shields.io/badge/style-ruff-41B5BE)](https://github.com/astral-sh/ruff)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=400)

# Ship Production AI Agents on Solana That **Remember, Pay, and Execute**

Stop wrestling with stateless LLMs that forget context after every turn. 

**Solana Agent** is the complete runtime for building **persistent, production-grade AI agents** that integrate natively with the Solana economy.

- **Persistent Memory**: Your agents remember user history, preferences, and past actions across sessions — powered by our hosted AGI service.
- **Seamless x402 Payments**: Real Solana-based payments with private keys or Privy wallets. No more fake demo mode.
- **Real Execution**: Hosted chat and memory SKUs orchestrate the Solana toolchain server-side, while the public SDK stays thin and wallet-aware.
- **Near AGI Performance**: Proven benchmark leadership with 0.935 mean human ratio on calibrated agentic tasks. Memory, planning, debugging, and tool use that actually works.

Whether you're building trading copilots, DeFi automations, or intelligent wallets, Solana Agent lets you ship reliable agents **in minutes, not months**.

`pip install solana-agent` — Get started with 5 lines of code.

## Why Solana Agent

- It remembers what happened earlier instead of starting from zero every turn.
- It can use real Solana payment and wallet flows, not just talk about them.
- It keeps the public SDK small: hosted chat/memory, wallet lifecycle, MCP, and raw x402 requests.
- It defaults to the hosted service, so getting started is simple.
- It is built for real agent workflows, not demo-only chat.

## Why Solana Builders Choose Us

Generic LLM wrappers fail at real agent work. They forget context, hallucinate tool calls, can't handle payments or onchain actions reliably, and break on long-running tasks.

**Solana Agent solves this with a complete, battle-tested stack:**

- **Persistent Remote Memory** that maintains coherent context across days or weeks — critical for personalized agents and ongoing workflows.
- **Native Solana Execution** with x402 payments using real wallets (private key or Privy). Agents can swap, transfer, lend, and interact with protocols without human-in-the-loop.
- **Hosted Solana Orchestration** — the built-in Jupiter, Kamino, Birdeye, and other Solana workflows now live behind the hosted SKUs instead of shipping as public SDK tool modules.
- **Proven Agent Intelligence** — Our hosted service delivers **near AGI performance** on the benchmarks that matter for agents: memory retention, multi-step planning, code repair, and commercial decision-making.
- **Simple SDK** with powerful runtime context for conversation isolation, memory tiers, and wallet delegation.

**The result?** Agents that ship faster, cost less to maintain, perform better in production, and actually move value on Solana.

## Proven Near AGI Agent Performance

Don't trust marketing claims. Trust the numbers.

Our hosted service leads on the hardest agent benchmarks because it was *built* for memory, continuity, tool orchestration, and recovery — not retrofitted.

**Human-Calibrated Cross-Domain Battery** (Latest): **0.842** weighted mean
- **Mean human ratio: 0.935** on the calibrated battery
- Minimum human ratio: 0.852
- **Verdict: Production-ready for serious agent workloads**

**Key Domain Wins:**
- Code debugging and repair: **1.000**
- Tool-using planning: **0.833–0.944**
- LongMemEval (memory): **0.776–0.840**
- Finance reforecasting: **0.889–1.000**
- Security incident response: **0.778–0.944**

**Best internal run: 0.889** aggregate across 8 domains. Median latency ~6s on complex memory tasks.

These aren't cherry-picked chat benchmarks. They test exactly what breaks most agents: staying coherent over long sessions, using tools correctly, fixing their own mistakes, and making sound financial/security decisions.

This is why Solana Agent isn't just "another LLM wrapper" — it's the runtime purpose-built for the next generation of onchain AI agents.

> "The hosted service is already a stronger fit for serious Solana agents than the usual stateless model wrapper."

## Transparent, Usage-Based Pricing

Pay only for what you use. Two SKUs optimized for different workloads:

| SKU | Best For | Input (est.) | Output (reserved) | Memory Surcharge |
|-----|----------|--------------|-------------------|------------------|
| **solana-agent-memory** (default) | Persistent agents, conversations, workflows | $10 / 1M tokens | $60 / 1M tokens | $0.50 per request (project tier, 30-day retention) |
| **solana-agent-chat** (`stateless`) | One-off queries, high-volume | $5 / 1M tokens | $30 / 1M tokens | None |

- `memory_ttl_tier="work"` (default, 7 days) or `"project"` for longer context of 30 days.
- x402 settlement in USDC on Solana — transparent, onchain, no surprise bills.
- Dynamic rate limits, service tiers, and admission controls protect both users and the upstream providers.

**Pro tip:** Start with the `solana-agent-memory` SKU for most agent use cases. The persistent context delivers far higher ROI than the modest price difference. Use `service_tier: "priority"` when you need the fastest possible responses.

The hosted service estimates cost *before* running inference (including any priority multiplier) and challenges with exact x402 payment details. Fair, predictable, and Solana-native.

## Service Tiers and Rate Limits

Our API gives you control over speed vs cost with two simple service tiers. No complex technical jargon — just choose what fits your needs.

### Standard vs Priority Service Tiers

Every request can include a `service_tier` field:

- **Standard** (default): The best value option. You pay the base rates shown in the table above. If the service is busy, your request waits a short time in our intelligent queue (usually just 1-5 seconds).

- **Priority**: For when speed matters most. This applies **2x pricing** but gives you priority queuing and much shorter (or zero) wait times. The system checks live conditions and will tell you in the response if it could deliver priority or had to fall back to standard.

**You only pay the higher rate when you actually receive priority service.**

The response always includes `service_tier_metadata` so you know exactly what happened:

```json
{
  "id": "chatcmpl-...",
  "choices": [ ... ],
  "service_tier_metadata": {
    "requested": "priority",
    "effective": "priority",
    "priority_available": true,
    "priority_reason": null
  }
}
```

Or if it was busy:

```json
"service_tier_metadata": {
  "requested": "priority",
  "effective": "standard",
  "priority_available": false,
  "priority_reason": "current load exceeds priority capacity"
}
```

This metadata appears in both regular JSON responses and streaming SSE chunks.

### Simple Rate Limits That Actually Work

We avoid the usual "retry on 429" frustration with smart design:

- **Dynamic Quotas**: Limits adjust automatically based on real-time load, your wallet's payment history, IP behavior, and upstream provider health. Per-wallet limits are much more generous once you've paid via x402.

- **Server-Side Queuing**: Short traffic spikes are absorbed by our servers with a brief wait instead of rejecting your request. This means fewer failed payments and better reliability.

Check live status, current quotas, and system health at any time with `GET /healthz`.

This system ensures the API stays fast and available even under heavy demand, while the priority tier (at 2x cost) lets you buy your way to the front of the line when it counts.

## Get Started in Seconds

```bash
pip install solana-agent
```

**Set your environment variables** (get your keys from your Solana wallet or Privy dashboard):

```bash
export PRIVY_APP_ID="your_privy_app_id"
export PRIVY_APP_SECRET="your_privy_app_secret"
# or for an operator-managed wallet:
# export X402_PRIVATE_KEY="your_base58_private_key_here"
```

Then run one of the Quick Start examples above.

For contributors and local development:
```bash
git clone https://github.com/truemagic-coder/solana-agent.git
cd solana-agent
poetry install
cp .env.example .env  # configure your test keys
make x402-sdk-smoke  # run the full validation suite
```

**Pro move:** Start with the memory-enabled trading example above. You'll immediately see the power of persistent context + real tool execution.

## Dead-Simple Setup for Powerful Agents

You only need to configure **three things**:

1. **Payment method** (`x402_private_key` or `x402_privy`)
2. **Memory behavior** (default = persistent `solana-agent-memory`)
3. **Agent personality** via natural language instructions

Everything else — hosted endpoint, tool routing, memory sessions, budgeting, observability — is handled for you.

**`config["ai"]`** is the canonical configuration key (with backward compatibility for `config["openai"]`).

### Config Cheatsheet

| Key | Recommendation | Why It Matters |
|-----|----------------|---------------|
| `auth_mode` | `"x402_private_key"` or `"x402_privy"` | Enables seamless onchain payments and wallet actions |
| `model` | Omit (defaults to memory) or `"stateless"` | Controls persistent memory vs one-shot responses |
| `base_url` | Omit it | Uses our production hosted service at `https://ai.solana-agent.com/v1` |
| `x402_preferred_asset` | Optional default: `"USDC"` | Explicitly pins hosted settlement to USDC; most apps can leave it unset |
| `memory_ttl_tier` (in runtime_context) | `"work"` or `"project"` | Balances retention vs cost for your use case |

## Quick Start: Ship Your First Agent in < 2 Minutes

### Private Key x402

```python
import os
from solana_agent import SolanaAgent

config = {
    "ai": {
        "auth_mode": "x402_private_key",
        "private_key": os.environ["X402_PRIVATE_KEY"],  # Base58 Solana key
    },
    "agents": [{
        "name": "default_agent",
        "instructions": "You are an expert Solana trading and portfolio assistant. Be concise, accurate, and proactive about opportunities.",
        "specialization": "trading",
    }]
}

agent = SolanaAgent(config=config)

# Your agent now has memory, tools, and payment capability
async for chunk in agent.process(
    user_id="trader_42",
    prompt="Analyze my wallet for any unusual token activity and suggest 2 high-conviction trades.",
    runtime_context={
        "conversation_id": "session-trader42-2026",
    }
):
    print(chunk, end="")
```

**This single agent can:** analyze wallets, surface market insights via Birdeye/Vybe, execute Jupiter swaps or Kamino positions, remember your risk preferences across sessions, and suggest actions proactively.

### Privy Wallet x402 (Recommended for embedded-wallet apps)

```python
import os
from solana_agent import SolanaAgent

config = {
    "ai": {
        "auth_mode": "x402_privy",
        "privy_app_id": os.environ["PRIVY_APP_ID"],
        "privy_app_secret": os.environ["PRIVY_APP_SECRET"],
    },
    "agents": [{
        "name": "trading_agent",
        "instructions": "You are a world-class Solana DeFi strategist. Prioritize user safety, clear explanations, and optimal execution.",
        "specialization": "defi",
    }]
}

agent = SolanaAgent(config=config)

runtime_context = {
    "privy_wallet_id": "user-wallet-from-privy",
    "conversation_id": "defi-session-xyz-2026",
    "memory_ttl_tier": "project",  # Longer retention for ongoing strategies
}

async for chunk in agent.process(
    user_id="user_789",
    prompt="Review my current positions, check for any impermanent loss risks on Kamino, and recommend rebalancing if needed.",
    runtime_context=runtime_context,
):
    print(chunk, end="")
```

Hosted x402 settlement is USDC-only on Solana. Most apps can leave `x402_preferred_asset` unset; if you keep it for explicitness or backward compatibility, it must be `USDC`.

If you want the hosted search add-on on a specific request, pass `search_enabled=True` directly to `process()`. The SDK forwards the flag automatically, and because search-enabled hosted requests are non-streaming in v1, the final answer is returned as a single text chunk.

```python
async for chunk in agent.process(
    user_id="user_789",
    message="Summarize the latest developments around Solana stablecoin flows.",
    runtime_context={"conversation_id": "market-brief-2026-05-04"},
    search_enabled=True,
):
    print(chunk, end="")
```

For the CLI, the same add-on is exposed as `solana-agent chat --search-enabled`.

## Master Your Agent's Memory

Memory is what separates toys from production agents. 

`conversation_id` + `memory_ttl_tier` give you **full control** over persistence without complex infrastructure:

- **Same `conversation_id`** = continuous memory thread (the agent "remembers" previous interactions perfectly).
- **Unique `conversation_id`** per user/session = perfect isolation (ideal for multi-tenant apps, tests, or isolated workflows).
- Pass via `runtime_context` on every `process()` call — the SDK forwards it to our hosted memory service automatically.
- Choose `"work"` (7-day retention, default) or `"project"` (30-day + $0.50 surcharge) based on your needs.

**Example with persistent memory:**

```python
runtime_context = {
    "conversation_id": "portfolio-alice-0426",  # Reuse this ID for Alice's ongoing conversation
    "memory_ttl_tier": "project"  # 30 days of context history
}

response = await agent.process(
    user_id="alice",
    prompt="Based on our last discussion about high-yield vaults, what's the current best opportunity?",
    runtime_context=runtime_context
)
```

This is the secret sauce behind coherent, long-running agents that feel truly intelligent. No more "as I explained earlier..." failures.

## Inspect Billing Without a Dashboard

The hosted wallet account surface is available directly from the SDK when you use `x402_private_key` or `x402_privy` auth.

The SDK now handles the hosted wallet challenge flow for you. Account reads automatically fetch a challenge, sign it with the configured wallet, and send the required headers. Hosted chat requests in wallet-backed x402 modes also attach the same wallet auth headers so Explorer allowance and monthly volume discounts can be quoted against the correct wallet account before settlement.

```python
# Private-key x402 mode: no extra runtime auth context required.
summary = await agent.get_account_summary()
pricing = await agent.get_pricing_info()

# Privy x402 mode: include the runtime wallet identity.
runtime_context = {"privy_wallet_id": "user-wallet-from-privy"}

usage = await agent.get_usage_report(
    "day",
    group_by="conversation",
    runtime_context=runtime_context,
)
forecast = await agent.get_usage_forecast(
    window_days=30,
    runtime_context=runtime_context,
)
```

For `x402_privy`, continue passing `runtime_context = {"privy_wallet_id": "..."}` on both account reads and hosted requests so the SDK can resolve the correct embedded wallet signer.

Use `group_by="conversation"` when your requests include `conversation_id` in `runtime_context` and you want spend and token usage broken out by conversation.

The same hosted account surface is available from the CLI:

```bash
solana-agent account summary --config config.json
solana-agent account usage --config config.json --granularity month --group-by conversation
solana-agent account forecast --config config.json --window-days 30
solana-agent account pricing --config config.json

# Privy mode
solana-agent account summary --config config.json --privy-wallet-id wallet-123
```

Wallet lifecycle and self-custody export are also available directly from the SDK and CLI:

```python
wallet = await agent.create_wallet(user_id="did:privy:user123")
address = await agent.get_wallet_address(user_id="did:privy:user123")
private_key = await agent.export_wallet_private_key(wallet_id=wallet["wallet_id"])
```

```bash
solana-agent wallet create --config config.json --user-id did:privy:user123
solana-agent wallet address --config config.json --user-id did:privy:user123
solana-agent wallet export --config config.json --wallet-id wallet-123
```

Hosted search can also be exercised from the CLI and smoke harness:

```bash
solana-agent chat --config config.json --search-enabled

/home/bevan/solana-agent/.venv/bin/python scripts/openai_x402_smoke.py \
    --scenario sdk-stateless \
    --search-enabled
```

## Public SDK Surface

The public package is intentionally small now:

- Hosted `solana-agent-memory` and `solana-agent-chat` requests
- Wallet create, address lookup, and Privy private-key export
- `mcp` for external MCP servers
- `x402_request` for direct paid HTTP access

The built-in Solana workflows are hosted capabilities, not public SDK tool modules. That keeps the package boundary aligned with the product you can actually rely on.

## Local Development & Validation

The production hosted endpoint (`https://ai.solana-agent.com/v1`) is the default for all examples.

For custom local testing, override `base_url` in your `ai` config to point at your own compatible endpoint.

**Quick Validation Commands:**

```bash
# Basic HTTP smoke tests
make x402-smoke

# Full SDK end-to-end tests (recommended)
make x402-sdk-smoke

# Run specific memory-focused scenario
poetry run python scripts/openai_x402_smoke.py --scenario memory-stream
```

These harnesses test memory recall, tool execution, payment flows, budgeting, and error handling — giving you confidence before deploying to production.

## Ready to Build the Future of Onchain Intelligence?

Solana Agent is more than a library — it's the foundation for the next wave of autonomous economic agents on Solana.

**What will you build?**
- Autonomous trading agents
- Personalized DeFi advisors
- Onchain customer support with wallet context
- Multi-agent research teams
- NFT collection managers with memory of owner preferences

**Next Steps:**
1. Install and run the quick start example above
2. Review the [full documentation](docs/index.rst) and bundled tools
3. Join the conversation in our community (link in repo)
4. Deploy your first production agent today

The scores prove it works. The tools prove it's powerful. The simplicity proves it's ready for *you*.

---

**Star the repo if you're building the onchain AI future.** Contributions welcome — especially around new tools, benchmark improvements, and real-world agent patterns.

*Solana Agent — Memory. Execution. Results.*

```bash
make x402-sdk-smoke SDK_SCENARIO=sdk-memory
```

For SDK memory smoke runs, the harness auto-generates a unique `conversation_id` when you do not provide one so the recall check is isolated from prior runs.

If you want to reuse a specific hosted memory thread across repeated smoke runs, set `X402_SMOKE_CONVERSATION_ID` in `.env` or pass `--conversation-id` directly.

The difference is intentional and ensures both the low-level payment contract and the full high-level SDK (with memory forwarding, tool orchestration, and `config["ai"]` resolution) are validated.

**For full details on development, testing, and contribution, see `docs/development.md` or the Makefile.**

## Migration from Previous Versions

- Use **`config["ai"]`** as the canonical key (the legacy `config["openai"]` alias remains for compatibility with a deprecation warning).
- Configure x402 auth modes instead of raw provider API keys.
- Leverage hosted remote memory via `runtime_context["conversation_id"]` instead of local stores.
- Default to the memory SKU for most agents; use `model: "stateless"` only when persistence is not needed.
- All Solana tools are now included — no separate Agent Kit installation required.

See `V34_AGI_X402_CHECKLIST.md` for the full transition guide.

**Live Documentation:** https://docs.solana-agent.com

## License

MIT. See [`LICENSE`](LICENSE) for details.

---

**Built for the Solana economy. Powered by real intelligence, real memory, and real execution.**

*Start building agents that matter.*

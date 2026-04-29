# Solana Agent v34.0.0 AGI x402 Checklist

Phased execution checklist for converging Solana Agent onto the Solana Agent AGI runtime and tool surface.

Current checkpoint:

- 2026-04-29: `pytest tests/unit -q -W error` passed (`209 passed`).
- 2026-04-29: local smoke `--scenario all` passed via `source ~/.zshrc` with `OPENAI_API_OPS_ACCESS_TOKEN=123` (`18/18 checks passed`).
- 2026-04-29: Phase 3 runtime cut landed: local `mongo`/`zep` config is rejected at the factory boundary and local history APIs are explicitly remote-only.
- 2026-04-29: Phase 1 tool surface merge landed: 35 first-party plugins now register from `solana_agent.tools`, import cleanly, and are covered by focused contract tests.

## Locked Decisions

- [x] OpenAI-compatible transport remains the client protocol surface.
- [x] Solana Agent AGI is the only AI provider target for v34.0.0.
- [x] Local development targets the AGI-compatible service running locally first.
- [x] Production target is `https://agi.solana-agent.com/v1` after local validation is complete.
- [x] Default model behavior uses remote memory via `solana-agent-memory` unless explicitly overridden.
- [x] Stateless behavior remains available through explicit override to `solana-agent-chat`.
- [x] Remote memory is the only memory mode for v34.0.0.
- [x] Zep is deleted from the public contract and implementation.
- [x] Mongo is removed from the Solana Agent runtime path.
- [x] x402-funded calls use a configured private Solana wallet key or Privy-backed runtime wallet export.
- [x] The AGI smoke harness is reused in Solana Agent for local validation.

## Phase 0: Contract Freeze

- [x] Freeze the Solana Agent v34.0.0 target contract in code and docs before porting.
- [x] Treat AGI as the only supported upstream AI runtime.
- [x] Freeze the default remote model ID to `solana-agent-memory`.
- [x] Freeze the explicit stateless override model ID to `solana-agent-chat`.
- [x] Freeze the config requirement that a private Solana key or Privy-backed runtime wallet is needed for funded x402 AI calls.
- [x] Freeze the rule that Solana Agent no longer owns long-term memory locally.
- [x] Freeze the rule that local-first smoke validation is required before any AGI production cutover.

Exit criteria:

- [x] One written contract exists for provider, auth, model defaults, and memory ownership.
- [x] No implementation phase proceeds with unresolved ambiguity about local versus remote memory behavior.

## Phase 1: Tool Surface Merge

- [x] Inventory the current Solana Agent Kit surface that is not yet first-party in Solana Agent.
- [x] Inventory the already-merged AGI tool surface and use it as the source of truth.
- [x] Port missing official tools into the Solana Agent package under a first-party tools namespace.
- [x] Port plugin entry points and packaging metadata needed for the merged tools.
- [x] Remove any remaining runtime assumption that users must install Solana Agent Kit separately.
- [x] Add or port focused tests covering the merged official tools.
- [ ] Decide the post-merge lifecycle of the standalone Solana Agent Kit repo.

Exit criteria:

- [x] Solana Agent ships the full intended official tool surface from one package line.
- [x] Tool tests pass from the Solana Agent repo without requiring Solana Agent Kit as a separate runtime dependency.

## Phase 2: AGI x402 Provider Path

- [x] Port the AGI x402-capable OpenAI-compatible adapter behavior into Solana Agent.
- [x] Support `auth_mode: x402_private_key` for AI calls.
- [x] Support `auth_mode: x402_privy` for AI calls.
- [x] Accept a base58 private Solana key through config or env for funded AI requests.
- [x] Set the default OpenAI-compatible `base_url` to the local AGI service during development.
- [ ] Set the production-ready target `base_url` to `https://agi.solana-agent.com/v1`.
- [x] Set the default model to `solana-agent-memory`.
- [x] Support explicit per-config override to `solana-agent-chat`.
- [x] Remove legacy provider selection branches for OpenAI, Cerebras, Groq, Grok, and other direct model vendors from the public Solana Agent contract.

Exit criteria:

- [ ] A Solana Agent config can make AI calls using only the AGI x402 transport path.
- [x] No public Solana Agent example still presents direct OpenAI, Cerebras, Groq, or Grok as first-class LLM options.

## Phase 3: Remote Memory-Only Migration

- [x] Remove Zep-backed memory code paths.
- [x] Remove Mongo-backed conversational memory code paths from the runtime flow.
- [x] Replace local memory assumptions with AGI remote memory defaults.
- [x] Ensure the default Solana Agent query path targets `solana-agent-memory`.
- [x] Ensure stateless override targets `solana-agent-chat` without local memory side effects.
- [x] Remove any double-memory behavior where Solana Agent would both inject local memory and call the remote memory SKU.
- [x] Audit delete-history and history-related client methods for compatibility with remote-only memory behavior.

Exit criteria:

- [x] Solana Agent no longer depends on local Mongo or Zep memory for the default runtime behavior.
- [x] Remote memory is the only default memory system in the client contract.

## Phase 4: Config Shape And Defaults

- [x] Define the canonical v34 config shape for AGI x402-only usage.
- [x] Define `base_url`, `auth_mode`, and x402 payer credential support in the OpenAI-compatible config surface.
- [x] Default `model` to `solana-agent-memory` when omitted.
- [x] Provide a documented override for `solana-agent-chat`.
- [x] Remove stale config examples that still mention Zep or Mongo for default memory.
- [x] Remove stale config examples that still mention OpenAI, Cerebras, Groq, or Grok as primary providers.
- [x] Add config validation that fails fast when the x402 private key is missing.
- [x] Support Privy app credentials plus runtime `privy_wallet_id` for funded AI requests.

Suggested target config:

```python
config = {
    "openai": {
        "auth_mode": "x402_private_key",
        "private_key": "your-base58-solana-private-key",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "solana-agent-memory",
        "stateless_model": "solana-agent-chat",
    }
}
```

Exit criteria:

- [x] One canonical v34 config example exists and matches implementation.
- [x] Misconfigured private-key auth fails with explicit validation errors.

## Phase 5: Smoke Harness Reuse

- [x] Reuse the AGI x402 smoke harness from the current workspace in Solana Agent.
- [x] Adapt the harness so it can validate Solana Agent against a local AGI-compatible endpoint.
- [x] Keep the fast success bundle for routine local validation.
- [x] Keep targeted memory, memory-stream, duplicate, and error scenarios available for deeper checks.
- [x] Add Solana Agent-side instructions for running the smoke harness locally.
- [x] Ensure local smoke validation is the release gate before testing against production AGI.

Required local checks:

- [x] `health`
- [x] `stateless`
- [x] `memory`
- [x] `memory-stream`
- [x] `duplicate`
- [x] `stream`
- [x] error scenarios relevant to x402 settlement, idempotency, memory unavailable, and upstream failures

Exit criteria:

- [x] Solana Agent can be validated locally against the AGI-compatible service with the shared harness.
- [x] The same harness contract is ready to point at production later without redesign.

## Phase 6: Docs And Migration

- [x] Rewrite the Solana Agent README to describe AGI x402-only AI usage.
- [x] Remove provider marketing and examples for direct OpenAI, Cerebras, Groq, and Grok.
- [x] Remove Zep and Mongo from the default stack description.
- [x] Add a migration section from legacy v33 config to v34 config.
- [x] Add a tool-merge migration note for users who still think in terms of Solana Agent Kit.
- [x] Add explicit local-first setup steps for AGI-compatible testing.
- [x] Add explicit production cutover steps for `https://agi.solana-agent.com/v1` after local validation.

Exit criteria:

- [ ] README, package metadata, and examples all describe the same v34 contract.
- [ ] A user can migrate without reading AGI internals or old Kit docs.

## Phase 7: Release Prep

- [x] Run focused regression tests for provider routing, merged tools, and client config validation.
- [x] Run the shared smoke harness locally against the AGI-compatible service until green.
- [x] Confirm the local smoke contract before pointing at production AGI.
- [ ] Bump package version from `33.3.2` to `34.0.0`.
- [ ] Write release notes describing the breaking contract changes.
- [ ] Call out removed dependencies and removed provider modes explicitly.

Breaking changes to announce:

- [ ] Zep removed
- [ ] Mongo removed from the Solana Agent runtime path
- [ ] AGI x402 is the only AI provider contract
- [ ] `solana-agent-memory` is the default model behavior
- [ ] private Solana key is required for funded AI requests
- [ ] Solana Agent Kit is no longer a separate required runtime surface

Exit criteria:

- [ ] The repo is releasable as `v34.0.0`.
- [ ] The release notes clearly explain migration and removed behaviors.

## Open Questions To Resolve During Execution

- [ ] Decide whether the OpenAI-compatible Python SDK remains the permanent transport dependency or only an interim transport layer.
- [ ] Decide whether any non-AI outbound x402 tools remain in scope for v34 or should be deferred.
- [ ] Decide whether client history APIs should become pass-through wrappers to remote AGI behavior or be reduced in scope.
- [ ] Decide the archive or freeze posture for the standalone Solana Agent Kit repo after merge completion.

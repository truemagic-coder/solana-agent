Solana Agent Documentation
=========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

Overview
--------

Solana Agent runs on a local-first AGI x402 runtime contract.

Current public runtime expectations:

- the only supported LLM config section is ``openai``
- ``auth_mode: "x402_private_key"`` and ``auth_mode: "x402_privy"`` are the funded AGI paths
- ``solana-agent-memory`` is the default model behavior
- ``solana-agent-chat`` is the explicit stateless override
- remote AGI memory is the only supported memory mode
- local Mongo and Zep are not part of the runtime path
- the official first-party tool surface ships in this package and registers through ``solana_agent.plugins``

Quick Start
-----------

Install local development dependencies:

.. code-block:: bash

   poetry install
   cp .env.example .env

Set at least these values in ``.env``:

- ``X402_PRIVATE_KEY``
- ``SOLANA_RPC_URL`` when your local x402 settlement flow needs an explicit RPC endpoint

Minimal Config
--------------

.. code-block:: python

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

Model Selection
---------------

In AGI x402 mode:

- omit ``model`` or set it to ``"memory"`` to use ``solana-agent-memory``
- set ``model`` to ``"stateless"`` to resolve through ``stateless_model``
- set ``model`` directly to ``"solana-agent-chat"`` for the explicit stateless SKU

Privy Payer Example
-------------------

.. code-block:: python

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

Bundled Tool Surface
--------------------

The official first-party tools are bundled directly in ``solana_agent.tools``. Do not install Solana Agent Kit as a separate runtime dependency for v34.

The bundled surface includes Birdeye, Jupiter, Kamino, MCP, Privy account and transaction tools, Solana transfer and swap flows, Rugcheck, Search, Technical Analysis, Token Math, Vybe, Image Generation, and x402 request helpers.

Local Smoke Validation
----------------------

Run the fast local success bundle:

.. code-block:: bash

   make x402-smoke

Run the full local matrix:

.. code-block:: bash

   make x402-smoke SCENARIO=all

Run one targeted scenario directly:

.. code-block:: bash

   poetry run python scripts/openai_x402_smoke.py --scenario memory-stream

The smoke harness covers:

- ``health``
- ``stateless``
- ``memory``
- ``memory-stream``
- ``duplicate``
- ``stream``
- public error scenarios
- internal fault-injection scenarios when the local AGI service enables them

After local validation is green, point ``base_url`` at ``https://agi.solana-agent.com/v1`` and rerun the same scenarios before production use.

Migration Notes
---------------

When moving from the v33 contract to the v34 runtime path:

- replace direct provider sections such as ``groq``, ``cerebras``, and ``grok`` with a single ``openai`` transport section
- replace upstream model-provider API keys with x402 payer configuration
- replace local-memory assumptions with the AGI remote-memory default
- use ``model: "stateless"`` or ``model: "solana-agent-chat"`` when you need the stateless SKU
- use the bundled first-party tools in this package rather than installing a separate Solana Agent Kit runtime
- treat conversation history and memory as remote-only runtime behavior

The broader migration plan is tracked in ``V34_AGI_X402_CHECKLIST.md``.

Development
-----------

Useful local commands:

.. code-block:: bash

   poetry run pytest tests/unit -q -W error
   poetry run python scripts/openai_x402_smoke.py --scenario all
   make livehtml

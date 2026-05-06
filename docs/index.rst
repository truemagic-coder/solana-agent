Solana Agent Documentation
==========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

Overview
--------

Solana Agent's public SDK targets the hosted AGI runtime.

Current public runtime expectations:

- the canonical AI config section is ``ai``
- ``openai`` remains supported as a compatibility alias
- hosted wallet and Privy integration stay on the service side
- ``solana-agent-memory`` is the default model behavior
- ``solana-agent-chat`` is the explicit stateless override
- remote AGI memory is the only supported memory mode
- local Mongo and Zep are not part of the runtime path
- the official first-party tool surface ships in this package and registers through ``solana_agent.plugins``

Development Setup
-----------------

Install local development dependencies:

.. code-block:: bash

   poetry install
   cp .env.example .env

Quick Start
-----------

Create a hosted Privy user from the CLI:

.. code-block:: bash

   solana-agent wallet menu

Then pass that ``privy_user_id`` into the SDK:

.. code-block:: python

   from solana_agent import SolanaAgent

   solana_agent = SolanaAgent(
      instructions="You are a helpful Solana AI assistant.",
      privy_user_id="did:privy:user123",
   )

   context = await solana_agent.context(
      conversation_id="wallet-session-1",
      model="chat",
   )

   response = await solana_agent.message(
      message="Check my wallet activity.",
      **context,
   )

   print(response)

``context()``, ``message()``, ``process()``, and ``process_message()`` already
read request identity from ``config.ai.privy_user_id`` or the constructor
``privy_user_id``. Do not pass ``privy_user_id`` again, and do not nest request
metadata under a ``runtime_context`` keyword.

Model Selection
---------------

In hosted AGI mode:

- omit ``model`` or set it to ``"memory"`` to use ``solana-agent-memory``
- set ``model`` to ``"chat"`` to use ``solana-agent-chat``
- set ``model`` to ``"stateless"`` to resolve through ``stateless_model``
- set ``model`` directly to ``"solana-agent-chat"`` for the explicit stateless SKU

Public SDK Surface
------------------

The public SDK surface is intentionally narrow. Use the hosted chat and memory SKUs for built-in Solana workflows.

The supported public modules are:

- ``mcp`` for external MCP servers
- hosted Privy user and wallet lifecycle methods exposed on ``SolanaAgent``

The public SDK does not support ``x402_request`` as a client-side tool.

MCP Tools
---------

Connect Streamable HTTP MCP servers by enabling the ``mcp`` tool and adding
server config:

.. code-block:: python

   import os
   from solana_agent import SolanaAgent

   solana_agent = SolanaAgent(
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

   context = await solana_agent.context(
      conversation_id="mcp-demo",
      model="chat",
   )

   response = await solana_agent.message(
      "Use the connected MCP tools to summarize my latest CRM tasks.",
      **context,
   )

   print(response)

Wallet And Billing Helpers
--------------------------

Set ``config.ai.privy_user_id`` before calling account helpers. The hosted
service manages a single active wallet for that user, with previous rotated
wallets returned as ``old_wallets``.

.. code-block:: python

   privy_user = await solana_agent.create_privy_user()

   wallet_address = await solana_agent.get_wallet_address()

   private_key = await solana_agent.export_wallet_private_key()

   rotated_wallet = await solana_agent.rotate_wallet(
      privy_user_id=privy_user["privy_user_id"],
   )

   summary = await solana_agent.get_account_summary()
   report = await solana_agent.get_usage_report("month")
   forecast = await solana_agent.get_usage_forecast()
   pricing = await solana_agent.get_pricing_info()

   print(privy_user["privy_user_id"])
   print(wallet_address)
   print(private_key)
   print(summary)
   print(report)
   print(forecast)
   print(pricing)

Use ``solana-agent wallet export --yes`` to export from the CLI. Pass
``--wallet-id`` when exporting an older rotated wallet from ``old_wallets``.

The hosted account service passes ``tooling`` and ``protocols`` sections through
on summary, usage, and forecast responses. Gasless metrics are reported in
lamports as ``gasless_savings_lamports`` for users and
``gasless_fee_payer_cost_lamports`` for the hosted fee payer.

Migration Notes
---------------

When moving from the v33 contract to the v34 runtime path:

- replace direct provider sections such as ``groq``, ``cerebras``, and ``grok`` with a single ``ai`` transport section
- move single-agent instructions and model selection into the ``ai`` section
- replace local-memory assumptions with the AGI remote-memory default
- use ``model: "stateless"`` or ``model: "solana-agent-chat"`` when you need the stateless SKU
- use hosted Solana workflows or external MCP servers instead of expecting bundled public tool modules in this package
- treat conversation history and memory as remote-only runtime behavior

Development
-----------

Useful local commands:

.. code-block:: bash

   poetry run pytest tests/unit -q -W error
   make livehtml

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

- the canonical AI config section is ``ai``
- ``openai`` remains supported as a compatibility alias
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
      "ai": {
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
      "ai": {
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

Public SDK Surface
------------------

The public SDK surface is intentionally narrow. Use the hosted chat and memory SKUs for built-in Solana workflows.

The supported public modules are:

- ``mcp`` for external MCP servers
- ``x402_request`` for direct paid HTTP access
- hosted wallet lifecycle methods exposed on ``SolanaAgent``

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

After local validation is green, point ``base_url`` at ``https://ai.solana-agent.com/v1`` and rerun the same scenarios before production use.

Migration Notes
---------------

When moving from the v33 contract to the v34 runtime path:

- replace direct provider sections such as ``groq``, ``cerebras``, and ``grok`` with a single ``ai`` transport section
- replace upstream model-provider API keys with x402 payer configuration
- replace local-memory assumptions with the AGI remote-memory default
- use ``model: "stateless"`` or ``model: "solana-agent-chat"`` when you need the stateless SKU
- use hosted Solana workflows or external MCP servers instead of expecting bundled public tool modules in this package
- treat conversation history and memory as remote-only runtime behavior

The broader migration plan is tracked in ``V34_AGI_X402_CHECKLIST.md``.

Development
-----------

Useful local commands:

.. code-block:: bash

   poetry run pytest tests/unit -q -W error
   poetry run python scripts/openai_x402_smoke.py --scenario all
   make livehtml
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

- the canonical AI config section is ``ai``
- ``openai`` remains supported as a compatibility alias
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
      "ai": {
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
      "ai": {
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

Public SDK Surface
------------------

The public SDK surface is intentionally narrow. Use the hosted chat and memory SKUs for built-in Solana workflows.

The supported public modules are:

- ``mcp`` for external MCP servers
- ``x402_request`` for direct paid HTTP access
- hosted wallet lifecycle methods exposed on ``SolanaAgent``

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

After local validation is green, point ``base_url`` at ``https://ai.solana-agent.com/v1`` and rerun the same scenarios before production use.

Migration Notes
---------------

When moving from the v33 contract to the v34 runtime path:

- replace direct provider sections such as ``groq``, ``cerebras``, and ``grok`` with a single ``ai`` transport section
- replace upstream model-provider API keys with x402 payer configuration
- replace local-memory assumptions with the AGI remote-memory default
- use ``model: "stateless"`` or ``model: "solana-agent-chat"`` when you need the stateless SKU
- use hosted Solana workflows or external MCP servers instead of expecting bundled public tool modules in this package
- treat conversation history and memory as remote-only runtime behavior

The broader migration plan is tracked in ``V34_AGI_X402_CHECKLIST.md``.

Development
-----------

Useful local commands:

.. code-block:: bash

   poetry run pytest tests/unit -q -W error
   poetry run python scripts/openai_x402_smoke.py --scenario all
   make livehtml
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

Overview
--------

Solana Agent runs on a local-first AGI x402 runtime contract.

Current public runtime expectations:

- the canonical AI config section is ``ai``
- ``openai`` remains supported as a compatibility alias
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
               # Example using the built-in PII guardrail for output (with defaults)
               {
                  "class": "solana_agent.guardrails.pii.PII"
                  # No config needed to use defaults
               }
         ]
      },
   }

Example Custom Guardrails - Optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Guardrails don't apply to structured outputs.

.. code-block:: python

   from solana_agent import InputGuardrail, OutputGuardrail
   import logging

   logger = logging.getLogger(__name__)

   class MyInputGuardrail(InputGuardrail):
      def __init__(self, config=None):
         super().__init__(config)
         self.setting1 = self.config.get("setting1", "default_value")
         logger.info(f"MyInputGuardrail initialized with setting1: {self.setting1}")

      async def process(self, text: str) -> str:
         # Example: Convert input to lowercase
         processed_text = text.lower()
         logger.debug(f"Input Guardrail processed: {text} -> {processed_text}")
         return processed_text

   class MyOutputGuardrail(OutputGuardrail):
      def __init__(self, config=None):
         super().__init__(config)
         self.filter_level = self.config.get("filter_level", "low")
         logger.info(f"MyOutputGuardrail initialized with filter_level: {self.filter_level}")

      async def process(self, text: str) -> str:
         # Example: Basic profanity filtering (replace with a real library)
         if self.filter_level == "high" and "badword" in text:
               processed_text = text.replace("badword", "*******")
               logger.warning(f"Output Guardrail filtered content.")
               return processed_text
         logger.debug("Output Guardrail passed text through.")
         return text


Tools
~~~~~~

Solana Agent Kit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install sakit

Inline Tool Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from solana_agent import SolanaAgent, Tool

   class TestTool(Tool):
      def __init__(self):
         # your tool initialization - delete the following pass
         pass

      @property
      def name(self) -> str:
         return "test_function"

      @property
      def description(self) -> str:
         return "Test function for Solana Agent"

      def configure(self, config: Dict[str, Any]) -> None:
         """Configure with all possible API key locations."""
         super().configure(config)

         # read your config values - delete the following pass
         pass

      def get_schema(self) -> Dict[str, Any]:
         # this is an example schema
         return {
            "type": "object",
            "properties": {
               "query": {"type": "string", "description": "Search query text"},
               "user_id": {"type": "string", "description": "User ID for the search session"}
            },
            "required": ["query", "user_id"],
            "additionalProperties": False,
         }

      async def execute(self, **params) -> Dict[str, Any]:
         try:
            # your tool logic
            result = "Your tool results"

            return {
               "status": "success",
               "result": result,
            }
         except Exception as e:
            return {
               "status": "error",
               "message": f"Error: {str(e)}",
            }

   config = {
      "ai": {
         "api_key": "your-openai-api-key",
      },
      "agents": [
         {
            "name": "research_specialist",
            "instructions": "You are an expert researcher who synthesizes complex information clearly.",
            "specialization": "Research and knowledge synthesis",
         },
         {
            "name": "customer_support",
            "instructions": "You provide friendly, helpful customer support responses.",
            "specialization": "Customer inquiries",
         }
      ],
   }

   solana_agent = SolanaAgent(config=config)

   test_tool = TestTool()

   solana_agent.register_tool(test_tool)

   async for response in solana_agent.process("user123", "What are the latest AI developments?"):
      print(response, end="")


Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prompt Injection at Runtime Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Useful for adding additional context to agent responses.

.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "ai": {
         "api_key": "your-openai-api-key",
      },
      "agents": [
         {
            "name": "research_specialist",
            "instructions": "You are an expert researcher who synthesizes complex information clearly.",
            "specialization": "Research and knowledge synthesis",
         },
         {
            "name": "customer_support",
            "instructions": "You provide friendly, helpful customer support responses.",
            "specialization": "Customer inquiries",
         }
      ],
   }

   solana_agent = SolanaAgent(config=config)

   async for response in solana_agent.process("user123", "What are the latest AI developments?", "Always end your sentences with eh?"):
      print(response, end="")

Custom Routing Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   
   from solana_agent import SolanaAgent
   from solana_agent.interfaces.services.routing import RoutingService as RoutingServiceInterface

   config = {
      "ai": {
         "api_key": "your-openai-api-key",
      },
      "agents": [
         {
            "name": "research_specialist",
            "instructions": "You are an expert researcher who synthesizes complex information clearly.",
            "specialization": "Research and knowledge synthesis",
         },
         {
            "name": "customer_support",
            "instructions": "You provide friendly, helpful customer support responses.",
            "specialization": "Customer inquiries",
         }
      ],
   }

   class Router(RoutingServiceInterface)
      def __init__(self):
         # your router initialization - delete the following pass
         pass

      async def route_query(self, query: str) -> str:
         # a simple example to route always to customer_support agent
         return "customer_support"

   router = Router()

   solana_agent = SolanaAgent(config=config)

   async for response in solana_agent.process("user123", "What are the latest AI developments?", router=router):
      print(response, end="")

API Reference
------------

Check out the :doc:`api/index` for detailed documentation of all modules and classes.

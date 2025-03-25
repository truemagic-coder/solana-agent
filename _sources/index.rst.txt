Solana Agent Documentation
=========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

Welcome to Solana Agent's documentation.

Getting Started
--------------

Installation
~~~~~~~~~~~

.. code-block:: bash

   pip install solana-agent

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "business": {
         "mission": "To provide users with a one-stop shop for their queries.",
         "values": {
               "Friendliness": "Users must be treated fairly, openly, and with friendliness.",
               "Ethical": "Agents must use a strong ethical framework in their interactions with users.",
         },
         "goals": [
               "Empower users with great answers to their queries.",
         ],
         "voice": "The voice of the brand is that of a research business."
      },
      "mongo": {
         "connection_string": "mongodb://localhost:27017",
         "database": "solana_agent"
      },
      "openai": {
         "api_key": "your-openai-api-key",
      },
      "zep": { # optional
         "api_key": "your-zep-api-key",
         "base_url": "your-zep-base-url", # not applicable if using Zep Cloud
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

   async for response in solana_agent.process("user123", "What are the latest AI developments?"):
      print(response, end="")


Plugin Usage
~~~~~~~~~~

Plugins like Solana Agent Kit (sakit) integrate automatically with Solana Agent.

.. code-block:: bash

   pip install sakit

.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "business": {
         "mission": "To provide users with a one-stop shop for their queries.",
         "values": {
               "Friendliness": "Users must be treated fairly, openly, and with friendliness.",
               "Ethical": "Agents must use a strong ethical framework in their interactions with users.",
         },
         "goals": [
               "Empower users with great answers to their queries.",
         ],
         "voice": "The voice of the brand is that of a research business."
      },
      "mongo": {
         "connection_string": "mongodb://localhost:27017",
         "database": "solana_agent"
      },
      "openai": {
         "api_key": "your-openai-api-key",
      },
      "zep": { # optional
         "api_key": "your-zep-api-key",
         "base_url": "your-zep-base-url", # not applicable if using Zep Cloud
      },
      "tools": {
        "search_internet": {
            "api_key": "your-perplexity-key", # Required
            "citations": True, # Optional, defaults to True
            "model": "sonar"  # Optional, defaults to "sonar"
        },
      },
      "agents": [
         {
               "name": "research_specialist",
               "instructions": "You are an expert researcher who synthesizes complex information clearly.",
               "specialization": "Research and knowledge synthesis",
               "tools": ["search_internet"],
         },
         {
               "name": "customer_support",
               "instructions": "You provide friendly, helpful customer support responses.",
               "specialization": "Customer inquiries",
         }
      ],
   }

   solana_agent = SolanaAgent(config=config)

   async for response in solana_agent.process("user123", "What are the latest AI developments?"):
      print(response, end="")

To create a plugin like Solana Agent Kit - read the code at https://github.com/truemagic-coder/solana-agent-kit

Custom Inline Tool Usage
~~~~~~~~~~

Creating a custom inline tool will register it will all agents.

.. code-block:: python

   from solana_agent import SolanaAgent
   from solana_agent.interfaces.plugins.plugins import Tool

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
               "required": ["query", "user_id"]
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
      "business": {
         "mission": "To provide users with a one-stop shop for their queries.",
         "values": {
               "Friendliness": "Users must be treated fairly, openly, and with friendliness.",
               "Ethical": "Agents must use a strong ethical framework in their interactions with users.",
         },
         "goals": [
               "Empower users with great answers to their queries.",
         ],
         "voice": "The voice of the brand is that of a research business."
      },
      "mongo": {
         "connection_string": "mongodb://localhost:27017",
         "database": "solana_agent"
      },
      "openai": {
         "api_key": "your-openai-api-key",
      },
      "zep": { # optional
         "api_key": "your-zep-api-key",
         "base_url": "your-zep-base-url", # not applicable if using Zep Cloud
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

Notes on Tools
-----
* Solana Agent agents can only call one tool per response.
* Solana Agent agents choose the best tool for the job.
* Solana Agent tools do not use OpenAI function calling.
* Solana Agent tools are async functions.

API Reference
------------

Check out the :doc:`api/index` for detailed documentation of all modules and classes.

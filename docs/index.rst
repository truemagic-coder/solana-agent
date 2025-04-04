Solana Agent Documentation
=========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

Welcome to Solana Agent's documentation. Solana Agent enables you to build an AI business in three lines of code!

Getting Started
--------------

Installation
~~~~~~~~~~~

.. code-block:: bash

   pip install solana-agent

Text/Text Streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "openai": {
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

   async for response in solana_agent.process("user123", "What are the latest AI developments?"):
      print(response, end="")

Audio/Audio Streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "openai": {
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

   audio_content = audio_file.read()

   async for response in solana_agent.process("user123", audio_content, output_format="audio", audio_voice="nova", audio_input_format="webm", audio_output_format="aac"):
      print(response, end="")


Text/Audio Streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "openai": {
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

   async for response in solana_agent.process("user123", "What is the latest news on Elon Musk?", output_format="audio", audio_voice="nova", audio_output_format="aac"):
      print(response, end="")

Audio/Text Streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "openai": {
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

   audio_content = audio_file.read()

   async for response in solana_agent.process("user123", audio_content, audio_input_format="aac"):
      print(response, end="")

Business Alignment Config - Optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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
   }

Conversational History Config - Optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = {
      "mongo": {
         "connection_string": "mongodb://localhost:27017",
         "database": "solana_agent"
      },
   }

Conversational Memory Config - Optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   
   config = {
      "zep": {
         "api_key": "your-zep-api-key",
      },
   }

Disable Internet Searching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async for response in solana_agent.process("user123", "Write me a poem.", internet_search=False):
      print(response, end="")


Customize Speech
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is an audio to audio example using the `audio_instructions` parameter.

You can prompt to control aspects of speech, including:

* Accent
* Emotional range
* Intonation
* Impressions
* Speed of speech
* Tone
* Whispering

.. code-block:: python
   
   async for response in solana_agent.process("user123", audio_content, output_format="audio", audio_voice="nova", audio_input_format="webm", audio_output_format="aac", audio_instructions="You speak with an American southern accent"):
      print(response, end="")


Tool Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash
   
   pip install sakit

.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "openai": {
         "api_key": "your-openai-api-key",
      },
      "tools": {
         "search_internet": {
               "api_key": "your-perplexity-api-key",
         },
      },
      "agents": [
         {
               "name": "research_specialist",
               "instructions": "You are an expert researcher who synthesizes complex information clearly. You use your search_internet tool to get the latest information.",
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

   async for response in solana_agent.process("user123", "What are the latest AI developments?", internet_search=False):
      print(response, end="")


Custom Inline Tool Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
      "openai": {
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

Custom Prompt Injection at Runtime Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Useful for Knowledge Base answers and FAQs.

.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "openai": {
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
      "openai": {
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

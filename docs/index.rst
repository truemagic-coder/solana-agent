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


Flows
~~~~~~

In both flows of single and multiple agents - it is one user query to one agent using one tool (if needed).

An agent can have multiple tools and will choose the best one to answer the user query.

Routing is determined by optimal domain expertise of the agent for the user query.

When the agent uses a tool it feeds the tool output back into the agent to generate the final response.

This is important as tools generally output unstructured and unformatted data that the agent needs to prepare for the user.

Keep this in mind while designing your agentic systems using Solana Agent.

.. code-block:: ascii

                        Single Agent                                     
                                                                           
      ┌────────┐        ┌─────────┐        ┌────────┐                    
      │        │        │         │        │        │                    
      │        │        │         │        │        │                    
      │  User  │◄──────►│  Agent  │◄──────►│  Tool  │                    
      │        │        │         │        │        │                    
      │        │        │         │        │        │                    
      └────────┘        └─────────┘        └────────┘                    
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                        Multiple Agents                                   
                                                                           
        ┌────────┐        ┌──────────┐        ┌─────────┐        ┌────────┐
        │        │        │          │        │         │        │        │
        │        │        │          │        │         │        │        │
   ┌───►│  User  ├───────►│  Router  ├───────►│  Agent  │◄──────►│  Tool  │
   │    │        │        │          │        │         │        │        │
   │    │        │        │          │        │         │        │        │
   │    └────────┘        └──────────┘        └────┬────┘        └────────┘
   │                                               │                       
   │                                               │                       
   │                                               │                       
   │                                               │                       
   └───────────────────────────────────────────────┘                       


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


agent/Audio Streaming
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
         "connection_string": "your-mongo-connection-string",
         "database": "your-database-name",
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

Internet Search (Plugin Tool Example)
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
            "api_key": "your-openai-api-key",
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

   async for response in solana_agent.process("user123", "What are the latest AI developments?"):
      print(response, end="")


Inline Tool Example
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

Prompt Injection at Runtime Example
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

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

In both flows of single and multiple agents - it is one user query to one agent using one or many tools (if needed).

An agent can have multiple tools and will choose the best ones to fulfill the user's query.

Routing is determined by optimal domain expertise of the agent for the user's query.

When the agent uses tools it feeds the tools output back to itself to generate the final response.

This is important as tools generally output unstructured and unformatted data that the agent needs to prepare for the user.

Keep this in mind while designing your agentic systems using Solana Agent.

.. code-block:: ascii

                        Single Agent                                     
                                                                           
      ┌────────┐        ┌─────────┐        ┌────────-┐                    
      │        │        │         │        │         │                    
      │        │        │         │        │         │                    
      │  User  │◄──────►│  Agent  │◄──────►│  Tools  │                    
      │        │        │         │        │         │                    
      │        │        │         │        │         │                    
      └────────┘        └─────────┘        └────────-┘                    
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                        Multiple Agents                                   
                                                                           
        ┌────────┐        ┌──────────┐        ┌─────────┐        ┌────────-┐
        │        │        │          │        │         │        │         │
        │        │        │          │        │         │        │         │
   ┌───►│  User  ├───────►│  Router  ├───────►│  Agent  │◄──────►│  Tools  │
   │    │        │        │          │        │         │        │         │
   │    │        │        │          │        │         │        │         │
   │    └────────┘        └──────────┘        └────┬────┘        └────────-┘
   │                                               │                       
   │                                               │                       
   │                                               │                       
   │                                               │                       
   └───────────────────────────────────────────────┘                       

Usage
~~~~~~

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


Image/Text Streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "openai": {
         "api_key": "your-openai-api-key",
      },
      "agents": [
         {
               "name": "vision_expert",
               "instructions": "You are an expert at analyzing images and answering questions about them.",
               "specialization": "Image analysis",
         }
      ],
   }

   solana_agent = SolanaAgent(config=config)

   # Example with an image URL
   image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

   # Example reading image bytes from a file
   image_bytes = await image_file.read()

   # You can mix URLs and bytes in the list
   images_to_process = [
      image_url,
      image_bytes,
   ]

   async for response in solana_agent.process("user123", "What is in this image? Describe the scene.", images=images_to_process):
      print(response, end="")


Agentic Forms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can attach a JSON Schema to any agent in your config so it can collect structured data conversationally.

.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "openai": {
         "api_key": "your-openai-api-key",
      },
      "agents": [
         {
            "name": "customer_support",
            "instructions": "You provide friendly, helpful customer support responses.",
            "specialization": "Customer inquiries",
            "capture_name": "contact_info",
            "capture_schema": {
               "type": "object",
               "properties": {
                  "email": { "type": "string" },
                  "phone": { "type": "string" },
                  "newsletter_subscribe": { "type": "boolean" }
               },
               "required": ["email"]
            }
         }
      ]
   }

   solana_agent = SolanaAgent(config=config)

   async for response in solana_agent.process("user123", "What are the latest AI developments?"):
      print(response, end="")


Command Line Interface (CLI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solana Agent includes a command-line interface (CLI) for text-based chat using a configuration file.

Ensure you have a valid configuration file (e.g., `config.json`) containing at least your OpenAI API key and agent definitions.

.. code-block:: json

   // config.json
   {
      "openai": {
         "api_key": "your-openai-api-key"
      },
      "agents": [
         {
               "name": "default_agent",
               "instructions": "You are a helpful AI assistant.",
               "specialization": "general"
         }
      ]
   }

Also ensure that you have `pip install uv` to call `uvx`.

.. code-block:: bash

   uvx solana-agent [OPTIONS]

   Options:

   --user-id TEXT: The user ID for the conversation (default: cli_user).
   --config TEXT: Path to the configuration JSON file (default: config.json).
   --prompt TEXT: Optional system prompt override for the agent.
   --help: Show help message and exit.

   # Using default config.json and user_id
   uvx solana-agent

   # Specifying user ID and config path
   uvx solana-agent --user-id my_cli_session --config ./my_agent_config.json

Optional Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Business Alignment - Optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Conversational History - Optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = {
      "mongo": {
         "connection_string": "your-mongo-connection-string",
         "database": "your-database-name",
      },
   }

Conversational Memory - Optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   
   config = {
      "zep": {
         "api_key": "your-zep-api-key",
      },
   }

Observability and Tracing - Optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = {
      "logfire": {
         "api_key": "your-logfire-write-token",
      },
   }

Guardrails - Optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Guardrails allow you to process and potentially modify user input before it reaches the agent (Input Guardrails) and agent output before it's sent back to the user (Output Guardrails). This is useful for implementing safety checks, content moderation, data sanitization, or custom transformations.

Guardrails don't apply to structured outputs.

Solana Agent provides a built-in PII scrubber based on scrubadub.

.. code-block:: python

   from solana_agent import SolanaAgent

   config = {
      "guardrails": {
         "input": [
               # Example using a custom input guardrail
               {
                  "class": "MyInputGuardrail",
                  "config": {"setting1": "value1"}
               },
               # Example using the built-in PII guardrail for input
               {
                  "class": "solana_agent.guardrails.pii.PII",
                  "config": {
                     "locale": "en_GB", # Optional: Specify locale (default: en_US)
                     "replacement": "[REDACTED]" # Optional: Custom replacement format
                  }
               }
         ],
         "output": [
               # Example using a custom output guardrail
               {
                  "class": "MyOutputGuardrail",
                  "config": {"filter_level": "high"}
               },
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


Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prompt Injection at Runtime Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Useful for adding additional context to agent responses.

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

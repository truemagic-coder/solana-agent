# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/solana-agent)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/ci.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/ci.yml)
[![Ruff Style](https://img.shields.io/badge/style-ruff-41B5BE)](https://github.com/astral-sh/ruff)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

## AI Agents for Solana

Build your AI agents in three lines of code!

## Why?
* Three lines of code setup
* Simple Agent Definition
* Streaming Responses
* Solana Integration
* Multi-Agent Swarm
* Multi-Modal (Images & Audio & Text)
* Conversational Memory & History
* Internet Search
* Intelligent Routing
* Business Alignment
* Extensible Tooling
* Autonomous Operation
* Smart Workflows
* Agentic Forms
* MCP Support
* Guardrails
* Image Generation
* Pydantic Logfire
* Tested & Secure
* Built in Python
* Powers [CometHeart](https://cometheart.com)
* Powers [Solana Agent Bot][https://t.me/solana_agent_bot]

## Unique Selling Proposition (USP) - Smart Workflows

Solana Agent is the first AI agent framework to deliver truly intelligent, dynamic workflows.

With Solana Agent, you can seamlessly define and integrate tools — such as Zapier MCP (for sending emails via Mailgun) and the Solana Balance tool — directly into your agent’s capabilities.

Then prompt your agent with natural language, for example:  
“Get my balances for my Solana wallet and then email them to me.”

Solana Agent will automatically orchestrate the workflow:  
It will first use the Solana Balance tool to retrieve your balances, then invoke the Zapier MCP tool to send the results via email — all without manual intervention or brittle, hardcoded logic.

You can also chain tool outputs (structured or unstructured) as inputs for subsequent tasks, enabling complex, multi-step automations with ease.

**The result?**  
A framework that is both powerful and simple — eliminating the need for static, fragile workflow definitions.  
Smart workflows are as easy as combining your tools and prompts.

## Features

* Easy three lines of code setup
* Simple agent definition using JSON
* Designed for a multi-agent swarm 
* Fast multi-modal processing of text, audio, and images
* Smart workflows that keep flows simple and smart
* Interact with the Solana blockchain with many useful tools
* MCP tool usage with first-class support for [Zapier](https://zapier.com/mcp)
* Integrated observability and tracing via [Pydantic Logfire](https://pydantic.dev/logfire)
* Persistent memory that preserves context across all agent interactions
* Quick Internet search to answer users' queries
* Streamlined message history for all agent interactions
* Intelligent query routing to agents with optimal domain expertise or your own custom routing
* Unified value system ensuring brand-aligned agent responses
* Powerful tool integration using standard Python packages and/or inline tools
* Assigned tools are utilized by agents automatically and effectively
* Input and output guardrails for content filtering, safety, and data sanitization
* Generate custom images based on text prompts with storage on S3 compatible services
* Deterministic agentic form filling in natural conversation
* Combine with event-driven systems to create autonomous agents

## Stack

### Tech

* [Python](https://python.org) - Programming Language
* [OpenAI](https://openai.com) - AI Model Provider
* [MongoDB](https://mongodb.com) - Conversational History (optional)
* [Zep Cloud](https://getzep.com) - Conversational Memory (optional)
* [Zapier](https://zapier.com) - App Integrations (optional)
* [Pydantic Logfire](https://pydantic.dev/logfire) - Observability and Tracing (optional)

### AI Models Used

**OpenAI**
* [gpt-5.2](https://platform.openai.com/docs/models/gpt-5.2) (agent & router)
* [text-embedding-3-large](https://platform.openai.com/docs/models/text-embedding-3-large) (embedding)
* [tts-1](https://platform.openai.com/docs/models/tts-1) (audio TTS)
* [gpt-4o-mini-transcribe](https://platform.openai.com/docs/models/gpt-4o-mini-transcribe) (audio transcription)

## Installation

You can install Solana Agent using pip:

`pip install solana-agent`

## Flows

In both flows of single and multiple agents - it is one user query to one agent using one or many tools (if needed).

An agent can have multiple tools and will choose the best ones to fulfill the user's query.

Routing is determined by optimal domain expertise of the agent for the user's query.

When the agent uses tools it feeds the tools output back to itself to generate the final response.

This is important as tools generally output unstructured and unformatted data that the agent needs to prepare for the user.

Keep this in mind while designing your agentic systems using Solana Agent.

```ascii
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
```

## Usage

### Text/Text Streaming

```python
from solana_agent import SolanaAgent

config = {
    "openai": {
        "api_key": "your-openai-api-key",
        "model": "gpt-5.2",  # Optional, defaults to gpt-5.2
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
```

### Audio/Audio Streaming

```python
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

audio_content = await audio_file.read()

async for response in solana_agent.process("user123", audio_content, output_format="audio", audio_voice="nova", audio_input_format="webm", audio_output_format="aac"):
    print(response, end="")
```

### Text/Audio Streaming

```python
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
```

### Audio/Text Streaming

```python
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

audio_content = await audio_file.read()

async for response in solana_agent.process("user123", audio_content, audio_input_format="aac"):
    print(response, end="")
```

### Image/Text Streaming

```python
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
```

### Agentic Forms

You can attach a JSON Schema to any agent in your config so it can collect structured data conversationally.

```python
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
```

### Command Line Interface (CLI)

Solana Agent includes a command-line interface (CLI) for text-based chat using a configuration file.

Ensure you have a valid configuration file (e.g., `config.json`) containing at least your OpenAI API key and agent definitions.

**./config.json**
```json
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
```

Also ensure that you have `pip install uv` to call `uvx`.

```bash
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
```

## Optional Feature Configs

### Business Alignment

```python
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
```

### Conversational History

```python
config = {
    "mongo": {
        "connection_string": "your-mongo-connection-string",
        "database": "your-database-name"
    },
}
```

### Conversational Memory

```python
config = {
    "zep": {
        "api_key": "your-zep-cloud-api-key",
    },
}
```

### Observability and Tracing

```python
config = {
    "logfire": {
        "api_key": "your-logfire-write-token",
    },
}
```

### Guardrails

Guardrails allow you to process and potentially modify user input before it reaches the agent (Input Guardrails) and agent output before it's sent back to the user (Output Guardrails). This is useful for implementing safety checks, content moderation, data sanitization, or custom transformations.

Guardrails don't work with structured outputs.

Solana Agent provides a built-in PII scrubber based on [scrubadub](https://github.com/LeapBeyond/scrubadub).

```python
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
```

#### Example Custom Guardrails

Guardrails don't work with structured outputs.

```python
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
```

## Tools

Tools empower agents to interact with external systems, fetch data, or perform actions. They can be used reactively within a user conversation or proactively when an agent is triggered autonomously.

Tools can be used from plugins like Solana Agent Kit (sakit) or via inline tools. Tools available via plugins integrate automatically with Solana Agent.

### Solana Agent Kit

[Solana Agent Kit](https://github.com/truemagic-coder/solana-agent-kit)

```bash
pip install sakit
```

### Inline Tool Example

```python
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

solana_agent.register_tool("customer_support", test_tool)

async for response in solana_agent.process("user123", "What are the latest AI developments?"):
    print(response, end="")
```

## Autonomous Operation & Event-Driven Agents

While Solana Agent facilitates request-response interactions, the underlying architecture supports building autonomous agents. You can achieve autonomy by orchestrating calls based on external triggers rather than direct user input.

**Key Concepts:**

*   **External Triggers:** Use schedulers like cron, message queues (RabbitMQ, Kafka), monitoring systems, webhooks, or other event sources to initiate agent actions.
*   **Programmatic Calls:** Instead of a user typing a message, your triggering system calls with a specific message (acting as instructions or data for the task) and potentially a dedicated user representing the autonomous process.
*   **Tool-Centric Tasks:** Autonomous agents often focus on executing specific tools. The prompt can instruct the agent to use a particular tool with given parameters derived from the triggering event.
*   **Example Scenario:** An agent could be triggered hourly by a scheduler. The `message` could be "Check the SOL balance for wallet X using the `solana` tool." The agent executes the tool, and the result could be logged or trigger another tool (e.g., using `mcp` to send an alert if the balance is low).

By combining Solana Agent's agent definitions, tool integration, and routing with external orchestration, you can create sophisticated autonomous systems.

## Advanced Customization

### Runtime Prompt Injection

```python
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
        }
    ],
}

solana_agent = SolanaAgent(config=config)

async for response in solana_agent.process("user123", "How do replace the latch on my dishwasher?", "This is my corporate appliance fixing FAQ"):
    print(response, end="")
```

### Custom Routing

In advanced cases like implementing a ticketing system on-top of Solana Agent - you can use your own router.

```python
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
```

## API Documentation

The official up-to-date documentation site

[Solana Agent Documentation Site](https://docs.solana-agent.com)

## Official Tools

The official collection of tools in one plugin

[Solana Agent Kit](https://github.com/truemagic-coder/solana-agent-kit)

## Example App

The official example app written in FastAPI and Next.js

[Solana Agent Example App](https://github.com/truemagic-coder/solana-agent-app)

## Demo App

The official demo app written in FastAPI and Next.js

[Solana Agent Demo App](https://demo.solana-agent.com)

## Agent Framework Comparisons

[Compare Python Agent Frameworks](https://github.com/truemagic-coder/solana-agent/wiki/Agent-Framework-Comparisons)

## Contributing

If you have a question, feedback, or feature request - please open a GitHub discussion.

If you find a bug - please open a GitHub issue.

We are currently accepting PRs if approved in discussions. Make sure all tests pass and the README & docs are updated.

To run the documentation site locally run `make livehtml` in the root directory.

To run the test suite locally run `poetry run pytest --cov=solana_agent --cov-report=html` in the root directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/solana-agent)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/ci.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/ci.yml)
[![Lines of Code](https://tokei.rs/b1/github/truemagic-coder/solana-agent?type=python&category=code&style=flat)](https://github.com/truemagic-coder/solana-agent)
[![Libraries.io dependency status for GitHub repo](https://img.shields.io/librariesio/github/truemagic-coder/solana-agent)](https://libraries.io/pypi/solana-agent)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

## Agentic IQ

Build your AI business in three lines of code!

## Why?
* Three lines of code setup
* Multi-Agent Swarm
* Multi-Modal Streaming (Text & Audio)
* Conversational Memory & History
* Built-in Internet Search
* Intelligent Routing
* Business Alignment
* Extensible Tooling
* Simple Business Definition
* Tested & Secure
* Built in Python
* Powers [CometHeart](https://cometheart.com) & [WalletBubbles](https://walletbubbles.com)

## Features

* Easy three lines of code setup
* Designed for a multi-agent swarm 
* Seamless text and audio streaming with real-time multi-modal processing
* Configurable audio voice characteristics via prompting
* Persistent memory that preserves context across all agent interactions
* Quick built-in Internet search to answer your queries
* Streamlined message history for all agent interactions
* Intelligent query routing to agents with optimal domain expertise or your own custom routing
* Unified value system ensuring brand-aligned agent responses
* Powerful tool integration using standard Python packages and/or inline tools
* Assigned tools are utilized by agents automatically and effectively
* Simple business definition using JSON

## Stack

* [Python](https://python.org) - Programming Language
* [OpenAI](https://openai.com) - LLM Provider
* [MongoDB](https://mongodb.com) - Conversational History (optional)
* [Zep Cloud](https://getzep.com) - Conversational Memory (optional)

## Installation

You can install Solana Agent using pip:

`pip install solana-agent`

## Usage

### Text/Text Streaming

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

audio_content = audio_file.read()

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

audio_content = audio_file.read()

async for response in solana_agent.process("user123", audio_content, audio_input_format="aac"):
    print(response, end="")
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
        "connection_string": "mongodb://localhost:27017",
        "database": "solana_agent"
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

### Disable Internet Searching

```python
async for response in solana_agent.process("user123", "Write me a poem.", internet_search=False):
    print(response, end="")
```

### Customize Speech

This is an audio to audio example using the `audio_instructions` parameter.

You can prompt to control aspects of speech, including:

* Accent
* Emotional range
* Intonation
* Impressions
* Speed of speech
* Tone
* Whispering

```python
async for response in solana_agent.process("user123", audio_content, output_format="audio", audio_voice="nova", audio_input_format="webm", audio_output_format="aac", audio_instructions="You speak with an American southern accent"):
    print(response, end="")
```

## Tools

Tools can be used from plugins like Solana Agent Kit (sakit) or via custom inline tools. Tools available via plugins integrate automatically with Solana Agent.

* Agents can only call one tool per response
* Agents choose the best tool for the job
* Tools do not use OpenAI function calling
* Tools are async functions

### Tool Example

`pip install sakit`

```python
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
```

### Custom Inline Tool Example

```python
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

solana_agent.register_tool("customer_support", test_tool)

async for response in solana_agent.process("user123", "What are the latest AI developments?"):
    print(response, end="")
```

## Agent Training

Many use-cases for Solana Agent require training your agents on your company data.

This can be accomplished via runtime prompt injection. Integrations that work well with this method are vector stores like Pinecone and FAQs.

This knowledge applies to all your AI agents.

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

## Custom Routing

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Agent Framework Comparisons

### Solana Agent vs OpenAI Agents SDK

| Feature                | Solana Agent                                   | OpenAI Agents SDK                              |
|------------------------|-----------------------------------------------|-----------------------------------------------|
| **Architecture**       | Service-oriented with query routing           | Agent-based with explicit handoffs            |
| **Configuration**      | JSON-based config with minimal code           | Python code-based agent definitions           |
| **Multi-Agent**        | Automatic specialization routing              | Direct agent-to-agent handoffs                |
| **Memory**             | Integrated MongoDB/Zep persistence            | In-context memory within message history      |
| **Multi-Modal**        | Full audio/text streaming built-in            | Optional voice support via add-on package     |
| **Model Support**      | Currently OpenAI focused                      | Any provider with OpenAI-compatible API       |
| **Tool Integration**   | Class-based tools with registry               | Function decorators with `@function_tool`     |
| **Debugging**          | Standard logging                              | Advanced tracing with visualization           |
| **Safety**             | Basic error handling                          | Configurable guardrails for I/O validation    |
| **Output Handling**    | Streaming yield pattern                       | Structured output types with validation       |
| **Business Focus**     | Business mission/values framework             | General purpose agent framework               |

---

### Solana Agent vs LangGraph

| Feature                | Solana Agent                                   | LangGraph                                      |
|------------------------|-----------------------------------------------|-----------------------------------------------|
| **Architecture**       | Service-oriented with agents                  | Graph-based state machine                     |
| **Workflow Design**    | Implicit routing by specialization            | Explicit node connections and state transitions |
| **Learning Curve**     | Simple setup with config objects              | Steeper with graph concepts and states        |
| **Streaming**          | Native streaming for all I/O                  | Requires additional configuration             |
| **Visualization**      | None built-in                                 | Graph visualization of agent workflows        |
| **State Management**   | Implicit state via memory                     | Explicit state transitions and persistence    |
| **Integration**        | Standalone framework                          | Part of LangChain ecosystem                   |
| **Flexibility**        | Fixed routing paradigm                        | Highly customizable flow control              |

---

### Solana Agent vs CrewAI

| Feature                | Solana Agent                                   | CrewAI                                        |
|------------------------|-----------------------------------------------|-----------------------------------------------|
| **Multi-Agent Design** | Specialist agents with router                 | Agent crews with explicit roles              |
| **Agent Interaction**  | Query router determines agent                 | Direct agent-to-agent delegation             |
| **Configuration**      | JSON-based configuration                      | Python class-based agent definitions         |
| **Task Structure**     | Query-based interactions                      | Task-based with goals and workflows          |
| **Memory Sharing**     | Shared memory store                           | Agent-specific memories                      |
| **Human Interaction**  | Built for direct human queries                | Designed for autonomous operation            |
| **Streaming**          | Native streaming support                      | Limited streaming support                    |
| **Team Dynamics**      | Flat specialist structure                     | Hierarchical with managers and workers       |

---

### Solana Agent vs PydanticAI

| Feature                | Solana Agent                                   | PydanticAI                                   |
|------------------------|-----------------------------------------------|---------------------------------------------|
| **Multi-Modal**        | Full audio/text streaming built-in            | Text output only, input depends on LLM      |
| **Memory**             | Built-in conversation history                 | Not included                                |
| **Multi-Agent**        | First-class multi-agent support               | Single agent focus with composition patterns|
| **Tool Creation**      | Python classes with execute method            | Function decorators with schema             |
| **Model Support**      | Currently OpenAI focused                      | Integrates with many LLMs                   |
| **Debugging**          | Standard logging                              | Pydantic Logfire integration                |
| **Flow Control**       | Implicit routing                              | Python control flow with graph support      |

---

### When to Use Each Framework

#### Choose **Solana Agent** when:
- You need a simple, quick-to-deploy agent system.
- Multi-modal support (text/audio) is essential.
- You want automatic routing between specialized agents.
- Business mission alignment is important.
- You prefer configuration over code.
- Persistent memory across conversations is needed.
- You want streaming responses out of the box.

#### Choose **OpenAI Agents SDK** when:
- You need detailed tracing for debugging complex agent workflows.
- You want explicit control over agent handoffs.
- Your architecture requires structured output validation.
- You're using multiple LLM providers with OpenAI-compatible APIs.
- You need configurable guardrails for safety.
- You prefer a code-first approach to agent definition.

#### Choose **LangGraph** when:
- You need complex, multi-step workflows with branching logic.
- Your use case requires explicit state management.
- You want to visualize the flow of your agent system.
- You're already in the LangChain ecosystem.
- You need fine-grained control over agent decision paths.
- Your application has complex conditional flows.
- You want to model your agent system as a state machine.

#### Choose **CrewAI** when:
- You need agents to work together with minimal human input.
- Your use case involves complex team collaboration.
- You need hierarchical task delegation between agents.
- You want agents with specific roles and responsibilities.
- Your application requires autonomous operation.
- You need explicit agent-to-agent communication.
- Your workflow involves complex multi-step tasks.

#### Choose **PydanticAI** when:
- You want to use multiple LLM providers in one codebase.
- You need real-time debugging and monitoring of agent behavior.
- You require structured responses with validation guarantees.
- Your application needs dependency injection for easier testing.
- You want to leverage your existing Pydantic knowledge.
- You need both simple control flow and complex graph capabilities.
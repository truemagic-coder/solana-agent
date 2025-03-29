# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/solana-agent?color=yellow)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-orange.svg)](https://www.python.org/downloads/)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/ci.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/ci.yml)
[![Lines of Code](https://tokei.rs/b1/github/truemagic-coder/solana-agent?type=python&category=code&style=flat)](https://github.com/truemagic-coder/solana-agent)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

## Agentic IQ

Build your AI business in three lines of code!

## Why?
* Three lines of code setup
* Multi-Agent Swarm
* Multi-Modal Streaming
* Conversational Memory & History
* Intelligent Routing
* Business Alignment
* Extensible Tooling
* Simple Business Definition
* Tested & Secure
* Support for MCP Servers
* Built in Python
* Deployed by [CometHeart](https://cometheart.com) & [WalletBubbles](https://walletbubbles.com)

## Features

* Easy three lines of code setup
* Designed for a multi-agent swarm 
* Seamless text and audio streaming with real-time multi-modal processing
* Persistent memory that preserves context across all agent interactions
* Streamlined message history for all agent interactions
* Intelligent query routing to agents with optimal domain expertise or your own custom routing
* Unified value system ensuring brand-aligned agent responses
* Powerful tool integration using standard Python packages and/or inline tools
* Assigned tools are utilized by agents automatically and effectively
* Simple business definition using JSON
* Ability to access any MCP server via URL

## Stack

* [Python](https://python.org) - Programming Language
* [OpenAI](https://openai.com) - LLMs
* [MongoDB](https://mongodb.com) - Conversational History (optional)
* [Zep](https://getzep.com) - Conversational Memory (optional)

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
        "api_key": "your-zep-api-key",
        "base_url": "your-zep-base-url", # not applicable if using Zep Cloud
    },
}
```

## Plugins

Plugins like Solana Agent Kit (sakit) integrate automatically with Solana Agent.

`pip install sakit`

### Internet Search
```python
from solana_agent import SolanaAgent

config = {
    "openai": {
        "api_key": "your-openai-api-key",
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
```

### MCP
```python
from solana_agent import SolanaAgent

config = {
    "openai": {
        "api_key": "your-openai-api-key",
    },
    "tools": {
        "mcp": {
            "server_urls": [
                "http://mcp-server1.com/mcp",
                "http://mcp-server2.com/mcp",
                "http://mcp-server3.com/mcp"
            ]
        }
    },
    "agents": [
        {
            "name": "research_specialist",
            "instructions": "You are an expert researcher who synthesizes complex information clearly.",
            "specialization": "Research and knowledge synthesis",
            "tools": ["mcp"],
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

To create a plugin like Solana Agent Kit - read the [code](https://github.com/truemagic-coder/solana-agent-kit)

## Advanced

### Custom Inline Tools

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

solana_agent.register_tool(test_tool)

async for response in solana_agent.process("user123", "What are the latest AI developments?"):
    print(response, end="")
```

### Custom Prompt Injection at Runtime

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

async for response in solana_agent.process("user123", "What are the latest AI developments?", "Always end your sentences with eh?"):
    print(response, end="")
```

### Custom Routing

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

## Notes
* Solana Agent agents can only call one tool per response.
* Solana Agent agents choose the best tool for the job.
* Solana Agent tools do not use OpenAI function calling.
* Solana Agent tools are async functions.

## Local Setup

A Docker Compose and Zep Config file is available at the root of this project

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

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
* Fast Responses
* Solana Ecosystem Integration
* Multi-Agent Swarm
* Multi-Modal Streaming (Text & Audio)
* Conversational Memory & History
* Internet Search
* Intelligent Routing
* Business Alignment
* Extensible Tooling
* Simple Business Definition
* Knowledge Base with PDF support
* MCP Support
* Tested & Secure
* Built in Python
* Powers [CometHeart](https://cometheart.com) & [WalletBubbles](https://walletbubbles.com)

## Features

* Easy three lines of code setup
* Fast AI responses
* Solana Ecosystem Integration via [AgentiPy](https://github.com/niceberginc/agentipy)
* MCP tool usage with first-class support for [Zapier](https://zapier.com/mcp)
* Designed for a multi-agent swarm 
* Seamless text and audio streaming with real-time multi-modal processing
* Configurable audio voice characteristics via prompting
* Persistent memory that preserves context across all agent interactions
* Quick Internet search to answer users' queries
* Streamlined message history for all agent interactions
* Intelligent query routing to agents with optimal domain expertise or your own custom routing
* Unified value system ensuring brand-aligned agent responses
* Powerful tool integration using standard Python packages and/or inline tools
* Assigned tools are utilized by agents automatically and effectively
* Simple business definition using JSON
* Integrated Knowledge Base with semantic search and automatic PDF chunking

## Stack

### Tech

* [Python](https://python.org) - Programming Language
* [OpenAI](https://openai.com), [Google](https://ai.google.dev), [xAI](https://x.ai) - LLM Providers
* [MongoDB](https://mongodb.com) - Conversational History (optional)
* [Zep Cloud](https://getzep.com) - Conversational Memory (optional)
* [Pinecone](https://pinecone.io) - Knowledge Base (optional)

### LLMs

* [gpt-4.1-mini](https://platform.openai.com/docs/models/gpt-4.1-mini) (agent)
* [gpt-4.1-nano](https://platform.openai.com/docs/models/gpt-4.1-nano) (router)
* [text-embedding-3-large](https://platform.openai.com/docs/models/text-embedding-3-large) or [text-embedding-3-small](https://platform.openai.com/docs/models/text-embedding-3-small) (embedding)
* [tts-1](https://platform.openai.com/docs/models/tts-1) (audio TTS)
* [gpt-4o-mini-transcribe](https://platform.openai.com/docs/models/gpt-4o-mini-transcribe) (audio transcription)
* [gemini-2.0-flash](https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash) (optional)
* [grok-3-mini-fast-beta](https://docs.x.ai/docs/models#models-and-pricing) (optional)

## Installation

You can install Solana Agent using pip:

`pip install solana-agent`

## Flows

In both flows of single and multiple agents - it is one user query to one agent using one tool (if needed).

An agent can have multiple tools and will choose the best one to answer the user query.

Routing is determined by optimal domain expertise of the agent for the user query.

When the agent uses a tool it feeds the tool output back to itself to generate the final response.

This is important as tools generally output unstructured and unformatted data that the agent needs to prepare for the user.

Keep this in mind while designing your agentic systems using Solana Agent.

```ascii
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
```

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

### Gemini

This allows Gemini to replace OpenAI for agent and router.

```python
config = {
    "gemini": {
        "api_key": "your-gemini-api-key",
    },
}
```

### Grok

This allows Grok to replace OpenAI (or Gemini) for agent.

```python
config = {
    "grok": {
        "api_key": "your-grok-api-key",
    },
}
```

### Knowledge Base

The Knowledge Base (KB) is meant to store text values and/or small PDFs.

```python
config = {
    "knowledge_base": {
        "pinecone": {
            "api_key": "your-pinecone-api-key",
            "index_name": "your-pinecone-index-name",
        }
    },
    "mongo": {
        "connection_string": "your-mongo-connection-string",
        "database": "your-database-name"
    },
}
```

#### Example for KB (text)

```python
from solana_agent import SolanaAgent

config = {
    "openai": {
        "api_key": "your-openai-api-key",
    },
    "knowledge_base": {
        "pinecone": {
            "api_key": "your-pinecone-api-key",
            "index_name": "your-pinecone-index-name",
        }
    },
    "mongo": {
        "connection_string": "your-mongo-connection-string",
        "database": "your-database-name"
    },
    "agents": [
        {
            "name": "kb_expert",
            "instructions": "You answer questions based on the provided knowledge base documents.",
            "specialization": "Company Knowledge",
        }
    ]
}

solana_agent = SolanaAgent(config=config)

doc_text = "Solana Agent is a Python framework for building multi-agent AI systems."
doc_metadata = {
    "source": "internal_docs",
    "version": "1.0",
    "tags": ["framework", "python", "ai"]
}
await solana_agent.kb_add_document(text=doc_text, metadata=doc_metadata)

async for response in solana_agent.process("user123", "What is Solana Agent?"):
    print(response, end="")
```

#### Example for KB (pdf)

```python
from solana_agent import SolanaAgent

config = {
    "openai": {
        "api_key": "your-openai-api-key",
    },
    "knowledge_base": {
        "pinecone": {
            "api_key": "your-pinecone-api-key",
            "index_name": "your-pinecone-index-name",
        }
    },
    "mongo": {
        "connection_string": "your-mongo-connection-string",
        "database": "your-database-name"
    },
    "agents": [
        {
            "name": "kb_expert",
            "instructions": "You answer questions based on the provided knowledge base documents.",
            "specialization": "Company Knowledge",
        }
    ]
}

solana_agent = SolanaAgent(config=config)

pdf_bytes = await pdf_file.read()

pdf_metadata = {
    "source": "annual_report_2024.pdf",
    "year": 2024,
    "tags": ["finance", "report"]
}

await solana_agent.kb_add_pdf_document(
    pdf_data=pdf_bytes,
    metadata=pdf_metadata,
)

async for response in solana_agent.process("user123", "Summarize the annual report for 2024."):
    print(response, end="")
```

## Tools

Tools can be used from plugins like Solana Agent Kit (sakit) or via inline tools. Tools available via plugins integrate automatically with Solana Agent.

* Agents can only call one tool per response
* Agents choose the best tool for the job
* Solana Agent doesn't use OpenAI function calling (tools) as they don't support async functions
* Solana Agent tools are async functions

### Solana

`pip install sakit`

```python
config = {
    "tools": {
        "solana": {
            "private_key": "your-solana-wallet-private-key", # base58 encoded string
            "rpc_url": "your-solana-rpc-url",
        },
    },
    "ai_agents": [
        {
            "name": "solana_expert",
            "instructions": "You are an expert Solana blockchain assistant. You always use the Solana tool to perform actions on the Solana blockchain.",
            "specialization": "Solana blockchain interaction",
            "tools": ["solana"],  # Enable the tool for this agent
        }
    ]
}

solana_agent = SolanaAgent(config=config)

async for response in solana_agent.process("user123", "What is my SOL balance?"):
    print(response, end="")
```

### Internet Search

`pip install sakit`

```python
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
            "name": "news_specialist",
            "instructions": "You are an expert news agent. You use your search_internet tool to get the latest information.",
            "specialization": "News researcher and specialist",
            "tools": ["search_internet"], # Enable the tool for this agent
        }
    ],
}

solana_agent = SolanaAgent(config=config)

async for response in solana_agent.process("user123", "What is the latest news on Elon Musk?"):
    print(response, end="")
```

### MCP

[Zapier](https://zapier.com/mcp) MCP has been tested, works, and is supported.

Zapier integrates over 7,000+ apps with 30,000+ actions that your Solana Agent can utilize.

Other MCP servers may work but are not supported.

`pip install sakit`

```python

from solana_agent import SolanaAgent

config = {
    "tools": {
        "mcp": {
            "urls": ["my-zapier-mcp-url"],
        }
    },
    "agents": [
        {
            "name": "zapier_expert",
            "instructions": "You are an expert in using Zapier integrations using MCP. You always use the mcp tool to perform Zapier AI like actions.",
            "specialization": "Zapier service integration expert",
            "tools": ["mcp"],  # Enable the tool for this agent
        }
    ]
}

solana_agent = SolanaAgent(config=config)

async for response in solana_agent.process("user123", "Send an email to bob@bob.com to clean his room!"):
    print(response, end="")
```

### Inline Tool Example

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

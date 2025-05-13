# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/solana-agent)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/ci.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/ci.yml)
[![Lines of Code](https://tokei.rs/b1/github/truemagic-coder/solana-agent?type=python&category=code&style=flat)](https://github.com/truemagic-coder/solana-agent)
[![Ruff Style](https://img.shields.io/badge/style-ruff-41B5BE)](https://github.com/astral-sh/ruff)
[![Libraries.io dependency status for GitHub repo](https://img.shields.io/librariesio/github/truemagic-coder/solana-agent)](https://libraries.io/pypi/solana-agent)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

## AI Agents for Solana

Build your AI agents in three lines of code!

## Why?
* Three lines of code setup
* Simple Agent Definition
* Fast Responses
* Multi-Vendor Support
* Solana Ecosystem Integration
* Multi-Agent Swarm
* Multi-Modal (Images & Audio & Text)
* Conversational Memory & History
* Internet Search
* Intelligent Routing
* Business Alignment
* Extensible Tooling
* Automatic Tool Workflows
* Autonomous Operation
* Knowledge Base
* MCP Support
* Guardrails
* Image Generation
* Pydantic Logfire
* Tested & Secure
* Built in Python
* Powers [CometHeart](https://cometheart.com)

## Features

* Easy three lines of code setup
* Simple agent definition using JSON
* Fast AI responses
* Multi-vendor support including OpenAI, Grok, and Gemini AI services
* Solana Ecosystem Integration via [AgentiPy](https://github.com/niceberginc/agentipy)
* MCP tool usage with first-class support for [Zapier](https://zapier.com/mcp)
* Integrated observability and tracing via [Pydantic Logfire](https://pydantic.dev/logfire)
* Designed for a multi-agent swarm 
* Seamless streaming with real-time multi-modal processing of text, audio, and images
* Persistent memory that preserves context across all agent interactions
* Quick Internet search to answer users' queries
* Streamlined message history for all agent interactions
* Intelligent query routing to agents with optimal domain expertise or your own custom routing
* Unified value system ensuring brand-aligned agent responses
* Powerful tool integration using standard Python packages and/or inline tools
* Assigned tools are utilized by agents automatically and effectively
* Integrated Knowledge Base with semantic search and automatic PDF chunking
* Input and output guardrails for content filtering, safety, and data sanitization
* Generate custom images based on text prompts with storage on S3 compatible services
* Automatic sequential tool workflows allowing agents to chain multiple tools
* Combine with event-driven systems to create autonomous agents

## Stack

### Tech

* [Python](https://python.org) - Programming Language
* [OpenAI](https://openai.com) - AI Model Provider
* [MongoDB](https://mongodb.com) - Conversational History (optional)
* [Zep Cloud](https://getzep.com) - Conversational Memory (optional)
* [Pinecone](https://pinecone.io) - Knowledge Base (optional)
* [AgentiPy](https://agentipy.fun) - Solana Ecosystem (optional)
* [Zapier](https://zapier.com) - App Integrations (optional)
* [Pydantic Logfire](https://pydantic.dev/logfire) - Observability and Tracing (optional)

### AI Models Used

**OpenAI**
* [gpt-4.1](https://platform.openai.com/docs/models/gpt-4.1) (agent - can be overridden)
* [gpt-4.1-nano](https://platform.openai.com/docs/models/gpt-4.1-nano) (router)
* [text-embedding-3-large](https://platform.openai.com/docs/models/text-embedding-3-large) (embedding)
* [tts-1](https://platform.openai.com/docs/models/tts-1) (audio TTS)
* [gpt-4o-mini-transcribe](https://platform.openai.com/docs/models/gpt-4o-mini-transcribe) (audio transcription)
* [gpt-image-1](https://platform.openai.com/docs/models/gpt-image-1) (image generation - can be overridden)
* [gpt-4o-mini-search-preview](https://platform.openai.com/docs/models/gpt-4o-mini-search-preview) (Internet search)

**Grok**
* [grok-3-fast](https://x.ai/api#pricing) (agent - optional)
* [grok-2-image](https://x.ai/api#pricing) (image generation - optional)

**Gemini**
* [gemini-2.5-flash-preview-04-17](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-preview) (agent - optional)
* [imagen-3.0-generate-002](https://ai.google.dev/gemini-api/docs/models#imagen-3) (image generation - optional)

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

### Grok

`grok-3-fast` can be used instead of `gpt-4.1` for the agent model

```python
config = {
    "grok": {
        "api_key": "your-grok-api-key",
    },
}
```

### Gemini

`gemini-2.5-pro-preview-03-25` can be used instead of `gpt-4.1` for the agent model

```python
config = {
    "gemini": {
        "api_key": "your-gemini-api-key",
    },
}
```

### Knowledge Base

The Knowledge Base (KB) is meant to store text values and/or PDFs (extracts text) - can handle very large PDFs.

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

### Guardrails

Guardrails allow you to process and potentially modify user input before it reaches the agent (Input Guardrails) and agent output before it's sent back to the user (Output Guardrails). This is useful for implementing safety checks, content moderation, data sanitization, or custom transformations.

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

* Agents can use multiple tools per response and should apply the right sequential order (like send an email to bob@bob.com with the latest news on Solana)
* Agents choose the best tools for the job
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
    "agents": [
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

### Image Generation

Generate images using OpenAI, Grok, or Gemini image models and upload them to S3-compatible storage.

`pip install sakit`

```python
from solana_agent import SolanaAgent

config = {
    "tools": {
        "image_gen": {
            "provider": "openai",                                        # Required: either "openai", "grok", or "gemini"
            "api_key": "your-api-key",                                   # Required: your OpenAI or Grok or Gemini API key
            "s3_endpoint_url": "https://your-s3-endpoint.com",           # Required: e.g., https://nyc3.digitaloceanspaces.com
            "s3_access_key_id": "YOUR_S3_ACCESS_KEY",                    # Required: Your S3 access key ID
            "s3_secret_access_key": "YOUR_S3_SECRET_KEY",                # Required: Your S3 secret access key
            "s3_bucket_name": "your-bucket-name",                        # Required: The name of your S3 bucket
            "s3_region_name": "your-region",                             # Optional: e.g., "nyc3", needed by some providers
            "s3_public_url_base": "https://your-cdn-or-bucket-url.com/", # Optional: Custom base URL for public links (include trailing slash). If omitted, a standard URL is constructed.
        }
    },
    "agents": [
        {
            "name": "image_creator",
            "instructions": "You are a creative assistant that generates images based on user descriptions. Use the image_gen tool to create and store the image.",
            "specialization": "Image generation and storage",
            "tools": ["image_gen"],  # Enable the tool for this agent
        }
    ]
}

solana_agent = SolanaAgent(config=config)

async for response in solana_agent.process("user123", "Generate an image of a smiling unicorn!"):
    print(response, end="")
```

### Nemo Agent

This plugin allows the agent to generate python programs using [Nemo Agent](https://nemo-agent.com) and uploads the files in a ZIP file to s3-compatible storage. It returns the public URL of the zip file.

This has been tested using [Cloudflare R2](https://developers.cloudflare.com/r2/).

```python
from solana_agent import SolanaAgent

config = {
    "tools": {
        "nemo_agent": {
            "provider": "openai",                                        # Required: either "openai" or "gemini"
            "api_key": "your-api-key",                                   # Required: your OpenAI or Gemini API key
            "s3_endpoint_url": "https://your-s3-endpoint.com",           # Required: e.g., https://nyc3.digitaloceanspaces.com
            "s3_access_key_id": "YOUR_S3_ACCESS_KEY",                    # Required: Your S3 access key ID
            "s3_secret_access_key": "YOUR_S3_SECRET_KEY",                # Required: Your S3 secret access key
            "s3_bucket_name": "your-bucket-name",                        # Required: The name of your S3 bucket
            "s3_region_name": "your-region",                             # Optional: e.g., "nyc3", needed by some providers
            "s3_public_url_base": "https://your-cdn-or-bucket-url.com/", # Optional: Custom base URL for public links (include trailing slash). If omitted, a standard URL is constructed.
        }
    },
    "agents": [
        {
            "name": "python_dev",
            "instructions": "You are an expert Python Developer. You always use your nemo_agent tool to generate python code.",
            "specialization": "Python Developer",
            "tools": ["nemo_agent"],  # Enable the tool for this agent
        }
    ]
}

solana_agent = SolanaAgent(config=config)

async for response in solana_agent.process("user123", "Generate an example leetcode medium in Python and give me the zip file."):
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

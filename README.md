# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/solana-agent?color=yellow)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-orange.svg)](https://www.python.org/downloads/)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/test.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/test.yml)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

## Agentic IQ

Solana Agent is an AI Agent framework defined on five key dimensions:

* **Multi-Modal Intelligence:** Agents process both text and audio seamlessly. Users interact naturally through their preferred communication channel.

* **Episodic Memory:** Agents remember past conversations with users. This eliminates repetitive explanations and builds meaningful relationships.

* **Cognitive Routing:** Queries go to the most qualified agent automatically. Each specialist has domain-specific knowledge for optimal responses.

* **Value Alignment:** Every agent follows clear organizational values and ethics. This ensures consistent, principled responses aligned with human priorities.

* **Extensible Tools:** Agents recognize when specialized tools are needed. They extend capabilities beyond conversation to take actions via Python package integration.

This approach creates AI assistants that are intelligent, contextually aware, and aligned with human values.


## Features

* Seamless text and audio streaming with real-time multi-modal processing
* Persistent memory that preserves context across all agent interactions
* Intelligent query routing to agents with optimal domain expertise
* Unified value system ensuring aligned, principled agent responses
* Powerful tool integration that extends capabilities using standard Python packages

## Stack

* [Python](https://python.org) - programming language
* [OpenAI](https://openai.com) - LLMs
* [MongoDB](https://mongodb.com) - database
* [Zep](https://getzep.com) - conversational memory

## Installation

You can install Solana Agent using pip:

`pip install solana-agent`

## Documentation

Each public method has a docstring for real-time IDE hinting.

## Example App

```python
from solana_agent import SolanaAgent

config = {
    "organization": {
        "mission_statement": "To revolutionize knowledge work through AI-human collaboration that puts people first.",
        "values": {
            "Human-Centered": "Always prioritize human well-being and augmentation over replacement.",
            "Transparency": "Provide clear explanations for decisions and never hide information.",
            "Collective Intelligence": "Value diverse perspectives and combine human and AI strengths.",
            "Continuous Learning": "Embrace feedback and continuously improve based on experience."
        },
        "goals": [
            "Enable human experts to focus on high-value creative work",
            "Reduce administrative overhead through intelligent automation",
            "Create seamless knowledge flow across the organization"
        ],
        "guidance": "When making decisions, prioritize long-term user success over short-term efficiency."
    },
    "mongo": {
        "connection_string": "mongodb://localhost:27017",
        "database": "solana_agent"
    },
    "openai": {
        "api_key": "your-openai-key",
    },
    "agents": [
        {
            "name": "research_specialist",
            "instructions": "You are an expert researcher who synthesizes complex information clearly.",
            "specialization": "Research and knowledge synthesis",
            "tools": ["some_tool"]
        },
        {
            "name": "customer_support",
            "instructions": "You provide friendly, helpful customer support responses.",
            "specialization": "Customer inquiries",
        }
    ],
}

# Create agent with configuration
solana_agent = SolanaAgent(config=config)

# Process a query that can use tools
async for response in solana_agent.process("user123", "What are the latest AI developments?"):
    print(response, end="")
```

## Models Used
* `gpt-4o-mini`
* `gpt-4o-mini-transcribe`
* `tts-1`

## Solana Agent Kit

A collection of Solana Agent tools

[Solana Agent Kit](https://github.com/truemagic-coder/solana-agent-kit)

## Example App

A Solana Agent example app written in FastAPI and Next.js

[Solana Agent Example App](https://github.com/truemagic-coder/solana-agent-app)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

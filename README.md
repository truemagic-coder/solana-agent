# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/solana-agent?color=yellow)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-orange.svg)](https://www.python.org/downloads/)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/ci.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/ci.yml)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

## Agentic IQ

Power your business using Solana Agent!

* **Brand:** AI agents will speak in your brand voice
* **Expert:** The AI agent with the most expertise will answer your users' inquiries
* **Extend:** Assign tools for your AI agents to perform actions

## Features

* Seamless text and audio streaming with real-time multi-modal processing
* Persistent memory that preserves context across all agent interactions (optional)
* Intelligent query routing to agents with optimal domain expertise
* Unified value system ensuring brand-aligned agent responses
* Powerful tool integration using standard Python packages
* Assigned tools are utilized by agents automatically and effectively

## Stack

* [Python](https://python.org) - Programming Language
* [OpenAI](https://openai.com) - LLMs
* [MongoDB](https://mongodb.com) - Database
* [Zep](https://getzep.com) - Conversational Memory (optional)

## Installation

You can install Solana Agent using pip:

`pip install solana-agent`

## Example App

```python
from solana_agent import SolanaAgent

config = {
    "organization": {
        "mission_statement": "To provide users with a one-stop shop for their queries.",
        "values": {
            "Friendliness": "Users must be treated fairly, openly, and with friendliness.",
            "Ethical": "Agents must use a strong ethical framework in their interactions with users.",
        },
        "goals": [
            "Empower users with great answers to their queries.",
        ],
        "guidance": "Align all interactions with the success of the user while respecting human dignity."
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
            "tools": ["some_tool"]
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

## Solana Agent Kit

A collection of Solana Agent tools

[Solana Agent Kit](https://github.com/truemagic-coder/solana-agent-kit)

## Example App

A Solana Agent example app written in FastAPI and Next.js

[Solana Agent Example App](https://github.com/truemagic-coder/solana-agent-app)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

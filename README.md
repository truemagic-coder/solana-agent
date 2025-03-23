# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/solana-agent?color=yellow)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-orange.svg)](https://www.python.org/downloads/)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/test.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/test.yml)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

## Agentic IQ

Solana Agent is an AI framework built on three pillars:

* **Swarm Intelligence:** Specialized agents collaborate and share memory for text and audio. Intelligent routing ensures the most qualified agent handles each query.
* **Organizational Alignment:** Agents follow a core set of values and ethics. This guarantees responses remain consistent and principled.
* **Extensible Tooling:** Agents recognize when tools are needed and integrate them seamlessly. They act beyond conversation to address user needs.

This approach produces AI Agents that are intelligent, cooperative, ethical, and aligned with organizational goals.


## Features

* Seamless text and audio streaming with real-time multi-modal processing
* Persistent memory that preserves context across all agent interactions
* Intelligent query routing to agents with optimal domain expertise
* Unified value system ensuring aligned, principled agent responses
* Powerful tool integration using standard Python packages

## Stack

* [Python](https://python.org) - programming language
* [OpenAI](https://openai.com) - LLMs
* [MongoDB](https://mongodb.com) - database
* [Zep](https://getzep.com) - conversational memory

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

solana_agent = SolanaAgent(config=config)

async for response in solana_agent.process("user123", "What are the latest AI developments?"):
    print(response, end="")
```

## Documentation

Each public method has a docstring for real-time IDE hinting.

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

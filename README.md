# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/solana-agent?color=yellow)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-orange.svg)](https://www.python.org/downloads/)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/test.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/test.yml)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

## Features

* Text streaming messages by AI Agents
* Conversational memory per user shared by all AI Agents
* Routing based on AI Agent specializations
* Built-in Internet Search for all AI Agents
* Organizational mission, values, goals, and guidance for all AI Agents
* Robust AI Agent tool plugins based on standard python packages

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
        "default_model": "gpt-4o-mini"
    },
    "agents": [
        {
            "name": "research_specialist",
            "instructions": "You are an expert researcher who synthesizes complex information clearly.",
            "specialization": "Research and knowledge synthesis",
            "model": "o3-mini",
            "tools": ["some_tool"]
        },
        {
            "name": "customer_support",
            "instructions": "You provide friendly, helpful customer support responses.",
            "specialization": "Customer inquiries",
            "model": "gpt-4o-mini"
        }
    ],
}

# Create agent with configuration
solana_agent = SolanaAgent(config=config)

# Process a query that can use tools
async for response in solana_agent.process("user123", "What are the latest AI developments?"):
    print(response, end="")
```

## Solana Agent Kit

[Solana Agent Kit](https://github.com/truemagic-coder/solana-agent-kit)

## Example App

[Solana Agent Example App](https://github.com/truemagic-coder/solana-agent-app)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

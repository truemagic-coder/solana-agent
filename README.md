# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-orange.svg)](https://www.python.org/downloads/)
[![codecov](https://img.shields.io/codecov/c/github/truemagic-coder/solana-agent/main.svg)](https://codecov.io/gh/truemagic-coder/solana-agent)
[![Build Status](https://img.shields.io/github/actions/workflow/status/truemagic-coder/solana-agent/test.yml?branch=main)](https://github.com/truemagic-coder/solana-agent/actions/workflows/test.yml)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

## Technical Features

- **🗣️ Advanced Interaction Layer:**  
    Streaming text-based conversations with real-time thinking.  
    Multi-turn context preservation and reasoning.

- **🧠 Distributed Intelligence Capabilities:**  
    Cross-domain knowledge integration from multiple sources.  
    Self-organizing information architecture.  
    Autonomous knowledge extraction and refinement.  
    Time-aware contextual responses.  
    Self-critical improvement systems. 

- **🛡️ Governance Framework:**  
    Define organization-wide values and operating principles in code.  
    Consistent decision-making aligned with organizational priorities.  
    Privacy-preserving knowledge sharing with configurable controls.  
    Transparent insight extraction with review capabilities.  
    Performance analytics across the agent network.

- **🔌 Standard Python Plugin System:**  
    Extensible architecture using Python's native package ecosystem.  
    PyPI-compatible plugin distribution with standard dependency management.  
    Entry point registration for seamless discovery.  
    Tool registry for AI agent capability extension.  
    Permission-based tool access for security and control.  
    Clean interface for third-party integrations through standard Python APIs.  
    Runtime tool discovery without service restarts.

## Implementation Technologies

Solana Agent leverages multiple technologies to enable these capabilities:

- **Knowledge Integration:**  
    Zep memory and Pinecone or Qdrant vector search.
- **Collaborative Intelligence:**  
    Multi-agent swarm architecture with specialized expertise domains.
- **Organization Alignment:**  
    Unified mission framework, critic system, and collective memory.
- **External Integration:**  
    Plugin system for extensible tool capabilities and API connections.

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
    "pinecone": {
        "api_key": "your-pinecone-key",
        "index": "your-index"
    },
    "stalled_ticket_timeout": 60,
    "ai_agents": [
        {
            "name": "research_specialist",
            "instructions": "You are an expert researcher who synthesizes complex information clearly.",
            "specialization": "Research and knowledge synthesis",
            "model": "o3-mini",
            "tools": ["search_internet"]
        },
        {
            "name": "customer_support",
            "instructions": "You provide friendly, helpful customer support responses.",
            "specialization": "Customer inquiries",
            "model": "gpt-4o-mini"
        }
    ],
    "human_agents": [
        {
            "agent_id": "expert_dev",
            "name": "Senior Developer", 
            "specialization": "Complex technical issues"
        },
        {
            "agent_id": "support_lead",
            "name": "Support Team Lead",
            "specialization": "Escalated customer issues"
        }
    ],
    "perplexity_api_key": "your-perplexity-key"  # For internet search plugin
}

# Create agent with configuration
solana_agent = SolanaAgent(config=config)

# Process a query that can use tools
async for response in solana_agent.process("user123", "What are the latest AI developments?"):
    print(response, end="")
```

## Solana Agent Kit (plugins collection)

[Solana Agent Kit](https://github.com/truemagic-coder/solana-agent-kit)

## Example App

[Solana Agent Example App](https://github.com/truemagic-coder/solana-agent-app)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

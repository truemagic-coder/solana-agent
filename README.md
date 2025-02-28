# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

Solana Agent is the first self-learning AI Agent Swarm framework.

## Why Solana Agent?

### 🧬 The First Self-Learning AI Agent Swarm Framework

Unlike traditional AI assistants that forget conversations after each session, Solana Agent maintains a rich, searchable memory system that grows smarter with every interaction.

**Why This Matters:**
- 🤖 **Swarm Intelligence**: Shares memory across all agents in the swarm
- 📈 **Continuous Learning**: Evolves with every new interaction
- 🎯 **Context-Aware**: Recalls past interactions for more relevant responses
- 🔄 **Self-Improving**: Builds knowledge and improves reasoning automatically
- 🧠 **Knowledge Base**: Add domain-specific knowledge for better reasoning
- 🛡️ **Secure**: Secure and private memory and data storage 

**Experience Agentic IQ!**

## Features

🔄 **Real-time AI Interactions**
- Streaming text-based conversations
- Real-time voice-to-voice conversations

🧠 **Memory System and Extensibility**
- Advanced AI memory combining conversational context, knowledge base, and parallel tool calling
- Create custom tools for extending the Agent's capabilities like further API integrations

🤖 **Multi-Agent Swarms**
- Create specialized agents with different expertise domains
- Automatic routing of queries to the most appropriate specialist
- Seamless handoffs between agents for complex multi-domain questions
- Shared memory and context across the entire agent swarm

🔍 **Multi-Source Search and Reasoning**
- Internet search via Perplexity
- X (Twitter) search using Grok
- Conversational memory powered by Zep
- Conversational message history using MongoDB (on-prem or hosted)
- Knowledge Base (KB) using Pinecone with reranking - available globally or user-specific
- Upload CSVs to be processed into summary reports and stored in the Knowledge Base (KB) using Gemini
- Comprehensive reasoning combining multiple data sources

## Why Choose Solana Agent Over LangChain?

### 🎯 Key Differentiators

🧠 **Advanced Memory Architecture**
   - Built-in episodic memory vs LangChain's basic memory types
   - Persistent cross-session knowledge retention
   - Automatic self-learning from conversations
   - Knowledge Base to add domain specific knowledge
   - CSV file uploads to perform document context search 

🤝 **Intelligent Multi-Agent Systems**
   - First-class support for specialized agent swarms
   - Dynamic inter-agent routing based on query complexity
   - Seamless handoffs with continuous memory preservation
   - Single unified interface for the entire agent network
   - No custom coding required for agent coordination

🏢 **Enterprise Focus**
   - Production-ready out of the box in a few lines of code
   - Enterprise-grade deployment options for all components and services
   - Simple conventions over complex configurations

🛠️ **Simplified Development**
   - No chain building required
   - Python plain functions vs complex chaining
   - Fewer moving parts equals more stable applications
   - Smaller repo size by 1000x: Solana Agent @ ~500 LOC vs LangChain @ ~500,000 LOC

🚀 **Performance**
   - Optimized for real-time streaming responses
   - Built-in voice processing capabilities
   - Multi-source search with automatic reasoning synthesis

## Installation

You can install Solana Agent using pip:

```bash
pip install solana-agent
```

## Documentation
* Each public method has a docstring for real-time IDE hinting

## Production Apps
* [Solana Agent Copilot](https://ai.solana-agent.com) - Solana Token AI Copilot using streaming text conversations
* [CometHeart](https://cometheart.com) - AI Companion and Business Coach on mobile using voice-to-voice conversations

## Example Apps
* [Solana Agent Example App](https://github.com/truemagic-coder/solana-agent-app) - See as source of documentation

## Contributing

Contributions to Solana Agent are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

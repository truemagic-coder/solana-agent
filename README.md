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
- 💭 **Collective Insights**: Extracts valuable knowledge from all user interactions
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

⚙️ **Task Management and Automation**
- Schedule one-time or recurring tasks for future execution
- Run computationally intensive operations as background tasks
- Support for long-running operations without blocking user interactions
- Task status tracking and delivery of results when completed
- Intelligent notification system for completed tasks

🌐 **Collective Swarm Intelligence**
- Hybrid semantic-keyword search for collective knowledge discovery
- Automatic extraction of valuable insights from all conversations
- Self-learning knowledge base that improves with each user interaction
- Cross-user knowledge sharing while preserving privacy
- Semantic search for finding conceptually related information even with different wording
- Automatic ranking of insights by relevance and importance

## Privacy and Collective Memory

Solana Agent's collective memory system is designed with privacy in mind:

- **Selective Knowledge Extraction**: Only factual, non-personal information is extracted
- **Privacy Filtering**: The AI is instructed to exclude user-specific details, opinions, and sensitive information
- **Optional Feature**: Easily disable collective memory with the `enable_collective_memory=False` parameter
- **Transparency**: Extracted insights are accessible and reviewable
- **Customizable Thresholds**: Adjust what qualifies as a valuable insight worth sharing

When collective memory is enabled, the system extracts valuable factual knowledge that can benefit all users, while carefully avoiding personal or sensitive information. For environments with stricter privacy requirements, the feature can be completely disabled.

## Why Choose Solana Agent Over LangChain?

### 🎯 Key Differentiators

🧠 **Advanced Memory Architecture**
   - Built-in episodic memory vs LangChain's basic memory types
   - Persistent cross-session knowledge retention
   - Automatic self-learning from conversations
   - Knowledge Base to add domain specific knowledge
   - CSV file uploads to perform document context search 
   - Collective swarm memory that learns from all user interactions

🤝 **Intelligent Multi-Agent Systems**
   - First-class support for specialized agent swarms
   - Dynamic inter-agent routing based on query complexity
   - Seamless handoffs with continuous memory preservation
   - Single unified interface for the entire agent network
   - No custom coding required for agent coordination
   - Shared collective intelligence across all agents

🏢 **Enterprise Focus**
   - Production-ready out of the box in a few lines of code
   - Enterprise-grade deployment options for all components and services
   - Simple conventions over complex configurations
   - Asynchronous task management for high-load scenarios

🛠️ **Simplified Development**
   - No chain building required
   - Python plain functions vs complex chaining
   - Fewer moving parts equals more stable applications
   - Smaller repo size by 1000x: Solana Agent @ ~500 LOC vs LangChain @ ~500,000 LOC
   - Built-in task scheduling and background processing

🚀 **Performance**
   - Optimized for real-time streaming responses
   - Built-in voice processing capabilities
   - Multi-source search with automatic reasoning synthesis
   - Efficient handling of long-running and scheduled tasks

## Installation

You can install Solana Agent using pip:

```bash
pip install solana-agent
```

## Documentation
* Each public method has a docstring for real-time IDE hinting

## Production Apps
* [CometHeart](https://cometheart.com) - AI Companion and Business Coach on mobile using voice-to-voice conversations

## Example Apps
* [Solana Agent Example App](https://github.com/truemagic-coder/solana-agent-app) - See as source of documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

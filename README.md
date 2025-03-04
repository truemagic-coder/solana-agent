# Solana Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-orange.svg)](https://www.python.org/downloads/)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

## Why Solana Agent?

### The Enterprise AGI Framework

Forget off-the-shelf chatbots. Solana Agent is an AGI framework engineered to transform how your business interacts, learns, and grows—at any scale.

**Five Reasons You’ll Love It:**

- **Empower Teams, Slash Costs**  
   Scale to thousands of inquiries without adding headcount, thanks to human-AI handoffs and automated routing.

- **Smarter Every Day**  
   Agents learn from each conversation, refining their expertise and successively handling more complex tasks.

- **Seamless Human Collaboration**  
   When AI hits a limit, it taps a human partner—passing along full context so no user repeats themselves.

- **Data-Driven Insights**  
   Extract actionable analytics directly from conversations, fueling business intelligence and better decisions.

- **Effortless Expansion**  
   Plug in specialized agents or new data sources at any time—your swarm evolves alongside your business needs.

## Features

### 🗣️ **Advanced Interaction Layer**
- Streaming text-based conversations with real-time thinking
- Voice-to-voice conversations with natural cadence
- Multi-turn context preservation and reasoning

### 🤖 **Multi-Agent Swarm Architecture**
- Dynamic specialized agent creation and coordination
- Intelligent routing based on query content and agent expertise
- Seamless handoffs with comprehensive context passing
- Parallel processing capabilities for complex multi-part questions
- Shared memory and collective intelligence across the entire swarm

### 👥 **Human Agent Integration**
- Hybrid AI-human collaboration system with seamless handoffs
- Ticket management for complex or sensitive inquiries
- Multiple handoff patterns (AI↔human, human↔human)
- In-chat command system for human agents to manage tickets
- Real-time notification system for ticket assignment
- Availability status management for human operators
- Intelligent fallback mechanisms when humans are unavailable
- Complete conversation context preservation during handoffs

### 🔍 **Multi-Source Knowledge Integration**
- Real-time internet search via Perplexity API
- Social media monitoring via X/Twitter (Grok API)
- Long-term conversational memory with Zep's graph-based storage
- Structured data analysis with CSV processing and summarization
- Vector-based knowledge retrieval with semantic search via Pinecone
- Hybrid search combining vector similarity and text relevance via MongoDB and Pinecone

### 🌐 **Collective Swarm Intelligence**
- Hybrid semantic-keyword search spanning all user interactions
- Privacy-preserving knowledge extraction from conversations
- Self-organizing knowledge base with automatic categorization
- Cross-domain insight discovery and connection building
- Autonomous knowledge refinement through verification tasks

### 🕰️ **Time-Aware Intelligence**
- User-specific timezone handling for personalized time references
- Chronologically accurate responses across all conversations
- Default timezone fallbacks at both agent and swarm levels
- Seamless timezone preservation during agent handoffs

### ⚖️ **Self-Critical System**
- AI critic that evaluates agent responses for quality and accuracy
- Autonomous identification of improvement areas across all agents
- Prioritized feedback based on severity and impact
- System-wide trend analysis to track collective improvement
- Integrates feedback automatically to improve future answers

### 🎯 **Unified Mission Framework**
- Define organization-wide values, goals, and operating principles
- Automatically applied to all agents in the swarm
- Visually distinct formatting for clarity and emphasis
- Consistent decision-making aligned with organizational priorities
- Individual agent specializations work within the directive framework

## Privacy and Collective Memory

Solana Agent's collective memory system is designed with privacy in mind:

- **Selective Knowledge Extraction**: Only factual, non-personal information is extracted
- **Privacy Filtering**: The AI is instructed to exclude user-specific details, opinions, and sensitive information
- **Optional Feature**: Easily disable collective memory with the `enable_collective_memory=False` parameter
- **Transparency**: Extracted insights are accessible and reviewable
- **Customizable Thresholds**: Adjust what qualifies as a valuable insight worth sharing

When collective memory is enabled, the system extracts valuable factual knowledge that can benefit all users, while carefully avoiding personal or sensitive information. For environments with stricter privacy requirements, the feature can be completely disabled.

## Time Awareness

Solana Agent's time awareness system ensures your AI assistants always provide chronologically accurate responses based on each user's timezone:

- **User-Based Time Context**: Pass user timezone directly in conversation requests
- **Global Time Management**: Get precise time in any timezone using NTP synchronization
- **Timezone Preservation**: Maintains timezone context during agent handoffs
- **Default Fallbacks**: Configurable defaults at both agent and swarm levels
- **Contextual Date Awareness**: All agents understand "today," "next week," or "in 3 days" relative to user's timezone

## Human-AI Collaboration

Solana Agent's human agent integration creates a seamless collaboration between AI systems and human operators:

- **Intelligent Handoffs**: AI agents automatically detect when human expertise is needed
- **Ticket Management**: Comprehensive ticketing system for tracking and prioritizing inquiries
- **Contextual Awareness**: Full conversation history is preserved during handoffs
- **In-Chat Commands**: Human agents manage tickets directly through intuitive chat commands
- **Fallback Intelligence**: System gracefully handles scenarios when humans are unavailable
- **Privacy Controls**: Configurable limits on what information is shared during handoffs
- **Notification System**: Customizable alerts when human expertise is requested
- **Availability Management**: Human agents can set their status to manage workload

This hybrid approach combines AI speed and scalability with human judgment and expertise, creating a system that's greater than the sum of its parts.

## Why Choose Solana Agent Over LangChain?

### 🎯 Key Differentiators

#### 💡 **AGI-First Architecture**
- Designed for autonomous intelligence and self-improvement
- Framework for agents to improve themselves without human intervention

#### 🧠 **Advanced Memory Architecture**
- Sophisticated episodic and semantic memory vs. basic memory types
- Cross-session knowledge retention with importance-based storage
- Autonomous extraction of insights from conversations
- Collective memory across users and agents

#### 🤝 **True Multi-Agent Orchestration**
- First-class swarm intelligence with specialized agent teams
- Dynamic routing based on expertise and query complexity
- Seamless context preservation during handoffs
- Single unified interface for the entire agent ecosystem
- Unified mission framework for consistent decision making

#### 👥 **Human-AI Collaboration**
- Hybrid team of AI and human agents working together
- Dynamic handoffs between humans and AI based on need
- Integrated ticketing system for complex inquiries
- In-chat command system for human operators
- Fallback mechanisms ensuring uninterrupted service

#### 🏢 **Enterprise Production Readiness**
- Deployment-ready with minimal configuration
- Comprehensive error handling and recovery mechanisms
- Scalable architecture supporting high-concurrency environments
- Privacy-preserving design with configurable controls

#### 🛠️ **Developer Experience**
- Clean, intuitive API without excessive abstractions
- Standard Python functions vs complex chaining constructs
- Minimal boilerplate for common operations
- ~2000 vs ~500,000 lines of code

## Installation

You can install Solana Agent using pip:

`pip install solana-agent`

## Documentation
* Each public method has a docstring for real-time IDE hinting

## Production Apps
* [CometHeart](https://cometheart.com) - AI Companion and Business Coach on mobile using voice-to-voice conversations

## Example Apps
* [Solana Agent Example App](https://github.com/truemagic-coder/solana-agent-app) - See as source of documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

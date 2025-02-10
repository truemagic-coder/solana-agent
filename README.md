# Solana-Agent

[![PyPI - Version](https://img.shields.io/pypi/v/solana-agent)](https://pypi.org/project/solana-agent/)

![Solana Agent Logo](https://dl.walletbubbles.com/solana-agent-logo.png?width=200)

Solana Agent is the best AI Agent framework.

## Features

- Streaming text-based conversations with AI
- Audio transcription and streaming text-to-speech conversion
- Thread management for maintaining conversation context
- Message persistence using SQLite or MongoDB
- Custom tool integration for extending AI capabilities
- The best memory context currently available for AI Agents
- Zep integration for tracking facts
- Search Internet with Perplexity tool
- Search Zep facts tool
- Search X with Grok tool
- Reasoning tool that combines OpenAI model reasoning, Zep facts, Internet search, and X search.
- Solana tools upcoming...

## Installation

You can install Solana Agent using pip:

```bash
pip install solana-agent
```

## Usage

Here's a basic example of how to use Solana Agent:

```python
from solana-agent import AI, SQLiteDatabase

async def main():
    database = SQLiteDatabase("conversations.db")
    async with AI("your_openai_api_key", "AI Assistant", "Your instructions here", database) as ai:
        user_id = "user123"
        response = await ai.text(user_id, "Hello, AI!")
        async for chunk in response:
            print(chunk, end="", flush=True)
        print()

# Run the async main function
import asyncio
asyncio.run(main())
```

## Contributing

Contributions to Solana Agent are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pydantic import BaseModel
from typing import List, Optional

from solana_agent.adapters.llm_adapter import OpenAIAdapter

# Test Models


class TestStructuredOutput(BaseModel):
    message: str
    confidence: float
    tags: List[str]

# Fixtures


@pytest.fixture
def mock_openai():
    with patch('solana_agent.adapters.llm_adapter.OpenAI') as mock:
        yield mock


@pytest.fixture
def adapter(mock_openai):
    return OpenAIAdapter(api_key="test-key", model="gpt-4o-mini")

# Helper for creating mock stream responses


def create_mock_chunk(content: str):
    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta.content = content
    return chunk

# Tests


@pytest.mark.asyncio
async def test_generate_text_basic(adapter, mock_openai):
    # Setup mock response
    mock_chunks = [
        create_mock_chunk("Hello"),
        create_mock_chunk(" world"),
        create_mock_chunk("!")
    ]
    mock_openai.return_value.chat.completions.create.return_value = mock_chunks

    # Test basic text generation
    result = ""
    async for chunk in adapter.generate_text("Hi there"):
        result += chunk

    assert result == "Hello world!"
    mock_openai.return_value.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_generate_text_with_search(adapter, mock_openai):
    # Setup mock response
    mock_chunks = [create_mock_chunk("Latest news")]
    mock_openai.return_value.chat.completions.create.return_value = mock_chunks

    # Test with search enabled
    result = ""
    async for chunk in adapter.generate_text(
        "What's new?",
        needs_search=True
    ):
        result += chunk

    # Verify search parameters were included
    call_kwargs = mock_openai.return_value.chat.completions.create.call_args[1]
    assert "web_search_options" in call_kwargs
    assert call_kwargs["model"] == "gpt-4o-mini-search-preview"


@pytest.mark.asyncio
async def test_generate_text_with_system_prompt(adapter, mock_openai):
    mock_chunks = [create_mock_chunk("Response")]
    mock_openai.return_value.chat.completions.create.return_value = mock_chunks

    async for _ in adapter.generate_text(
        "Hello",
        system_prompt="You are a helpful assistant"
    ):
        pass

    call_kwargs = mock_openai.return_value.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_generate_embedding(adapter, mock_openai):
    # Setup mock embedding response
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
    mock_openai.return_value.embeddings.create.return_value = mock_response

    # Test embedding generation
    embedding = adapter.generate_embedding("Test text")

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_parse_structured_output(adapter, mock_openai):
    # Setup mock parsed response
    mock_parsed = TestStructuredOutput(
        message="Test message",
        confidence=0.95,
        tags=["test", "example"]
    )

    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.parsed = mock_parsed

    mock_openai.return_value.beta.chat.completions.parse.return_value = mock_completion

    # Test structured output parsing
    result = await adapter.parse_structured_output(
        prompt="Test prompt",
        system_prompt="Test system prompt",
        model_class=TestStructuredOutput
    )

    assert isinstance(result, TestStructuredOutput)
    assert result.message == "Test message"
    assert result.confidence == 0.95
    assert result.tags == ["test", "example"]


@pytest.mark.asyncio
async def test_generate_text_error_handling(adapter, mock_openai):
    # Setup mock to raise an exception
    mock_openai.return_value.chat.completions.create.side_effect = Exception(
        "Test error")

    # Test error handling
    result = ""
    async for chunk in adapter.generate_text("Test"):
        result += chunk

    assert "I apologize" in result
    assert "Test error" in result


def test_generate_embedding_error_handling(adapter, mock_openai):
    # Setup mock to raise an exception
    mock_openai.return_value.embeddings.create.side_effect = Exception(
        "Test error")

    # Test error handling
    embedding = adapter.generate_embedding("Test")

    assert len(embedding) == 1536  # Default fallback size
    assert all(x == 0.0 for x in embedding)

# Run tests with: pytest tests/test_llm_adapter.py -v

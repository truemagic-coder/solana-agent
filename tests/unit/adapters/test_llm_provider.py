"""
Tests for LLM provider adapters.

This module tests the adapters that implement the LLMProvider interface.
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import List
from pydantic import BaseModel

from solana_agent.adapters.llm_adapter import OpenAIAdapter


class MockEmbeddingResponse:
    """Mock response for embeddings."""

    def __init__(self, embedding_size=1536):
        self.data = [MagicMock(embedding=[0.1] * embedding_size)]


class MockCompletionChunk:
    """Mock streaming chunk response."""

    def __init__(self, content=None):
        self.choices = [MagicMock(delta=MagicMock(content=content))]


class MockCompletion:
    """Mock completion response."""

    def __init__(self, content="This is a test response"):
        self.choices = [MagicMock(message=MagicMock(content=content))]


class MockParsedResponse:
    """Mock parsed response for the beta api."""

    def __init__(self, parsed_data):
        self.choices = [MagicMock(message=MagicMock(parsed=parsed_data))]


class TestModel(BaseModel):
    """Test Pydantic model for structured output testing."""
    name: str = ""
    value: int = 0
    items: List[str] = []


@pytest.fixture
def openai_adapter():
    """Create an OpenAIAdapter instance."""
    with patch('solana_agent.adapters.llm_adapter.OpenAI') as mock_openai:
        # Create a mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Return the adapter with mocked client
        adapter = OpenAIAdapter(api_key="test-api-key", model="gpt-4o-mini")

        # Expose the mock client for assertions
        adapter.mock_client = mock_client

        yield adapter


# --------------------------
# Initialization Tests
# --------------------------

def test_init():
    """Test adapter initialization."""
    with patch('solana_agent.adapters.llm_adapter.OpenAI') as mock_openai:
        adapter = OpenAIAdapter(api_key="test-api-key", model="gpt-4o")

        # Check OpenAI client was initialized with the correct API key
        mock_openai.assert_called_once_with(api_key="test-api-key")

        # Check model was set correctly
        assert adapter.model == "gpt-4o"


# --------------------------
# Embedding Tests
# --------------------------

def test_generate_embedding(openai_adapter):
    """Test generating embeddings."""
    # Setup mock response
    mock_response = MockEmbeddingResponse()
    openai_adapter.mock_client.embeddings.create.return_value = mock_response

    # Call the method
    embedding = openai_adapter.generate_embedding("Test text")

    # Assertions
    openai_adapter.mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input="Test text"
    )

    # Should return the embedding from the response
    assert len(embedding) == 1536
    assert embedding[0] == 0.1


def test_generate_embedding_error(openai_adapter):
    """Test error handling when generating embeddings."""
    # Setup mock to raise exception
    openai_adapter.mock_client.embeddings.create.side_effect = Exception(
        "API error")

    # Call the method - should not raise exception
    embedding = openai_adapter.generate_embedding("Test text")

    # Should return zero vector as fallback
    assert len(embedding) == 1536
    assert embedding[0] == 0.0


# --------------------------
# Text Generation Tests
# --------------------------

@pytest.mark.asyncio
async def test_generate_text_streaming(openai_adapter):
    """Test generating text with streaming enabled."""
    # Setup mock chunks
    chunks = [
        MockCompletionChunk("Hello"),
        MockCompletionChunk(" world"),
        MockCompletionChunk("!"),
        MockCompletionChunk(None)  # Empty chunk to test filtering
    ]

    # Configure mock response
    openai_adapter.mock_client.chat.completions.create.return_value = chunks

    # Call the method and collect results
    result = []
    async for chunk in openai_adapter.generate_text("user1", "Tell me a joke", "Be funny"):
        result.append(chunk)

    # Assertions
    openai_adapter.mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Be funny"},
            {"role": "user", "content": "Tell me a joke"}
        ],
        stream=True,
        temperature=0.7,
        max_tokens=None,
        response_format=None
    )

    # Should have collected all non-empty chunks
    assert result == ["Hello", " world", "!"]


@pytest.mark.asyncio
async def test_generate_text_non_streaming(openai_adapter):
    """Test generating text without streaming."""
    # Setup mock response
    mock_response = MockCompletion("This is a joke response")
    openai_adapter.mock_client.chat.completions.create.return_value = mock_response

    # Call the method and collect results
    result = []
    async for chunk in openai_adapter.generate_text("user1", "Tell me a joke", stream=False):
        result.append(chunk)

    # Assertions
    openai_adapter.mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Tell me a joke"}
        ],
        stream=False,
        temperature=0.7,
        max_tokens=None,
        response_format=None
    )

    # Should have collected the complete response
    assert result == ["This is a joke response"]


@pytest.mark.asyncio
async def test_generate_text_with_custom_params(openai_adapter):
    """Test generating text with custom parameters."""
    # Setup mock response
    mock_response = MockCompletion()
    openai_adapter.mock_client.chat.completions.create.return_value = mock_response

    # Call the method with custom parameters
    custom_params = {
        "model": "gpt-4o",
        "temperature": 0.3,
        "max_tokens": 500,
        "response_format": {"type": "json_object"}
    }

    result = []
    async for chunk in openai_adapter.generate_text(
        "user1",
        "Generate JSON",
        stream=False,
        **custom_params
    ):
        result.append(chunk)

    # Assertions
    openai_adapter.mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Generate JSON"}
        ],
        stream=False,
        temperature=0.3,
        max_tokens=500,
        response_format={"type": "json_object"}
    )


# --------------------------
# Structured Output Tests
# --------------------------

@pytest.mark.asyncio
async def test_parse_structured_output_beta_api(openai_adapter):
    """Test parsing structured output with the beta API."""
    # Setup expected output
    expected_model = TestModel(name="test", value=42, items=["item1", "item2"])

    # Configure mock response
    mock_response = MockParsedResponse(expected_model)
    openai_adapter.mock_client.beta.chat.completions.parse.return_value = mock_response

    # Call the method
    result = await openai_adapter.parse_structured_output(
        prompt="Generate test data",
        system_prompt="Return structured data",
        model_class=TestModel
    )

    # Assertions
    openai_adapter.mock_client.beta.chat.completions.parse.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return structured data"},
            {"role": "user", "content": "Generate test data"}
        ],
        response_format=TestModel,
        temperature=0.2
    )

    # Should return the parsed model
    assert result == expected_model

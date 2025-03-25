import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO

from solana_agent.services.agent import AgentService
from solana_agent.domains.agent import AIAgent, OrganizationMission

# Test Data
TEST_AGENT = AIAgent(
    name="test_agent",
    instructions="Test instructions",
    specialization="testing",
)

TEST_MISSION = OrganizationMission(
    mission_statement="Test mission",
    values=[{"name": "integrity", "description": "Be honest"}],
    goals=["Help users"],
    voice="Talk in a friendly tone"
)

TEST_AUDIO_CONTENT = b"fake audio data"
TEST_TRANSCRIPTION = "This is a test query"
TEST_RESPONSE = "This is a test response."


class AsyncIterator:
    """Helper class to create proper async iterators for testing."""

    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        async def generator():
            for item in self.items:
                yield item
        return generator()


@pytest.fixture
def mock_llm_provider():
    with patch('solana_agent.services.agent.LLMProvider') as MockLLM:
        provider = MockLLM.return_value

        # Mock text generation
        async def async_text_gen(*args, **kwargs):
            for chunk in ["This ", "is ", "a ", "test ", "response."]:
                yield chunk
        provider.generate_text.return_value = async_text_gen()

        # Mock audio transcription
        async def async_transcribe(*args, **kwargs):
            yield TEST_TRANSCRIPTION
        provider.transcribe_audio.return_value = async_transcribe()

        # Mock TTS
        async def async_tts(*args, **kwargs):
            for chunk in [b"audio_chunk_1", b"audio_chunk_2"]:
                yield chunk
        provider.tts.return_value = async_tts()

        yield provider


@pytest.fixture
def agent_service(mock_llm_provider):
    """Create agent service with default configuration."""
    service = AgentService(
        llm_provider=mock_llm_provider,
        organization_mission=TEST_MISSION,
        config={
            "tools": {
                "test_tool": {
                    "param1": "value1"
                }
            }
        }
    )

    # Register the test agent manually
    service.register_ai_agent(
        name=TEST_AGENT.name,
        instructions=TEST_AGENT.instructions,
        specialization=TEST_AGENT.specialization
    )

    return service


@pytest.mark.asyncio
async def test_generate_response_text(agent_service):
    """Test text-to-text response generation."""
    response_chunks = []
    async for chunk in agent_service.generate_response(
        agent_name="test_agent",
        user_id="test_user",
        query="test query",
        output_format="text"
    ):
        response_chunks.append(chunk)

    assert "".join(response_chunks) == TEST_RESPONSE
    assert all(isinstance(chunk, str) for chunk in response_chunks)


@pytest.mark.asyncio
async def test_generate_response_audio_input(agent_service, mock_llm_provider):
    """Test audio-to-text response generation."""
    audio_file = BytesIO(TEST_AUDIO_CONTENT)

    response_chunks = []
    async for chunk in agent_service.generate_response(
        agent_name="test_agent",
        user_id="test_user",
        query=audio_file,
        output_format="text"
    ):
        response_chunks.append(chunk)

    mock_llm_provider.transcribe_audio.assert_called_once()
    assert "".join(response_chunks) == TEST_RESPONSE


@pytest.mark.asyncio
async def test_generate_response_audio_output(agent_service):
    """Test text-to-audio response generation."""
    audio_chunks = []
    async for chunk in agent_service.generate_response(
        agent_name="test_agent",
        user_id="test_user",
        query="test query",
        output_format="audio"
    ):
        audio_chunks.append(chunk)

    assert all(isinstance(chunk, bytes) for chunk in audio_chunks)


def test_get_agent_system_prompt(agent_service):
    """Test system prompt generation."""
    prompt = agent_service.get_agent_system_prompt("test_agent")

    assert TEST_AGENT.name in prompt
    assert TEST_AGENT.instructions in prompt
    assert TEST_MISSION.mission_statement in prompt
    assert TEST_MISSION.values[0]["name"] in prompt
    assert TEST_MISSION.goals[0] in prompt


def test_get_all_ai_agents(agent_service):
    """Test retrieving all AI agents."""
    agents = agent_service.get_all_ai_agents()

    assert len(agents) == 1
    assert TEST_AGENT.name in agents
    assert agents[TEST_AGENT.name] == TEST_AGENT


def test_assign_tool_for_agent(agent_service):
    """Test tool assignment using internal tool registry."""
    # Mock the assign_tool_to_agent method
    agent_service.tool_registry.assign_tool_to_agent = Mock(return_value=True)

    success = agent_service.assign_tool_for_agent(
        "test_agent",
        "test_tool"
    )

    assert success
    agent_service.tool_registry.assign_tool_to_agent.assert_called_once_with(
        "test_agent",
        "test_tool"
    )

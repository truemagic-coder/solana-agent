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
    guidance="Be helpful"
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
def mock_agent_repository():
    repo = Mock()
    repo.get_ai_agent_by_name = Mock(return_value=TEST_AGENT)
    repo.get_all_ai_agents = Mock(return_value=[TEST_AGENT])
    repo.save_ai_agent = Mock()
    return repo


@pytest.fixture
def agent_service(mock_llm_provider, mock_agent_repository):
    """Create agent service with default configuration."""
    return AgentService(
        llm_provider=mock_llm_provider,
        agent_repository=mock_agent_repository,
        organization_mission=TEST_MISSION,
        config={
            "tools": {
                "test_tool": {
                    "param1": "value1"
                }
            }
        }
    )


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


@pytest.mark.asyncio
async def test_handle_tool_call(agent_service):
    """Test tool call handling with internal tool registry."""
    tool_call = {
        "tool_call": {
            "name": "test_tool",
            "parameters": {"param1": "value1"}
        }
    }

    # Mock tool registry methods
    mock_tool = Mock()
    mock_tool.execute = Mock(
        return_value={"status": "success", "result": "tool result"})

    agent_service.tool_registry.get_tool = Mock(return_value=mock_tool)
    agent_service.tool_registry.get_agent_tools = Mock(return_value=[{
        "name": "test_tool",
        "description": "Test tool",
        "parameters": {}
    }])

    # First assign the tool to the agent
    agent_service.assign_tool_for_agent("test_agent", "test_tool")

    result = await agent_service._handle_tool_call(
        "test_agent",
        json.dumps(tool_call)
    )

    assert result == "tool result"
    mock_tool.execute.assert_called_once_with(param1="value1")


def test_get_agent_system_prompt(agent_service):
    """Test system prompt generation."""
    prompt = agent_service.get_agent_system_prompt("test_agent")

    assert TEST_AGENT.name in prompt
    assert TEST_AGENT.instructions in prompt
    assert TEST_MISSION.mission_statement in prompt
    assert TEST_MISSION.values[0]["name"] in prompt
    assert TEST_MISSION.goals[0] in prompt


def test_register_ai_agent(agent_service, mock_agent_repository):
    """Test AI agent registration."""
    agent_service.register_ai_agent(
        name="new_agent",
        instructions="New instructions",
        specialization="new_spec"
    )

    mock_agent_repository.save_ai_agent.assert_called_once()


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

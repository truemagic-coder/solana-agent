import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, AsyncGenerator

from solana_agent.services.agent import AgentService
from solana_agent.domains import AIAgent, OrganizationMission
from solana_agent.plugins.registry import ToolRegistry


class AsyncIterator:
    """Helper class to create async iterators for testing."""

    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


# Test Data
TEST_AGENT = AIAgent(
    name="test_agent",
    instructions="Test instructions",
    specialization="test_spec",
    model="gpt-4o-mini"
)

TEST_MISSION = OrganizationMission(
    mission_statement="Test mission",
    values=[{"name": "value1", "description": "desc1"}],
    goals=["goal1"]
)

# Fixtures


@pytest.fixture
def mock_llm_provider():
    provider = Mock()
    return provider


@pytest.fixture
def mock_agent_repository():
    repo = Mock()
    repo.get_ai_agent_by_name.return_value = TEST_AGENT
    repo.get_all_ai_agents.return_value = [TEST_AGENT]
    return repo


@pytest.fixture
def mock_tool_registry():
    registry = Mock(spec=ToolRegistry)

    # Define the search tool
    search_tool = {
        "name": "search_internet",
        "description": "Search the internet",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            }
        }
    }

    # Setup registry methods
    registry.get_agent_tools.return_value = [search_tool]
    registry.get_tool.return_value = Mock(
        execute=Mock(
            return_value={"status": "success", "result": "Tool result"})
    )

    return registry


@pytest.fixture
def agent_service(mock_llm_provider, mock_agent_repository, mock_tool_registry):
    return AgentService(
        llm_provider=mock_llm_provider,
        agent_repository=mock_agent_repository,
        organization_mission=TEST_MISSION,
        tool_registry=mock_tool_registry
    )

# Tests


def test_register_ai_agent(agent_service, mock_agent_repository):
    agent_service.register_ai_agent(
        name="test",
        instructions="test instructions",
        specialization="test spec"
    )
    mock_agent_repository.save_ai_agent.assert_called_once()


def test_get_agent_system_prompt(agent_service):
    prompt = agent_service.get_agent_system_prompt("test_agent")
    assert "You are test_agent" in prompt
    assert "Test instructions" in prompt
    assert "Test mission" in prompt
    assert "value1" in prompt
    assert "goal1" in prompt


def test_get_all_ai_agents(agent_service):
    agents = agent_service.get_all_ai_agents()
    assert len(agents) == 1
    assert "test_agent" in agents
    assert isinstance(agents["test_agent"], AIAgent)


def test_get_specializations(agent_service):
    specs = agent_service.get_specializations()
    assert "test_spec" in specs
    assert "AI expertise" in specs["test_spec"]


def test_find_agents_by_specialization(agent_service):
    agents = agent_service.find_agents_by_specialization("test_spec")
    assert len(agents) == 1
    assert "test_agent" in agents


def test_assign_tool_for_agent(agent_service, mock_tool_registry):
    mock_tool_registry.assign_tool_to_agent.return_value = True
    result = agent_service.assign_tool_for_agent(
        "test_agent", "search_internet")
    assert result is True
    mock_tool_registry.assign_tool_to_agent.assert_called_once()


@pytest.mark.asyncio
async def test_generate_response_normal_text(agent_service, mock_llm_provider):
    # Setup the mock to return an async iterator
    mock_llm_provider.generate_text.return_value = AsyncIterator(
        ["Hello", " world"])

    result = ""
    async for chunk in agent_service.generate_response(
        agent_name="test_agent",
        user_id="user1",
        query="Hi"
    ):
        result += chunk

    assert result == "Hello world"


@pytest.mark.asyncio
async def test_generate_response_error_handling(agent_service, mock_llm_provider):
    # Setup the mock to raise an exception
    mock_llm_provider.generate_text.side_effect = Exception("Test error")

    result = ""
    async for chunk in agent_service.generate_response(
        agent_name="test_agent",
        user_id="user1",
        query="Hi"
    ):
        result += chunk

    assert "I apologize" in result
    assert "Test error" in result


def test_tool_usage_prompt_generation(agent_service):
    prompt = agent_service._get_tool_usage_prompt("test_agent")
    assert "AVAILABLE TOOLS" in prompt
    assert "search_internet" in prompt
    assert "TOOL USAGE FORMAT" in prompt
    assert "RESPONSE RULES" in prompt


def test_execute_tool(agent_service, mock_tool_registry):
    mock_tool = Mock()
    mock_tool.execute.return_value = {"status": "success", "result": "Done"}
    mock_tool_registry.get_tool.return_value = mock_tool

    result = agent_service.execute_tool(
        agent_name="test_agent",
        tool_name="search_internet",
        parameters={"query": "test"}
    )

    assert result["status"] == "success"
    assert result["result"] == "Done"

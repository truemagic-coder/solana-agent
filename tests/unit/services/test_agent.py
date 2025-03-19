"""
Tests for the AgentService implementation.

This module tests the agent service functionality including registration,
response generation, and tool management.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any

from solana_agent.services.agent import AgentService
from solana_agent.domains import AIAgent, HumanAgent, OrganizationMission
from solana_agent.interfaces import LLMProvider, AgentRepository, ToolRegistry


# ---------------------
# Fixtures
# ---------------------

@pytest.fixture
def mock_llm_provider():
    """Return a mock LLM provider."""
    provider = Mock(spec=LLMProvider)

    # Setup the generate_text method as an AsyncMock that yields text
    async def mock_generate(*args, **kwargs):
        yield "This is a test response"
        yield " from the mock LLM provider."

    provider.generate_text = mock_generate
    return provider


# Fix your mock_agent_repository fixture
@pytest.fixture
def mock_agent_repository():
    """Return a mock agent repository."""
    # Create a MagicMock with all needed methods
    repo = MagicMock(spec=AgentRepository)

    # Set up default return values
    repo.get_ai_agent_by_name.return_value = None
    repo.get_all_ai_agents.return_value = []
    repo.get_all_human_agents.return_value = []
    repo.save_ai_agent.return_value = None
    repo.save_human_agent.return_value = None

    return repo


@pytest.fixture
def mock_tool_registry():
    """Return a mock tool registry."""
    registry = Mock(spec=ToolRegistry)
    registry.get_agent_tools.return_value = []
    return registry


@pytest.fixture
def basic_organization_mission():
    """Return a basic organization mission."""
    return OrganizationMission(
        mission_statement="To provide excellent automated assistance",
        values=[
            {"name": "quality", "description": "High quality responses"},
            {"name": "efficiency", "description": "Fast and efficient service"}
        ],
        goals=["90% customer satisfaction", "50% reduction in response time"]
    )


@pytest.fixture
def sample_ai_agent():
    """Return a sample AI agent."""
    return AIAgent(
        name="test_ai_agent",
        instructions="You are a test AI agent",
        specialization="testing",
        model="gpt-4o-mini"
    )


@pytest.fixture
def sample_human_agent():
    """Return a sample human agent."""
    return HumanAgent(
        id="human123",
        name="Test Human",
        specializations=["customer_support", "testing"],
        availability=True
    )


@pytest.fixture
def agent_service(mock_llm_provider, mock_agent_repository, mock_tool_registry, basic_organization_mission):
    """Return an agent service with mocked dependencies."""
    service = AgentService(
        llm_provider=mock_llm_provider,
        agent_repository=mock_agent_repository,
        organization_mission=basic_organization_mission,
        tool_registry=mock_tool_registry  # Inject the mock tool registry directly
    )
    return service


@pytest.fixture
def populated_agent_service(agent_service, sample_ai_agent, sample_human_agent):
    """Return an agent service with pre-populated agents."""
    # Setup the repository mock to return our sample agents
    agent_service.agent_repository.get_all_ai_agents.return_value = [
        sample_ai_agent]
    agent_service.agent_repository.get_all_human_agents.return_value = [
        sample_human_agent]
    agent_service.agent_repository.get_ai_agent_by_name.return_value = sample_ai_agent

    return agent_service


# ---------------------
# Agent Registration Tests
# ---------------------

def test_register_ai_agent(agent_service):
    """Test registering an AI agent."""
    # Act
    agent_service.register_ai_agent(
        name="test_agent",
        instructions="Test instructions",
        specialization="testing",
        model="gpt-4o"
    )

    # Assert
    agent_service.agent_repository.save_ai_agent.assert_called_once()
    saved_agent = agent_service.agent_repository.save_ai_agent.call_args[0][0]
    assert saved_agent.name == "test_agent"
    assert saved_agent.instructions == "Test instructions"
    assert saved_agent.specialization == "testing"
    assert saved_agent.model == "gpt-4o"


def test_register_human_agent(agent_service):
    """Test registering a human agent."""
    # Act
    agent_id = agent_service.register_human_agent(
        agent_id="human123",
        name="Test Human",
        specialization="customer_support"
    )

    # Assert
    agent_service.agent_repository.save_human_agent.assert_called_once()
    saved_agent = agent_service.agent_repository.save_human_agent.call_args[0][0]
    assert saved_agent.id == "human123"
    assert saved_agent.name == "Test Human"
    assert "customer_support" in saved_agent.specializations
    assert agent_id == "human123"


# ---------------------
# Agent Retrieval Tests
# ---------------------

def test_get_agent_by_name_ai_agent(populated_agent_service, sample_ai_agent):
    """Test getting an AI agent by name."""
    # Arrange
    populated_agent_service.agent_repository.get_ai_agent_by_name.return_value = sample_ai_agent

    # Act
    agent = populated_agent_service.get_agent_by_name("test_ai_agent")

    # Assert
    assert agent == sample_ai_agent
    populated_agent_service.agent_repository.get_ai_agent_by_name.assert_called_once_with(
        "test_ai_agent")


def test_get_agent_by_name_human_agent(populated_agent_service, sample_human_agent):
    """Test getting a human agent by name."""
    # Arrange
    populated_agent_service.agent_repository.get_ai_agent_by_name.return_value = None

    # Act
    agent = populated_agent_service.get_agent_by_name("Test Human")

    # Assert
    assert agent == sample_human_agent
    populated_agent_service.agent_repository.get_ai_agent_by_name.assert_called_once_with(
        "Test Human")
    populated_agent_service.agent_repository.get_all_human_agents.assert_called_once()


def test_get_agent_by_name_not_found(populated_agent_service):
    """Test getting an agent that doesn't exist."""
    # Arrange
    populated_agent_service.agent_repository.get_ai_agent_by_name.return_value = None

    # Act
    agent = populated_agent_service.get_agent_by_name("nonexistent_agent")

    # Assert
    assert agent is None
    populated_agent_service.agent_repository.get_ai_agent_by_name.assert_called_once_with(
        "nonexistent_agent")
    populated_agent_service.agent_repository.get_all_human_agents.assert_called_once()


def test_get_agent_system_prompt(populated_agent_service, sample_ai_agent):
    """Test getting an agent's system prompt."""
    # Act
    system_prompt = populated_agent_service.get_agent_system_prompt(
        "test_ai_agent")

    # Assert
    assert "You are test_ai_agent" in system_prompt
    assert "You are a test AI agent" in system_prompt
    assert "ORGANIZATION MISSION" in system_prompt
    assert "ORGANIZATION VALUES" in system_prompt
    assert "ORGANIZATION GOALS" in system_prompt
    populated_agent_service.agent_repository.get_ai_agent_by_name.assert_called_with(
        "test_ai_agent")


def test_get_agent_system_prompt_not_found(populated_agent_service):
    """Test getting system prompt for an agent that doesn't exist."""
    # Arrange
    populated_agent_service.agent_repository.get_ai_agent_by_name.return_value = None

    # Act
    system_prompt = populated_agent_service.get_agent_system_prompt(
        "nonexistent_agent")

    # Assert
    assert system_prompt == ""
    populated_agent_service.agent_repository.get_ai_agent_by_name.assert_called_once_with(
        "nonexistent_agent")


def test_get_all_ai_agents(populated_agent_service, sample_ai_agent):
    """Test getting all AI agents."""
    # Act
    agents = populated_agent_service.get_all_ai_agents()

    # Assert
    assert len(agents) == 1
    assert "test_ai_agent" in agents
    assert agents["test_ai_agent"] == sample_ai_agent
    populated_agent_service.agent_repository.get_all_ai_agents.assert_called_once()


def test_get_all_human_agents(populated_agent_service, sample_human_agent):
    """Test getting all human agents."""
    # Act
    agents = populated_agent_service.get_all_human_agents()

    # Assert
    assert len(agents) == 1
    assert "human123" in agents
    assert agents["human123"] == sample_human_agent
    populated_agent_service.agent_repository.get_all_human_agents.assert_called_once()


# ---------------------
# Specialization Tests
# ---------------------

def test_get_specializations(populated_agent_service):
    """Test getting all agent specializations."""
    # Act
    specializations = populated_agent_service.get_specializations()

    # Assert
    assert "testing" in specializations
    assert "customer_support" in specializations
    assert "AI expertise in testing" in specializations.values()
    assert "Human expertise in customer_support" in specializations.values()


def test_find_agents_by_specialization(populated_agent_service):
    """Test finding agents by specialization."""
    # Act
    testing_agents = populated_agent_service.find_agents_by_specialization(
        "testing")
    support_agents = populated_agent_service.find_agents_by_specialization(
        "customer_support")
    nonexistent_agents = populated_agent_service.find_agents_by_specialization(
        "nonexistent")

    # Assert
    assert "test_ai_agent" in testing_agents
    assert "human123" in testing_agents
    assert "human123" in support_agents
    assert "test_ai_agent" not in support_agents
    assert len(nonexistent_agents) == 0


def test_has_specialization(populated_agent_service, sample_ai_agent):
    """Test checking if an agent has a specialization."""
    # Override mock behavior for specialization checks
    # First reset any previous calls
    populated_agent_service.agent_repository.get_ai_agent_by_name.reset_mock()

    # Setup side effect to return different responses based on the agent name
    def get_agent_side_effect(name):
        if name == "test_ai_agent":
            return sample_ai_agent
        elif name == "nonexistent":
            return None
        else:
            return None

    populated_agent_service.agent_repository.get_ai_agent_by_name.side_effect = get_agent_side_effect

    # Act & Assert
    assert populated_agent_service.has_specialization(
        "test_ai_agent", "testing") is True
    assert populated_agent_service.has_specialization(
        "test_ai_agent", "customer_support") is False
    assert populated_agent_service.has_specialization(
        "human123", "customer_support") is True
    assert populated_agent_service.has_specialization(
        "human123", "nonexistent") is False
    assert populated_agent_service.has_specialization(
        "nonexistent", "testing") is False

# ---------------------
# Response Generation Tests
# ---------------------


@pytest.mark.asyncio
async def test_generate_response(populated_agent_service):
    """Test generating a response from an agent."""
    # Act
    response_chunks = []
    async for chunk in populated_agent_service.generate_response(
        agent_name="test_ai_agent",
        user_id="user123",
        query="Test query"
    ):
        response_chunks.append(chunk)

    # Assert
    assert len(response_chunks) > 0
    assert "".join(
        response_chunks) == "This is a test response from the mock LLM provider."


@pytest.mark.asyncio
async def test_generate_response_with_memory(populated_agent_service):
    """Test generating a response with memory context."""
    # Act
    response_chunks = []
    async for chunk in populated_agent_service.generate_response(
        agent_name="test_ai_agent",
        user_id="user123",
        query="Test query",
        memory_context="Previous conversation context."
    ):
        response_chunks.append(chunk)

    # Assert
    assert len(response_chunks) > 0
    # Check that the response contains our test text
    assert "".join(
        response_chunks) == "This is a test response from the mock LLM provider."


@pytest.mark.asyncio
async def test_generate_response_agent_not_found(populated_agent_service):
    """Test generating a response for an agent that doesn't exist."""
    # Arrange
    populated_agent_service.agent_repository.get_ai_agent_by_name.return_value = None

    # Act
    response_chunks = []
    async for chunk in populated_agent_service.generate_response(
        agent_name="nonexistent_agent",
        user_id="user123",
        query="Test query"
    ):
        response_chunks.append(chunk)

    # Assert
    assert len(response_chunks) == 1
    assert "not found" in response_chunks[0]


# ---------------------
# Tool Management Tests
# ---------------------

def test_assign_tool_for_agent(populated_agent_service):
    """Test registering a tool for an agent."""
    # Arrange
    populated_agent_service.tool_registry.assign_tool_to_agent.return_value = True

    # Act
    result = populated_agent_service.assign_tool_for_agent(
        "test_ai_agent", "test_tool")

    # Assert
    assert result is True
    populated_agent_service.tool_registry.assign_tool_to_agent.assert_called_once_with(
        "test_ai_agent", "test_tool")


def test_assign_tool_agent_not_found(populated_agent_service):
    """Test registering a tool for an agent that doesn't exist."""
    # Arrange
    populated_agent_service.agent_repository.get_ai_agent_by_name.return_value = None

    # Act
    result = populated_agent_service.assign_tool_for_agent(
        "nonexistent_agent", "test_tool")

    # Assert
    assert result is False
    populated_agent_service.tool_registry.assign_tool_to_agent.assert_not_called()


def test_get_agent_tools(populated_agent_service):
    """Test getting tools for an agent."""
    # Arrange
    populated_agent_service.tool_registry.get_agent_tools.return_value = [
        {"name": "test_tool", "description": "A test tool"}
    ]

    # Act
    tools = populated_agent_service.get_agent_tools("test_ai_agent")

    # Assert
    assert len(tools) == 1
    assert tools[0]["name"] == "test_tool"
    populated_agent_service.tool_registry.get_agent_tools.assert_called_once_with(
        "test_ai_agent")


def test_execute_tool_success(populated_agent_service):
    """Test executing a tool successfully."""
    # Arrange
    tool_mock = Mock()
    tool_mock.execute.return_value = {
        "status": "success", "result": "tool executed"}

    populated_agent_service.tool_registry.get_tool.return_value = tool_mock
    populated_agent_service.tool_registry.get_agent_tools.return_value = [
        {"name": "test_tool", "description": "A test tool"}
    ]

    # Act
    result = populated_agent_service.execute_tool(
        agent_name="test_ai_agent",
        tool_name="test_tool",
        parameters={"param": "value"}
    )

    # Assert
    assert result["status"] == "success"
    assert result["result"] == "tool executed"
    tool_mock.execute.assert_called_once_with(param="value")


def test_execute_tool_not_found(populated_agent_service):
    """Test executing a tool that doesn't exist."""
    # Arrange
    populated_agent_service.tool_registry.get_tool.return_value = None

    # Act
    result = populated_agent_service.execute_tool(
        agent_name="test_ai_agent",
        tool_name="nonexistent_tool",
        parameters={"param": "value"}
    )

    # Assert
    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_execute_tool_no_access(populated_agent_service):
    """Test executing a tool the agent doesn't have access to."""
    # Arrange
    tool_mock = Mock()
    populated_agent_service.tool_registry.get_tool.return_value = tool_mock
    populated_agent_service.tool_registry.get_agent_tools.return_value = [
        {"name": "other_tool", "description": "Another tool"}
    ]

    # Act
    result = populated_agent_service.execute_tool(
        agent_name="test_ai_agent",
        tool_name="test_tool",
        parameters={"param": "value"}
    )

    # Assert
    assert result["status"] == "error"
    assert "doesn't have access" in result["message"]
    tool_mock.execute.assert_not_called()


def test_execute_tool_exception(populated_agent_service):
    """Test executing a tool that raises an exception."""
    # Arrange
    tool_mock = Mock()
    tool_mock.execute.side_effect = Exception("Test exception")

    populated_agent_service.tool_registry.get_tool.return_value = tool_mock
    populated_agent_service.tool_registry.get_agent_tools.return_value = [
        {"name": "test_tool", "description": "A test tool"}
    ]

    # Act
    result = populated_agent_service.execute_tool(
        agent_name="test_ai_agent",
        tool_name="test_tool",
        parameters={"param": "value"}
    )

    # Assert
    assert result["status"] == "error"
    assert "Error executing tool" in result["message"]
    tool_mock.execute.assert_called_once_with(param="value")


# ---------------------
# Agent Status Tests
# ---------------------

def test_agent_exists(populated_agent_service):
    """Test checking if an agent exists."""
    # Reset any previous calls and configure mock behavior
    populated_agent_service.agent_repository.get_ai_agent_by_name.reset_mock()
    populated_agent_service.agent_repository.get_all_human_agents.reset_mock()

    # Set up side effect to properly handle different agent names
    def get_ai_agent_side_effect(name):
        if name == "test_ai_agent":
            return populated_agent_service.agent_repository.get_all_ai_agents()[0]
        else:
            return None

    populated_agent_service.agent_repository.get_ai_agent_by_name.side_effect = get_ai_agent_side_effect

    # Act & Assert
    assert populated_agent_service.agent_exists("test_ai_agent") is True
    assert populated_agent_service.agent_exists("human123") is True
    assert populated_agent_service.agent_exists("nonexistent_agent") is False


def test_list_active_agents(populated_agent_service, sample_ai_agent, sample_human_agent):
    """Test listing active agents."""
    # Arrange - Make one agent inactive
    inactive_human = HumanAgent(
        id="inactive123",
        name="Inactive Agent",
        specializations=["testing"],
        availability=False
    )
    populated_agent_service.agent_repository.get_all_human_agents.return_value = [
        sample_human_agent, inactive_human
    ]

    # Act
    active_agents = populated_agent_service.list_active_agents()

    # Assert
    assert "test_ai_agent" in active_agents
    assert "human123" in active_agents
    assert "inactive123" not in active_agents
    assert len(active_agents) == 2

from typing import Dict, List
import pytest
from unittest.mock import Mock, AsyncMock

from solana_agent.services.routing import RoutingService
from solana_agent.domains import QueryAnalysis, AIAgent

# Test Data
TEST_QUERY = "How do I setup a Solana validator?"
TEST_USER_ID = "test_user"

TEST_ANALYSIS = QueryAnalysis(
    primary_specialization="validator",
    secondary_specializations=["technical", "infrastructure"],
    complexity_level=4,
    topics=["solana", "validator", "setup"],
    confidence=0.9
)

TEST_AGENTS = {
    "validator_expert": AIAgent(
        name="validator_expert",
        instructions="Validator setup specialist",
        specialization="validator",
        model="gpt-4o",
        secondary_specializations=["technical", "infrastructure"]
    ),
    "general_ai": AIAgent(
        name="general_ai",
        instructions="General purpose agent",
        specialization="general",
        model="gpt-4o"
    )
}


@pytest.fixture
def mock_llm_provider():
    provider = Mock()
    provider.parse_structured_output = AsyncMock(return_value=TEST_ANALYSIS)
    return provider


@pytest.fixture
def mock_agent_service():
    service = Mock()
    service.get_all_ai_agents.return_value = TEST_AGENTS
    return service


@pytest.fixture
def routing_service(mock_llm_provider, mock_agent_service):
    return RoutingService(
        llm_provider=mock_llm_provider,
        agent_service=mock_agent_service
    )


@pytest.mark.asyncio
async def test_analyze_query_success(routing_service, mock_llm_provider):
    """Test successful query analysis."""
    analysis = await routing_service._analyze_query(TEST_QUERY)

    assert analysis["primary_specialization"] == "validator"
    assert "technical" in analysis["secondary_specializations"]
    assert analysis["complexity_level"] == 4
    assert "solana" in analysis["topics"]
    assert analysis["confidence"] == 0.9

    mock_llm_provider.parse_structured_output.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_query_error(routing_service, mock_llm_provider):
    """Test error handling in query analysis."""
    mock_llm_provider.parse_structured_output.side_effect = Exception(
        "Test error")

    analysis = await routing_service._analyze_query(TEST_QUERY)

    assert analysis["primary_specialization"] == "general"
    assert analysis["secondary_specializations"] == []
    assert analysis["complexity_level"] == 1
    assert analysis["topics"] == []
    assert analysis["confidence"] == 0.0


@pytest.mark.asyncio
async def test_route_query_to_specialist(routing_service):
    """Test routing to specialist agent."""
    agent_name = await routing_service.route_query(TEST_USER_ID, TEST_QUERY)
    assert agent_name == "validator_expert"


@pytest.mark.asyncio
async def test_route_query_fallback(routing_service, mock_agent_service):
    """Test fallback to general AI when no specialist found."""
    # Override both analysis and agent service for proper fallback testing
    routing_service._analyze_query = AsyncMock(return_value={
        "primary_specialization": "unknown_specialization",
        "secondary_specializations": [],
        "complexity_level": 1,
        "topics": [],
        "confidence": 0.5
    })

    # Override agent service to only return general AI
    mock_agent_service.get_all_ai_agents.return_value = {
        "general_ai": TEST_AGENTS["general_ai"]
    }

    agent_name = await routing_service.route_query(TEST_USER_ID, TEST_QUERY)
    assert agent_name == "general_ai"


@pytest.mark.asyncio
async def test_find_best_ai_agent_with_secondary_match(routing_service):
    """Test agent selection with secondary specialization match."""
    agent = await routing_service._find_best_ai_agent(
        primary_specialization="technical",
        secondary_specializations=["validator"]
    )
    assert agent == "validator_expert"


@pytest.mark.asyncio
async def test_find_best_ai_agent_no_agents(routing_service, mock_agent_service):
    """Test handling when no agents are available."""
    mock_agent_service.get_all_ai_agents.return_value = {}

    agent = await routing_service._find_best_ai_agent(
        primary_specialization="technical",
        secondary_specializations=[]
    )
    assert agent is None


@pytest.mark.asyncio  # Add asyncio marker
async def test_agent_scoring(routing_service):
    """Test the agent scoring mechanism."""
    result = await routing_service._find_best_ai_agent(  # Add await here
        primary_specialization="validator",
        secondary_specializations=["technical"]
    )

    # Validator expert should win with highest score
    assert result == "validator_expert"


@pytest.mark.asyncio
async def test_agent_scoring_no_match(routing_service, mock_agent_service):
    """Test scoring when no good matches are found."""
    # Override agents to only have general AI
    mock_agent_service.get_all_ai_agents.return_value = {
        "general_ai": TEST_AGENTS["general_ai"]
    }

    result = await routing_service._find_best_ai_agent(
        primary_specialization="unknown",
        secondary_specializations=["unknown"]
    )

    # Should return general AI as fallback
    assert result == "general_ai"

"""
Tests for the RoutingService implementation.

This module tests routing queries to appropriate agents based on
specializations, availability, and query analysis.
"""
import pytest
import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from typing import Dict, List, Any, Optional

from solana_agent.services.routing import RoutingService
from solana_agent.domains import Ticket, ScheduledTask, ScheduledTaskStatus, QueryAnalysis
from solana_agent.interfaces import LLMProvider, AgentService, TicketService, SchedulingService


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = Mock()
    provider.parse_structured_output = AsyncMock()
    return provider


@pytest.fixture
def mock_agent_service():
    """Create a mock agent service with some agent data."""
    service = Mock()

    # Create sample AI agents
    service._ai_agents = {
        "solana": MagicMock(specialization="solana"),
        "frontend": MagicMock(specialization="frontend"),
        "backend": MagicMock(specialization="backend"),
        "general": MagicMock(specialization="general"),
    }

    # Create sample human agents
    service._human_agents = {
        "human1": MagicMock(
            specializations=["solana", "blockchain"],
            availability=True
        ),
        "human2": MagicMock(
            specializations=["frontend", "design"],
            availability=True
        ),
        "human3": MagicMock(
            specializations=["backend", "database"],
            availability=False  # unavailable
        )
    }

    # Set up methods to get agents
    service.get_all_human_agents = Mock(return_value=service._human_agents)
    service.get_all_ai_agents = Mock(return_value=service._ai_agents)

    return service


@pytest.fixture
def mock_ticket_service():
    """Create a mock ticket service."""
    service = Mock()
    service.get_or_create_ticket = AsyncMock()
    service.assign_ticket = Mock(return_value=True)
    service.add_note_to_ticket = Mock()
    service.get_ticket_by_id = Mock()
    return service


@pytest.fixture
def mock_scheduling_service():
    """Create a mock scheduling service."""
    service = Mock()
    service.schedule_task = AsyncMock()
    service.get_agent_tasks = AsyncMock(return_value=[])
    service.repository = Mock()
    service.repository.get_tasks_by_metadata = Mock(return_value=[])
    service.repository.update_scheduled_task = Mock()
    service.get_agent_schedule = AsyncMock()
    return service


@pytest.fixture
def sample_ticket():
    """Create a sample ticket."""
    return Ticket(
        id="ticket-123",
        user_id="user-456",
        title="Help with Solana integration",
        description="I'm having trouble integrating my dApp with Solana.",
        status="new",
        created_at=datetime.datetime.now(datetime.timezone.utc),
        priority='medium'
    )


@pytest.fixture
def routing_service(mock_llm_provider, mock_agent_service, mock_ticket_service):
    """Create a routing service without scheduling service."""
    return RoutingService(
        llm_provider=mock_llm_provider,
        agent_service=mock_agent_service,
        ticket_service=mock_ticket_service,
    )


@pytest.fixture
def routing_service_with_scheduling(mock_llm_provider, mock_agent_service,
                                    mock_ticket_service, mock_scheduling_service):
    """Create a routing service with scheduling service."""
    return RoutingService(
        llm_provider=mock_llm_provider,
        agent_service=mock_agent_service,
        ticket_service=mock_ticket_service,
        scheduling_service=mock_scheduling_service,
    )


@pytest.fixture
def sample_query_analysis():
    """Create a sample query analysis result."""
    return QueryAnalysis(
        primary_specialization="solana",
        secondary_specializations=["blockchain", "frontend"],
        complexity_level=3,
        requires_human=False,
        topics=["solana", "web3", "integration"],
        confidence=0.8
    )


@pytest.fixture
def sample_complex_query_analysis():
    """Create a sample complex query analysis result."""
    return QueryAnalysis(
        primary_specialization="solana",
        secondary_specializations=["blockchain", "frontend"],
        complexity_level=5,
        requires_human=True,
        topics=["solana", "web3", "integration", "security"],
        confidence=0.9
    )


# -------------------------
# Initialization Tests
# -------------------------

def test_routing_service_initialization(routing_service):
    """Test that the routing service initializes correctly."""
    assert routing_service.scheduling_service is None


def test_routing_service_with_scheduling(routing_service_with_scheduling, mock_scheduling_service):
    """Test initialization with scheduling service."""
    assert routing_service_with_scheduling.scheduling_service == mock_scheduling_service


# -------------------------
# Query Analysis Tests
# -------------------------

@pytest.mark.asyncio
async def test_analyze_query_success(routing_service, mock_llm_provider, sample_query_analysis):
    """Test successful query analysis."""
    # Setup the mock to return our sample analysis
    mock_llm_provider.parse_structured_output.return_value = sample_query_analysis

    # Call the method
    result = await routing_service.analyze_query("How do I integrate with Solana?")

    # Assertions
    assert result["primary_specialization"] == "solana"
    assert "blockchain" in result["secondary_specializations"]
    assert result["complexity_level"] == 3
    assert result["requires_human"] is False
    assert "solana" in result["topics"]
    assert result["confidence"] == 0.8

    # Check LLM was called with appropriate prompt
    mock_llm_provider.parse_structured_output.assert_called_once()
    call_args = mock_llm_provider.parse_structured_output.call_args
    assert "How do I integrate with Solana?" in call_args[1]["prompt"]
    assert call_args[1]["model_class"] == QueryAnalysis


@pytest.mark.asyncio
async def test_analyze_query_error(routing_service, mock_llm_provider):
    """Test error handling during query analysis."""
    # Setup the mock to raise an exception
    mock_llm_provider.parse_structured_output.side_effect = Exception(
        "LLM Error")

    # Call the method
    result = await routing_service.analyze_query("How do I integrate with Solana?")

    # Should return default values on error
    assert result["primary_specialization"] == "general"
    assert result["complexity_level"] == 1
    assert result["requires_human"] is False
    assert result["confidence"] == 0.0


# -------------------------
# Query Routing Tests
# -------------------------

@pytest.mark.asyncio
async def test_route_query_basic(routing_service, mock_llm_provider,
                                 mock_ticket_service, sample_query_analysis,
                                 sample_ticket):
    """Test basic query routing to an AI agent."""
    # Setup mocks
    mock_llm_provider.parse_structured_output.return_value = sample_query_analysis
    mock_ticket_service.get_or_create_ticket.return_value = sample_ticket

    # Call the method
    agent_name, ticket = await routing_service.route_query(
        "user-456", "How do I integrate with Solana?"
    )

    # Assertions
    assert agent_name == "solana"  # Should route to Solana agent
    assert ticket == sample_ticket

    # Check that ticket was assigned
    mock_ticket_service.assign_ticket.assert_called_once_with(
        sample_ticket.id, "solana")
    mock_ticket_service.add_note_to_ticket.assert_called_once()


@pytest.mark.asyncio
async def test_route_complex_query_to_human(routing_service, mock_llm_provider,
                                            mock_ticket_service, sample_complex_query_analysis,
                                            sample_ticket):
    """Test routing complex queries to human agents."""
    # Setup mocks
    mock_llm_provider.parse_structured_output.return_value = sample_complex_query_analysis
    mock_ticket_service.get_or_create_ticket.return_value = sample_ticket

    # Set up patch to simulate finding a human agent
    with patch.object(routing_service, '_find_best_human_agent',
                      AsyncMock(return_value="human1")):
        # Call the method
        agent_name, ticket = await routing_service.route_query(
            "user-456", "Complex Solana security issue with integration"
        )

        # Assertions
        assert agent_name == "human1"  # Should route to human agent
        assert ticket == sample_ticket

        # Check that ticket was assigned to human agent
        mock_ticket_service.assign_ticket.assert_called_once_with(
            sample_ticket.id, "human1")


@pytest.mark.asyncio
async def test_route_ai_query_no_scheduling(routing_service_with_scheduling,
                                            mock_llm_provider, mock_ticket_service,
                                            mock_scheduling_service, sample_query_analysis,
                                            sample_ticket):
    """Test AI query routing doesn't use scheduling even when available."""
    # Setup mocks
    mock_llm_provider.parse_structured_output.return_value = sample_query_analysis
    mock_ticket_service.get_or_create_ticket.return_value = sample_ticket

    # Call the method
    agent_name, ticket = await routing_service_with_scheduling.route_query(
        "user-456", "How do I integrate with Solana?"
    )

    # Assertions
    assert agent_name == "solana"  # Should route to Solana agent
    assert ticket == sample_ticket

    # Check that scheduling was NOT used for AI agent
    mock_scheduling_service.schedule_task.assert_not_called()
    mock_ticket_service.add_note_to_ticket.assert_called_once()
    note_text = mock_ticket_service.add_note_to_ticket.call_args[0][1]
    assert "Assigned to" in note_text
    assert "Scheduled for" not in note_text


@pytest.mark.asyncio
async def test_route_human_query_with_scheduling(routing_service_with_scheduling,
                                                 mock_llm_provider, mock_ticket_service,
                                                 mock_scheduling_service, sample_complex_query_analysis,
                                                 sample_ticket):
    """Test human query routing with scheduling service."""
    # Setup mocks
    mock_llm_provider.parse_structured_output.return_value = sample_complex_query_analysis
    mock_ticket_service.get_or_create_ticket.return_value = sample_ticket

    # Set up patch to simulate finding a human agent
    with patch.object(routing_service_with_scheduling, '_find_best_human_agent',
                      AsyncMock(return_value="human1")):

        # Create a scheduled task as the return value
        scheduled_task = ScheduledTask(
            task_id=f"ticket_{sample_ticket.id}",
            title=f"Handle ticket: {sample_ticket.title}",
            description=sample_ticket.description,
            status=ScheduledTaskStatus.SCHEDULED,
            priority=5,
            estimated_minutes=45,
            assigned_to="human1",  # Human agent
            scheduled_start=datetime.datetime.now(
                datetime.timezone.utc) + datetime.timedelta(minutes=30)
        )
        mock_scheduling_service.schedule_task.return_value = scheduled_task

        # Call the method
        agent_name, ticket = await routing_service_with_scheduling.route_query(
            "user-456", "Complex Solana security issue with integration"
        )

        # Assertions
        assert agent_name == "human1"  # Should route to human agent
        assert ticket == sample_ticket

        # Check that scheduling was used for human agent
        mock_scheduling_service.schedule_task.assert_called_once()
        mock_ticket_service.add_note_to_ticket.assert_called_once()
        note_text = mock_ticket_service.add_note_to_ticket.call_args[0][1]
        assert "Scheduled for" in note_text


# -------------------------
# Ticket Rerouting Tests
# -------------------------

@pytest.mark.asyncio
async def test_reroute_ticket_success(routing_service, mock_ticket_service, sample_ticket):
    """Test successful ticket rerouting."""
    # Setup mock
    mock_ticket_service.get_ticket_by_id.return_value = sample_ticket

    # Call the method
    result = await routing_service.reroute_ticket(
        "ticket-123", "frontend", "Needs frontend expertise"
    )

    # Assertions
    assert result is True
    mock_ticket_service.assign_ticket.assert_called_once_with(
        "ticket-123", "frontend")
    mock_ticket_service.add_note_to_ticket.assert_called_once()
    note_text = mock_ticket_service.add_note_to_ticket.call_args[0][1]
    assert "Rerouted to frontend" in note_text
    assert "Needs frontend expertise" in note_text


@pytest.mark.asyncio
async def test_reroute_ticket_to_human_with_scheduling(routing_service_with_scheduling,
                                                       mock_ticket_service, mock_scheduling_service,
                                                       sample_ticket):
    """Test rerouting to human agent with scheduling."""
    # Setup mocks
    mock_ticket_service.get_ticket_by_id.return_value = sample_ticket

    # Create a scheduled task for this ticket
    task = ScheduledTask(
        task_id="task-1",
        title="Handle ticket",
        description="Test task",
        status=ScheduledTaskStatus.SCHEDULED,
        priority=5,
        assigned_to="solana",  # Current AI agent
        metadata={"ticket_id": "ticket-123"}
    )
    mock_scheduling_service.repository.get_tasks_by_metadata.return_value = [
        task]

    # Patch is_human_agent to return True for human2
    routing_service_with_scheduling.agent_service.get_all_human_agents = Mock(
        return_value={"human2": MagicMock()}
    )

    # Call the method to reroute to human agent
    result = await routing_service_with_scheduling.reroute_ticket(
        "ticket-123", "human2", "Needs human expertise"
    )

    # Assertions
    assert result is True

    # Check that task was updated for human
    mock_scheduling_service.repository.update_scheduled_task.assert_called_once()
    assert task.assigned_to == "human2"

    # Check that ticket was reassigned
    mock_ticket_service.assign_ticket.assert_called_once_with(
        "ticket-123", "human2")


@pytest.mark.asyncio
async def test_reroute_ticket_to_ai_agent(routing_service_with_scheduling,
                                          mock_ticket_service, mock_scheduling_service,
                                          sample_ticket):
    """Test rerouting to AI agent without scheduling."""
    # Setup mocks
    mock_ticket_service.get_ticket_by_id.return_value = sample_ticket

    # Create a scheduled task for this ticket with human agent
    task = ScheduledTask(
        task_id="task-1",
        title="Handle ticket",
        description="Test task",
        status=ScheduledTaskStatus.SCHEDULED,
        priority=5,
        assigned_to="human1",  # Current human agent
        metadata={"ticket_id": "ticket-123"}
    )
    mock_scheduling_service.repository.get_tasks_by_metadata.return_value = [
        task]

    # Patch is_human_agent to return False for solana
    routing_service_with_scheduling.agent_service.get_all_human_agents = Mock(
        return_value={})
    routing_service_with_scheduling.agent_service.get_all_ai_agents = Mock(
        return_value={"solana": MagicMock()}
    )

    # Call the method to reroute to AI agent
    result = await routing_service_with_scheduling.reroute_ticket(
        "ticket-123", "solana", "Can be handled by AI"
    )

    # Assertions
    assert result is True

    # Verify the task was completed since AI doesn't need scheduling
    assert task.status == ScheduledTaskStatus.COMPLETED
    mock_scheduling_service.repository.update_scheduled_task.assert_called_once()

    # Check that ticket was reassigned
    mock_ticket_service.assign_ticket.assert_called_once_with(
        "ticket-123", "solana")


@pytest.mark.asyncio
async def test_reroute_nonexistent_ticket(routing_service, mock_ticket_service):
    """Test rerouting a ticket that doesn't exist."""
    # Setup mock to return None for nonexistent ticket
    mock_ticket_service.get_ticket_by_id.return_value = None

    # Call the method
    result = await routing_service.reroute_ticket(
        "nonexistent-ticket", "frontend", "Needs frontend expertise"
    )

    # Assertions
    assert result is False
    mock_ticket_service.assign_ticket.assert_not_called()
    mock_ticket_service.add_note_to_ticket.assert_not_called()


# -------------------------
# Agent Finding Tests
# -------------------------

@pytest.mark.asyncio
async def test_find_best_human_agent(routing_service, mock_agent_service, sample_ticket):
    """Test finding the best human agent."""
    # Call the method
    result = await routing_service._find_best_human_agent(
        "solana", ["blockchain", "frontend"]
    )

    # Should find human1 who has solana specialization and is available
    assert result == "human1"


@pytest.mark.asyncio
async def test_find_human_agent_with_schedule(routing_service_with_scheduling,
                                              mock_scheduling_service, sample_ticket):
    """Test finding human agent with schedule check."""
    # Set up agent schedule mock
    today = datetime.datetime.now(datetime.timezone.utc).date()

    # Configure mock schedule responses for different agents
    def mock_agent_schedule(agent_id, date):
        schedule = MagicMock()
        if agent_id == "human1":
            # human1 is available
            schedule.has_availability_today.return_value = True
            schedule.is_available_at.return_value = True
        else:
            # human2 is not available according to schedule
            schedule.has_availability_today.return_value = False
            schedule.is_available_at.return_value = False
        return schedule

    mock_scheduling_service.get_agent_schedule = AsyncMock(
        side_effect=mock_agent_schedule)

    # Call the method
    result = await routing_service_with_scheduling._find_best_human_agent(
        "solana", ["blockchain"]
    )

    # Should find human1 who is available based on schedule
    assert result == "human1"

    # Verify schedule was checked for human agents
    mock_scheduling_service.get_agent_schedule.assert_any_call("human1", today)


@pytest.mark.asyncio
async def test_find_human_agent_with_workload(routing_service_with_scheduling,
                                              mock_scheduling_service, sample_ticket):
    """Test finding human agent considering workload."""
    # Set up tasks for agents to simulate workload
    mock_scheduling_service.get_agent_tasks.side_effect = lambda agent_id, **kwargs: (
        # human1 has no tasks, human2 has 2
        [] if agent_id == "human1" else [MagicMock(), MagicMock()]
    )

    # Set up schedule - both agents are available
    mock_schedule = MagicMock()
    mock_schedule.has_availability_today.return_value = True
    mock_schedule.is_available_at.return_value = True
    mock_scheduling_service.get_agent_schedule.return_value = mock_schedule

    # Call the method
    result = await routing_service_with_scheduling._find_best_human_agent(
        "frontend", ["design"]
    )

    # Should find human2 despite workload since it's the only match for frontend
    assert result == "human2"

    # Test with equal specialization match where workload matters
    # Both human1 and human2 match "blockchain" as secondary, but human1 has no tasks
    result = await routing_service_with_scheduling._find_best_human_agent(
        "blockchain", ["design"]
    )

    # Should prefer human1 due to lower workload
    assert result == "human1"

    # Verify task count was checked
    mock_scheduling_service.get_agent_tasks.assert_any_call(
        "human1",
        start_date=datetime.datetime.now(datetime.timezone.utc),
        include_completed=False
    )


@pytest.mark.asyncio
async def test_find_best_ai_agent(routing_service, sample_ticket):
    """Test finding the best AI agent."""
    # Call the method
    agent_name, is_scheduled, task = await routing_service._find_best_ai_agent(
        "solana", ["blockchain", "frontend"]
    )

    # Should find the solana agent without scheduling
    assert agent_name == "solana"
    assert is_scheduled is False
    assert task is None


@pytest.mark.asyncio
async def test_find_best_ai_agent_with_scheduling_disabled(routing_service_with_scheduling,
                                                           mock_scheduling_service, sample_ticket):
    """Test finding AI agent with scheduling service available but not used."""
    # Call the method
    agent_name, is_scheduled, task = await routing_service_with_scheduling._find_best_ai_agent(
        "solana", ["blockchain", "frontend"]
    )

    # Should find the solana agent but NOT schedule it
    assert agent_name == "solana"
    assert is_scheduled is False
    assert task is None

    # Schedule_task should never be called for AI agents
    mock_scheduling_service.schedule_task.assert_not_called()


def test_get_priority_from_complexity(routing_service):
    """Test priority calculation from complexity."""
    # Test all complexity levels
    assert routing_service._get_priority_from_complexity(1) == 1
    assert routing_service._get_priority_from_complexity(2) == 2
    assert routing_service._get_priority_from_complexity(3) == 5
    assert routing_service._get_priority_from_complexity(4) == 8
    assert routing_service._get_priority_from_complexity(5) == 10

    # Test default case
    assert routing_service._get_priority_from_complexity(999) == 5

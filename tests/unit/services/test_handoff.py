"""
Tests for the HandoffService implementation.

This module tests handoff evaluation, human assistance requests,
and agent-to-agent transfers.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from solana_agent.services.handoff import HandoffService
from solana_agent.domains import (
    TicketStatus, TicketNote, HandoffEvaluation, Handoff,
    AIAgent, HumanAgent
)


# ---------------------
# Fixtures
# ---------------------

@pytest.fixture
def mock_handoff_repository():
    """Return a mock handoff repository."""
    repo = Mock()
    repo.record = Mock(return_value=True)
    repo.get_by_ticket = Mock(return_value=[])
    repo.get_stats = Mock(return_value={
        "total_handoffs": 5,
        "human_escalations": 2,
        "successful_handoffs": 4
    })
    return repo


@pytest.fixture
def mock_ticket_repository():
    """Return a mock ticket repository."""
    repo = Mock()

    # Create a test ticket
    ticket = Mock()
    ticket.id = "ticket123"
    ticket.title = "Test Ticket"
    ticket.status = TicketStatus.ASSIGNED
    ticket.assigned_to = "ai_agent_1"
    ticket.user_id = "user123"
    ticket.created_at = datetime(2025, 3, 1, 10, 0)
    ticket.updated_at = datetime(2025, 3, 1, 10, 30)
    ticket.notes = []

    # Set up repository methods
    repo.get_by_id = Mock(return_value=ticket)
    repo.update = Mock(return_value=True)
    repo.add_note = Mock(return_value=True)

    return repo


@pytest.fixture
def mock_llm_provider():
    """Return a mock LLM provider."""
    provider = Mock()

    # Set up parse_structured_output for handoff evaluations
    async def mock_parse_handoff(**kwargs):
        # By default, return a handoff is not needed
        return HandoffEvaluation(
            handoff_needed=False,
            confidence=0.8,
            target_agent=None,
            reason=None
        )

    provider.parse_structured_output = AsyncMock(
        side_effect=mock_parse_handoff)
    return provider


@pytest.fixture
def mock_agent_service(mock_llm_provider):
    """Return a mock agent service."""
    service = Mock()
    service.llm_provider = mock_llm_provider

    # Set up AI agents
    ai_agent_1 = AIAgent(
        name="ai_agent_1",
        instructions="General assistant",
        specialization="general",
        model="gpt-4o-mini"
    )

    ai_agent_2 = AIAgent(
        name="ai_agent_2",
        instructions="Technical specialist",
        specialization="programming",
        model="gpt-4o"
    )

    # Set up human agents
    human_agent_1 = HumanAgent(
        id="human_agent_1",
        name="Human Support",
        specializations=["customer_support", "general"],
        availability=True
    )

    human_agent_2 = HumanAgent(
        id="human_agent_2",
        name="Technical Expert",
        specializations=["programming", "technical"],
        availability=True
    )

    unavailable_human = HumanAgent(
        id="human_agent_3",
        name="Unavailable Human",
        specializations=["general"],
        availability=False
    )

    # Set up agent service methods
    service.get_all_ai_agents = Mock(return_value={
        "ai_agent_1": ai_agent_1,
        "ai_agent_2": ai_agent_2
    })

    service.get_all_human_agents = Mock(return_value={
        "human_agent_1": human_agent_1,
        "human_agent_2": human_agent_2,
        "human_agent_3": unavailable_human
    })

    return service


@pytest.fixture
def handoff_service(mock_handoff_repository, mock_ticket_repository, mock_agent_service):
    """Return a handoff service with mocked dependencies."""
    return HandoffService(
        handoff_repository=mock_handoff_repository,
        ticket_repository=mock_ticket_repository,
        agent_service=mock_agent_service
    )


# ---------------------
# Initialization Tests
# ---------------------

def test_handoff_service_initialization(mock_handoff_repository, mock_ticket_repository, mock_agent_service):
    """Test that the handoff service initializes properly."""
    service = HandoffService(
        handoff_repository=mock_handoff_repository,
        ticket_repository=mock_ticket_repository,
        agent_service=mock_agent_service
    )

    assert service.handoff_repository == mock_handoff_repository
    assert service.ticket_repository == mock_ticket_repository
    assert service.agent_service == mock_agent_service
    assert service.llm_provider == mock_agent_service.llm_provider


# ---------------------
# Handoff Evaluation Tests
# ---------------------

@pytest.mark.asyncio
async def test_evaluate_handoff_not_needed(handoff_service):
    """Test evaluation when handoff is not needed."""
    # Use a longer query to bypass the length check in evaluate_handoff_needed
    query = "What's the weather today in San Francisco? I'm planning a trip and need to know if I should bring an umbrella or sunglasses."
    response = "The weather today in San Francisco is sunny with a high of 75°F. You should bring sunglasses and perhaps a light jacket for the evening fog."
    current_agent = "ai_agent_1"

    # Act
    needed, target, reason = await handoff_service.evaluate_handoff_needed(
        query, response, current_agent
    )

    # Assert
    assert needed is False
    assert target is None
    assert reason is None
    handoff_service.llm_provider.parse_structured_output.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_handoff_needed(handoff_service):
    """Test evaluation when handoff is needed."""
    # Sample conversation
    query = "Can you help me debug this complex Python error in my Solana smart contract?"
    response = "I'm not specialized in Solana development. Let me get someone more technical."
    current_agent = "ai_agent_1"

    # Configure mock to indicate handoff is needed
    async def mock_handoff_needed(**kwargs):
        return HandoffEvaluation(
            handoff_needed=True,
            confidence=0.9,
            target_agent="ai_agent_2",
            reason="Technical programming question beyond current agent's expertise"
        )

    handoff_service.llm_provider.parse_structured_output.side_effect = mock_handoff_needed

    # Act
    needed, target, reason = await handoff_service.evaluate_handoff_needed(
        query, response, current_agent
    )

    # Assert
    assert needed is True
    assert target == "ai_agent_2"
    assert "Technical programming" in reason
    handoff_service.llm_provider.parse_structured_output.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_handoff_error(handoff_service):
    """Test evaluation error handling."""
    # Use a longer query to bypass the length check
    query = "I'm encountering a complex issue with my smart contract on Solana. When I try to deploy it, I get this error message that I don't understand."
    response = "I'll help you troubleshoot that Solana smart contract deployment issue. First, let me understand the error message you're seeing."
    current_agent = "ai_agent_1"

    # Configure mock to throw an exception
    handoff_service.llm_provider.parse_structured_output.side_effect = Exception(
        "Model error")

    # Act
    needed, target, reason = await handoff_service.evaluate_handoff_needed(
        query, response, current_agent
    )

    # Assert
    assert needed is False
    assert target is None
    assert reason is None
    handoff_service.llm_provider.parse_structured_output.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_handoff_short_query(handoff_service):
    """Test evaluation with short queries that skip evaluation."""
    # Sample conversation with very short query
    query = "Thanks!"
    response = "You're welcome!"
    current_agent = "ai_agent_1"

    # Act
    needed, target, reason = await handoff_service.evaluate_handoff_needed(
        query, response, current_agent
    )

    # Assert
    assert needed is False
    assert target is None
    assert reason is None
    # Verify we skipped the LLM call for short queries
    handoff_service.llm_provider.parse_structured_output.assert_not_called()


# ---------------------
# Human Help Request Tests
# ---------------------

@pytest.mark.asyncio
async def test_request_human_help_with_matching_specialization(handoff_service):
    """Test requesting human help with a matching specialization."""
    # Act
    success = await handoff_service.request_human_help(
        ticket_id="ticket123",
        reason="Technical question",
        specialization="programming"
    )

    # Assert
    assert success is True
    handoff_service.ticket_repository.update.assert_called_once()
    handoff_service.ticket_repository.add_note.assert_called_once()
    handoff_service.handoff_repository.record.assert_called_once()

    # Verify the ticket was assigned to the correct human
    update_call = handoff_service.ticket_repository.update.call_args[0]
    assert update_call[0] == "ticket123"
    # The technical expert
    assert update_call[1]["assigned_to"] == "human_agent_2"
    assert update_call[1]["status"] == TicketStatus.WAITING_FOR_HUMAN


@pytest.mark.asyncio
async def test_request_human_help_no_matching_specialization(handoff_service):
    """Test requesting human help with no matching specialization."""
    # Act
    success = await handoff_service.request_human_help(
        ticket_id="ticket123",
        reason="General question",
        specialization="marketing"  # No agent has this specialization
    )

    # Assert
    assert success is True
    handoff_service.ticket_repository.update.assert_called_once()

    # Verify any available human was assigned
    update_call = handoff_service.ticket_repository.update.call_args[0]
    assert update_call[0] == "ticket123"
    assert update_call[1]["assigned_to"] in ["human_agent_1", "human_agent_2"]
    assert update_call[1]["status"] == TicketStatus.WAITING_FOR_HUMAN


@pytest.mark.asyncio
async def test_request_human_help_no_humans_available(handoff_service, mock_agent_service):
    """Test requesting human help when no humans are available."""
    # Modify mock to return no available humans
    no_humans_available = {}
    mock_agent_service.get_all_human_agents.return_value = no_humans_available

    # Act
    success = await handoff_service.request_human_help(
        ticket_id="ticket123",
        reason="General question"
    )

    # Assert
    assert success is False
    handoff_service.ticket_repository.update.assert_not_called()
    handoff_service.ticket_repository.add_note.assert_called_once()


@pytest.mark.asyncio
async def test_request_human_help_nonexistent_ticket(handoff_service):
    """Test requesting human help for a nonexistent ticket."""
    # Modify mock to return None for ticket
    handoff_service.ticket_repository.get_by_id.return_value = None

    # Act
    success = await handoff_service.request_human_help(
        ticket_id="nonexistent",
        reason="General question"
    )

    # Assert
    assert success is False
    handoff_service.ticket_repository.update.assert_not_called()
    handoff_service.ticket_repository.add_note.assert_not_called()


# ---------------------
# Handoff Handling Tests
# ---------------------

@pytest.mark.asyncio
async def test_handle_handoff_to_ai_agent(handoff_service):
    """Test handling handoff to an AI agent."""
    # Act
    success = await handoff_service.handle_handoff(
        ticket_id="ticket123",
        target_agent="ai_agent_2",
        reason="Technical query"
    )

    # Assert
    assert success is True
    handoff_service.ticket_repository.update.assert_called_once()
    handoff_service.ticket_repository.add_note.assert_called_once()
    handoff_service.handoff_repository.record.assert_called_once()

    # Verify the ticket was assigned to the correct agent
    update_call = handoff_service.ticket_repository.update.call_args[0]
    assert update_call[0] == "ticket123"
    assert update_call[1]["assigned_to"] == "ai_agent_2"
    assert update_call[1]["status"] == TicketStatus.ASSIGNED


@pytest.mark.asyncio
async def test_handle_handoff_to_human_agent_by_id(handoff_service):
    """Test handling handoff to a human agent by ID."""
    # This should delegate to request_human_help
    with patch.object(handoff_service, 'request_human_help', AsyncMock(return_value=True)) as mock_request:
        # Act
        success = await handoff_service.handle_handoff(
            ticket_id="ticket123",
            target_agent="human_agent_1",
            reason="Need human help"
        )

        # Assert
        assert success is True
        # Fix: Use keyword arguments to match the actual implementation
        mock_request.assert_called_once_with(
            ticket_id="ticket123", reason="Need human help")


@pytest.mark.asyncio
async def test_handle_handoff_to_human_agent_by_name(handoff_service):
    """Test handling handoff to a human agent by name."""
    # This should delegate to request_human_help
    with patch.object(handoff_service, 'request_human_help', AsyncMock(return_value=True)) as mock_request:
        # Act
        success = await handoff_service.handle_handoff(
            ticket_id="ticket123",
            target_agent="Human Support",  # Name instead of ID
            reason="Need human help"
        )

        # Assert
        assert success is True
        mock_request.assert_called_once()


@pytest.mark.asyncio
async def test_handle_handoff_to_generic_human(handoff_service):
    """Test handling handoff to a generic human."""
    # This should delegate to request_human_help
    with patch.object(handoff_service, 'request_human_help', AsyncMock(return_value=True)) as mock_request:
        # Act
        success = await handoff_service.handle_handoff(
            ticket_id="ticket123",
            target_agent="human",  # Generic request
            reason="Need human help"
        )

        # Assert
        assert success is True

        # Fix: Use keyword arguments to match the actual implementation
        mock_request.assert_called_once_with(
            ticket_id="ticket123", reason="Need human help")


@pytest.mark.asyncio
async def test_handle_handoff_nonexistent_agent(handoff_service):
    """Test handling handoff to nonexistent agent."""
    # Act
    success = await handoff_service.handle_handoff(
        ticket_id="ticket123",
        target_agent="nonexistent_agent",
        reason="Wrong agent"
    )

    # Assert
    assert success is False
    handoff_service.ticket_repository.add_note.assert_called_once()
    note_call = handoff_service.ticket_repository.add_note.call_args[0]
    assert "not found" in note_call[1].content


@pytest.mark.asyncio
async def test_handle_handoff_nonexistent_ticket(handoff_service):
    """Test handling handoff for nonexistent ticket."""
    # Modify mock to return None for ticket
    handoff_service.ticket_repository.get_by_id.return_value = None

    # Act
    success = await handoff_service.handle_handoff(
        ticket_id="nonexistent",
        target_agent="ai_agent_2",
        reason="Wrong ticket"
    )

    # Assert
    assert success is False
    handoff_service.ticket_repository.update.assert_not_called()
    handoff_service.ticket_repository.add_note.assert_not_called()


# ---------------------
# Combined Workflow Tests
# ---------------------

@pytest.mark.asyncio
async def test_evaluate_and_handle_handoff_workflow(handoff_service):
    """Test the full workflow from evaluation to handoff."""
    # Configure evaluation mock to recommend handoff
    async def mock_handoff_needed(**kwargs):
        return HandoffEvaluation(
            handoff_needed=True,
            confidence=0.9,
            target_agent="ai_agent_2",
            reason="Technical question beyond current agent's expertise"
        )

    handoff_service.llm_provider.parse_structured_output.side_effect = mock_handoff_needed

    # Test the workflow
    # 1. Evaluate if handoff is needed
    query = "How do I implement a Solana smart contract?"
    response = "I'm not specialized in Solana development."
    current_agent = "ai_agent_1"

    needed, target, reason = await handoff_service.evaluate_handoff_needed(
        query, response, current_agent
    )

    assert needed is True
    assert target == "ai_agent_2"

    # 2. Handle the handoff based on evaluation
    if needed:
        success = await handoff_service.handle_handoff(
            ticket_id="ticket123",
            target_agent=target,
            reason=reason
        )
        assert success is True

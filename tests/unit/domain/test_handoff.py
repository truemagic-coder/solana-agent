"""
Tests for the handoff domain models.

This module tests the Handoff and HandoffEvaluation domain models 
using both standard pytest tests and property-based testing with hypothesis.
"""
import pytest
from datetime import datetime, timedelta
from hypothesis import given, strategies as st
from pydantic import ValidationError

from solana_agent.domains import Handoff, HandoffEvaluation

# ---------------------
# Fixtures
# ---------------------


@pytest.fixture
def basic_handoff():
    """Return a basic handoff for testing."""
    return Handoff(
        id="handoff123",
        from_agent="agent_alpha",
        to_agent="agent_bravo",
        ticket_id="ticket456",
        reason="Question requires specialized knowledge",
        notes="Please handle with priority"
    )


@pytest.fixture
def basic_handoff_evaluation():
    """Return a basic handoff evaluation for testing."""
    return HandoffEvaluation(
        handoff_needed=True,
        target_agent="agent_charlie",
        reason="Query is better handled by a specialist",
        confidence=0.85
    )


@pytest.fixture
def handoff_factory():
    """Return a factory function for creating handoffs."""
    def _create_handoff(
        handoff_id="handoff123",
        from_agent="agent_alpha",
        to_agent="agent_bravo",
        ticket_id="ticket456",
        reason="Standard handoff",
        timestamp=None,
        successful=True,
        notes=None
    ):
        if timestamp is None:
            timestamp = datetime.now()

        return Handoff(
            id=handoff_id,
            from_agent=from_agent,
            to_agent=to_agent,
            ticket_id=ticket_id,
            reason=reason,
            timestamp=timestamp,
            successful=successful,
            notes=notes
        )
    return _create_handoff


@pytest.fixture
def handoff_evaluation_factory():
    """Return a factory function for creating handoff evaluations."""
    def _create_handoff_evaluation(
        handoff_needed=True,
        target_agent="agent_specialist",
        reason="Specialized knowledge required",
        confidence=0.75
    ):
        return HandoffEvaluation(
            handoff_needed=handoff_needed,
            target_agent=target_agent,
            reason=reason,
            confidence=confidence
        )
    return _create_handoff_evaluation


# ---------------------
# Basic Creation Tests
# ---------------------

def test_handoff_creation(basic_handoff):
    """Test creating a Handoff with valid parameters."""
    assert basic_handoff.id == "handoff123"
    assert basic_handoff.from_agent == "agent_alpha"
    assert basic_handoff.to_agent == "agent_bravo"
    assert basic_handoff.ticket_id == "ticket456"
    assert basic_handoff.reason == "Question requires specialized knowledge"
    assert basic_handoff.successful is True
    assert basic_handoff.notes == "Please handle with priority"
    assert isinstance(basic_handoff.timestamp, datetime)


def test_handoff_evaluation_creation(basic_handoff_evaluation):
    """Test creating a HandoffEvaluation with valid parameters."""
    assert basic_handoff_evaluation.handoff_needed is True
    assert basic_handoff_evaluation.target_agent == "agent_charlie"
    assert basic_handoff_evaluation.reason == "Query is better handled by a specialist"
    assert basic_handoff_evaluation.confidence == 0.85


# ---------------------
# Factory Usage Tests
# ---------------------

def test_handoff_factory(handoff_factory):
    """Test creating handoffs with the factory."""
    # Create with defaults
    default_handoff = handoff_factory()
    assert default_handoff.id == "handoff123"
    assert default_handoff.from_agent == "agent_alpha"
    assert default_handoff.to_agent == "agent_bravo"
    assert default_handoff.successful is True

    # Create with custom values
    custom_handoff = handoff_factory(
        handoff_id="custom_id",
        from_agent="custom_source",
        to_agent="custom_target",
        ticket_id="custom_ticket",
        reason="Custom reason",
        successful=False,
        notes="Handoff failed due to agent unavailability"
    )
    assert custom_handoff.id == "custom_id"
    assert custom_handoff.from_agent == "custom_source"
    assert custom_handoff.to_agent == "custom_target"
    assert custom_handoff.successful is False
    assert custom_handoff.notes == "Handoff failed due to agent unavailability"


def test_handoff_evaluation_factory(handoff_evaluation_factory):
    """Test creating handoff evaluations with the factory."""
    # Create with defaults
    default_evaluation = handoff_evaluation_factory()
    assert default_evaluation.handoff_needed is True
    assert default_evaluation.target_agent == "agent_specialist"
    assert default_evaluation.confidence == 0.75

    # Create with custom values
    custom_evaluation = handoff_evaluation_factory(
        handoff_needed=False,
        target_agent=None,
        reason="Current agent can handle the query",
        confidence=0.95
    )
    assert custom_evaluation.handoff_needed is False
    assert custom_evaluation.target_agent is None
    assert custom_evaluation.reason == "Current agent can handle the query"
    assert custom_evaluation.confidence == 0.95


# ---------------------
# Validation Tests
# ---------------------

def test_handoff_required_fields():
    """Test validation of required Handoff fields."""
    # Missing required fields
    with pytest.raises(ValidationError):
        Handoff()

    # Missing from_agent
    with pytest.raises(ValidationError):
        Handoff(
            to_agent="agent_target",
            ticket_id="ticket123",
            reason="Test reason"
        )

    # Missing to_agent
    with pytest.raises(ValidationError):
        Handoff(
            from_agent="agent_source",
            ticket_id="ticket123",
            reason="Test reason"
        )

    # Valid minimum fields
    handoff = Handoff(
        from_agent="agent_source",
        to_agent="agent_target",
        ticket_id="ticket123",
        reason="Test reason"
    )
    assert handoff.id is None  # Optional field
    assert handoff.from_agent == "agent_source"
    assert handoff.to_agent == "agent_target"
    assert handoff.ticket_id == "ticket123"
    assert handoff.reason == "Test reason"
    assert isinstance(handoff.timestamp, datetime)  # Default value
    assert handoff.successful is True  # Default value
    assert handoff.notes is None  # Optional field


def test_handoff_evaluation_required_fields():
    """Test validation of required HandoffEvaluation fields."""
    # Missing required fields
    with pytest.raises(ValidationError):
        HandoffEvaluation()

    # Missing handoff_needed
    with pytest.raises(ValidationError):
        HandoffEvaluation(
            target_agent="agent_target",
            confidence=0.8
        )

    # Valid minimum fields
    evaluation = HandoffEvaluation(
        handoff_needed=True,
        confidence=0.8
    )
    assert evaluation.handoff_needed is True
    assert evaluation.target_agent is None  # Optional field
    assert evaluation.reason is None  # Optional field
    assert evaluation.confidence == 0.8


def test_handoff_evaluation_confidence_bounds():
    """Test validation of confidence bounds in HandoffEvaluation."""
    # Confidence too low
    with pytest.raises(ValidationError):
        HandoffEvaluation(
            handoff_needed=True,
            confidence=-0.1  # Should be at least 0
        )

    # Confidence too high
    with pytest.raises(ValidationError):
        HandoffEvaluation(
            handoff_needed=True,
            confidence=1.1  # Should be at most 1
        )

    # Valid boundary values
    min_confidence = HandoffEvaluation(
        handoff_needed=False,
        confidence=0.0
    )
    assert min_confidence.confidence == 0.0

    max_confidence = HandoffEvaluation(
        handoff_needed=True,
        confidence=1.0
    )
    assert max_confidence.confidence == 1.0


# ---------------------
# Property-Based Tests
# ---------------------

@given(
    from_agent=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    to_agent=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    ticket_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    reason=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    successful=st.booleans(),
    notes=st.one_of(st.none(), st.text(min_size=0, max_size=1000))
)
def test_handoff_properties(from_agent, to_agent, ticket_id, reason, successful, notes):
    """Test Handoff with various generated properties."""
    handoff = Handoff(
        from_agent=from_agent,
        to_agent=to_agent,
        ticket_id=ticket_id,
        reason=reason,
        successful=successful,
        notes=notes
    )

    assert handoff.from_agent == from_agent
    assert handoff.to_agent == to_agent
    assert handoff.ticket_id == ticket_id
    assert handoff.reason == reason
    assert handoff.successful is successful
    assert handoff.notes == notes
    assert isinstance(handoff.timestamp, datetime)


@given(
    handoff_needed=st.booleans(),
    confidence=st.floats(min_value=0, max_value=1),
    # Using none() or text for optional fields
    target_agent=st.one_of(st.none(), st.text(
        min_size=1, max_size=100).filter(lambda x: x.strip())),
    reason=st.one_of(st.none(), st.text(
        min_size=1, max_size=500).filter(lambda x: x.strip()))
)
def test_handoff_evaluation_properties(handoff_needed, target_agent, reason, confidence):
    """Test HandoffEvaluation with various generated properties."""
    # For no handoff, target_agent and reason should be None
    if not handoff_needed:
        target_agent = None
        reason = None

    evaluation = HandoffEvaluation(
        handoff_needed=handoff_needed,
        target_agent=target_agent,
        reason=reason,
        confidence=confidence
    )

    assert evaluation.handoff_needed is handoff_needed
    assert evaluation.target_agent == target_agent
    assert evaluation.reason == reason
    assert evaluation.confidence == confidence


# ---------------------
# Edge Cases
# ---------------------

def test_handoff_same_agent():
    """Test handoff where source and target are the same agent."""
    # This might be a business rule violation, but the model itself might allow it
    same_agent = "agent_same"
    handoff = Handoff(
        from_agent=same_agent,
        to_agent=same_agent,
        ticket_id="ticket123",
        reason="Self-handoff for testing"
    )

    assert handoff.from_agent == handoff.to_agent == same_agent


def test_handoff_timestamps():
    """Test handoff with specific timestamps."""
    # Create handoff with past timestamp
    past_time = datetime.now() - timedelta(days=7)
    handoff = Handoff(
        from_agent="agent_source",
        to_agent="agent_target",
        ticket_id="ticket123",
        reason="Past handoff",
        timestamp=past_time
    )
    assert handoff.timestamp == past_time

    # Create handoff with future timestamp
    future_time = datetime.now() + timedelta(days=1)
    handoff = Handoff(
        from_agent="agent_source",
        to_agent="agent_target",
        ticket_id="ticket123",
        reason="Scheduled handoff",
        timestamp=future_time
    )
    assert handoff.timestamp == future_time


def test_high_low_confidence_evaluations():
    """Test handoff evaluations at confidence extremes."""
    # Very low confidence (but valid)
    low_confidence = HandoffEvaluation(
        handoff_needed=True,
        target_agent="agent_target",
        reason="Low confidence handoff",
        confidence=0.01
    )
    assert low_confidence.handoff_needed is True
    assert low_confidence.confidence == 0.01

    # Maximum confidence
    high_confidence = HandoffEvaluation(
        handoff_needed=True,
        target_agent="agent_expert",
        reason="Very confident handoff",
        confidence=1.0
    )
    assert high_confidence.handoff_needed is True
    assert high_confidence.confidence == 1.0

    # No handoff with high confidence
    confident_no_handoff = HandoffEvaluation(
        handoff_needed=False,
        confidence=0.99
    )
    assert confident_no_handoff.handoff_needed is False
    assert confident_no_handoff.target_agent is None
    assert confident_no_handoff.confidence == 0.99

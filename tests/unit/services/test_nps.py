"""
Tests for the NPSService implementation.

This module tests NPS management, feedback collection, and metrics calculation.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta
import copy

from solana_agent.services.nps import NPSService
from solana_agent.domains import UserFeedback, NPSRating, FeedbackType


# ---------------------
# Fixtures
# ---------------------

@pytest.fixture
def mock_feedback_repository():
    """Return a mock feedback repository."""
    repo = Mock()

    # Mock store_feedback method
    repo.store_feedback = Mock(return_value="feedback-123")

    # Mock get_user_feedback method
    sample_feedback_history = [
        UserFeedback(
            id="feedback-101",
            user_id="user-123",
            type=FeedbackType.NPS,
            nps_rating=NPSRating(score=9),
            timestamp=datetime.now() - timedelta(days=5)
        ),
        UserFeedback(
            id="feedback-102",
            user_id="user-123",
            type=FeedbackType.TEXT,
            text="Great experience with your wallet service!",
            timestamp=datetime.now() - timedelta(days=3)
        )
    ]
    repo.get_user_feedback = Mock(return_value=sample_feedback_history)

    # Mock get_average_nps method
    repo.get_average_nps = Mock(return_value=8.5)

    # Mock get_nps_distribution method
    sample_distribution = {
        6: 1,  # Detractor
        7: 2,  # Passive
        8: 3,  # Passive
        9: 3,  # Promoter
        10: 1  # Promoter
    }
    repo.get_nps_distribution = Mock(return_value=sample_distribution)

    return repo


@pytest.fixture
def nps_service(mock_feedback_repository):
    """Return an NPS service with mocked dependencies."""
    return NPSService(feedback_repository=mock_feedback_repository)


@pytest.fixture
def sample_user_id():
    """Return a sample user ID."""
    return "user-123"


@pytest.fixture
def sample_nps_rating():
    """Return a sample NPS rating."""
    return NPSRating(score=9)


# ---------------------
# Initialization Tests
# ---------------------

def test_nps_service_initialization(mock_feedback_repository):
    """Test that the NPS service initializes properly."""
    service = NPSService(feedback_repository=mock_feedback_repository)

    assert service.feedback_repository == mock_feedback_repository


# ---------------------
# Store NPS Rating Tests
# ---------------------

def test_store_nps_rating(nps_service, sample_user_id):
    """Test storing an NPS rating."""
    # Act
    feedback_id = nps_service.store_nps_rating(
        user_id=sample_user_id,
        score=9,
        ticket_id="ticket-123"
    )

    # Assert
    assert feedback_id == "feedback-123"
    nps_service.feedback_repository.store_feedback.assert_called_once()

    # Check that the feedback object was created correctly
    call_args = nps_service.feedback_repository.store_feedback.call_args[0][0]
    assert call_args.user_id == sample_user_id
    assert call_args.type == FeedbackType.NPS
    assert call_args.ticket_id == "ticket-123"
    assert call_args.nps_rating.score == 9


def test_store_nps_rating_validation_error_too_high(nps_service, sample_user_id):
    """Test validation error when NPS score is too high."""
    # Act & Assert
    with pytest.raises(ValueError) as excinfo:
        nps_service.store_nps_rating(user_id=sample_user_id, score=11)

    assert "NPS score must be between 0 and 10" in str(excinfo.value)
    nps_service.feedback_repository.store_feedback.assert_not_called()


def test_store_nps_rating_validation_error_too_low(nps_service, sample_user_id):
    """Test validation error when NPS score is too low."""
    # Act & Assert
    with pytest.raises(ValueError) as excinfo:
        nps_service.store_nps_rating(user_id=sample_user_id, score=-1)

    assert "NPS score must be between 0 and 10" in str(excinfo.value)
    nps_service.feedback_repository.store_feedback.assert_not_called()


# ---------------------
# Store Feedback Tests
# ---------------------

def test_store_feedback(nps_service, sample_user_id):
    """Test storing textual feedback."""
    # Act
    feedback_id = nps_service.store_feedback(
        user_id=sample_user_id,
        feedback_text="Very intuitive interface!",
        ticket_id="ticket-456"
    )

    # Assert
    assert feedback_id == "feedback-123"
    nps_service.feedback_repository.store_feedback.assert_called_once()

    # Check that the feedback object was created correctly
    call_args = nps_service.feedback_repository.store_feedback.call_args[0][0]
    assert call_args.user_id == sample_user_id
    assert call_args.type == FeedbackType.TEXT
    assert call_args.ticket_id == "ticket-456"
    assert call_args.text == "Very intuitive interface!"


# ---------------------
# Get User Feedback Tests
# ---------------------

def test_get_user_feedback_history(nps_service, sample_user_id):
    """Test retrieving user feedback history."""
    # Act
    feedback_history = nps_service.get_user_feedback_history(sample_user_id)

    # Assert
    assert len(feedback_history) == 2
    assert feedback_history[0].id == "feedback-101"
    assert feedback_history[0].type == FeedbackType.NPS
    assert feedback_history[0].nps_rating.score == 9

    assert feedback_history[1].id == "feedback-102"
    assert feedback_history[1].type == FeedbackType.TEXT
    assert feedback_history[1].text == "Great experience with your wallet service!"

    nps_service.feedback_repository.get_user_feedback.assert_called_once_with(
        sample_user_id)


# ---------------------
# Average NPS Tests
# ---------------------

def test_get_average_nps_default_days(nps_service):
    """Test getting average NPS with default days parameter."""
    # Act
    average = nps_service.get_average_nps()

    # Assert
    assert average == 8.5
    nps_service.feedback_repository.get_average_nps.assert_called_once_with(30)


def test_get_average_nps_custom_days(nps_service):
    """Test getting average NPS with custom days parameter."""
    # Act
    average = nps_service.get_average_nps(days=90)

    # Assert
    assert average == 8.5
    nps_service.feedback_repository.get_average_nps.assert_called_once_with(90)


# ---------------------
# NPS Distribution Tests
# ---------------------

def test_get_nps_distribution_default_days(nps_service):
    """Test getting NPS distribution with default days parameter."""
    # Act
    distribution = nps_service.get_nps_distribution()

    # Assert
    assert distribution == {6: 1, 7: 2, 8: 3, 9: 3, 10: 1}
    nps_service.feedback_repository.get_nps_distribution.assert_called_once_with(
        30)


def test_get_nps_distribution_custom_days(nps_service):
    """Test getting NPS distribution with custom days parameter."""
    # Act
    distribution = nps_service.get_nps_distribution(days=14)

    # Assert
    assert distribution == {6: 1, 7: 2, 8: 3, 9: 3, 10: 1}
    nps_service.feedback_repository.get_nps_distribution.assert_called_once_with(
        14)


# ---------------------
# Calculate NPS Score Tests
# ---------------------

def test_calculate_nps_score(nps_service):
    """Test calculating NPS score with mixed distribution."""
    # Act
    result = nps_service.calculate_nps_score()

    # Assert
    assert result["promoters"] == 4
    assert result["passives"] == 5
    assert result["detractors"] == 1
    assert result["total_responses"] == 10

    # Check percentages
    assert result["promoter_percent"] == 40.0
    assert result["passive_percent"] == 50.0
    assert result["detractor_percent"] == 10.0

    # Check NPS score
    assert result["nps_score"] == 30.0  # 40% promoters - 10% detractors


def test_calculate_nps_score_empty_distribution(nps_service):
    """Test calculating NPS score with no responses."""
    # Arrange
    nps_service.feedback_repository.get_nps_distribution = Mock(
        return_value={})

    # Act
    result = nps_service.calculate_nps_score()

    # Assert
    assert result["promoters"] == 0
    assert result["passives"] == 0
    assert result["detractors"] == 0
    assert result["total_responses"] == 0

    # Check percentages
    assert result["promoter_percent"] == 0
    assert result["passive_percent"] == 0
    assert result["detractor_percent"] == 0

    # Check NPS score
    assert result["nps_score"] == 0


def test_calculate_nps_score_all_promoters(nps_service):
    """Test calculating NPS score with all promoters."""
    # Arrange
    nps_service.feedback_repository.get_nps_distribution = Mock(return_value={
                                                                9: 5, 10: 5})

    # Act
    result = nps_service.calculate_nps_score()

    # Assert
    assert result["promoters"] == 10
    assert result["passives"] == 0
    assert result["detractors"] == 0
    assert result["total_responses"] == 10

    # Check percentages
    assert result["promoter_percent"] == 100.0
    assert result["passive_percent"] == 0.0
    assert result["detractor_percent"] == 0.0

    # Check NPS score
    assert result["nps_score"] == 100.0  # 100% promoters - 0% detractors


def test_calculate_nps_score_all_detractors(nps_service):
    """Test calculating NPS score with all detractors."""
    # Arrange
    nps_service.feedback_repository.get_nps_distribution = Mock(
        return_value={0: 3, 3: 3, 6: 4})

    # Act
    result = nps_service.calculate_nps_score()

    # Assert
    assert result["promoters"] == 0
    assert result["passives"] == 0
    assert result["detractors"] == 10
    assert result["total_responses"] == 10

    # Check percentages
    assert result["promoter_percent"] == 0.0
    assert result["passive_percent"] == 0.0
    assert result["detractor_percent"] == 100.0

    # Check NPS score
    assert result["nps_score"] == -100.0  # 0% promoters - 100% detractors


def test_calculate_nps_score_custom_days(nps_service):
    """Test calculating NPS score with custom days parameter."""
    # Act
    result = nps_service.calculate_nps_score(days=60)

    # Assert
    nps_service.feedback_repository.get_nps_distribution.assert_called_once_with(
        60)
    # Using same distribution as default test
    assert result["nps_score"] == 30.0

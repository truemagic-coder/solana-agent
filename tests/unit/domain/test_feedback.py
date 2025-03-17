"""
Tests for the feedback domain models.

This module tests the UserFeedback, NPSRating, and FeedbackType
domain models using both standard pytest tests and property-based
testing with hypothesis.
"""
import pytest
from datetime import datetime, timedelta
from hypothesis import given, strategies as st
from pydantic import ValidationError

from solana_agent.domains import UserFeedback, NPSRating, FeedbackType

# ---------------------
# Fixtures
# ---------------------


@pytest.fixture
def nps_rating():
    """Return a basic NPS rating for testing."""
    return NPSRating(score=9)


@pytest.fixture
def basic_nps_feedback():
    """Return a basic NPS feedback for testing."""
    return UserFeedback(
        id="feedback123",
        user_id="user456",
        type=FeedbackType.NPS,
        ticket_id="ticket789",
        nps_rating=NPSRating(score=9),
        text="Great experience overall"
    )


@pytest.fixture
def basic_text_feedback():
    """Return a basic text feedback for testing."""
    return UserFeedback(
        id="feedback456",
        user_id="user789",
        type=FeedbackType.TEXT,
        ticket_id="ticket123",
        text="This was very helpful, thank you!"
    )


@pytest.fixture
def basic_rating_feedback():
    """Return a basic rating feedback for testing."""
    return UserFeedback(
        id="feedback789",
        user_id="user123",
        type=FeedbackType.RATING,
        ticket_id="ticket456",
        text="Good service",
        metadata={"rating": 4}
    )


@pytest.fixture
def basic_issue_feedback():
    """Return a basic issue feedback for testing."""
    return UserFeedback(
        id="feedback101",
        user_id="user202",
        type=FeedbackType.ISSUE,
        ticket_id="ticket303",
        text="I encountered a bug when trying to submit my request",
        metadata={"severity": "medium", "browser": "Chrome"}
    )


# Factory fixtures


@pytest.fixture
def feedback_factory():
    """Return a factory function for creating user feedback."""
    def _create_feedback(
        feedback_id="feedback123",
        user_id="user456",
        feedback_type=FeedbackType.TEXT,
        ticket_id="ticket789",
        text="Test feedback",
        nps_rating=None,
        metadata=None,
        timestamp=None
    ):
        if metadata is None:
            metadata = {}
        if timestamp is None:
            timestamp = datetime.now()

        return UserFeedback(
            id=feedback_id,
            user_id=user_id,
            type=feedback_type,
            ticket_id=ticket_id,
            text=text,
            nps_rating=nps_rating,
            metadata=metadata,
            timestamp=timestamp
        )
    return _create_feedback


# ---------------------
# Basic Creation Tests
# ---------------------

def test_nps_rating_creation(nps_rating):
    """Test creating NPSRating with valid parameters."""
    assert nps_rating.score == 9
    assert nps_rating.category == "promoter"


def test_nps_feedback_creation(basic_nps_feedback):
    """Test creating UserFeedback with NPS type."""
    assert basic_nps_feedback.id == "feedback123"
    assert basic_nps_feedback.user_id == "user456"
    assert basic_nps_feedback.type == FeedbackType.NPS
    assert basic_nps_feedback.ticket_id == "ticket789"
    assert basic_nps_feedback.nps_rating.score == 9
    assert basic_nps_feedback.text == "Great experience overall"
    assert isinstance(basic_nps_feedback.timestamp, datetime)


def test_text_feedback_creation(basic_text_feedback):
    """Test creating UserFeedback with TEXT type."""
    assert basic_text_feedback.id == "feedback456"
    assert basic_text_feedback.user_id == "user789"
    assert basic_text_feedback.type == FeedbackType.TEXT
    assert basic_text_feedback.ticket_id == "ticket123"
    assert basic_text_feedback.text == "This was very helpful, thank you!"
    assert basic_text_feedback.nps_rating is None


def test_rating_feedback_creation(basic_rating_feedback):
    """Test creating UserFeedback with RATING type."""
    assert basic_rating_feedback.id == "feedback789"
    assert basic_rating_feedback.user_id == "user123"
    assert basic_rating_feedback.type == FeedbackType.RATING
    assert basic_rating_feedback.ticket_id == "ticket456"
    assert basic_rating_feedback.text == "Good service"
    assert basic_rating_feedback.metadata.get("rating") == 4


def test_issue_feedback_creation(basic_issue_feedback):
    """Test creating UserFeedback with ISSUE type."""
    assert basic_issue_feedback.id == "feedback101"
    assert basic_issue_feedback.user_id == "user202"
    assert basic_issue_feedback.type == FeedbackType.ISSUE
    assert basic_issue_feedback.ticket_id == "ticket303"
    assert "bug" in basic_issue_feedback.text.lower()
    assert basic_issue_feedback.metadata.get("severity") == "medium"
    assert basic_issue_feedback.metadata.get("browser") == "Chrome"


# ---------------------
# Factory Usage Tests
# ---------------------

def test_feedback_factory(feedback_factory):
    """Test creating feedback with the factory."""
    # Create with defaults
    default_feedback = feedback_factory()
    assert default_feedback.id == "feedback123"
    assert default_feedback.user_id == "user456"
    assert default_feedback.type == FeedbackType.TEXT

    # Create NPS feedback
    nps_feedback = feedback_factory(
        feedback_type=FeedbackType.NPS,
        nps_rating=NPSRating(score=8),
        text=None
    )
    assert nps_feedback.type == FeedbackType.NPS
    assert nps_feedback.nps_rating.score == 8
    assert nps_feedback.nps_rating.category == "passive"

    # Create with custom metadata
    custom_feedback = feedback_factory(
        feedback_id="custom123",
        feedback_type=FeedbackType.RATING,
        text="Custom rating",
        metadata={"rating": 5, "feature": "search"}
    )
    assert custom_feedback.id == "custom123"
    assert custom_feedback.metadata.get("rating") == 5
    assert custom_feedback.metadata.get("feature") == "search"


# ---------------------
# Validation Tests
# ---------------------

def test_nps_rating_validation():
    """Test validation of NPSRating."""
    # Score too low
    with pytest.raises(ValidationError):
        NPSRating(score=-1)  # Should be at least 0

    # Score too high
    with pytest.raises(ValidationError):
        NPSRating(score=11)  # Should be at most 10

    # Valid boundary values
    min_rating = NPSRating(score=0)
    assert min_rating.score == 0
    assert min_rating.category == "detractor"

    max_rating = NPSRating(score=10)
    assert max_rating.score == 10
    assert max_rating.category == "promoter"


def test_user_feedback_required_fields():
    """Test validation of required UserFeedback fields."""
    # Missing required fields
    with pytest.raises(ValidationError):
        UserFeedback()

    # Missing user_id
    with pytest.raises(ValidationError):
        UserFeedback(
            id="test_feedback",
            type=FeedbackType.TEXT,
            text="Test content"
        )

    # Missing type
    with pytest.raises(ValidationError):
        UserFeedback(
            id="test_feedback",
            user_id="test_user",
            text="Test content"
        )

    # Valid minimum fields
    feedback = UserFeedback(
        user_id="test_user",
        type=FeedbackType.TEXT
    )
    assert feedback.id == ""  # Default empty string
    # Should have default value
    assert isinstance(feedback.timestamp, datetime)


def test_feedback_type_consistency():
    """Test consistency between feedback type and provided data."""
    # NPS type should have nps_rating
    nps_feedback = UserFeedback(
        user_id="test_user",
        type=FeedbackType.NPS,
        nps_rating=NPSRating(score=8)
    )
    assert nps_feedback.nps_rating is not None
    assert nps_feedback.nps_rating.score == 8


# ---------------------
# Property-Based Tests
# ---------------------

@given(
    score=st.integers(min_value=0, max_value=10)
)
def test_nps_rating_properties(score):
    """Test NPSRating with various scores."""
    rating = NPSRating(score=score)
    assert rating.score == score

    # Test category property
    if score >= 9:
        assert rating.category == "promoter"
    elif score >= 7:
        assert rating.category == "passive"
    else:
        assert rating.category == "detractor"


@given(
    user_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    feedback_type=st.sampled_from([
        FeedbackType.TEXT,
        FeedbackType.RATING,
        FeedbackType.ISSUE
    ]),
    ticket_id=st.one_of(
        st.none(),
        st.text(min_size=1, max_size=100).filter(lambda x: x.strip())
    ),
    text=st.one_of(
        st.none(),
        st.text(min_size=0, max_size=1000)
    )
)
def test_user_feedback_properties(user_id, feedback_type, ticket_id, text):
    """Test UserFeedback with various generated properties."""
    feedback = UserFeedback(
        user_id=user_id,
        type=feedback_type,
        ticket_id=ticket_id,
        text=text
    )

    assert feedback.user_id == user_id
    assert feedback.type == feedback_type
    assert feedback.ticket_id == ticket_id
    assert feedback.text == text
    assert isinstance(feedback.timestamp, datetime)


@given(
    user_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    score=st.integers(min_value=0, max_value=10),
    text=st.one_of(
        st.none(),
        st.text(min_size=0, max_size=1000)
    )
)
def test_nps_user_feedback(user_id, score, text):
    """Test NPS-type UserFeedback with various scores."""
    nps_feedback = UserFeedback(
        user_id=user_id,
        type=FeedbackType.NPS,
        nps_rating=NPSRating(score=score),
        text=text
    )

    assert nps_feedback.user_id == user_id
    assert nps_feedback.type == FeedbackType.NPS
    assert nps_feedback.nps_rating.score == score
    assert nps_feedback.text == text

    # Check proper NPS category
    if score >= 9:
        assert nps_feedback.nps_rating.category == "promoter"
    elif score >= 7:
        assert nps_feedback.nps_rating.category == "passive"
    else:
        assert nps_feedback.nps_rating.category == "detractor"


# ---------------------
# Edge Cases
# ---------------------

def test_nps_rating_categories():
    """Test NPSRating category boundaries."""
    # Test each category boundary
    detractor = NPSRating(score=6)
    assert detractor.category == "detractor"

    passive_lower = NPSRating(score=7)
    assert passive_lower.category == "passive"

    passive_upper = NPSRating(score=8)
    assert passive_upper.category == "passive"

    promoter = NPSRating(score=9)
    assert promoter.category == "promoter"


def test_feedback_with_empty_metadata():
    """Test feedback with empty metadata."""
    feedback = UserFeedback(
        user_id="test_user",
        type=FeedbackType.TEXT,
        metadata={}
    )
    assert feedback.metadata == {}

    # Ensure we can add to the metadata
    feedback.metadata["new_key"] = "new_value"
    assert feedback.metadata["new_key"] == "new_value"


def test_feedback_with_complex_metadata(feedback_factory):
    """Test feedback with complex metadata structures."""
    # Create feedback with nested metadata
    feedback = feedback_factory(
        feedback_type=FeedbackType.ISSUE,
        metadata={
            "device": {
                "os": "iOS",
                "version": "16.2",
                "model": "iPhone 14"
            },
            "app_version": "2.3.1",
            "steps_to_reproduce": [
                "Login to account",
                "Navigate to settings",
                "Click on profile picture"
            ],
            "tags": ["ui", "profile", "bug"]
        }
    )

    # Verify nested structures
    assert feedback.metadata["device"]["os"] == "iOS"
    assert feedback.metadata["device"]["version"] == "16.2"
    assert len(feedback.metadata["steps_to_reproduce"]) == 3
    assert "ui" in feedback.metadata["tags"]


def test_feedback_timestamps():
    """Test feedback timestamp handling."""
    # Create feedback with specific timestamp
    past_time = datetime.now() - timedelta(days=7)
    feedback = UserFeedback(
        user_id="test_user",
        type=FeedbackType.TEXT,
        timestamp=past_time
    )
    assert feedback.timestamp == past_time

    # Create feedback with default timestamp
    feedback_now = UserFeedback(
        user_id="test_user",
        type=FeedbackType.TEXT
    )
    # Should be very close to now
    time_diff = datetime.now() - feedback_now.timestamp
    assert time_diff.total_seconds() < 1  # Less than 1 second difference

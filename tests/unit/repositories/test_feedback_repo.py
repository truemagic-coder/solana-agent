"""
Tests for feedback repository implementations.

This module contains unit tests for MongoFeedbackRepository.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta

from solana_agent.repositories.feedback import MongoFeedbackRepository
from solana_agent.domains import UserFeedback, FeedbackType, NPSRating


@pytest.fixture
def mock_db_adapter():
    """Create a mock database adapter."""
    adapter = Mock()
    adapter.create_collection = Mock()
    adapter.create_index = Mock()
    adapter.insert_one = Mock()
    adapter.find = Mock()
    adapter.aggregate = Mock()
    return adapter


@pytest.fixture
def feedback_repository(mock_db_adapter):
    """Create a repository with mocked database adapter."""
    return MongoFeedbackRepository(mock_db_adapter)


@pytest.fixture
def sample_text_feedback():
    """Create a sample text feedback for testing."""
    return UserFeedback(
        user_id="user123",
        type=FeedbackType.TEXT,
        text="This feature is really helpful!",
        context="feature_suggestion",
        timestamp=datetime.now(),
        metadata={"source": "chat"}
    )


@pytest.fixture
def sample_nps_feedback():
    """Create a sample NPS feedback for testing."""
    return UserFeedback(
        user_id="user456",
        type=FeedbackType.NPS,
        text="Great product overall",
        nps_rating=NPSRating(
            score=9,
            reason="Easy to use and intuitive"
        ),
        timestamp=datetime.now(),
        metadata={"source": "email_survey"}
    )


class TestMongoFeedbackRepository:
    """Tests for the MongoFeedbackRepository implementation."""

    def test_init(self, mock_db_adapter):
        """Test repository initialization."""
        repo = MongoFeedbackRepository(mock_db_adapter)

        # Verify collection is created
        mock_db_adapter.create_collection.assert_called_once_with("feedback")

        # Verify indexes are created
        assert mock_db_adapter.create_index.call_count == 3
        mock_db_adapter.create_index.assert_any_call(
            "feedback", [("user_id", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "feedback", [("timestamp", -1)])
        mock_db_adapter.create_index.assert_any_call("feedback", [("type", 1)])

    def test_store_feedback_with_id(self, feedback_repository, mock_db_adapter, sample_text_feedback):
        """Test storing feedback when ID is already present."""
        # Set an ID
        feedback_id = "existing_id"
        sample_text_feedback.id = feedback_id

        # Store feedback
        result = feedback_repository.store_feedback(sample_text_feedback)

        # Verify result
        assert result == feedback_id

        # Verify DB operation
        mock_db_adapter.insert_one.assert_called_once()
        collection, data = mock_db_adapter.insert_one.call_args[0]
        assert collection == "feedback"
        assert data["id"] == feedback_id
        assert data["user_id"] == "user123"
        assert data["type"] == FeedbackType.TEXT
        assert data["text"] == "This feature is really helpful!"

    def test_store_feedback_without_id(self, feedback_repository, mock_db_adapter, sample_text_feedback):
        """Test storing feedback when ID is not present."""
        # Ensure no ID
        sample_text_feedback.id = None

        # Store feedback
        result = feedback_repository.store_feedback(sample_text_feedback)

        # Verify ID was generated (UUID format)
        assert result is not None
        assert len(result) > 0

        # Verify DB operation
        mock_db_adapter.insert_one.assert_called_once()
        collection, data = mock_db_adapter.insert_one.call_args[0]
        assert collection == "feedback"
        assert data["id"] == result
        assert data["user_id"] == "user123"

    def test_store_nps_feedback(self, feedback_repository, mock_db_adapter, sample_nps_feedback):
        """Test storing NPS feedback."""
        # Store feedback
        result = feedback_repository.store_feedback(sample_nps_feedback)

        # Verify DB operation
        mock_db_adapter.insert_one.assert_called_once()
        collection, data = mock_db_adapter.insert_one.call_args[0]
        assert collection == "feedback"
        assert data["type"] == FeedbackType.NPS
        assert data["nps_rating"]["score"] == 9
        assert data["nps_rating"]["reason"] == "Easy to use and intuitive"

    def test_get_user_feedback(self, feedback_repository, mock_db_adapter):
        """Test retrieving feedback for a user."""
        user_id = "user123"

        # Configure mock to return feedback data
        mock_db_adapter.find.return_value = [
            {
                "id": "fb1",
                "user_id": user_id,
                "type": FeedbackType.TEXT,
                "content": "Feedback 1",
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "fb2",
                "user_id": user_id,
                "type": FeedbackType.NPS,
                "content": "NPS Feedback",
                "nps_rating": {
                    "score": 8,
                    "reason": "Good experience"
                },
                "timestamp": datetime.now().isoformat()
            }
        ]

        # Get the feedback
        feedback_items = feedback_repository.get_user_feedback(user_id)

        # Verify DB query
        mock_db_adapter.find.assert_called_once_with(
            "feedback",
            {"user_id": user_id},
            sort=[("timestamp", -1)]
        )

        # Verify results
        assert len(feedback_items) == 2
        assert feedback_items[0].id == "fb1"
        assert feedback_items[0].type == FeedbackType.TEXT
        assert feedback_items[1].id == "fb2"
        assert feedback_items[1].type == FeedbackType.NPS
        assert feedback_items[1].nps_rating.score == 8
        assert feedback_items[1].nps_rating.reason == "Good experience"

    def test_get_user_feedback_handles_invalid_feedback(self, feedback_repository, mock_db_adapter):
        """Test that get_user_feedback handles invalid feedback gracefully."""
        user_id = "user123"

        # Configure mock to return one valid and one invalid feedback
        mock_db_adapter.find.return_value = [
            {
                "id": "fb1",
                "user_id": user_id,
                "type": FeedbackType.TEXT,
                "content": "Valid Feedback",
                "timestamp": datetime.now().isoformat()
            },
            {
                # Missing required fields
                "id": "fb2",
                "user_id": user_id
            }
        ]

        # Get the feedback - should not raise an exception
        feedback_items = feedback_repository.get_user_feedback(user_id)

        # Verify only valid feedback is returned
        assert len(feedback_items) == 1
        assert feedback_items[0].id == "fb1"

    def test_get_average_nps(self, feedback_repository, mock_db_adapter):
        """Test calculating average NPS score."""
        # Configure mock to return aggregation result
        mock_db_adapter.aggregate.return_value = [
            {
                "_id": None,
                "avg_score": 8.5,
                "count": 10
            }
        ]

        # Get average NPS
        average = feedback_repository.get_average_nps(days=30)

        # Verify result
        assert average == 8.5

        # Verify DB query
        mock_db_adapter.aggregate.assert_called_once()
        collection, pipeline = mock_db_adapter.aggregate.call_args[0]
        assert collection == "feedback"
        assert pipeline[0]["$match"]["type"] == FeedbackType.NPS

        # Verify time period filter is applied
        assert "$gte" in pipeline[0]["$match"]["timestamp"]

        # Verify aggregation groups and calculates average
        assert pipeline[1]["$group"]["avg_score"] == {
            "$avg": "$nps_rating.score"}

    def test_get_average_nps_no_data(self, feedback_repository, mock_db_adapter):
        """Test calculating average NPS with no data."""
        # Configure mock to return empty result
        mock_db_adapter.aggregate.return_value = []

        # Get average NPS
        average = feedback_repository.get_average_nps(days=30)

        # Verify default result when no data
        assert average == 0.0

    def test_get_nps_distribution(self, feedback_repository, mock_db_adapter):
        """Test getting NPS score distribution."""
        # Configure mock to return aggregation result
        mock_db_adapter.aggregate.return_value = [
            {"_id": 8, "count": 3},
            {"_id": 9, "count": 5},
            {"_id": 10, "count": 2}
        ]

        # Get distribution
        distribution = feedback_repository.get_nps_distribution(days=30)

        # Verify results include returned scores
        assert distribution[8] == 3
        assert distribution[9] == 5
        assert distribution[10] == 2

        # Verify missing scores are set to 0
        for score in range(8):
            assert distribution[score] == 0

        # Verify DB query
        mock_db_adapter.aggregate.assert_called_once()
        collection, pipeline = mock_db_adapter.aggregate.call_args[0]
        assert collection == "feedback"
        assert pipeline[0]["$match"]["type"] == FeedbackType.NPS

        # Verify time period filter is applied
        assert "$gte" in pipeline[0]["$match"]["timestamp"]

        # Verify aggregation groups by score
        assert pipeline[1]["$group"]["_id"] == "$nps_rating.score"

    def test_get_nps_distribution_custom_days(self, feedback_repository, mock_db_adapter):
        """Test getting NPS distribution with custom time period."""
        # Configure mock
        mock_db_adapter.aggregate.return_value = []

        # Get distribution for 90 days
        feedback_repository.get_nps_distribution(days=90)

        # Verify time period
        collection, pipeline = mock_db_adapter.aggregate.call_args[0]
        time_filter = pipeline[0]["$match"]["timestamp"]["$gte"]
        expected_cutoff = datetime.now() - timedelta(days=90)

        # Allow small difference due to test execution time
        time_diff = abs((time_filter - expected_cutoff).total_seconds())
        assert time_diff < 1  # Within 1 second

    def test_get_nps_distribution_empty(self, feedback_repository, mock_db_adapter):
        """Test getting NPS distribution with no data."""
        # Configure mock to return empty result
        mock_db_adapter.aggregate.return_value = []

        # Get distribution
        distribution = feedback_repository.get_nps_distribution()

        # Verify all scores are 0
        for score in range(11):
            assert distribution[score] == 0

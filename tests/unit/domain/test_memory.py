"""
Tests for the memory domain models.

This module tests the MemoryInsight, MemorySearchResult, and MemoryInsightsResponse
domain models using both standard pytest tests and property-based testing with hypothesis.
"""
import pytest
from datetime import datetime, timedelta
from hypothesis import given, strategies as st
from pydantic import ValidationError

from solana_agent.domains import MemoryInsight, MemorySearchResult, MemoryInsightsResponse


# ---------------------
# Fixtures
# ---------------------


@pytest.fixture
def basic_memory_insight():
    """Return a basic memory insight for testing."""
    return MemoryInsight(
        content="User prefers email notifications over SMS",
        category="preferences",
        confidence=0.85,
        source="conversation_20230615"
    )


@pytest.fixture
def basic_memory_search_result():
    """Return a basic memory search result for testing."""
    insight = MemoryInsight(
        content="User reported an issue with the checkout process",
        category="issue",
        confidence=0.92,
        source="ticket_123"
    )
    return MemorySearchResult(
        insight=insight,
        relevance_score=0.78
    )


@pytest.fixture
def basic_insights_response():
    """Return a basic memory insights response for testing."""
    return MemoryInsightsResponse(
        insights=[
            {
                "content": "User has a premium subscription",
                "category": "account_status",
                "confidence": 0.95
            },
            {
                "content": "User prefers dark theme in the app",
                "category": "preferences",
                "confidence": 0.82
            }
        ]
    )


# Factory fixtures


@pytest.fixture
def memory_insight_factory():
    """Return a factory function for creating memory insights."""
    def _create_insight(
        content="Test insight content",
        category="general",
        confidence=0.75,
        source="test_source",
        timestamp=None,
        metadata=None
    ):
        if timestamp is None:
            timestamp = datetime.now()
        if metadata is None:
            metadata = {}

        return MemoryInsight(
            content=content,
            category=category,
            confidence=confidence,
            source=source,
            timestamp=timestamp,
            metadata=metadata
        )
    return _create_insight


@pytest.fixture
def memory_search_result_factory(memory_insight_factory):
    """Return a factory function for creating memory search results."""
    def _create_search_result(
        insight=None,
        relevance_score=0.8
    ):
        if insight is None:
            insight = memory_insight_factory()

        return MemorySearchResult(
            insight=insight,
            relevance_score=relevance_score
        )
    return _create_search_result


@pytest.fixture
def insights_response_factory():
    """Return a factory function for creating memory insights responses."""
    def _create_insights_response(
        insights=None
    ):
        if insights is None:
            insights = [
                {
                    "content": "Default insight 1",
                    "category": "general",
                    "confidence": 0.7
                },
                {
                    "content": "Default insight 2",
                    "category": "general",
                    "confidence": 0.8
                }
            ]

        return MemoryInsightsResponse(
            insights=insights
        )
    return _create_insights_response


# ---------------------
# Basic Creation Tests
# ---------------------

def test_memory_insight_creation(basic_memory_insight):
    """Test creating a MemoryInsight with valid parameters."""
    assert basic_memory_insight.content == "User prefers email notifications over SMS"
    assert basic_memory_insight.category == "preferences"
    assert basic_memory_insight.confidence == 0.85
    assert basic_memory_insight.source == "conversation_20230615"
    assert isinstance(basic_memory_insight.timestamp, datetime)
    assert isinstance(basic_memory_insight.metadata, dict)
    assert len(basic_memory_insight.metadata) == 0


def test_memory_search_result_creation(basic_memory_search_result):
    """Test creating a MemorySearchResult with valid parameters."""
    assert basic_memory_search_result.relevance_score == 0.78
    assert isinstance(basic_memory_search_result.insight, MemoryInsight)
    assert basic_memory_search_result.insight.content == "User reported an issue with the checkout process"
    assert basic_memory_search_result.insight.category == "issue"
    assert basic_memory_search_result.insight.confidence == 0.92
    assert basic_memory_search_result.insight.source == "ticket_123"


def test_insights_response_creation(basic_insights_response):
    """Test creating a MemoryInsightsResponse with valid parameters."""
    assert len(basic_insights_response.insights) == 2
    assert basic_insights_response.insights[0]["content"] == "User has a premium subscription"
    assert basic_insights_response.insights[0]["category"] == "account_status"
    assert basic_insights_response.insights[0]["confidence"] == 0.95
    assert basic_insights_response.insights[1]["content"] == "User prefers dark theme in the app"
    assert basic_insights_response.insights[1]["category"] == "preferences"
    assert basic_insights_response.insights[1]["confidence"] == 0.82


# ---------------------
# Factory Usage Tests
# ---------------------

def test_memory_insight_factory(memory_insight_factory):
    """Test creating memory insights with the factory."""
    # Create with defaults
    default_insight = memory_insight_factory()
    assert default_insight.content == "Test insight content"
    assert default_insight.category == "general"
    assert default_insight.confidence == 0.75
    assert default_insight.source == "test_source"

    # Create with custom values
    custom_insight = memory_insight_factory(
        content="Custom insight content",
        category="custom_category",
        confidence=0.95,
        source="custom_source",
        metadata={"importance": "high", "verified": True}
    )
    assert custom_insight.content == "Custom insight content"
    assert custom_insight.category == "custom_category"
    assert custom_insight.confidence == 0.95
    assert custom_insight.source == "custom_source"
    assert custom_insight.metadata["importance"] == "high"
    assert custom_insight.metadata["verified"] is True


def test_memory_search_result_factory(memory_search_result_factory, memory_insight_factory):
    """Test creating memory search results with the factory."""
    # Create with default insight
    default_result = memory_search_result_factory()
    assert default_result.relevance_score == 0.8
    assert isinstance(default_result.insight, MemoryInsight)
    assert default_result.insight.content == "Test insight content"

    # Create with custom insight and score
    custom_insight = memory_insight_factory(
        content="Custom insight for search",
        category="search_test"
    )
    custom_result = memory_search_result_factory(
        insight=custom_insight,
        relevance_score=0.65
    )
    assert custom_result.relevance_score == 0.65
    assert custom_result.insight.content == "Custom insight for search"
    assert custom_result.insight.category == "search_test"


def test_insights_response_factory(insights_response_factory):
    """Test creating memory insights responses with the factory."""
    # Create with defaults
    default_response = insights_response_factory()
    assert len(default_response.insights) == 2
    assert default_response.insights[0]["content"] == "Default insight 1"
    assert default_response.insights[1]["content"] == "Default insight 2"

    # Create with custom insights
    custom_insights = [
        {
            "content": "Custom insight 1",
            "category": "custom",
            "confidence": 0.9,
            "metadata": {"tags": ["important", "verified"]}
        }
    ]
    custom_response = insights_response_factory(insights=custom_insights)
    assert len(custom_response.insights) == 1
    assert custom_response.insights[0]["content"] == "Custom insight 1"
    assert custom_response.insights[0]["category"] == "custom"
    assert custom_response.insights[0]["metadata"]["tags"][0] == "important"


# ---------------------
# Validation Tests
# ---------------------

def test_memory_insight_required_fields():
    """Test validation of required MemoryInsight fields."""
    # Missing required fields
    with pytest.raises(ValidationError):
        MemoryInsight()

    # Missing content
    with pytest.raises(ValidationError):
        MemoryInsight(
            category="test_category",
            confidence=0.8
        )

    # Valid minimum fields
    insight = MemoryInsight(content="Test content")
    assert insight.content == "Test content"
    assert insight.category is None  # Optional field
    assert insight.confidence == 0.0  # Default value
    assert insight.source is None  # Optional field
    assert isinstance(insight.timestamp, datetime)  # Default value
    assert insight.metadata == {}  # Default value


def test_memory_search_result_required_fields():
    """Test validation of required MemorySearchResult fields."""
    # Missing required fields
    with pytest.raises(ValidationError):
        MemorySearchResult()

    # Missing insight
    with pytest.raises(ValidationError):
        MemorySearchResult(
            relevance_score=0.8
        )

    # Missing relevance_score
    with pytest.raises(ValidationError):
        MemorySearchResult(
            insight=MemoryInsight(content="Test insight")
        )

    # Valid minimum fields
    search_result = MemorySearchResult(
        insight=MemoryInsight(content="Test insight"),
        relevance_score=0.8
    )
    assert search_result.insight.content == "Test insight"
    assert search_result.relevance_score == 0.8


def test_confidence_and_relevance_bounds():
    """Test validation of confidence and relevance bounds."""
    # Confidence too low
    with pytest.raises(ValidationError):
        MemoryInsight(
            content="Test content",
            confidence=-0.1  # Should be at least 0
        )

    # Confidence too high
    with pytest.raises(ValidationError):
        MemoryInsight(
            content="Test content",
            confidence=1.1  # Should be at most 1
        )

    # Relevance score too low
    with pytest.raises(ValidationError):
        MemorySearchResult(
            insight=MemoryInsight(content="Test insight"),
            relevance_score=-0.1  # Should be at least 0
        )

    # Relevance score too high
    with pytest.raises(ValidationError):
        MemorySearchResult(
            insight=MemoryInsight(content="Test insight"),
            relevance_score=1.1  # Should be at most 1
        )

    # Valid boundary values
    min_values = MemorySearchResult(
        insight=MemoryInsight(content="Test insight", confidence=0.0),
        relevance_score=0.0
    )
    assert min_values.insight.confidence == 0.0
    assert min_values.relevance_score == 0.0

    max_values = MemorySearchResult(
        insight=MemoryInsight(content="Test insight", confidence=1.0),
        relevance_score=1.0
    )
    assert max_values.insight.confidence == 1.0
    assert max_values.relevance_score == 1.0


# ---------------------
# Property-Based Tests
# ---------------------

@given(
    content=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
    category=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    confidence=st.floats(min_value=0, max_value=1),
    source=st.one_of(st.none(), st.text(min_size=0, max_size=200))
)
def test_memory_insight_properties(content, category, confidence, source):
    """Test MemoryInsight with various generated properties."""
    insight = MemoryInsight(
        content=content,
        category=category,
        confidence=confidence,
        source=source
    )

    assert insight.content == content
    assert insight.category == category
    assert insight.confidence == confidence
    assert insight.source == source
    assert isinstance(insight.timestamp, datetime)
    assert isinstance(insight.metadata, dict)


@given(
    relevance_score=st.floats(min_value=0, max_value=1)
)
def test_memory_search_result_properties(relevance_score):
    """Test MemorySearchResult with various generated properties."""
    # Create insight directly instead of using fixture
    insight = MemoryInsight(
        content="Test insight content",
        category="general",
        confidence=0.75,
        source="test_source"
    )

    search_result = MemorySearchResult(
        insight=insight,
        relevance_score=relevance_score
    )

    assert search_result.insight.content == "Test insight content"
    assert search_result.relevance_score == relevance_score


@given(
    insights=st.lists(
        st.dictionaries(
            keys=st.sampled_from(
                ["content", "category", "confidence", "source"]),
            values=st.one_of(
                st.text(min_size=1, max_size=100),
                st.floats(min_value=0, max_value=1)
            ),
            min_size=1
        ),
        min_size=0,
        max_size=10
    )
)
def test_memory_insights_response_properties(insights):
    """Test MemoryInsightsResponse with various generated properties."""
    # Ensure each insight dictionary has at least a content key
    valid_insights = []
    for insight in insights:
        if "content" not in insight:
            insight["content"] = "Generated content"
        valid_insights.append(insight)

    response = MemoryInsightsResponse(insights=valid_insights)

    assert len(response.insights) == len(valid_insights)
    for i, insight in enumerate(response.insights):
        assert insight["content"] == valid_insights[i]["content"]


# ---------------------
# Edge Cases
# ---------------------

def test_empty_insights_response():
    """Test MemoryInsightsResponse with empty insights list."""
    empty_response = MemoryInsightsResponse(insights=[])
    assert len(empty_response.insights) == 0

    # Should be able to add insights later
    empty_response.insights.append({
        "content": "Added after creation",
        "confidence": 0.9
    })
    assert len(empty_response.insights) == 1
    assert empty_response.insights[0]["content"] == "Added after creation"


def test_memory_insight_with_complex_metadata():
    """Test MemoryInsight with complex metadata structures."""
    complex_metadata = {
        "user_info": {
            "preferences": ["email", "dark_theme"],
            "history": {
                "first_seen": "2023-01-15",
                "interactions": 27
            }
        },
        "context": ["support", "billing_question"],
        "priority_score": 0.85,
        "tags": ["important", "follow_up_required"]
    }

    insight = MemoryInsight(
        content="User has a complex history",
        metadata=complex_metadata
    )

    assert insight.metadata["user_info"]["preferences"][0] == "email"
    assert insight.metadata["user_info"]["history"]["interactions"] == 27
    assert insight.metadata["priority_score"] == 0.85
    assert "important" in insight.metadata["tags"]


def test_memory_insight_timestamp_handling():
    """Test MemoryInsight timestamp handling."""
    # Test with specific timestamp
    past_time = datetime.now() - timedelta(days=30)
    insight = MemoryInsight(
        content="Historical insight",
        timestamp=past_time
    )
    assert insight.timestamp == past_time

    # Test with default (current) timestamp
    now_insight = MemoryInsight(content="Current insight")
    time_diff = datetime.now() - now_insight.timestamp
    assert time_diff.total_seconds() < 1  # Less than 1 second difference


def test_memory_insight_equality():
    """Test equality comparison of memory insights."""
    # Two insights with same content but different metadata
    insight1 = MemoryInsight(
        content="Same content",
        category="category1",
        confidence=0.8
    )

    insight2 = MemoryInsight(
        content="Same content",
        category="category2",
        confidence=0.7
    )

    # In Pydantic v2, equality checks all fields
    assert insight1 != insight2

    # Create a copy with same values
    insight3 = MemoryInsight(
        content=insight1.content,
        category=insight1.category,
        confidence=insight1.confidence,
        source=insight1.source,
        timestamp=insight1.timestamp,
        metadata=insight1.metadata.copy()
    )

    # Should be equal if all fields match
    assert insight1 == insight3


def test_memory_search_result_with_extreme_relevance():
    """Test MemorySearchResult with extreme relevance scores."""
    insight = MemoryInsight(content="Test insight")

    # Perfect relevance
    perfect_match = MemorySearchResult(
        insight=insight,
        relevance_score=1.0
    )
    assert perfect_match.relevance_score == 1.0

    # Zero relevance (still valid)
    zero_match = MemorySearchResult(
        insight=insight,
        relevance_score=0.0
    )
    assert zero_match.relevance_score == 0.0


def test_insights_response_with_rich_content():
    """Test MemoryInsightsResponse with rich insight content."""
    rich_insights = [
        {
            "content": "User has multiple accounts",
            "category": "account_info",
            "confidence": 0.95,
            "relationships": {
                "accounts": ["acc123", "acc456"],
                "primary": "acc123"
            }
        },
        {
            "content": "User frequently asks about premium features",
            "category": "interests",
            "confidence": 0.82,
            "feature_mentions": ["export", "analytics", "automation"],
            "sentiment": "positive",
            "frequency": "high"
        }
    ]

    response = MemoryInsightsResponse(insights=rich_insights)
    assert len(response.insights) == 2
    assert response.insights[0]["relationships"]["accounts"][1] == "acc456"
    assert response.insights[1]["feature_mentions"][0] == "export"

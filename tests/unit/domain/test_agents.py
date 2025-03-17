"""
Tests for the agents domain models.

This module tests the AIAgent, HumanAgent, and OrganizationMission
domain models using both standard pytest tests and property-based
testing with hypothesis.
"""
import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from solana_agent.domains import AIAgent, HumanAgent, OrganizationMission

# ---------------------
# Fixtures
# ---------------------


@pytest.fixture
def basic_ai_agent():
    """Return a basic AI agent for testing."""
    return AIAgent(
        name="test_agent",
        instructions="This is a test agent for unit tests",
        specialization="testing",
        model="gpt-4o-mini"
    )


@pytest.fixture
def basic_human_agent():
    """Return a basic human agent for testing."""
    return HumanAgent(
        id="human123",
        name="Test User",
        specializations=["customer_support", "testing"],
        availability=True
    )


@pytest.fixture
def basic_organization_mission():
    """Return a basic organization mission for testing."""
    return OrganizationMission(
        mission_statement="To provide excellent automated assistance",
        values=[
            {"name": "quality", "description": "High quality responses"},
            {"name": "efficiency", "description": "Fast and efficient service"}
        ],
        goals=["90% customer satisfaction", "50% reduction in response time"],
        guidance="Always be helpful and accurate"
    )

# Specialized fixtures


@pytest.fixture
def development_ai_agent():
    """Return a development-specialized AI agent."""
    return AIAgent(
        name="dev_assistant",
        instructions="Help with software development tasks",
        specialization="software_development",
        model="gpt-4o"
    )


@pytest.fixture
def support_ai_agent():
    """Return a support-specialized AI agent."""
    return AIAgent(
        name="support_assistant",
        instructions="Help with customer support inquiries",
        specialization="customer_support",
        model="gpt-4o-mini"
    )


@pytest.fixture
def available_human_agent():
    """Return an available human agent."""
    return HumanAgent(
        id="human_available",
        name="Available Agent",
        specializations=["general", "management"],
        availability=True
    )


@pytest.fixture
def unavailable_human_agent():
    """Return an unavailable human agent."""
    return HumanAgent(
        id="human_unavailable",
        name="Unavailable Agent",
        specializations=["development", "design"],
        availability=False
    )

# Factory fixtures (for creating customized instances)


@pytest.fixture
def ai_agent_factory():
    """Return a factory function for creating AI agents."""
    def _create_ai_agent(name="test_agent", instructions="Test instructions",
                         specialization="general", model="gpt-4o-mini"):
        return AIAgent(
            name=name,
            instructions=instructions,
            specialization=specialization,
            model=model
        )
    return _create_ai_agent


@pytest.fixture
def human_agent_factory():
    """Return a factory function for creating human agents."""
    def _create_human_agent(agent_id="human1", name="Test Human",
                            specializations=None, availability=True):
        if specializations is None:
            specializations = ["general"]
        return HumanAgent(
            id=agent_id,
            name=name,
            specializations=specializations,
            availability=availability
        )
    return _create_human_agent


@pytest.fixture
def organization_mission_factory():
    """Return a factory function for creating organization missions."""
    def _create_organization_mission(
        mission_statement="Test mission",
        values=None,
        goals=None,
        guidance=None
    ):
        if values is None:
            values = [{"name": "quality", "description": "High quality"}]
        if goals is None:
            goals = ["Improve service quality"]

        return OrganizationMission(
            mission_statement=mission_statement,
            values=values,
            goals=goals,
            guidance=guidance
        )
    return _create_organization_mission


# ---------------------
# Basic Creation Tests
# ---------------------

def test_ai_agent_creation(basic_ai_agent):
    """Test creating an AIAgent with valid parameters."""
    assert basic_ai_agent.name == "test_agent"
    assert basic_ai_agent.instructions == "This is a test agent for unit tests"
    assert basic_ai_agent.specialization == "testing"
    assert basic_ai_agent.model == "gpt-4o-mini"


def test_human_agent_creation(basic_human_agent):
    """Test creating a HumanAgent with valid parameters."""
    assert basic_human_agent.id == "human123"
    assert basic_human_agent.name == "Test User"
    assert "customer_support" in basic_human_agent.specializations
    assert "testing" in basic_human_agent.specializations
    assert basic_human_agent.availability is True


def test_organization_mission_creation(basic_organization_mission):
    """Test creating an OrganizationMission with valid parameters."""
    assert basic_organization_mission.mission_statement == "To provide excellent automated assistance"
    assert len(basic_organization_mission.values) == 2
    assert basic_organization_mission.values[0]["name"] == "quality"
    assert len(basic_organization_mission.goals) == 2


# ---------------------
# Factory Usage Tests
# ---------------------

def test_ai_agent_factory(ai_agent_factory):
    """Test creating AI agents with the factory."""
    # Create with defaults
    default_agent = ai_agent_factory()
    assert default_agent.name == "test_agent"
    assert default_agent.model == "gpt-4o-mini"

    # Create with custom values
    custom_agent = ai_agent_factory(
        name="custom_agent",
        instructions="Custom instructions",
        specialization="custom",
        model="claude-3-5-sonnet"
    )
    assert custom_agent.name == "custom_agent"
    assert custom_agent.model == "claude-3-5-sonnet"

# ---------------------
# Validation Tests
# ---------------------


def test_ai_agent_required_fields():
    """Test validation of required AIAgent fields."""
    # Missing required fields
    with pytest.raises(ValidationError):
        AIAgent()

    # Missing specialization
    with pytest.raises(ValidationError):
        AIAgent(name="test_agent", instructions="Test instructions")

    # Valid minimum fields
    agent = AIAgent(
        name="test_agent",
        instructions="Test instructions",
        specialization="general"
    )
    assert agent.model == "gpt-4o-mini"  # Default value


def test_human_agent_required_fields():
    """Test validation of required HumanAgent fields."""
    # Missing required fields
    with pytest.raises(ValidationError):
        HumanAgent()

    # Missing name
    with pytest.raises(ValidationError):
        HumanAgent(id="human1", specializations=["general"])

    # Valid minimum fields
    agent = HumanAgent(
        id="human1",
        name="Test Human",
        specializations=["general"]
    )
    assert agent.availability is True  # Default value


def test_organization_mission_validation():
    """Test validation rules for OrganizationMission."""
    # Values must be a list of dictionaries with name and description
    with pytest.raises(ValidationError):
        OrganizationMission(
            mission_statement="Test mission",
            values=["invalid value format"]
        )

    # Goals must be a list of strings
    with pytest.raises(ValidationError):
        OrganizationMission(
            mission_statement="Test mission",
            goals=[123, 456]  # Not strings
        )


# ---------------------
# Property-Based Tests
# ---------------------

@given(
    name=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    instructions=st.text(min_size=10, max_size=10000).filter(
        lambda x: x.strip()),
    specialization=st.text(min_size=1, max_size=50).filter(
        lambda x: x.strip()),
    model=st.sampled_from(
        ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "claude-3-opus"])
)
def test_ai_agent_properties(name, instructions, specialization, model):
    """Test AIAgent with various generated properties."""
    agent = AIAgent(
        name=name,
        instructions=instructions,
        specialization=specialization,
        model=model
    )

    assert agent.name == name
    assert agent.instructions == instructions
    assert agent.specialization == specialization
    assert agent.model == model


@given(
    agent_id=st.text(min_size=1, max_size=36).filter(lambda x: x.strip()),
    name=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    specializations=st.lists(
        st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        min_size=1,
        max_size=10
    ),
    availability=st.booleans()
)
def test_human_agent_properties(agent_id, name, specializations, availability):
    """Test HumanAgent with various generated properties."""
    agent = HumanAgent(
        id=agent_id,
        name=name,
        specializations=specializations,
        availability=availability
    )

    assert agent.id == agent_id
    assert agent.name == name
    assert all(spec in agent.specializations for spec in specializations)
    assert agent.availability is availability


@given(
    mission_statement=st.text(
        min_size=10, max_size=1000).filter(lambda x: x.strip()),
    values=st.lists(
        st.fixed_dictionaries({
            "name": st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
            "description": st.text(min_size=5, max_size=200).filter(lambda x: x.strip())
        }),
        min_size=0,
        max_size=10
    ),
    goals=st.lists(
        st.text(min_size=5, max_size=200).filter(lambda x: x.strip()),
        min_size=0,
        max_size=10
    )
)
def test_organization_mission_properties(mission_statement, values, goals):
    """Test OrganizationMission with various generated properties."""
    mission = OrganizationMission(
        mission_statement=mission_statement,
        values=values,
        goals=goals
    )

    assert mission.mission_statement == mission_statement
    assert len(mission.values) == len(values)
    assert len(mission.goals) == len(goals)


# ---------------------
# Edge Cases
# ---------------------

def test_ai_agent_edge_cases():
    """Test AIAgent with edge case inputs."""
    # Very long name
    very_long_name = "a" * 1000
    agent = AIAgent(
        name=very_long_name,
        instructions="Test instructions",
        specialization="general"
    )
    assert agent.name == very_long_name

    # Empty instructions (should fail validation)
    with pytest.raises(ValidationError):
        AIAgent(
            name="test_agent",
            instructions="",  # Empty instructions
            specialization="general"
        )


def test_human_agent_empty_specializations():
    """Test HumanAgent with empty specializations list."""
    # Empty specializations list (should fail validation)
    with pytest.raises(ValidationError):
        HumanAgent(
            id="human1",
            name="Test Human",
            specializations=[]  # Empty list
        )


def test_organization_mission_empty_fields():
    """Test OrganizationMission with empty fields."""
    # Empty mission statement (should fail validation)
    with pytest.raises(ValidationError):
        OrganizationMission(mission_statement="")

    # Empty values and goals are valid
    mission = OrganizationMission(
        mission_statement="Test mission",
        values=[],
        goals=[]
    )
    assert len(mission.values) == 0
    assert len(mission.goals) == 0

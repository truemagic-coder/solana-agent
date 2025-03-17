"""
Test fixtures for agent domain models.

This module contains fixtures for creating test instances of
AIAgent, HumanAgent, and OrganizationMission domain models.
"""
import pytest
from solana_agent.domains import AIAgent, HumanAgent, OrganizationMission

# Basic fixtures


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

"""
Tests for domain models in agent.py.

These tests verify the behavior of the BusinessMission and AIAgent models,
including all field validations and constraints.
"""

import pytest
from pydantic import ValidationError
from solana_agent.domains.agent import BusinessMission, AIAgent


class TestBusinessMission:
    """Test suite for the BusinessMission model."""

    def test_valid_business_mission_full(self):
        """Test creating a valid BusinessMission with all fields."""
        mission = BusinessMission(
            mission="To revolutionize the blockchain experience",
            values=[
                {"name": "Innovation", "description": "Pushing boundaries"},
                {"name": "Trust", "description": "Building reliable systems"},
            ],
            goals=["Achieve 1M users", "Launch 5 new features"],
            voice="Professional yet approachable",
        )

        assert mission.mission == "To revolutionize the blockchain experience"
        assert len(mission.values) == 2
        assert mission.values[0]["name"] == "Innovation"
        assert len(mission.goals) == 2
        assert mission.voice == "Professional yet approachable"

    def test_valid_business_mission_minimal(self):
        """Test creating a valid BusinessMission with only required fields."""
        mission = BusinessMission(
            mission="To revolutionize the blockchain experience", voice="Professional"
        )

        assert mission.mission == "To revolutionize the blockchain experience"
        assert mission.values == []
        assert mission.goals == []
        assert mission.voice == "Professional"

    def test_invalid_empty_mission(self):
        """Test validation error when mission is empty."""
        with pytest.raises(ValidationError) as excinfo:
            BusinessMission(mission="", voice="Professional")

        assert "Mission cannot be empty" in str(excinfo.value)

    def test_invalid_whitespace_mission(self):
        """Test validation error when mission is only whitespace."""
        with pytest.raises(ValidationError) as excinfo:
            BusinessMission(mission="   ", voice="Professional")

        assert "Mission cannot be empty" in str(excinfo.value)

    def test_invalid_empty_voice(self):
        """Test validation error when voice is empty."""
        with pytest.raises(ValidationError) as excinfo:
            BusinessMission(mission="Valid mission", voice="")

        assert "Voice cannot be empty" in str(excinfo.value)

    def test_invalid_whitespace_voice(self):
        """Test validation error when voice is only whitespace."""
        with pytest.raises(ValidationError) as excinfo:
            BusinessMission(mission="Valid mission", voice="   ")

        assert "Voice cannot be empty" in str(excinfo.value)

    def test_missing_voice(self):
        """Test that voice can be None."""
        mission = BusinessMission(mission="Valid mission")
        assert mission.voice is None

    def test_invalid_values_missing_name(self):
        """Test validation error when a value is missing a name."""
        with pytest.raises(ValidationError) as excinfo:
            BusinessMission(
                mission="Valid mission",
                voice="Professional",
                values=[{"description": "No name provided"}],
            )

        assert "Each value must have a name and description" in str(excinfo.value)

    def test_invalid_values_missing_description(self):
        """Test validation error when a value is missing a description."""
        with pytest.raises(ValidationError) as excinfo:
            BusinessMission(
                mission="Valid mission",
                voice="Professional",
                values=[{"name": "No description"}],
            )

        assert "Each value must have a name and description" in str(excinfo.value)


class TestAIAgent:
    """Test suite for the AIAgent model."""

    def test_valid_ai_agent(self):
        """Test creating a valid AIAgent."""
        agent = AIAgent(
            name="financial_expert",
            instructions="Provide financial advice and answer questions about investments.",
            specialization="Finance",
        )

        assert agent.name == "financial_expert"
        assert (
            agent.instructions
            == "Provide financial advice and answer questions about investments."
        )
        assert agent.specialization == "Finance"

    def test_invalid_empty_name(self):
        """Test validation error when name is empty."""
        with pytest.raises(ValidationError) as excinfo:
            AIAgent(
                name="",
                instructions="Valid instructions that are definitely long enough.",
                specialization="Finance",
            )

        assert "Field cannot be empty" in str(excinfo.value)

    def test_invalid_whitespace_name(self):
        """Test validation error when name is only whitespace."""
        with pytest.raises(ValidationError) as excinfo:
            AIAgent(
                name="   ",
                instructions="Valid instructions that are definitely long enough.",
                specialization="Finance",
            )

        assert "Field cannot be empty" in str(excinfo.value)

    def test_invalid_empty_instructions(self):
        """Test validation error when instructions are empty."""
        with pytest.raises(ValidationError) as excinfo:
            AIAgent(name="financial_expert", instructions="", specialization="Finance")

        assert "Instructions cannot be empty" in str(excinfo.value)

    def test_invalid_whitespace_instructions(self):
        """Test validation error when instructions are only whitespace."""
        with pytest.raises(ValidationError) as excinfo:
            AIAgent(
                name="financial_expert", instructions="   ", specialization="Finance"
            )

        assert "Instructions cannot be empty" in str(excinfo.value)

    def test_invalid_short_instructions(self):
        """Test validation error when instructions are too short."""
        with pytest.raises(ValidationError) as excinfo:
            AIAgent(
                name="financial_expert",
                instructions="Too short",
                specialization="Finance",
            )

        assert "Instructions must be at least 10 characters" in str(excinfo.value)

    def test_invalid_empty_specialization(self):
        """Test validation error when specialization is empty."""
        with pytest.raises(ValidationError) as excinfo:
            AIAgent(
                name="financial_expert",
                instructions="Valid instructions that are definitely long enough.",
                specialization="",
            )

        assert "Field cannot be empty" in str(excinfo.value)

    def test_invalid_whitespace_specialization(self):
        """Test validation error when specialization is only whitespace."""
        with pytest.raises(ValidationError) as excinfo:
            AIAgent(
                name="financial_expert",
                instructions="Valid instructions that are definitely long enough.",
                specialization="   ",
            )

        assert "Field cannot be empty" in str(excinfo.value)

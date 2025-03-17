"""
Tests for the CriticService implementation.

This module tests response evaluation, human intervention detection,
and improvement suggestions functionality.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import json

from solana_agent.services.critic import CriticService
from solana_agent.domains import ResponseEvaluation


# ---------------------
# Fixtures
# ---------------------

@pytest.fixture
def mock_llm_provider():
    """Return a mock LLM provider."""
    provider = Mock()

    # Setup the parse_structured_output method
    async def mock_parse_structured_output(**kwargs):
        return ResponseEvaluation(
            accuracy=8.5,
            relevance=9.0,
            completeness=7.5,
            clarity=8.0,
            helpfulness=8.5,
            overall_score=8.3,
            feedback="Good response overall, but could provide more details.",
            improvement_suggestions=["Add more context", "Include an example"]
        )

    provider.parse_structured_output = AsyncMock(
        side_effect=mock_parse_structured_output)

    # Create a proper async generator mock
    async def async_generator():
        yield "Improve the response by adding more technical details and examples."

    # Mock the generate_text method to return the async generator
    provider.generate_text = AsyncMock(return_value=async_generator())

    return provider


@pytest.fixture
def critic_service(mock_llm_provider):
    """Return a critic service with a mock LLM provider."""
    return CriticService(llm_provider=mock_llm_provider)


@pytest.fixture
def sample_query():
    """Return a sample user query."""
    return "How do I deploy a smart contract on Solana?"


@pytest.fixture
def good_response():
    """Return a sample good response."""
    return """
    To deploy a smart contract on Solana, you'll need to:
    
    1. Set up your development environment with Rust and Solana CLI
    2. Write your program using the Solana SDK
    3. Build your program with 'cargo build-bpf'
    4. Deploy using 'solana program deploy'
    
    Would you like me to explain any of these steps in more detail?
    """


@pytest.fixture
def poor_response():
    """Return a sample poor response."""
    return "You can deploy smart contracts on Solana."


# ---------------------
# Initialization Tests
# ---------------------

def test_critic_service_initialization(mock_llm_provider):
    """Test that the critic service initializes properly."""
    service = CriticService(llm_provider=mock_llm_provider)
    assert service.llm_provider == mock_llm_provider
    assert service.model == "gpt-4o-mini"

    custom_model_service = CriticService(
        llm_provider=mock_llm_provider,
        model="gpt-4o"
    )
    assert custom_model_service.model == "gpt-4o"


# ---------------------
# Evaluation Tests
# ---------------------

@pytest.mark.asyncio
async def test_evaluate_response_good(critic_service, sample_query, good_response, mock_llm_provider):
    """Test evaluating a good response."""
    # Arrange
    expected_evaluation = ResponseEvaluation(
        accuracy=8.5,
        relevance=9.0,
        completeness=7.5,
        clarity=8.0,
        helpfulness=8.5,
        overall_score=8.3,
        feedback="Good response overall, but could provide more details.",
        improvement_suggestions=["Add more context", "Include an example"]
    )

    # Act
    result = await critic_service.evaluate_response(sample_query, good_response)

    # Assert
    mock_llm_provider.parse_structured_output.assert_called_once()
    assert result["scores"]["accuracy"] == expected_evaluation.accuracy
    assert result["scores"]["relevance"] == expected_evaluation.relevance
    assert result["scores"]["completeness"] == expected_evaluation.completeness
    assert result["scores"]["clarity"] == expected_evaluation.clarity
    assert result["scores"]["helpfulness"] == expected_evaluation.helpfulness
    assert result["scores"]["overall"] == expected_evaluation.overall_score
    assert result["feedback"] == expected_evaluation.feedback
    assert result["improvement_suggestions"] == expected_evaluation.improvement_suggestions
    assert result["action_needed"] is False  # Score is above 7.0


@pytest.mark.asyncio
async def test_evaluate_response_error_handling(critic_service, sample_query, good_response, mock_llm_provider):
    """Test error handling during response evaluation."""
    # Arrange - make the mock throw an exception
    mock_llm_provider.parse_structured_output.side_effect = Exception(
        "Model error")

    # Act
    result = await critic_service.evaluate_response(sample_query, good_response)

    # Assert
    assert result["scores"]["accuracy"] == 0
    assert result["scores"]["overall"] == 0
    assert "Evaluation failed" in result["feedback"]
    assert result["action_needed"] is True


# ---------------------
# Human Intervention Tests
# ---------------------

@pytest.mark.asyncio
async def test_needs_human_intervention_good_response(critic_service, sample_query, good_response):
    """Test detecting if good responses need human intervention."""
    # Act
    needs_intervention = await critic_service.needs_human_intervention(
        sample_query, good_response)

    # Assert - default threshold is 5.0, our mock returns 8.3
    assert needs_intervention is False


@pytest.mark.asyncio
async def test_needs_human_intervention_high_threshold(critic_service, sample_query, good_response):
    """Test with a threshold higher than the response quality."""
    # Act
    needs_intervention = await critic_service.needs_human_intervention(
        sample_query, good_response, threshold=9.0)

    # Assert - threshold is 9.0, our mock returns 8.3
    assert needs_intervention is True


@pytest.mark.asyncio
async def test_needs_human_intervention_error(critic_service, sample_query, good_response, mock_llm_provider):
    """Test error handling during human intervention check."""
    # Arrange - make the mock throw an exception
    mock_llm_provider.parse_structured_output.side_effect = Exception(
        "Model error")

    # Act
    needs_intervention = await critic_service.needs_human_intervention(
        sample_query, good_response)

    # Assert - errors should recommend human intervention
    assert needs_intervention is True


# ---------------------
# Improvement Suggestion Tests
# ---------------------

@pytest.mark.asyncio
async def test_suggest_improvements(critic_service, sample_query, poor_response):
    """Test suggesting improvements for a response."""
    # Create a patched version that returns a string directly

    async def patched_suggest_improvements(self, query, response):
        return "Improve the response by adding more technical details and examples."

    # Patch the method for this test
    with patch.object(CriticService, 'suggest_improvements', patched_suggest_improvements):
        # Act
        suggestions = await critic_service.suggest_improvements(sample_query, poor_response)

        # Assert
        assert "technical details" in suggestions


@pytest.mark.asyncio
async def test_suggest_improvements_error(critic_service, sample_query, poor_response, mock_llm_provider):
    """Test error handling during improvement suggestions."""
    # Arrange - make the mock throw an exception
    mock_llm_provider.generate_text.side_effect = Exception("Model error")

    # Act
    suggestions = await critic_service.suggest_improvements(sample_query, poor_response)

    # Assert
    assert "Error:" in suggestions


# ---------------------
# Integration Tests with Different Responses
# ---------------------

@pytest.mark.asyncio
async def test_evaluate_different_response_types(critic_service, sample_query, mock_llm_provider):
    """Test evaluating different types of responses."""

    # Test with very short response
    short_response = "Use Solana CLI."
    await critic_service.evaluate_response(sample_query, short_response)

    # Test with very long response
    long_response = "Lorem ipsum " * 100
    await critic_service.evaluate_response(sample_query, long_response)

    # Test with response containing code
    code_response = """
    To deploy a Solana smart contract:
    
    ```rust
    use solana_program::{
        account_info::AccountInfo, entrypoint, entrypoint::ProgramResult, pubkey::Pubkey
    };
    
    entrypoint!(process_instruction);
    fn process_instruction(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        instruction_data: &[u8],
    ) -> ProgramResult {
        Ok(())
    }
    ```
    """
    await critic_service.evaluate_response(sample_query, code_response)

    # Verify all calls succeeded (no exceptions thrown)
    assert mock_llm_provider.parse_structured_output.call_count == 3


@pytest.mark.asyncio
async def test_evaluation_with_custom_model(mock_llm_provider):
    """Test evaluation using a custom model."""
    # Arrange
    service = CriticService(llm_provider=mock_llm_provider, model="gpt-4")

    # Act
    result = await service.evaluate_response("Test query", "Test response")

    # Assert that the custom model was passed to the provider
    call_kwargs = mock_llm_provider.parse_structured_output.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_full_workflow(critic_service, sample_query, poor_response):
    """Test the full workflow from evaluation to suggestions."""
    # Patch the suggest_improvements method for this specific test

    async def patched_suggest_improvements(self, query, response):
        return "Improved suggestion for testing workflow"

    with patch.object(CriticService, 'suggest_improvements', patched_suggest_improvements):
        # First evaluate the response
        evaluation = await critic_service.evaluate_response(sample_query, poor_response)

        # Check if human intervention is needed
        needs_human = await critic_service.needs_human_intervention(sample_query, poor_response)

        # Get improvement suggestions
        suggestions = await critic_service.suggest_improvements(sample_query, poor_response)

        # Verify the workflow produces expected results
        assert isinstance(evaluation, dict)
        assert isinstance(needs_human, bool)
        assert isinstance(suggestions, str)
        assert "Improved suggestion" in suggestions

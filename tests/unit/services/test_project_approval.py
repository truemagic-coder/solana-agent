"""
Tests for the ProjectApprovalService implementation.

This module tests project submission, review, and approval processes.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import uuid

from solana_agent.services.project_approval import ProjectApprovalService
from solana_agent.domains import (
    Project,
    ProjectStatus,
    ApprovalCriteria,
    ProjectApprovalResult,
    CriteriaEvaluation
)


# ---------------------
# Fixtures
# ---------------------

@pytest.fixture
def sample_criteria():
    """Return sample approval criteria."""
    return [
        ApprovalCriteria(
            id="crit-1",
            name="Technical Feasibility",
            description="Evaluates if the project is technically possible with available resources",
            weight=0.3
        ),
        ApprovalCriteria(
            id="crit-2",
            name="Community Impact",
            description="Evaluates the project's potential impact on the Solana ecosystem",
            weight=0.4
        ),
        ApprovalCriteria(
            id="crit-3",
            name="Resource Efficiency",
            description="Evaluates if the project uses resources efficiently",
            weight=0.3
        )
    ]


@pytest.fixture
def mock_project_repository():
    """Return a mock project repository."""
    repo = Mock()

    # Sample project for testing
    sample_project = Project(
        id="proj-123",
        name="Solana NFT Marketplace",
        description="A decentralized marketplace for NFTs on Solana",
        goals="Create an accessible platform for NFT trading with low fees",
        scope="MVP with basic trading functionality",
        resources_required="3 developers, 2 designers, 1 PM for 3 months",
        timeline="3 months",
        status=ProjectStatus.PENDING_REVIEW,
        submitted_at=datetime.now() - timedelta(days=2),
        submitter_id="user-456",
        reviews=[]
    )

    # Mock methods
    repo.create = Mock(return_value="proj-123")
    repo.get_by_id = Mock(return_value=sample_project)
    repo.update = Mock(return_value=True)
    repo.list_all = Mock(return_value=[sample_project])

    return repo


@pytest.fixture
def mock_llm_provider():
    """Return a mock LLM provider."""
    provider = Mock()

    # Sample AI review result
    sample_result = ProjectApprovalResult(
        criteria_evaluations=[
            CriteriaEvaluation(
                score=8.5, comments="Technically sound and feasible"),
            CriteriaEvaluation(
                score=9.0, comments="Strong potential community impact"),
            CriteriaEvaluation(
                score=7.5, comments="Reasonable resource allocation")
        ],
        overall_score=8.5,
        assessment="This is a promising project that aligns well with ecosystem goals"
    )

    # Mock parse_structured_output method
    provider.parse_structured_output = AsyncMock(return_value=sample_result)

    return provider


@pytest.fixture
def approval_service(mock_project_repository, mock_llm_provider, sample_criteria):
    """Return a project approval service with mocked dependencies."""
    return ProjectApprovalService(
        project_repository=mock_project_repository,
        llm_provider=mock_llm_provider,
        criteria=sample_criteria,
        auto_approve_threshold=8.0,
        require_human_approval=True
    )


@pytest.fixture
def sample_project():
    """Return a sample project for testing."""
    return Project(
        id="",  # Empty ID as it will be assigned by repository
        name="Solana Mobile Wallet",
        description="A secure mobile wallet for Solana assets",
        goals="Create an easy-to-use mobile wallet with biometric security",
        scope="iOS and Android apps with basic wallet functionality",
        resources_required="2 mobile devs, 1 designer for 2 months",
        timeline="2 months",
        submitter_id="user-789",
        reviews=[]
    )


# ---------------------
# Initialization Tests
# ---------------------

def test_service_initialization(mock_project_repository, mock_llm_provider, sample_criteria):
    """Test that the project approval service initializes properly."""
    # Act
    service = ProjectApprovalService(
        project_repository=mock_project_repository,
        llm_provider=mock_llm_provider,
        criteria=sample_criteria,
        auto_approve_threshold=7.5,
        require_human_approval=False
    )

    # Assert
    assert service.repository == mock_project_repository
    assert service.llm_provider == mock_llm_provider
    assert service.criteria == sample_criteria
    assert service.auto_approve_threshold == 7.5
    assert service.require_human_approval is False


def test_service_initialization_defaults(mock_project_repository):
    """Test service initialization with default values."""
    # Act
    service = ProjectApprovalService(
        project_repository=mock_project_repository)

    # Assert
    assert service.repository == mock_project_repository
    assert service.llm_provider is None
    assert service.criteria == []
    assert service.auto_approve_threshold == 8.0
    assert service.require_human_approval is True


# ---------------------
# Submit Project Tests
# ---------------------

@pytest.mark.asyncio
async def test_submit_project(approval_service, sample_project):
    """Test submitting a project for approval."""
    # Act
    project_id = await approval_service.submit_project(sample_project)

    # Assert
    assert project_id == "proj-123"
    assert sample_project.status == ProjectStatus.PENDING_REVIEW
    assert sample_project.submitted_at is not None
    approval_service.repository.create.assert_called_once_with(sample_project)


# ---------------------
# Get Project Tests
# ---------------------

@pytest.mark.asyncio
async def test_get_project_existing(approval_service):
    """Test getting an existing project."""
    # Act
    project = await approval_service.get_project("proj-123")

    # Assert
    assert project is not None
    assert project.id == "proj-123"
    assert project.name == "Solana NFT Marketplace"
    approval_service.repository.get_by_id.assert_called_once_with("proj-123")


@pytest.mark.asyncio
async def test_get_project_nonexistent(approval_service):
    """Test getting a nonexistent project."""
    # Arrange
    approval_service.repository.get_by_id = Mock(return_value=None)

    # Act
    project = await approval_service.get_project("nonexistent-id")

    # Assert
    assert project is None
    approval_service.repository.get_by_id.assert_called_once_with(
        "nonexistent-id")


# ---------------------
# Review Project Tests
# ---------------------

@pytest.mark.asyncio
async def test_review_project_human(approval_service):
    """Test reviewing a project as a human reviewer."""
    # Act
    overall_score, criteria_results = await approval_service.review_project(
        project_id="proj-123",
        reviewer_id="reviewer-001",
        is_human_reviewer=True
    )

    # Assert
    assert overall_score == 0.0
    assert len(criteria_results) == 3
    for i, result in enumerate(criteria_results):
        assert result["criterion_id"] == approval_service.criteria[i].id
        assert result["name"] == approval_service.criteria[i].name
        assert result["score"] is None  # To be filled by human
        assert result["comments"] == ""  # To be filled by human


@pytest.mark.asyncio
async def test_review_project_ai(approval_service):
    """Test AI-assisted project review."""
    # Act
    overall_score, criteria_results = await approval_service.review_project(
        project_id="proj-123",
        reviewer_id="ai-system",
        is_human_reviewer=False
    )

    # Assert
    assert overall_score == 8.5
    assert len(criteria_results) == 3
    assert criteria_results[0]["score"] == 8.5
    assert "Technically sound" in criteria_results[0]["comments"]
    assert criteria_results[1]["score"] == 9.0
    assert "Strong potential" in criteria_results[1]["comments"]
    assert criteria_results[2]["score"] == 7.5

    # Verify LLM was called
    approval_service.llm_provider.parse_structured_output.assert_called_once()


@pytest.mark.asyncio
async def test_review_project_nonexistent(approval_service):
    """Test reviewing a nonexistent project."""
    # Arrange
    approval_service.repository.get_by_id = Mock(return_value=None)

    # Act & Assert
    with pytest.raises(ValueError, match="Project not found"):
        await approval_service.review_project(
            project_id="nonexistent-id",
            reviewer_id="reviewer-001"
        )


@pytest.mark.asyncio
async def test_review_project_wrong_status(approval_service):
    """Test reviewing a project with wrong status."""
    # Arrange
    project = await approval_service.get_project("proj-123")
    project.status = ProjectStatus.APPROVED

    # Act & Assert
    with pytest.raises(ValueError, match="Project is not pending review"):
        await approval_service.review_project(
            project_id="proj-123",
            reviewer_id="reviewer-001"
        )


@pytest.mark.asyncio
async def test_ai_review_without_llm_provider(mock_project_repository, sample_criteria):
    """Test AI review without LLM provider."""
    # Arrange
    service = ProjectApprovalService(
        project_repository=mock_project_repository,
        criteria=sample_criteria
    )
    project = await service.get_project("proj-123")

    # Act & Assert
    with pytest.raises(ValueError, match="LLM provider is required"):
        await service._perform_ai_review(project)


# ---------------------
# Submit Review Tests
# ---------------------

@pytest.mark.asyncio
async def test_submit_review_successful(approval_service):
    """Test successful review submission."""
    # Arrange
    criteria_scores = [
        {"criterion_id": "crit-1", "name": "Technical Feasibility",
            "score": 8.0, "comments": "Good"},
        {"criterion_id": "crit-2", "name": "Community Impact",
            "score": 9.0, "comments": "Excellent"},
        {"criterion_id": "crit-3", "name": "Resource Efficiency",
            "score": 7.0, "comments": "Acceptable"}
    ]

    # Act
    result = await approval_service.submit_review(
        project_id="proj-123",
        reviewer_id="reviewer-001",
        criteria_scores=criteria_scores,
        overall_score=8.0,
        comments="Solid project overall"
    )

    # Assert
    assert result is True
    project = await approval_service.get_project("proj-123")
    assert len(project.reviews) == 1
    assert project.status == ProjectStatus.PENDING_APPROVAL  # Human approval required
    approval_service.repository.update.assert_called_once()


@pytest.mark.asyncio
async def test_submit_review_auto_approve(approval_service):
    """Test review submission with auto-approval."""
    # Arrange
    approval_service.require_human_approval = False
    criteria_scores = [
        {"criterion_id": "crit-1", "score": 9.0, "comments": "Excellent"},
        {"criterion_id": "crit-2", "score": 8.5, "comments": "Very good"},
        {"criterion_id": "crit-3", "score": 9.0, "comments": "Excellent"}
    ]

    # Act
    result = await approval_service.submit_review(
        project_id="proj-123",
        reviewer_id="reviewer-001",
        criteria_scores=criteria_scores,
        overall_score=8.8,
        comments="Outstanding project"
    )

    # Assert
    assert result is True
    project = await approval_service.get_project("proj-123")
    assert project.status == ProjectStatus.APPROVED  # Auto-approved due to high score
    approval_service.repository.update.assert_called_once()


@pytest.mark.asyncio
async def test_submit_review_auto_reject(approval_service):
    """Test review submission with auto-rejection."""
    # Arrange
    criteria_scores = [
        {"criterion_id": "crit-1", "score": 4.0, "comments": "Poor"},
        {"criterion_id": "crit-2", "score": 3.0, "comments": "Inadequate"},
        {"criterion_id": "crit-3", "score": 2.0, "comments": "Very poor"}
    ]

    # Act
    result = await approval_service.submit_review(
        project_id="proj-123",
        reviewer_id="reviewer-001",
        criteria_scores=criteria_scores,
        overall_score=3.0,
        comments="Does not meet standards"
    )

    # Assert
    assert result is True
    project = await approval_service.get_project("proj-123")
    assert project.status == ProjectStatus.REJECTED  # Auto-rejected due to low score
    approval_service.repository.update.assert_called_once()


@pytest.mark.asyncio
async def test_submit_review_nonexistent_project(approval_service):
    """Test submitting review for nonexistent project."""
    # Arrange
    approval_service.repository.get_by_id = Mock(return_value=None)

    # Act & Assert
    with pytest.raises(ValueError, match="Project not found"):
        await approval_service.submit_review(
            project_id="nonexistent-id",
            reviewer_id="reviewer-001",
            criteria_scores=[],
            overall_score=7.0
        )


# ---------------------
# Approve Project Tests
# ---------------------

@pytest.mark.asyncio
async def test_approve_project(approval_service):
    """Test approving a project."""
    # Act
    result = await approval_service.approve_project(
        project_id="proj-123",
        approver_id="approver-001",
        comments="Project approved"
    )

    # Assert
    assert result is True
    project = await approval_service.get_project("proj-123")
    assert project.status == ProjectStatus.APPROVED
    assert project.approved_at is not None
    assert project.approved_by == "approver-001"
    assert project.approval_comments == "Project approved"
    approval_service.repository.update.assert_called_once()


@pytest.mark.asyncio
async def test_approve_nonexistent_project(approval_service):
    """Test approving a nonexistent project."""
    # Arrange
    approval_service.repository.get_by_id = Mock(return_value=None)

    # Act & Assert
    with pytest.raises(ValueError, match="Project not found"):
        await approval_service.approve_project(
            project_id="nonexistent-id",
            approver_id="approver-001"
        )


# ---------------------
# Reject Project Tests
# ---------------------

@pytest.mark.asyncio
async def test_reject_project(approval_service):
    """Test rejecting a project."""
    # Act
    result = await approval_service.reject_project(
        project_id="proj-123",
        rejector_id="rejector-001",
        reason="Does not align with platform goals"
    )

    # Assert
    assert result is True
    project = await approval_service.get_project("proj-123")
    assert project.status == ProjectStatus.REJECTED
    assert project.rejected_at is not None
    assert project.rejected_by == "rejector-001"
    assert project.rejection_reason == "Does not align with platform goals"
    approval_service.repository.update.assert_called_once()


@pytest.mark.asyncio
async def test_reject_nonexistent_project(approval_service):
    """Test rejecting a nonexistent project."""
    # Arrange
    approval_service.repository.get_by_id = Mock(return_value=None)

    # Act & Assert
    with pytest.raises(ValueError, match="Project not found"):
        await approval_service.reject_project(
            project_id="nonexistent-id",
            rejector_id="rejector-001",
            reason="Invalid project"
        )


# ---------------------
# AI Review Error Handling
# ---------------------

@pytest.mark.asyncio
async def test_ai_review_error_handling(approval_service):
    """Test error handling during AI review."""
    # Arrange
    approval_service.llm_provider.parse_structured_output = AsyncMock(
        side_effect=Exception("LLM processing error")
    )

    # Act
    overall_score, criteria_results = await approval_service._perform_ai_review(
        await approval_service.get_project("proj-123")
    )

    # Assert
    assert overall_score == 0.0
    assert criteria_results == []

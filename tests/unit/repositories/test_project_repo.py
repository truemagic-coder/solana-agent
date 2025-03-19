"""
Tests for the MongoDB Project Repository.

This module tests the MongoProjectRepository implementation.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import uuid
import datetime

from solana_agent.repositories.project import MongoProjectRepository
from solana_agent.domains.projects import Project, ProjectStatus, ProjectReview, ApprovalCriteria


@pytest.fixture
def mock_mongodb_adapter():
    """Create a mock MongoDB adapter."""
    adapter = Mock()
    adapter.database = {}

    # Create mock collection
    collection = MagicMock()
    adapter.database["projects"] = collection

    return adapter


@pytest.fixture
def sample_project():
    """Create a sample project for testing."""
    return Project(
        name="Test Project",
        description="A test project description",
        submitter_id="user-123",
        goals="Sample goals",
        scope="Sample scope",
        resources_required="Sample resources",
        timeline="Sample timeline"
    )


@pytest.fixture
def sample_review():
    """Create a sample project review."""
    return ProjectReview(
        id=str(uuid.uuid4()),
        project_id="test-project-id",  # Added required field
        reviewer_id="reviewer-123",
        criteria_scores=[  # Updated to match the model
            {
                "criterion_id": "criterion-1",
                "name": "Feasibility",
                "score": 8.5,
                "comments": "Good feasibility"
            }
        ],
        overall_score=8.5,  # Added required field
        comments="Looks good to me"
        # timestamp will be default_factory
    )


class TestMongoProjectRepository:
    """Tests for the MongoProjectRepository."""

    def test_init(self, mock_mongodb_adapter):
        """Test repository initialization."""
        repo = MongoProjectRepository(mock_mongodb_adapter)
        assert repo.db_adapter == mock_mongodb_adapter
        assert repo.collection == mock_mongodb_adapter.database["projects"]

    def test_create(self, mock_mongodb_adapter, sample_project):
        """Test creating a project."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        mock_mongodb_adapter.database["projects"].insert_one.return_value = MagicMock(
            inserted_id="some-id"
        )

        # Execute
        project_id = repo.create(sample_project)

        # Verify
        assert project_id == sample_project.id
        assert mock_mongodb_adapter.database["projects"].insert_one.called

        # Verify ID was generated
        assert sample_project.id is not None

        # Get the passed data to insert_one
        inserted_data = mock_mongodb_adapter.database["projects"].insert_one.call_args[0][0]
        assert inserted_data == sample_project.model_dump()

    def test_get_existing(self, mock_mongodb_adapter, sample_project):
        """Test getting an existing project."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        project_dict = sample_project.model_dump()
        mock_mongodb_adapter.database["projects"].find_one.return_value = project_dict

        # Execute
        result = repo.get("test-id")

        # Verify
        assert result is not None
        assert result.name == sample_project.name
        assert result.description == sample_project.description
        mock_mongodb_adapter.database["projects"].find_one.assert_called_once_with({
                                                                                   "id": "test-id"})

    def test_get_nonexistent(self, mock_mongodb_adapter):
        """Test getting a non-existent project."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        mock_mongodb_adapter.database["projects"].find_one.return_value = None

        # Execute
        result = repo.get("nonexistent-id")

        # Verify
        assert result is None
        mock_mongodb_adapter.database["projects"].find_one.assert_called_once_with({
                                                                                   "id": "nonexistent-id"})

    def test_update_existing(self, mock_mongodb_adapter, sample_project):
        """Test updating an existing project."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        sample_project.id = "test-id"
        mock_mongodb_adapter.database["projects"].update_one.return_value = MagicMock(
            modified_count=1
        )

        # Execute
        result = repo.update(sample_project)

        # Verify
        assert result is True
        mock_mongodb_adapter.database["projects"].update_one.assert_called_once(
        )

    def test_update_nonexistent(self, mock_mongodb_adapter, sample_project):
        """Test updating a non-existent project."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        sample_project.id = "nonexistent-id"
        mock_mongodb_adapter.database["projects"].update_one.return_value = MagicMock(
            modified_count=0
        )

        # Execute
        result = repo.update(sample_project)

        # Verify
        assert result is False
        mock_mongodb_adapter.database["projects"].update_one.assert_called_once(
        )

    def test_delete_existing(self, mock_mongodb_adapter):
        """Test deleting an existing project."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        mock_mongodb_adapter.database["projects"].delete_one.return_value = MagicMock(
            deleted_count=1
        )

        # Execute
        result = repo.delete("test-id")

        # Verify
        assert result is True
        mock_mongodb_adapter.database["projects"].delete_one.assert_called_once_with({
                                                                                     "id": "test-id"})

    def test_delete_nonexistent(self, mock_mongodb_adapter):
        """Test deleting a non-existent project."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        mock_mongodb_adapter.database["projects"].delete_one.return_value = MagicMock(
            deleted_count=0
        )

        # Execute
        result = repo.delete("nonexistent-id")

        # Verify
        assert result is False
        mock_mongodb_adapter.database["projects"].delete_one.assert_called_once_with({
                                                                                     "id": "nonexistent-id"})

    def test_find_by_status(self, mock_mongodb_adapter, sample_project):
        """Test finding projects by status."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        projects_list = [sample_project.model_dump()]
        mock_mongodb_adapter.database["projects"].find.return_value = projects_list

        # Execute with string status
        result1 = repo.find_by_status("draft")

        # Verify
        assert len(result1) == 1
        assert result1[0].name == sample_project.name
        mock_mongodb_adapter.database["projects"].find.assert_called_with({
                                                                          "status": "draft"})

        # Execute with enum status
        result2 = repo.find_by_status(ProjectStatus.DRAFT)

        # Verify
        assert len(result2) == 1
        assert result2[0].name == sample_project.name

    def test_find_by_submitter(self, mock_mongodb_adapter, sample_project):
        """Test finding projects by submitter."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        projects_list = [sample_project.model_dump()]
        mock_mongodb_adapter.database["projects"].find.return_value = projects_list

        # Execute
        result = repo.find_by_submitter("user-123")

        # Verify
        assert len(result) == 1
        assert result[0].name == sample_project.name
        mock_mongodb_adapter.database["projects"].find.assert_called_once_with(
            {"submitter_id": "user-123"})

    def test_get_all(self, mock_mongodb_adapter, sample_project):
        """Test getting all projects."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        projects_list = [sample_project.model_dump()]
        mock_mongodb_adapter.database["projects"].find.return_value = projects_list

        # Execute
        result = repo.get_all()

        # Verify
        assert len(result) == 1
        assert result[0].name == sample_project.name
        mock_mongodb_adapter.database["projects"].find.assert_called_once_with(
        )

    def test_find_by_name_exact(self, mock_mongodb_adapter, sample_project):
        """Test finding projects by name with exact match."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        projects_list = [sample_project.model_dump()]
        mock_mongodb_adapter.database["projects"].find.return_value = projects_list

        # Execute
        result = repo.find_by_name("Test Project", exact=True)

        # Verify
        assert len(result) == 1
        assert result[0].name == sample_project.name
        mock_mongodb_adapter.database["projects"].find.assert_called_once_with(
            {"name": "Test Project"})

    def test_find_by_name_regex(self, mock_mongodb_adapter, sample_project):
        """Test finding projects by name with regex match."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        projects_list = [sample_project.model_dump()]
        mock_mongodb_adapter.database["projects"].find.return_value = projects_list

        # Execute
        result = repo.find_by_name("Test", exact=False)

        # Verify
        assert len(result) == 1
        assert result[0].name == sample_project.name
        mock_mongodb_adapter.database["projects"].find.assert_called_once_with(
            {"name": {"$regex": "Test", "$options": "i"}}
        )

    def test_find_by_reviewer(self, mock_mongodb_adapter, sample_project):
        """Test finding projects by reviewer."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        projects_list = [sample_project.model_dump()]
        mock_mongodb_adapter.database["projects"].find.return_value = projects_list

        # Execute
        result = repo.find_by_reviewer("reviewer-123")

        # Verify
        assert len(result) == 1
        assert result[0].name == sample_project.name
        mock_mongodb_adapter.database["projects"].find.assert_called_once_with(
            {"reviews.reviewer_id": "reviewer-123"}
        )

    def test_add_review(self, mock_mongodb_adapter, sample_project, sample_review):
        """Test adding a review to a project."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        review_dict = sample_review.model_dump()
        mock_mongodb_adapter.database["projects"].update_one.return_value = MagicMock(
            modified_count=1
        )

        # Execute
        result = repo.add_review("test-id", review_dict)

        # Verify
        assert result is True
        mock_mongodb_adapter.database["projects"].update_one.assert_called_once_with(
            {"id": "test-id"},
            {"$push": {"reviews": review_dict}}
        )

    def test_get_projects_awaiting_approval(self, mock_mongodb_adapter):
        """Test getting projects awaiting approval."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)

        # Need to patch find_by_status which we use internally
        with patch.object(repo, 'find_by_status') as mock_find_by_status:
            mock_find_by_status.return_value = []

            # Execute
            repo.get_projects_awaiting_approval()

            # Verify correct status was used
            mock_find_by_status.assert_called_once_with(
                ProjectStatus.PENDING_APPROVAL)

    def test_get_projects_by_approval_status_approved(self, mock_mongodb_adapter):
        """Test getting approved projects."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)

        # Need to patch find_by_status which we use internally
        with patch.object(repo, 'find_by_status') as mock_find_by_status:
            mock_find_by_status.return_value = []

            # Execute
            repo.get_projects_by_approval_status(approved=True)

            # Verify correct status was used
            mock_find_by_status.assert_called_once_with(ProjectStatus.APPROVED)

    def test_get_projects_by_approval_status_rejected(self, mock_mongodb_adapter):
        """Test getting rejected projects."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)

        # Need to patch find_by_status which we use internally
        with patch.object(repo, 'find_by_status') as mock_find_by_status:
            mock_find_by_status.return_value = []

            # Execute
            repo.get_projects_by_approval_status(approved=False)

            # Verify correct status was used
            mock_find_by_status.assert_called_once_with(ProjectStatus.REJECTED)

    def test_count_by_status(self, mock_mongodb_adapter):
        """Test counting projects by status."""
        # Setup
        repo = MongoProjectRepository(mock_mongodb_adapter)
        aggregate_result = [
            {"_id": "draft", "count": 5},
            {"_id": "approved", "count": 3},
            {"_id": "rejected", "count": 2}
        ]
        mock_mongodb_adapter.database["projects"].aggregate.return_value = aggregate_result

        # Execute
        result = repo.count_by_status()

        # Verify
        assert result["draft"] == 5
        assert result["approved"] == 3
        assert result["rejected"] == 2

        # All status values should be present even if zero
        for status in ProjectStatus:
            assert status.value in result

        # Verify correct aggregation pipeline
        mock_mongodb_adapter.database["projects"].aggregate.assert_called_once(
        )
        pipeline = mock_mongodb_adapter.database["projects"].aggregate.call_args[0][0]
        assert pipeline[0]["$group"]["_id"] == "$status"

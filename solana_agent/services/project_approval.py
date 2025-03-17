"""
Project approval service implementation.

This service manages the review and approval process for projects.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

from solana_agent.interfaces.services import ProjectApprovalService as ProjectApprovalServiceInterface
from solana_agent.interfaces.providers import LLMProvider
from solana_agent.interfaces.repositories import ProjectRepository
from solana_agent.domain.projects import Project, ProjectStatus, ApprovalCriteria, ProjectReview
from solana_agent.domain.models import ProjectApprovalResult


class ProjectApprovalService(ProjectApprovalServiceInterface):
    """Service for managing project approvals."""

    def __init__(
        self,
        project_repository: ProjectRepository,
        llm_provider: Optional[LLMProvider] = None,
        criteria: Optional[List[ApprovalCriteria]] = None,
        auto_approve_threshold: float = 8.0,
        require_human_approval: bool = True
    ):
        """Initialize the project approval service.

        Args:
            project_repository: Repository for project data
            llm_provider: Optional provider for AI-assisted reviews
            criteria: Optional list of approval criteria
            auto_approve_threshold: Score threshold for automatic approval
            require_human_approval: Whether human approval is required
        """
        self.repository = project_repository
        self.llm_provider = llm_provider
        self.criteria = criteria or []
        self.auto_approve_threshold = auto_approve_threshold
        self.require_human_approval = require_human_approval

    async def submit_project(self, project: Project) -> str:
        """Submit a project for approval.

        Args:
            project: Project to submit

        Returns:
            Project ID
        """
        # Set initial status
        project.status = ProjectStatus.PENDING_REVIEW
        project.submitted_at = datetime.now()

        # Store project
        return self.repository.create(project)

    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID.

        Args:
            project_id: Project ID

        Returns:
            Project or None if not found
        """
        return self.repository.get_by_id(project_id)

    async def review_project(
        self, project_id: str, reviewer_id: str, is_human_reviewer: bool = True
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Review a project against approval criteria.

        Args:
            project_id: Project ID
            reviewer_id: ID of the reviewer
            is_human_reviewer: Whether the reviewer is human

        Returns:
            Tuple of (overall_score, criteria_results)
        """
        project = await self.get_project(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        if project.status != ProjectStatus.PENDING_REVIEW:
            raise ValueError(f"Project is not pending review: {project_id}")

        # Initialize results
        criteria_results = []

        # If AI review is enabled, use LLM provider
        if not is_human_reviewer and self.llm_provider:
            return await self._perform_ai_review(project)

        # For human reviewers, just prepare empty criteria template
        for criterion in self.criteria:
            criteria_results.append({
                "criterion_id": criterion.id,
                "name": criterion.name,
                "description": criterion.description,
                "score": None,  # To be filled by human
                "comments": ""   # To be filled by human
            })

        return 0.0, criteria_results

    async def submit_review(
        self,
        project_id: str,
        reviewer_id: str,
        criteria_scores: List[Dict[str, Any]],
        overall_score: float,
        comments: str = ""
    ) -> bool:
        """Submit a completed review.

        Args:
            project_id: Project ID
            reviewer_id: ID of the reviewer
            criteria_scores: Scores for each criterion
            overall_score: Overall project score
            comments: Optional review comments

        Returns:
            True if review was submitted successfully
        """
        project = await self.get_project(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        # Create review object
        review = ProjectReview(
            id=str(uuid4()),
            project_id=project_id,
            reviewer_id=reviewer_id,
            criteria_scores=criteria_scores,
            overall_score=overall_score,
            comments=comments,
            timestamp=datetime.now()
        )

        # Add review to project
        project.reviews.append(review)

        # Update status based on score
        if not self.require_human_approval and overall_score >= self.auto_approve_threshold:
            project.status = ProjectStatus.APPROVED
        elif overall_score < 5.0:  # Below 5 is automatic rejection
            project.status = ProjectStatus.REJECTED
        else:
            # Keep in review state if human approval required
            project.status = ProjectStatus.PENDING_APPROVAL

        # Update project
        self.repository.update(project_id, project)

        return True

    async def approve_project(self, project_id: str, approver_id: str, comments: str = "") -> bool:
        """Approve a project.

        Args:
            project_id: Project ID
            approver_id: ID of the approver
            comments: Optional approval comments

        Returns:
            True if approval was successful
        """
        project = await self.get_project(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        # Update project status
        project.status = ProjectStatus.APPROVED
        project.approved_at = datetime.now()
        project.approved_by = approver_id
        project.approval_comments = comments

        # Update project
        self.repository.update(project_id, project)

        return True

    async def reject_project(self, project_id: str, rejector_id: str, reason: str) -> bool:
        """Reject a project.

        Args:
            project_id: Project ID
            rejector_id: ID of the rejector
            reason: Rejection reason

        Returns:
            True if rejection was successful
        """
        project = await self.get_project(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        # Update project status
        project.status = ProjectStatus.REJECTED
        project.rejected_at = datetime.now()
        project.rejected_by = rejector_id
        project.rejection_reason = reason

        # Update project
        self.repository.update(project_id, project)

        return True

    async def _perform_ai_review(self, project: Project) -> Tuple[float, List[Dict[str, Any]]]:
        """Perform an AI-assisted review of a project.

        Args:
            project: Project to review

        Returns:
            Tuple of (overall_score, criteria_results)
        """
        if not self.llm_provider:
            raise ValueError("LLM provider is required for AI reviews")

        # Format project details
        project_details = f"""
        Project Name: {project.name}
        Description: {project.description}
        Goals: {project.goals}
        Scope: {project.scope}
        Resources Required: {project.resources_required}
        Timeline: {project.timeline}
        """

        # Format criteria
        criteria_text = "\n".join([
            f"{i+1}. {c.name}: {c.description}" for i, c in enumerate(self.criteria)
        ])

        prompt = f"""
        Please review this project proposal against the approval criteria.
        
        PROJECT DETAILS:
        {project_details}
        
        APPROVAL CRITERIA:
        {criteria_text}
        
        Evaluate the project against each criterion, providing a score (0-10) and comments.
        Also provide an overall assessment and final score.
        """

        try:
            result = await self.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="You are an expert project evaluator. Assess projects objectively against criteria.",
                model_class=ProjectApprovalResult,
                temperature=0.2
            )

            # Format criteria results
            criteria_results = []
            for i, criterion in enumerate(self.criteria):
                if i < len(result.criteria_evaluations):
                    evaluation = result.criteria_evaluations[i]
                    criteria_results.append({
                        "criterion_id": criterion.id,
                        "name": criterion.name,
                        "score": evaluation.score,
                        "comments": evaluation.comments
                    })

            return result.overall_score, criteria_results
        except Exception as e:
            print(f"Error in AI project review: {e}")
            return 0.0, []

"""
Project repository implementation using MongoDB.

This module provides data access for projects.
"""
from typing import List, Optional, Dict, Any
import datetime
import uuid

from solana_agent.adapters.mongodb_adapter import MongoDBAdapter
from solana_agent.interfaces.repositories import ProjectRepository
from solana_agent.domains.projects import Project, ProjectStatus


class MongoProjectRepository(ProjectRepository):
    """MongoDB implementation of the ProjectRepository interface."""

    def __init__(self, db_adapter: MongoDBAdapter, collection_name: str = "projects"):
        """Initialize with a MongoDB adapter.

        Args:
            db_adapter: MongoDB adapter
            collection_name: Name of the collection to use
        """
        self.db_adapter = db_adapter
        self.collection = db_adapter.db[collection_name]

    def create(self, project: Project) -> str:
        """Create a new project.

        Args:
            project: Project to create

        Returns:
            ID of the created project
        """
        # Generate an ID if not provided
        if not project.id:
            project.id = str(uuid.uuid4())

        # Convert to dict for MongoDB
        project_dict = project.model_dump()

        # Insert into MongoDB
        result = self.collection.insert_one(project_dict)

        return project.id

    def get(self, project_id: str) -> Optional[Project]:
        """Get a project by ID.

        Args:
            project_id: ID of the project to retrieve

        Returns:
            Project or None if not found
        """
        # Try to find the project
        project_dict = self.collection.find_one({"id": project_id})

        if not project_dict:
            return None

        # Convert to Project object
        return Project(**project_dict)

    def update(self, project: Project) -> bool:
        """Update an existing project.

        Args:
            project: Project to update

        Returns:
            True if update was successful
        """
        project_dict = project.model_dump()
        project_dict["updated_at"] = datetime.datetime.now()

        # Update the project
        result = self.collection.update_one(
            {"id": project.id},
            {"$set": project_dict}
        )

        return result.modified_count > 0

    def delete(self, project_id: str) -> bool:
        """Delete a project.

        Args:
            project_id: ID of the project to delete

        Returns:
            True if deletion was successful
        """
        result = self.collection.delete_one({"id": project_id})
        return result.deleted_count > 0

    def find_by_status(self, status: str) -> List[Project]:
        """Find projects by status.

        Args:
            status: Project status

        Returns:
            List of matching projects
        """
        # Convert string to enum if needed
        if isinstance(status, str):
            try:
                status = ProjectStatus(status)
                status_value = status.value
            except ValueError:
                status_value = status
        else:
            # Assuming it's already an enum
            status_value = status.value

        # Find projects with the given status
        projects_dict = list(self.collection.find({"status": status_value}))

        # Convert to Project objects
        return [Project(**project_dict) for project_dict in projects_dict]

    def find_by_submitter(self, submitter_id: str) -> List[Project]:
        """Find projects by submitter.

        Args:
            submitter_id: ID of the submitter

        Returns:
            List of matching projects
        """
        # Find projects with the given submitter
        projects_dict = list(self.collection.find(
            {"submitter_id": submitter_id}))

        # Convert to Project objects
        return [Project(**project_dict) for project_dict in projects_dict]

    def get_all(self) -> List[Project]:
        """Get all projects.

        Returns:
            List of all projects
        """
        # Find all projects
        projects_dict = list(self.collection.find())

        # Convert to Project objects
        return [Project(**project_dict) for project_dict in projects_dict]

    def find_by_name(self, name: str, exact: bool = False) -> List[Project]:
        """Find projects by name.

        Args:
            name: Project name to search for
            exact: Whether to match exactly or use regex

        Returns:
            List of matching projects
        """
        if exact:
            query = {"name": name}
        else:
            # Case-insensitive regex search
            query = {"name": {"$regex": name, "$options": "i"}}

        # Find projects with matching name
        projects_dict = list(self.collection.find(query))

        # Convert to Project objects
        return [Project(**project_dict) for project_dict in projects_dict]

    def find_by_reviewer(self, reviewer_id: str) -> List[Project]:
        """Find projects by reviewer.

        Args:
            reviewer_id: ID of the reviewer

        Returns:
            List of projects reviewed by the user
        """
        # Find projects with reviews by the given reviewer
        projects_dict = list(self.collection.find({
            "reviews.reviewer_id": reviewer_id
        }))

        # Convert to Project objects
        return [Project(**project_dict) for project_dict in projects_dict]

    def add_review(self, project_id: str, review: Dict[str, Any]) -> bool:
        """Add a review to a project.

        Args:
            project_id: ID of the project
            review: Review to add

        Returns:
            True if review was added successfully
        """
        # Update the project
        result = self.collection.update_one(
            {"id": project_id},
            {"$push": {"reviews": review}}
        )

        return result.modified_count > 0

    def get_projects_awaiting_approval(self) -> List[Project]:
        """Get all projects awaiting approval.

        Returns:
            List of projects with PENDING_APPROVAL status
        """
        return self.find_by_status(ProjectStatus.PENDING_APPROVAL)

    def get_projects_by_approval_status(self, approved: bool) -> List[Project]:
        """Get projects by approval status.

        Args:
            approved: Whether to get approved or rejected projects

        Returns:
            List of approved or rejected projects
        """
        if approved:
            status = ProjectStatus.APPROVED
        else:
            status = ProjectStatus.REJECTED

        return self.find_by_status(status)

    def count_by_status(self) -> Dict[str, int]:
        """Count projects by status.

        Returns:
            Dictionary mapping status names to counts
        """
        # Use MongoDB's aggregation framework
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]

        result = list(self.collection.aggregate(pipeline))

        # Convert to a dictionary
        counts = {}
        for item in result:
            counts[item["_id"]] = item["count"]

        # Ensure all statuses are represented
        for status in ProjectStatus:
            if status.value not in counts:
                counts[status.value] = 0

        return counts

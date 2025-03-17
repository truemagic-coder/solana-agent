"""
MongoDB implementation of the scheduling repository.
"""
from datetime import datetime, date
from typing import Dict, List, Optional, Any

from solana_agent.domain.scheduling import ScheduledTask, AgentSchedule
from solana_agent.interfaces.repositories import SchedulingRepository


class MongoSchedulingRepository(SchedulingRepository):
    """MongoDB implementation of the SchedulingRepository interface."""

    def __init__(self, db_adapter):
        """Initialize the repository with a database adapter."""
        self.db = db_adapter
        self.tasks_collection = "scheduled_tasks"
        self.schedules_collection = "agent_schedules"

        # Ensure collections exist
        self.db.create_collection(self.tasks_collection)
        self.db.create_collection(self.schedules_collection)

        # Create indexes for tasks
        self.db.create_index(self.tasks_collection, [
                             ("task_id", 1)], unique=True)
        self.db.create_index(self.tasks_collection, [("assigned_to", 1)])
        self.db.create_index(self.tasks_collection, [("status", 1)])
        self.db.create_index(self.tasks_collection, [("scheduled_start", 1)])

        # Create indexes for schedules
        self.db.create_index(
            self.schedules_collection,
            [("agent_id", 1), ("date", 1)],
            unique=True
        )

    def get_scheduled_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a scheduled task by ID."""
        doc = self.db.find_one(self.tasks_collection, {"task_id": task_id})
        if not doc:
            return None

        # Convert string dates back to datetime
        self._convert_task_dates(doc)

        return ScheduledTask.model_validate(doc)

    def create_scheduled_task(self, task: ScheduledTask) -> str:
        """Create a new scheduled task and return its ID."""
        doc = task.model_dump()

        # Convert datetime to string for MongoDB
        self._prepare_task_dates(doc)

        # Check if task already exists
        existing = self.db.find_one(self.tasks_collection, {
                                    "task_id": task.task_id})
        if existing:
            self.db.update_one(
                self.tasks_collection,
                {"task_id": task.task_id},
                {"$set": doc}
            )
            return task.task_id
        else:
            self.db.insert_one(self.tasks_collection, doc)
            return task.task_id

    def update_scheduled_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update a scheduled task."""
        # Convert datetime objects to strings
        updates_copy = updates.copy()
        if "scheduled_start" in updates_copy and isinstance(updates_copy["scheduled_start"], datetime):
            updates_copy["scheduled_start"] = updates_copy["scheduled_start"].isoformat()

        if "scheduled_end" in updates_copy and isinstance(updates_copy["scheduled_end"], datetime):
            updates_copy["scheduled_end"] = updates_copy["scheduled_end"].isoformat()

        return self.db.update_one(
            self.tasks_collection,
            {"task_id": task_id},
            {"$set": updates_copy}
        )

    def delete_scheduled_task(self, task_id: str) -> bool:
        """Delete a scheduled task."""
        return self.db.delete_one(self.tasks_collection, {"task_id": task_id})

    def get_tasks_by_status(self, status: str) -> List[ScheduledTask]:
        """Get tasks with a specific status."""
        docs = self.db.find(self.tasks_collection, {"status": status})

        tasks = []
        for doc in docs:
            self._convert_task_dates(doc)
            tasks.append(ScheduledTask.model_validate(doc))

        return tasks

    def get_agent_tasks(
        self, agent_id: str, start_date: datetime, end_date: datetime
    ) -> List[ScheduledTask]:
        """Get tasks for an agent within a time period."""
        start_date_str = start_date.isoformat()
        end_date_str = end_date.isoformat()

        query = {
            "assigned_to": agent_id,
            "$or": [
                # Task starts within the time period
                {"scheduled_start": {"$gte": start_date_str, "$lt": end_date_str}},
                # Task ends within the time period
                {"scheduled_end": {"$gt": start_date_str, "$lte": end_date_str}},
                # Task spans the entire time period
                {"$and": [
                    {"scheduled_start": {"$lte": start_date_str}},
                    {"scheduled_end": {"$gte": end_date_str}}
                ]}
            ]
        }

        docs = self.db.find(self.tasks_collection, query)

        tasks = []
        for doc in docs:
            self._convert_task_dates(doc)
            tasks.append(ScheduledTask.model_validate(doc))

        return tasks

    def get_unscheduled_tasks(self) -> List[ScheduledTask]:
        """Get tasks that haven't been scheduled yet."""
        query = {
            "$or": [
                {"scheduled_start": None},
                {"scheduled_start": {"$exists": False}},
                {"assigned_to": None},
                {"assigned_to": {"$exists": False}}
            ]
        }

        docs = self.db.find(self.tasks_collection, query)

        tasks = []
        for doc in docs:
            self._convert_task_dates(doc)
            tasks.append(ScheduledTask.model_validate(doc))

        return tasks

    def get_agent_schedule(self, agent_id: str, date: datetime.date) -> Optional[AgentSchedule]:
        """Get schedule for an agent on a specific date."""
        date_str = date.isoformat()

        doc = self.db.find_one(
            self.schedules_collection,
            {"agent_id": agent_id, "date": date_str}
        )

        if not doc:
            return None

        # Convert date string back to date
        doc["date"] = datetime.date.fromisoformat(doc["date"])

        # Convert task dates
        for task in doc.get("tasks", []):
            self._convert_task_dates(task)

        return AgentSchedule.model_validate(doc)

    def get_all_agent_schedules(self, date: datetime.date) -> List[AgentSchedule]:
        """Get schedules for all agents on a specific date."""
        date_str = date.isoformat()

        docs = self.db.find(self.schedules_collection, {"date": date_str})

        schedules = []
        for doc in docs:
            # Convert date string back to date
            doc["date"] = datetime.date.fromisoformat(doc["date"])

            # Convert task dates
            for task in doc.get("tasks", []):
                self._convert_task_dates(task)

            schedules.append(AgentSchedule.model_validate(doc))

        return schedules

    def save_agent_schedule(self, schedule: AgentSchedule) -> bool:
        """Save an agent's schedule."""
        doc = schedule.model_dump()

        # Convert date to string
        doc["date"] = doc["date"].isoformat()

        # Convert task dates
        for task in doc.get("tasks", []):
            self._prepare_task_dates(task)

        # Check if schedule already exists
        existing = self.db.find_one(
            self.schedules_collection,
            {"agent_id": schedule.agent_id, "date": doc["date"]}
        )

        if existing:
            # Update existing schedule
            return self.db.update_one(
                self.schedules_collection,
                {"_id": existing["_id"]},
                {"$set": doc}
            )
        else:
            # Create new schedule
            self.db.insert_one(self.schedules_collection, doc)
            return True

    def _convert_task_dates(self, task_doc: Dict[str, Any]) -> None:
        """Convert string dates to datetime objects in a task document."""
        if task_doc.get("scheduled_start") and isinstance(task_doc["scheduled_start"], str):
            task_doc["scheduled_start"] = datetime.fromisoformat(
                task_doc["scheduled_start"])

        if task_doc.get("scheduled_end") and isinstance(task_doc["scheduled_end"], str):
            task_doc["scheduled_end"] = datetime.fromisoformat(
                task_doc["scheduled_end"])

    def _prepare_task_dates(self, task_doc: Dict[str, Any]) -> None:
        """Convert datetime objects to strings in a task document."""
        if task_doc.get("scheduled_start") and isinstance(task_doc["scheduled_start"], datetime):
            task_doc["scheduled_start"] = task_doc["scheduled_start"].isoformat()

        if task_doc.get("scheduled_end") and isinstance(task_doc["scheduled_end"], datetime):
            task_doc["scheduled_end"] = task_doc["scheduled_end"].isoformat()

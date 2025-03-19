"""
MongoDB implementation of the scheduling repository.
"""
from datetime import datetime, date
from typing import Dict, List, Optional, Any

from solana_agent.domains import ScheduledTask, AgentSchedule
from solana_agent.interfaces import SchedulingRepository


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
        self, agent_id: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, include_completed: bool = False
    ) -> List[ScheduledTask]:
        """Get tasks for an agent within a time period.

        Args:
            agent_id: ID of the agent to get tasks for
            start_time: Optional start date of the time period
            end_time: Optional end date of the time period
            include_completed: Whether to include completed tasks (default: False)

        Returns:
            List of scheduled tasks
        """
        # Start with basic agent query
        query = {"assigned_to": agent_id}

        # Filter out completed tasks if not requested
        if not include_completed:
            query["status"] = {"$ne": "completed"}

        # Add date constraints if provided
        if start_time is not None and end_time is not None:
            # Both dates provided
            start_time_str = start_time.isoformat()
            end_time_str = end_time.isoformat()

            query["$or"] = [
                # Task starts within the time period
                {"scheduled_start": {"$gte": start_time_str, "$lt": end_time_str}},
                # Task ends within the time period
                {"scheduled_end": {"$gt": start_time_str, "$lte": end_time_str}},
                # Task spans the entire time period
                {"$and": [
                    {"scheduled_start": {"$lte": start_time_str}},
                    {"scheduled_end": {"$gte": end_time_str}}
                ]}
            ]
        elif start_time is not None:
            # Only start date provided
            start_time_str = start_time.isoformat()
            query["scheduled_end"] = {"$gte": start_time_str}
        elif end_time is not None:
            # Only end date provided
            end_time_str = end_time.isoformat()
            query["scheduled_start"] = {"$lte": end_date_str}

        # Get tasks matching the query
        docs = self.db.find(self.tasks_collection, query)

        # Process results
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
            {"agent_id": agent_id, "working_hours.schedule_date": date_str}
        )

        if not doc:
            return None

        # Process availability patterns
        self._convert_schedule_times(doc)

        return AgentSchedule.model_validate(doc)

    def get_all_agent_schedules(self, date: datetime.date) -> List[AgentSchedule]:
        """Get schedules for all agents on a specific date."""
        date_str = date.isoformat()

        docs = self.db.find(self.schedules_collection, {
                            "working_hours.schedule_date": date_str})

        schedules = []
        for doc in docs:
            self._convert_schedule_times(doc)
            schedules.append(AgentSchedule.model_validate(doc))

        return schedules

    def save_agent_schedule(self, schedule: AgentSchedule) -> bool:
        """Save an agent's schedule."""
        doc = schedule.model_dump()
        schedule_date = doc["working_hours"]["schedule_date"]

        # Prepare data
        for pattern in doc["availability_patterns"]:
            pattern["start_time"] = pattern["start_time"].isoformat()
            pattern["end_time"] = pattern["end_time"].isoformat()

        # Check if schedule exists
        existing = self.db.find_one(
            self.schedules_collection,
            {"agent_id": doc["agent_id"],
                "working_hours.schedule_date": schedule_date}
        )

        if existing:
            return self.db.update_one(
                self.schedules_collection,
                {"_id": existing["_id"]},
                {"$set": doc}
            )
        else:
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

    def _convert_schedule_times(self, schedule_doc: Dict[str, Any]) -> None:
        """Convert string times to time objects in a schedule document."""
        from datetime import time

        # Convert availability pattern times
        if "availability_patterns" in schedule_doc:
            for pattern in schedule_doc["availability_patterns"]:
                if pattern.get("start_time") and isinstance(pattern["start_time"], str):
                    # Parse time from format like "08:00:00"
                    hours, minutes, seconds = map(
                        int, pattern["start_time"].split(":"))
                    pattern["start_time"] = time(hours, minutes, seconds)

                if pattern.get("end_time") and isinstance(pattern["end_time"], str):
                    hours, minutes, seconds = map(
                        int, pattern["end_time"].split(":"))
                    pattern["end_time"] = time(hours, minutes, seconds)

        # Keep working_hours times as strings since they're expected that way
        # in the model (based on the sample_schedule fixture)

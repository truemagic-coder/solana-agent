"""
Tests for the MongoSchedulingRepository implementation.

This module contains unit tests for the MongoDB-based scheduling repository.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime, date, timedelta
import uuid

from solana_agent.domains.scheduling import AgentAvailabilityPattern
from solana_agent.repositories.scheduling import MongoSchedulingRepository
from solana_agent.domains import ScheduledTask, AgentSchedule


@pytest.fixture
def mock_db_adapter():
    """Create a mock database adapter."""
    adapter = Mock()
    adapter.create_collection = Mock()
    adapter.create_index = Mock()
    adapter.insert_one = Mock(return_value="mock_id")
    adapter.find_one = Mock()
    adapter.find = Mock()
    adapter.update_one = Mock(return_value=True)
    adapter.delete_one = Mock(return_value=True)
    return adapter


@pytest.fixture
def scheduling_repo(mock_db_adapter):
    """Create a repository with mocked database adapter."""
    return MongoSchedulingRepository(mock_db_adapter)


@pytest.fixture
def sample_task():
    """Create a sample scheduled task for testing."""
    task_time = datetime.now() + timedelta(days=1)
    return ScheduledTask(
        task_id=f"task_{str(uuid.uuid4())[:8]}",
        title="Review code PR #1234",
        description="Review and approve pending code changes",
        priority=2,
        status="pending",
        assigned_to="agent_123",
        scheduled_start=task_time,
        scheduled_end=task_time + timedelta(hours=1),
        estimated_duration=60,  # minutes
        metadata={"repository": "main-app", "pr_number": "1234"}
    )


@pytest.fixture
def sample_schedule():
    """Create a sample agent schedule for testing."""
    from datetime import time  # Make sure to import time

    # Create proper AgentAvailabilityPattern objects
    availability_patterns = [
        AgentAvailabilityPattern(
            day_of_week=0,  # Monday (0-indexed)
            start_time=time(8, 0),  # 8:00 AM
            end_time=time(17, 0)  # 5:00 PM
        ),
        AgentAvailabilityPattern(
            day_of_week=1,  # Tuesday
            start_time=time(8, 0),
            end_time=time(17, 0)
        )
    ]

    # Working hours with schedule_date
    schedule_date = date.today() + timedelta(days=1)
    working_hours = {
        "timezone": "UTC",
        "schedule_date": schedule_date.isoformat(),
        "start_time": time(8, 0).isoformat(),
        "end_time": time(17, 0).isoformat()
    }

    return AgentSchedule(
        agent_id="agent_123",
        availability_patterns=availability_patterns,
        time_off_periods=[],
        working_hours=working_hours
    )


class TestMongoSchedulingRepository:
    """Tests for the MongoSchedulingRepository implementation."""

    def test_init(self, mock_db_adapter):
        """Test repository initialization."""
        repo = MongoSchedulingRepository(mock_db_adapter)

        # Verify collections are created
        mock_db_adapter.create_collection.assert_any_call("scheduled_tasks")
        mock_db_adapter.create_collection.assert_any_call("agent_schedules")
        assert mock_db_adapter.create_collection.call_count == 2

        # Verify indexes are created
        assert mock_db_adapter.create_index.call_count >= 5
        mock_db_adapter.create_index.assert_any_call(
            "scheduled_tasks", [("task_id", 1)], unique=True)
        mock_db_adapter.create_index.assert_any_call(
            "scheduled_tasks", [("assigned_to", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "scheduled_tasks", [("status", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "scheduled_tasks", [("scheduled_start", 1)])
        mock_db_adapter.create_index.assert_any_call(
            "agent_schedules", [("agent_id", 1), ("date", 1)], unique=True)

    # ScheduledTask Tests
    def test_get_scheduled_task_found(self, scheduling_repo, mock_db_adapter, sample_task):
        """Test retrieving an existing task."""
        task_id = sample_task.task_id

        # Configure mock to return the task
        task_dict = sample_task.model_dump()
        # Convert datetime objects to iso strings to simulate DB storage
        task_dict["scheduled_start"] = task_dict["scheduled_start"].isoformat()
        task_dict["scheduled_end"] = task_dict["scheduled_end"].isoformat()
        mock_db_adapter.find_one.return_value = task_dict

        # Get the task
        result = scheduling_repo.get_scheduled_task(task_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once_with(
            "scheduled_tasks", {"task_id": task_id})

        # Verify result
        assert result is not None
        assert result.task_id == task_id
        assert result.title == "Review code PR #1234"
        assert result.status == "pending"
        # Verify date conversion
        assert isinstance(result.scheduled_start, datetime)
        assert isinstance(result.scheduled_end, datetime)

    def test_get_scheduled_task_not_found(self, scheduling_repo, mock_db_adapter):
        """Test retrieving a non-existent task."""
        task_id = "nonexistent"

        # Configure mock to return None
        mock_db_adapter.find_one.return_value = None

        # Get the task
        result = scheduling_repo.get_scheduled_task(task_id)

        # Verify DB query
        mock_db_adapter.find_one.assert_called_once_with(
            "scheduled_tasks", {"task_id": task_id})

        # Verify result
        assert result is None

    def test_create_scheduled_task_new(self, scheduling_repo, mock_db_adapter, sample_task):
        """Test creating a new scheduled task."""
        # Configure mock to simulate task not existing
        mock_db_adapter.find_one.return_value = None

        # Create task
        result_id = scheduling_repo.create_scheduled_task(sample_task)

        # Verify result
        assert result_id == sample_task.task_id

        # Verify DB operations
        mock_db_adapter.find_one.assert_called_once_with(
            "scheduled_tasks", {"task_id": sample_task.task_id})
        mock_db_adapter.insert_one.assert_called_once()
        collection, data = mock_db_adapter.insert_one.call_args[0]

        assert collection == "scheduled_tasks"
        assert data["task_id"] == sample_task.task_id
        assert data["title"] == "Review code PR #1234"
        assert data["priority"] == 2
        # Verify conversion to string
        assert isinstance(data["scheduled_start"], str)
        assert isinstance(data["scheduled_end"], str)
        assert data["metadata"]["repository"] == "main-app"

    def test_create_scheduled_task_existing(self, scheduling_repo, mock_db_adapter, sample_task):
        """Test creating a task when it already exists (should update)."""
        # Configure mock to simulate task already existing
        mock_db_adapter.find_one.return_value = {
            "task_id": sample_task.task_id}

        # Create/update task
        result_id = scheduling_repo.create_scheduled_task(sample_task)

        # Verify result
        assert result_id == sample_task.task_id

        # Verify DB operations
        mock_db_adapter.find_one.assert_called_once()
        mock_db_adapter.update_one.assert_called_once()
        mock_db_adapter.insert_one.assert_not_called()

    def test_update_scheduled_task(self, scheduling_repo, mock_db_adapter):
        """Test updating a scheduled task."""
        task_id = "task_123"
        updates = {
            "status": "in_progress",
            "scheduled_start": datetime.now() + timedelta(hours=2)
        }

        # Update the task
        result = scheduling_repo.update_scheduled_task(task_id, updates)

        # Verify result
        assert result is True

        # Verify DB operation
        mock_db_adapter.update_one.assert_called_once()
        collection, query, update = mock_db_adapter.update_one.call_args[0]

        assert collection == "scheduled_tasks"
        assert query == {"task_id": task_id}
        assert "$set" in update
        assert update["$set"]["status"] == "in_progress"
        # Verify conversion to string
        assert isinstance(update["$set"]["scheduled_start"], str)

    def test_delete_scheduled_task(self, scheduling_repo, mock_db_adapter):
        """Test deleting a scheduled task."""
        task_id = "task_123"

        # Delete the task
        result = scheduling_repo.delete_scheduled_task(task_id)

        # Verify result
        assert result is True

        # Verify DB operation
        mock_db_adapter.delete_one.assert_called_once_with(
            "scheduled_tasks", {"task_id": task_id})

    def test_get_tasks_by_status(self, scheduling_repo, mock_db_adapter, sample_task):
        """Test getting tasks by status."""
        status = "pending"

        # Prepare task dict with iso dates
        task_dict = sample_task.model_dump()
        task_dict["scheduled_start"] = task_dict["scheduled_start"].isoformat()
        task_dict["scheduled_end"] = task_dict["scheduled_end"].isoformat()

        # Configure mock to return tasks
        mock_db_adapter.find.return_value = [task_dict]

        # Get tasks
        results = scheduling_repo.get_tasks_by_status(status)

        # Verify DB query
        mock_db_adapter.find.assert_called_once_with(
            "scheduled_tasks", {"status": status})

        # Verify results
        assert len(results) == 1
        assert results[0].task_id == sample_task.task_id
        assert results[0].status == status
        assert isinstance(results[0].scheduled_start,
                          datetime)  # Verify date conversion
        assert isinstance(results[0].scheduled_end, datetime)

    def test_get_agent_tasks(self, scheduling_repo, mock_db_adapter, sample_task):
        """Test getting tasks for an agent within a time period."""
        agent_id = "agent_123"
        start_date = datetime.now()
        end_date = start_date + timedelta(days=2)

        # Prepare task dict with iso dates
        task_dict = sample_task.model_dump()
        task_dict["scheduled_start"] = task_dict["scheduled_start"].isoformat()
        task_dict["scheduled_end"] = task_dict["scheduled_end"].isoformat()

        # Configure mock to return tasks
        mock_db_adapter.find.return_value = [task_dict]

        # Get agent tasks
        results = scheduling_repo.get_agent_tasks(
            agent_id, start_date, end_date)

        # Verify DB query was constructed correctly
        mock_db_adapter.find.assert_called_once()
        collection, query = mock_db_adapter.find.call_args[0]

        assert collection == "scheduled_tasks"
        assert query["assigned_to"] == agent_id
        assert "$or" in query
        assert len(query["$or"]) == 3  # Three time overlap conditions

        # Verify results
        assert len(results) == 1
        assert results[0].task_id == sample_task.task_id
        assert results[0].assigned_to == agent_id

    def test_get_unscheduled_tasks(self, scheduling_repo, mock_db_adapter):
        """Test getting unscheduled tasks."""
        # Create a sample unscheduled task
        unscheduled_task = {
            "task_id": "task_unscheduled",
            "title": "Unscheduled task",
            "status": "pending",
            "assigned_to": None,
            "priority": 3
        }

        # Configure mock to return tasks
        mock_db_adapter.find.return_value = [unscheduled_task]

        # Get unscheduled tasks
        results = scheduling_repo.get_unscheduled_tasks()

        # Verify DB query
        mock_db_adapter.find.assert_called_once()
        collection, query = mock_db_adapter.find.call_args[0]

        assert collection == "scheduled_tasks"
        assert "$or" in query
        assert len(query["$or"]) == 4  # Four conditions for unscheduled tasks

        # Verify results
        assert len(results) == 1
        assert results[0].task_id == "task_unscheduled"
        assert results[0].title == "Unscheduled task"
        assert results[0].assigned_to is None

    # AgentSchedule Tests
    def test_get_agent_schedule_found(self, scheduling_repo, mock_db_adapter, sample_schedule):
        """Test retrieving an existing agent schedule."""
        agent_id = sample_schedule.agent_id
        schedule_date = date.today() + timedelta(days=1)  # Define the date explicitly

        # Prepare schedule dict with proper serialization
        schedule_dict = sample_schedule.model_dump()

        # Convert availability_patterns to a serializable format
        for pattern in schedule_dict["availability_patterns"]:
            pattern["start_time"] = pattern["start_time"].isoformat()
            pattern["end_time"] = pattern["end_time"].isoformat()

        # Configure mock to return the schedule
        mock_db_adapter.find_one.return_value = schedule_dict

        # Get the schedule
        result = scheduling_repo.get_agent_schedule(agent_id, schedule_date)

        # Verify DB query - check working_hours.schedule_date, not date directly
        mock_db_adapter.find_one.assert_called_once()
        args = mock_db_adapter.find_one.call_args[0]
        assert args[0] == "agent_schedules"
        assert "agent_id" in args[1]
        assert "working_hours.schedule_date" in args[1]

        # Verify result
        assert result is not None
        assert result.agent_id == agent_id
        assert result.working_hours is not None
        assert result.working_hours["schedule_date"] == schedule_date.isoformat(
        )
        assert len(result.availability_patterns) == 2

    def test_get_agent_schedule_not_found(self, scheduling_repo, mock_db_adapter):
        """Test retrieving a non-existent agent schedule."""
        agent_id = "nonexistent"
        schedule_date = date.today()

        # Configure mock to return None
        mock_db_adapter.find_one.return_value = None

        # Get the schedule
        result = scheduling_repo.get_agent_schedule(agent_id, schedule_date)

        # Verify DB query - update to match the actual implementation
        mock_db_adapter.find_one.assert_called_once_with(
            "agent_schedules", {"agent_id": agent_id, "working_hours.schedule_date": schedule_date.isoformat()})

        # Verify result
        assert result is None

    def test_get_all_agent_schedules(self, scheduling_repo, mock_db_adapter, sample_schedule):
        """Test getting all agent schedules for a specific date."""
        schedule_date = date.today() + timedelta(days=1)

        # Create multiple agent schedules
        schedule_dict1 = sample_schedule.model_dump()
        # Convert availability patterns for JSON serialization
        for pattern in schedule_dict1["availability_patterns"]:
            pattern["start_time"] = pattern["start_time"].isoformat()
            pattern["end_time"] = pattern["end_time"].isoformat()

        schedule_dict2 = sample_schedule.model_dump()
        schedule_dict2["agent_id"] = "agent_456"
        # Convert availability patterns for JSON serialization
        for pattern in schedule_dict2["availability_patterns"]:
            pattern["start_time"] = pattern["start_time"].isoformat()
            pattern["end_time"] = pattern["end_time"].isoformat()

        # Configure mock to return schedules
        mock_db_adapter.find.return_value = [schedule_dict1, schedule_dict2]

        # Get all schedules
        results = scheduling_repo.get_all_agent_schedules(schedule_date)

        # Verify DB query uses working_hours.schedule_date
        mock_db_adapter.find.assert_called_once()
        args = mock_db_adapter.find.call_args[0]
        assert args[0] == "agent_schedules"
        assert "working_hours.schedule_date" in args[1]

        # Verify results
        assert len(results) == 2
        assert results[0].agent_id == "agent_123"
        assert results[1].agent_id == "agent_456"

    def test_save_agent_schedule_new(self, scheduling_repo, mock_db_adapter, sample_schedule):
        """Test saving a new agent schedule."""
        # Configure mock to simulate schedule not existing
        mock_db_adapter.find_one.return_value = None

        # Save schedule
        result = scheduling_repo.save_agent_schedule(sample_schedule)

        # Verify result
        assert result is True

        # Verify DB operations
        mock_db_adapter.find_one.assert_called_once()
        mock_db_adapter.insert_one.assert_called_once()

        # Verify the data structure being saved
        collection, data = mock_db_adapter.insert_one.call_args[0]
        assert collection == "agent_schedules"
        assert data["agent_id"] == sample_schedule.agent_id
        assert "working_hours" in data
        assert "schedule_date" in data["working_hours"]
        # Verify availability patterns are serialized
        assert isinstance(data["availability_patterns"], list)
        for pattern in data["availability_patterns"]:
            assert isinstance(pattern["start_time"], str)
            assert isinstance(pattern["end_time"], str)

    def test_save_agent_schedule_existing(self, scheduling_repo, mock_db_adapter, sample_schedule):
        """Test updating an existing agent schedule."""
        # Configure mock to simulate schedule already existing
        mock_db_adapter.find_one.return_value = {
            "_id": "existing_id", "agent_id": sample_schedule.agent_id}

        # Save/update schedule
        result = scheduling_repo.save_agent_schedule(sample_schedule)

        # Verify result
        assert result is True

        # Verify DB operations
        mock_db_adapter.find_one.assert_called_once()
        mock_db_adapter.update_one.assert_called_once()
        mock_db_adapter.insert_one.assert_not_called()

    def test_date_conversions(self, scheduling_repo):
        """Test date conversion utility methods."""
        # Test _convert_task_dates
        task_doc = {
            "task_id": "task_123",
            "scheduled_start": "2025-03-18T12:00:00",
            "scheduled_end": "2025-03-18T13:00:00"
        }

        scheduling_repo._convert_task_dates(task_doc)

        assert isinstance(task_doc["scheduled_start"], datetime)
        assert isinstance(task_doc["scheduled_end"], datetime)
        assert task_doc["scheduled_start"].hour == 12
        assert task_doc["scheduled_end"].hour == 13

        # Test _prepare_task_dates
        task_doc = {
            "task_id": "task_123",
            "scheduled_start": datetime(2025, 3, 18, 12, 0),
            "scheduled_end": datetime(2025, 3, 18, 13, 0)
        }

        scheduling_repo._prepare_task_dates(task_doc)

        assert isinstance(task_doc["scheduled_start"], str)
        assert isinstance(task_doc["scheduled_end"], str)
        assert "2025-03-18T12:00:00" in task_doc["scheduled_start"]
        assert "2025-03-18T13:00:00" in task_doc["scheduled_end"]

"""
Scheduling service implementation.

This service manages task scheduling, agent availability, and
coordination of work across the system.
"""
import uuid
import datetime
from typing import Dict, List, Optional, Any, Tuple

from solana_agent.interfaces import SchedulingService as SchedulingServiceInterface
from solana_agent.interfaces import TaskPlanningService, AgentService, ResourceService
from solana_agent.interfaces import SchedulingRepository
from solana_agent.domains import (
    AgentAvailabilityPattern, ScheduledTask, AgentSchedule, SchedulingEvent,
    TimeWindow, TimeOffRequest, TimeOffStatus
)


class SchedulingService(SchedulingServiceInterface):
    """Service for intelligent task scheduling and agent coordination."""

    def __init__(
        self,
        scheduling_repository: SchedulingRepository,
        task_planning_service: Optional[TaskPlanningService] = None,
        agent_service: Optional[AgentService] = None
    ):
        """Initialize the scheduling service.

        Args:
            scheduling_repository: Repository for scheduling data
            task_planning_service: Optional task planning service
            agent_service: Optional agent service
        """
        self.repository = scheduling_repository
        self.task_planning_service = task_planning_service
        self.agent_service = agent_service

    async def schedule_task(
        self,
        task: ScheduledTask,
        preferred_agent_id: str = None
    ) -> ScheduledTask:
        """Schedule a task with optimal time and agent assignment.

        Args:
            task: Task to schedule
            preferred_agent_id: Preferred agent ID

        Returns:
            Scheduled task
        """
        # First check if task already has a fixed schedule
        if task.scheduled_start and task.scheduled_end and task.assigned_to:
            # Task is already fully scheduled, just save it
            self.repository.update_scheduled_task(task)
            return task

        # Find best agent for task based on specialization and availability
        if not task.assigned_to:
            task.assigned_to = await self._find_optimal_agent(task, preferred_agent_id)

        # Find optimal time slot
        if not (task.scheduled_start and task.scheduled_end):
            time_window = await self._find_optimal_time_slot(task)
            if time_window:
                task.scheduled_start = time_window.start
                task.scheduled_end = time_window.end

        # Update task status
        if task.status == "pending":
            task.status = "scheduled"

        # Save the scheduled task
        self.repository.update_scheduled_task(task)

        # Log scheduling event
        self._log_scheduling_event(
            "task_scheduled",
            task.task_id,
            task.assigned_to,
            {"scheduled_start": task.scheduled_start.isoformat()
             if task.scheduled_start else None}
        )

        return task

    async def find_optimal_time_slot_with_resources(
        self,
        task: ScheduledTask,
        resource_service: ResourceService,
        agent_schedule: Optional[AgentSchedule] = None
    ) -> Optional[TimeWindow]:
        """Find the optimal time slot for a task based on both agent and resource availability.

        Args:
            task: Task to schedule
            resource_service: Resource service for checking resource availability
            agent_schedule: Optional cached agent schedule

        Returns:
            Optimal time window or None if not found
        """
        if not task.assigned_to:
            return None

        # First, find potential time slots based on agent availability
        agent_id = task.assigned_to
        duration = task.estimated_minutes or 30

        # Start no earlier than now
        start_after = datetime.datetime.now(datetime.timezone.utc)

        # Apply task constraints
        for constraint in task.constraints:
            if constraint.get("type") == "must_start_after" and constraint.get("time"):
                constraint_time = datetime.datetime.fromisoformat(
                    constraint["time"])
                if constraint_time > start_after:
                    start_after = constraint_time

        # Get potential time slots for the agent
        agent_slots = await self.find_available_time_slots(
            agent_id,
            duration,
            start_after,
            count=3,  # Get multiple slots to try with resources
            agent_schedule=agent_schedule
        )

        if not agent_slots:
            return None

        # Check if task has resource requirements
        required_resources = getattr(task, "required_resources", [])
        if not required_resources:
            # If no resources needed, return the first available agent slot
            return agent_slots[0]

        # For each potential time slot, check resource availability
        for time_slot in agent_slots:
            all_resources_available = True

            for resource_req in required_resources:
                resource_type = resource_req.get("resource_type")
                requirements = resource_req.get("requirements", "")
                quantity = resource_req.get("quantity", 1)

                # Find available resources for this time slot
                resources = await resource_service.find_available_resources(
                    start_time=time_slot.start,
                    end_time=time_slot.end,
                    resource_type=resource_type,
                    tags=requirements.split() if requirements else None
                )

                if len(resources) < quantity:
                    all_resources_available = False
                    break

            # If all resources are available, use this time slot
            if all_resources_available:
                return time_slot

        # If no time slot has all resources available, default to first slot
        return agent_slots[0]

    async def optimize_schedule(self) -> Dict[str, Any]:
        """Optimize the entire schedule to maximize efficiency.

        Returns:
            Optimization results
        """
        # Get all pending and scheduled tasks
        pending_tasks = self.repository.get_unscheduled_tasks()
        scheduled_tasks = self.repository.get_tasks_by_status("scheduled")

        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks_by_priority_and_dependencies(
            pending_tasks + scheduled_tasks
        )

        # Get all agent schedules
        agent_schedules = self.repository.get_all_agent_schedules()
        agent_schedule_map = {
            schedule.agent_id: schedule for schedule in agent_schedules}

        # Track changes for reporting
        changes = {
            "rescheduled_tasks": [],
            "reassigned_tasks": [],
            "unresolvable_conflicts": []
        }

        # Process each task in priority order
        for task in sorted_tasks:
            # Skip completed tasks
            if task.status in ["completed", "cancelled"]:
                continue

            original_agent = task.assigned_to
            original_start = task.scheduled_start

            # Find optimal agent and time
            best_agent_id = await self._find_optimal_agent(task)

            # If agent changed, update assignment
            if best_agent_id != original_agent and best_agent_id is not None:
                task.assigned_to = best_agent_id
                changes["reassigned_tasks"].append({
                    "task_id": task.task_id,
                    "original_agent": original_agent,
                    "new_agent": best_agent_id
                })

            # Find best time slot for this agent
            if task.assigned_to:
                # Use the agent's schedule from our map if available
                agent_schedule = agent_schedule_map.get(task.assigned_to)

                # Find optimal time considering the agent's schedule - pass the cached schedule
                time_window = await self._find_optimal_time_slot(task, agent_schedule)

                if time_window and (
                    not original_start or
                    time_window.start != original_start
                ):
                    task.scheduled_start = time_window.start
                    task.scheduled_end = time_window.end
                    changes["rescheduled_tasks"].append({
                        "task_id": task.task_id,
                        "original_time": original_start.isoformat() if original_start else None,
                        "new_time": time_window.start.isoformat()
                    })
            else:
                changes["unresolvable_conflicts"].append({
                    "task_id": task.task_id,
                    "reason": "No suitable agent found"
                })

            # Save changes
            self.repository.update_scheduled_task(task)

        return changes

    async def register_agent_schedule(self, schedule: AgentSchedule) -> bool:
        """Register or update an agent's schedule.

        Args:
            schedule: Agent schedule

        Returns:
            True if successful
        """
        return self.repository.save_agent_schedule(schedule)

    async def get_agent_schedule(self, agent_id: str) -> Optional[AgentSchedule]:
        """Get an agent's schedule.

        Args:
            agent_id: Agent ID

        Returns:
            Agent schedule or None if not found
        """
        return self.repository.get_agent_schedule(agent_id)

    async def get_agent_tasks(
        self,
        agent_id: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        include_completed: bool = False
    ) -> List[ScheduledTask]:
        """Get all tasks scheduled for an agent within a time range.

        Args:
            agent_id: Agent ID
            start_time: Optional start time filter
            end_time: Optional end time filter
            include_completed: Whether to include completed tasks

        Returns:
            List of tasks
        """
        return self.repository.get_agent_tasks(agent_id=agent_id, start_time=start_time, end_time=end_time, include_completed=include_completed)

    async def mark_task_started(self, task_id: str) -> bool:
        """Mark a task as started.

        Args:
            task_id: Task ID

        Returns:
            True if successful
        """
        task = self.repository.get_scheduled_task(task_id)
        if not task:
            return False

        task.status = "in_progress"
        task.actual_start = datetime.datetime.now(datetime.timezone.utc)
        self.repository.update_scheduled_task(task)

        self._log_scheduling_event(
            "task_started",
            task_id,
            task.assigned_to,
            {"actual_start": task.actual_start.isoformat()}
        )

        return True

    async def mark_task_completed(self, task_id: str) -> bool:
        """Mark a task as completed.

        Args:
            task_id: Task ID

        Returns:
            True if successful
        """
        task = self.repository.get_scheduled_task(task_id)
        if not task:
            return False

        task.status = "completed"
        task.actual_end = datetime.datetime.now(datetime.timezone.utc)
        self.repository.update_scheduled_task(task)

        # Calculate metrics
        duration_minutes = 0
        if task.actual_start:
            duration_minutes = int(
                (task.actual_end - task.actual_start).total_seconds() / 60)

        estimated_minutes = task.estimated_minutes or 0
        accuracy = 0
        if estimated_minutes > 0 and duration_minutes > 0:
            # Calculate how accurate the estimate was (1.0 = perfect, <1.0 = underestimate, >1.0 = overestimate)
            accuracy = estimated_minutes / duration_minutes

        self._log_scheduling_event(
            "task_completed",
            task_id,
            task.assigned_to,
            {
                "actual_end": task.actual_end.isoformat(),
                "duration_minutes": duration_minutes,
                "estimated_minutes": estimated_minutes,
                "estimate_accuracy": accuracy
            }
        )

        return True

    async def find_available_time_slots(
        self,
        agent_id: str,
        duration_minutes: int,
        start_after: datetime.datetime = None,
        end_before: datetime.datetime = None,
        count: int = 3,
        agent_schedule: Optional[AgentSchedule] = None
    ) -> List[TimeWindow]:
        """Find available time slots for an agent.

        Args:
            agent_id: Agent ID
            duration_minutes: Task duration in minutes
            start_after: Don't look for slots before this time
            end_before: Don't look for slots after this time
            count: Maximum number of slots to return
            agent_schedule: Optional cached agent schedule

        Returns:
            List of available time windows
        """
        # Default time bounds
        if not start_after:
            start_after = datetime.datetime.now(datetime.timezone.utc)
        if not end_before:
            end_before = start_after + datetime.timedelta(days=7)

        # Get agent schedule if not provided
        if not agent_schedule:
            agent_schedule = self.repository.get_agent_schedule(agent_id)

        if not agent_schedule:
            # If no schedule exists, assume standard business hours
            default_schedule = AgentSchedule(
                agent_id=agent_id,
                availability_patterns=[
                    # Monday-Friday, 9am-5pm
                    AgentAvailabilityPattern(
                        day_of_week=i,
                        start_time=datetime.time(9, 0),
                        end_time=datetime.time(17, 0)
                    ) for i in range(5)  # 0=Monday through 4=Friday
                ]
            )
            agent_schedule = default_schedule

        # Get all tasks in the time range
        existing_tasks = self.repository.get_agent_tasks(
            agent_id,
            start_after,
            end_before,
        )

        # Convert to time windows
        blocked_windows = []
        for task in existing_tasks:
            if task.scheduled_start and task.scheduled_end:
                blocked_windows.append(TimeWindow(
                    start=task.scheduled_start,
                    end=task.scheduled_end
                ))

        # Find slots with sufficient duration where the agent is available
        available_slots = []
        current_time = start_after

        # Look for slots until we have enough or reach end_before
        while len(available_slots) < count and current_time < end_before:
            # Check if agent is available at this time according to their schedule
            if agent_schedule.is_available_at(current_time):
                # Calculate potential slot end time
                slot_end = current_time + \
                    datetime.timedelta(minutes=duration_minutes)

                # Check if slot fits within schedule and before end_before
                if slot_end <= end_before:
                    # Check if agent is continuously available during this slot
                    is_available = True

                    # Check every hour during the slot
                    check_time = current_time
                    while check_time <= slot_end:
                        if not agent_schedule.is_available_at(check_time):
                            is_available = False
                            break
                        check_time += datetime.timedelta(hours=1)

                    # Check if slot overlaps with any blocked windows
                    if is_available:
                        potential_slot = TimeWindow(
                            start=current_time, end=slot_end)

                        for blocked in blocked_windows:
                            if potential_slot.overlaps_with(blocked):
                                is_available = False
                                # Move time pointer to end of this blocked slot
                                current_time = blocked.end
                                break

                        if is_available:
                            available_slots.append(potential_slot)
                            # Move time pointer to end of this slot for next iteration
                            current_time = slot_end
                            continue

            # If we get here, either not available or we found a slot
            # Increment by 30 minutes and try again
            current_time += datetime.timedelta(minutes=30)

        return available_slots

    async def resolve_scheduling_conflicts(self) -> Dict[str, Any]:
        """Detect and resolve scheduling conflicts.

        Returns:
            Conflict resolution results
        """
        # Get all scheduled tasks
        tasks = self.repository.get_tasks_by_status("scheduled")

        # Group tasks by agent
        agent_tasks = {}
        for task in tasks:
            if task.assigned_to:
                if task.assigned_to not in agent_tasks:
                    agent_tasks[task.assigned_to] = []
                agent_tasks[task.assigned_to].append(task)

        # Check for conflicts within each agent's schedule
        conflicts = []
        for agent_id, agent_task_list in agent_tasks.items():
            # Sort tasks by start time
            agent_task_list.sort(
                key=lambda t: t.scheduled_start or datetime.datetime.max)

            # Check for overlaps
            for i in range(len(agent_task_list) - 1):
                current = agent_task_list[i]
                next_task = agent_task_list[i + 1]

                if (current.scheduled_start and current.scheduled_end and
                        next_task.scheduled_start and next_task.scheduled_end):

                    current_window = TimeWindow(
                        start=current.scheduled_start, end=current.scheduled_end)
                    next_window = TimeWindow(
                        start=next_task.scheduled_start, end=next_task.scheduled_end)

                    if current_window.overlaps_with(next_window):
                        conflicts.append({
                            "agent_id": agent_id,
                            "task1": current.task_id,
                            "task2": next_task.task_id,
                            "start1": current.scheduled_start.isoformat(),
                            "end1": current.scheduled_end.isoformat(),
                            "start2": next_task.scheduled_start.isoformat(),
                            "end2": next_task.scheduled_end.isoformat()
                        })

                        # Try to resolve by moving the second task later
                        next_task.scheduled_start = current.scheduled_end
                        next_task.scheduled_end = next_task.scheduled_start + datetime.timedelta(
                            minutes=next_task.estimated_minutes or 30
                        )
                        self.repository.update_scheduled_task(next_task)

        # Log conflict resolution
        if conflicts:
            self._log_scheduling_event(
                "conflicts_resolved",
                None,
                None,
                {"conflict_count": len(conflicts)}
            )

        return {"conflicts_found": len(conflicts), "conflicts": conflicts}

    async def request_time_off(
        self,
        agent_id: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        reason: str = ""
    ) -> Tuple[bool, str, Optional[TimeOffRequest]]:
        """Request time off for an agent.

        Args:
            agent_id: Agent ID
            start_time: Start time
            end_time: End time
            reason: Optional reason for time off

        Returns:
            Tuple of (success, message, request_object)
        """
        # Validate time range
        if start_time >= end_time:
            return False, "Start time must be before end time", None

        if start_time < datetime.datetime.now(datetime.timezone.utc):
            return False, "Time off cannot be requested in the past", None

        # Check for conflicts with existing schedules
        conflicts = self.repository.find_conflicting_tasks(
            agent_id, start_time, end_time)

        # Create time off request
        request = TimeOffRequest(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time,
            reason=reason,
            status=TimeOffStatus.PENDING,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            conflicts=[task.task_id for task in conflicts]
        )

        # Store the request
        success = self.repository.save_time_off_request(request)

        # Log the request
        self._log_scheduling_event(
            "time_off_requested",
            None,
            agent_id,
            {
                "request_id": request.id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "conflicts": len(conflicts)
            }
        )

        # Return result
        conflict_message = ""
        if conflicts:
            conflict_message = f"Warning: Found {len(conflicts)} conflicting tasks."
        return success, conflict_message, request if success else None

    async def approve_time_off(self, request_id: str) -> bool:
        """Approve a time off request.

        Args:
            request_id: Time off request ID

        Returns:
            True if approval was successful
        """
        # Get request
        request = self.repository.get_time_off_request(request_id)
        if not request:
            return False

        # Update status
        request.status = TimeOffStatus.APPROVED
        request.processed_at = datetime.datetime.now(datetime.timezone.utc)
        success = self.repository.save_time_off_request(request)

        if success:
            # Update agent schedule
            agent_schedule = self.repository.get_agent_schedule(
                request.agent_id)
            if agent_schedule:
                # Add time off period to agent schedule
                agent_schedule.time_off_periods.append({
                    "start": request.start_time,
                    "end": request.end_time,
                    "reason": request.reason
                })
                self.repository.save_agent_schedule(agent_schedule)

            # Log the approval
            self._log_scheduling_event(
                "time_off_approved",
                None,
                request.agent_id,
                {"request_id": request.id}
            )

        return success

    async def deny_time_off(self, request_id: str, reason: str = "") -> bool:
        """Deny a time off request.

        Args:
            request_id: Time off request ID
            reason: Reason for denial

        Returns:
            True if denial was successful
        """
        # Get request
        request = self.repository.get_time_off_request(request_id)
        if not request:
            return False

        # Update status
        request.status = TimeOffStatus.DENIED
        request.processed_at = datetime.datetime.now(datetime.timezone.utc)
        request.denial_reason = reason
        success = self.repository.save_time_off_request(request)

        if success:
            # Log the denial
            self._log_scheduling_event(
                "time_off_denied",
                None,
                request.agent_id,
                {"request_id": request.id, "reason": reason}
            )

        return success

    async def get_time_off_requests(
        self,
        agent_id: Optional[str] = None,
        status: Optional[TimeOffStatus] = None
    ) -> List[TimeOffRequest]:
        """Get time off requests, optionally filtered by agent and status.

        Args:
            agent_id: Optional agent ID filter
            status: Optional status filter

        Returns:
            List of time off requests
        """
        return self.repository.get_time_off_requests(agent_id, status)

    async def cancel_time_off_request(self, request_id: str, reason: str = "") -> bool:
        """Cancel a pending time off request.

        Args:
            request_id: Time off request ID
            reason: Optional reason for cancellation

        Returns:
            True if cancellation was successful
        """
        # Get the request
        request = self.repository.get_time_off_request(request_id)
        if not request:
            return False

        # Can only cancel pending requests
        if request.status != TimeOffStatus.PENDING:
            return False

        # Update status
        request.status = TimeOffStatus.CANCELLED
        request.processed_at = datetime.datetime.now(datetime.timezone.utc)
        request.cancellation_reason = reason
        success = self.repository.save_time_off_request(request)

        if success:
            # Log the cancellation
            self._log_scheduling_event(
                "time_off_cancelled",
                None,
                request.agent_id,
                {"request_id": request.id, "reason": reason}
            )

        return success

    async def get_agent_time_off_history(
        self,
        agent_id: str,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        include_denied: bool = False,
        include_cancelled: bool = False
    ) -> List[TimeOffRequest]:
        """Get time off history for an agent with filtering options.

        Args:
            agent_id: Agent ID
            start_date: Optional start date filter
            end_date: Optional end date filter
            include_denied: Whether to include denied requests
            include_cancelled: Whether to include cancelled requests

        Returns:
            List of time off requests matching the criteria
        """
        # Get all time off requests for the agent
        all_requests = self.repository.get_time_off_requests(agent_id, None)

        # Filter by date if needed
        if start_date or end_date:
            filtered_requests = []

            for request in all_requests:
                request_start_date = request.start_time.date()
                request_end_date = request.end_time.date()

                # Filter by start date
                if start_date and request_end_date < start_date:
                    continue

                # Filter by end date
                if end_date and request_start_date > end_date:
                    continue

                filtered_requests.append(request)

            all_requests = filtered_requests

        # Filter by status
        status_filtered_requests = []
        for request in all_requests:
            # Always include approved and pending requests
            if request.status in [TimeOffStatus.APPROVED, TimeOffStatus.PENDING]:
                status_filtered_requests.append(request)
            # Include denied requests if requested
            elif request.status == TimeOffStatus.DENIED and include_denied:
                status_filtered_requests.append(request)
            # Include cancelled requests if requested
            elif request.status == TimeOffStatus.CANCELLED and include_cancelled:
                status_filtered_requests.append(request)

        # Sort by start date, most recent first
        status_filtered_requests.sort(
            key=lambda r: r.start_time,
            reverse=True
        )

        return status_filtered_requests

    async def _find_optimal_agent(
        self,
        task: ScheduledTask,
        preferred_agent_id: Optional[str] = None
    ) -> Optional[str]:
        """Find the optimal agent for a task based on specialization and availability.

        Args:
            task: Task to assign
            preferred_agent_id: Optional preferred agent ID

        Returns:
            Best agent ID or None if not found
        """
        # If there's a preferred agent, check if they're suitable first
        if preferred_agent_id:
            # Check agent availability if we have an estimated task duration
            if task.estimated_minutes and self._is_agent_available_for_task(preferred_agent_id, task):
                return preferred_agent_id
            elif not task.estimated_minutes:
                # No duration info, but still check their current workload
                # Arbitrary threshold
                if self._get_agent_current_workload(preferred_agent_id) < 3:
                    return preferred_agent_id

        # Get potential agents based on specialization
        potential_agents = []
        specialization = getattr(task, "specialization", None)

        if specialization and self.agent_service:
            potential_agents = self.agent_service.find_agents_by_specialization(
                specialization)

        if not potential_agents and self.agent_service:
            # Fall back to all available agents
            potential_agents = self.agent_service.list_active_agents()

        # If we still have no agents, we can't proceed
        if not potential_agents:
            return None

        # Score each agent based on multiple factors
        agent_scores = {}
        now = datetime.datetime.now(datetime.timezone.utc)

        for agent_id in potential_agents:
            # Skip if agent doesn't exist or is inactive
            if not self.agent_service.agent_exists(agent_id):
                continue

            # Base score starts at 0
            score = 0.0

            # Factor 1: Specialization match
            if specialization and self.agent_service.has_specialization(agent_id, specialization):
                score += 10.0

            # Factor 2: Current workload
            workload = self._get_agent_current_workload(agent_id)
            score -= workload * 2  # Penalty for higher workload

            # Factor 3: Availability for this task
            if task.scheduled_start and task.estimated_minutes:
                # For tasks with predefined times
                if self._is_agent_available_for_task(agent_id, task):
                    score += 5.0
                else:
                    # Major penalty for conflicts
                    score -= 20.0
            else:
                # For tasks without scheduled times, check for nearest available slot
                next_slot = await self._find_next_available_slot(agent_id, task.estimated_minutes or 30)
                if next_slot:
                    # Prefer agents who can start sooner
                    time_until_available = (
                        # hours
                        next_slot.start - now).total_seconds() / 3600.0
                    score -= time_until_available  # Smaller penalty for agents available sooner
                else:
                    # No available slots found in the scheduling window
                    score -= 15.0

            # Factor 4: Performance metrics (if tracked)
            if hasattr(self.agent_service, "get_agent_performance"):
                try:
                    performance = await self.agent_service.get_agent_performance(agent_id)
                    if performance and "success_rate" in performance:
                        score += performance["success_rate"] * 5.0
                except Exception:
                    # Ignore errors in performance tracking
                    pass

            # Factor 5: Recent experience with similar tasks
            if hasattr(self.repository, "get_agent_task_history"):
                try:
                    similar_tasks = self.repository.get_agent_task_history(
                        agent_id,
                        specialization=specialization,
                        limit=5
                    )
                    if similar_tasks and len(similar_tasks) > 0:
                        score += len(similar_tasks) * \
                            0.5  # Bonus for experience
                except Exception:
                    # Ignore errors in history tracking
                    pass

            # Store the final score
            agent_scores[agent_id] = score

        # Find the agent with the highest score
        if not agent_scores:
            return None

        best_agent = max(agent_scores.items(), key=lambda x: x[1])

        return best_agent[0]  # Return the agent ID

    def _get_agent_current_workload(self, agent_id: str) -> int:
        """Get the current workload for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Number of active and scheduled tasks
        """
        try:
            # Count scheduled and in-progress tasks
            tasks = self.repository.get_agent_tasks(
                agent_id,
                start_time=datetime.datetime.now(datetime.timezone.utc),
            )

            # Only count relevant tasks (not completed or cancelled)
            active_tasks = [t for t in tasks if t.status in [
                "scheduled", "in_progress"]]

            return len(active_tasks)
        except Exception as e:
            print(f"Error getting agent workload: {e}")
            return 0

    def _is_agent_available_for_task(self, agent_id: str, task: ScheduledTask) -> bool:
        """Check if an agent is available for a specific task.

        Args:
            agent_id: Agent ID
            task: Task to check

        Returns:
            True if agent is available
        """
        if not task.scheduled_start or not task.estimated_minutes:
            return True  # Can't check without timing info

        start_time = task.scheduled_start
        end_time = start_time + \
            datetime.timedelta(minutes=task.estimated_minutes)

        # Check for booking conflicts
        if self.repository._has_conflicting_bookings(agent_id, start_time, end_time):
            return False

        # Check agent schedule/availability patterns
        agent_schedule = self.repository.get_agent_schedule(agent_id)
        if agent_schedule:
            # Check each half hour during the task
            check_time = start_time
            while check_time < end_time:
                if not agent_schedule.is_available_at(check_time):
                    return False
                check_time += datetime.timedelta(minutes=30)

        return True

    async def _find_next_available_slot(
        self,
        agent_id: str,
        duration_minutes: int
    ) -> Optional[TimeWindow]:
        """Find the next available time slot for an agent.

        Args:
            agent_id: Agent ID
            duration_minutes: Required duration

        Returns:
            Next available time slot or None
        """
        start_after = datetime.datetime.now(datetime.timezone.utc)

        # Look up to 7 days ahead
        end_before = start_after + datetime.timedelta(days=7)

        # Use the existing method to find available slots
        slots = await self.find_available_time_slots(
            agent_id,
            duration_minutes,
            start_after,
            end_before,
            count=1
        )

        return slots[0] if slots else None

    async def _find_optimal_time_slot(
        self,
        task: ScheduledTask,
        agent_schedule: Optional[AgentSchedule] = None
    ) -> Optional[TimeWindow]:
        """Find the optimal time slot for a task based on constraints and agent availability.

        Args:
            task: Task to schedule
            agent_schedule: Optional cached agent schedule

        Returns:
            Optimal time window or None if not found
        """
        if not task.assigned_to:
            return None

        # Get estimated duration (default to 30 minutes if not specified)
        duration = task.estimated_minutes or 30

        # Start no earlier than now
        start_after = datetime.datetime.now(datetime.timezone.utc)

        # Apply task constraints
        for constraint in task.constraints:
            if constraint.get("type") == "must_start_after" and constraint.get("time"):
                constraint_time = datetime.datetime.fromisoformat(
                    constraint["time"])
                if constraint_time > start_after:
                    start_after = constraint_time

        # Find available time slots
        available_slots = await self.find_available_time_slots(
            task.assigned_to,
            duration,
            start_after=start_after,
            count=1,
            agent_schedule=agent_schedule
        )

        if available_slots:
            return available_slots[0]

        # No suitable slot found
        return None

    def _sort_tasks_by_priority_and_dependencies(self, tasks: List[ScheduledTask]) -> List[ScheduledTask]:
        """Sort tasks by priority and dependencies.

        Args:
            tasks: List of tasks to sort

        Returns:
            Sorted list of tasks
        """
        # Create a copy of the tasks to sort
        sorted_tasks = tasks.copy()

        # First sort by priority (higher priority first)
        sorted_tasks.sort(key=lambda t: -(t.priority or 0))

        # Then handle dependencies (this is a simplified approach)
        # A more complete solution would use topological sorting for the dependency graph

        # Map task_id to its position in the sorted list
        task_positions = {task.task_id: i for i,
                          task in enumerate(sorted_tasks)}

        # Track if we've made any swaps
        made_swaps = True
        max_iterations = len(sorted_tasks)  # Avoid infinite loop
        iteration = 0

        while made_swaps and iteration < max_iterations:
            made_swaps = False
            iteration += 1

            for i, task in enumerate(sorted_tasks):
                # Check if this task depends on others
                for dep_id in task.depends_on:
                    # Find the dependent task
                    dep_pos = task_positions.get(dep_id)
                    if dep_pos is not None and dep_pos > i:
                        # The dependency is after this task, swap them
                        sorted_tasks[i], sorted_tasks[dep_pos] = sorted_tasks[dep_pos], sorted_tasks[i]

                        # Update positions
                        task_positions[task.task_id] = dep_pos
                        task_positions[sorted_tasks[i].task_id] = i

                        made_swaps = True

        return sorted_tasks

    def _log_scheduling_event(
        self,
        event_type: str,
        task_id: Optional[str],
        agent_id: Optional[str],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a scheduling event.

        Args:
            event_type: Type of event
            task_id: Optional task ID
            agent_id: Optional agent ID
            details: Optional event details
        """
        event = SchedulingEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            task_id=task_id,
            agent_id=agent_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            details=details or {}
        )

        try:
            self.repository.save_scheduling_event(event)
        except Exception as e:
            print(f"Error logging scheduling event: {e}")

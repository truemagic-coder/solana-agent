

class SchedulingService:
    """Service for intelligent task scheduling and agent coordination."""

    def __init__(
        self,
        scheduling_repository: SchedulingRepository,
        task_planning_service: TaskPlanningService = None,
        agent_service: AgentService = None
    ):
        self.repository = scheduling_repository
        self.task_planning_service = task_planning_service
        self.agent_service = agent_service

    async def schedule_task(
        self,
        task: ScheduledTask,
        preferred_agent_id: str = None
    ) -> ScheduledTask:
        """Schedule a task with optimal time and agent assignment."""
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
        """Find the optimal time slot for a task based on both agent and resource availability."""
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
        """Optimize the entire schedule to maximize efficiency."""
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

    async def register_agent_schedule(self, schedule: AgentSchedule) -> bool:
        """Register or update an agent's schedule."""
        return self.repository.save_agent_schedule(schedule)

    async def get_agent_schedule(self, agent_id: str) -> Optional[AgentSchedule]:
        """Get an agent's schedule."""
        return self.repository.get_agent_schedule(agent_id)

    async def get_agent_tasks(
        self,
        agent_id: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        include_completed: bool = False
    ) -> List[ScheduledTask]:
        """Get all tasks scheduled for an agent within a time range."""
        status_filter = None if include_completed else "scheduled"
        return self.repository.get_agent_tasks(agent_id, start_time, end_time, status_filter)

    async def mark_task_started(self, task_id: str) -> bool:
        """Mark a task as started."""
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
        """Mark a task as completed."""
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
        """Find available time slots for an agent."""
        # Default time bounds
        if not start_after:
            start_after = datetime.datetime.now(datetime.timezone.utc)
        if not end_before:
            end_before = start_after + datetime.timedelta(days=7)

        # Get agent schedule if not provided
        if not agent_schedule:
            agent_schedule = self.repository.get_agent_schedule(agent_id)
        if not agent_schedule:
            return []

        # Rest of method unchanged...
    async def resolve_scheduling_conflicts(self) -> Dict[str, Any]:
        """Detect and resolve scheduling conflicts."""
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

    async def _find_optimal_agent(
        self,
        task: ScheduledTask,
        preferred_agent_id: str = None,
        excluded_agents: List[str] = None
    ) -> Optional[str]:
        """Find the optimal agent for a task based on specialization and availability."""
        if not self.agent_service:
            return preferred_agent_id

        # Initialize excluded agents list if not provided
        excluded_agents = excluded_agents or []

        # Get the specializations required for this task
        required_specializations = task.specialization_tags

        # Get all agent specializations
        agent_specializations = self.agent_service.get_specializations()

        # Start with the preferred agent if specified and not excluded
        if preferred_agent_id and preferred_agent_id not in excluded_agents:
            # Check if preferred agent has the required specialization
            if preferred_agent_id in agent_specializations:
                agent_spec = agent_specializations[preferred_agent_id]
                for req_spec in required_specializations:
                    if req_spec.lower() in agent_spec.lower():
                        # Check if the agent is available
                        schedule = self.repository.get_agent_schedule(
                            preferred_agent_id)
                        if schedule and (not task.scheduled_start or schedule.is_available_at(task.scheduled_start)):
                            return preferred_agent_id  # Missing return statement was here

        # Rank all agents based on specialization match and availability
        candidates = []

        # First, check AI agents (they typically have higher availability)
        for agent_id, specialization in agent_specializations.items():
            # Skip excluded agents
            if agent_id in excluded_agents:
                continue

            # Skip if we know it's a human agent (they have different availability patterns)
            is_human = False
            if self.agent_service.human_agent_registry:
                human_agents = self.agent_service.human_agent_registry.get_all_human_agents()
                is_human = agent_id in human_agents

                if is_human:
                    continue

            # Calculate specialization match score
            spec_match_score = 0
            for req_spec in required_specializations:
                if req_spec.lower() in specialization.lower():
                    spec_match_score += 1

            # Only consider agents with at least some specialization match
            if spec_match_score > 0:
                candidates.append({
                    "agent_id": agent_id,
                    "score": spec_match_score,
                    "is_human": is_human
                })

        # Then, check human agents (they typically have more limited availability)
        for agent_id, specialization in agent_specializations.items():
            # Skip excluded agents
            if agent_id in excluded_agents:
                continue

            # Skip if not a human agent
            is_human = False
            if self.agent_service.human_agent_registry:
                human_agents = self.agent_service.human_agent_registry.get_all_human_agents()
                is_human = agent_id in human_agents

                if not is_human:
                    continue

            # Calculate specialization match score
            spec_match_score = 0
            for req_spec in required_specializations:
                if req_spec.lower() in specialization.lower():
                    spec_match_score += 1

            # Only consider agents with at least some specialization match
            if spec_match_score > 0:
                candidates.append({
                    "agent_id": agent_id,
                    "score": spec_match_score,
                    "is_human": is_human
                })

        # Sort candidates by score (descending)
        candidates.sort(key=lambda c: c["score"], reverse=True)

        # Check availability for each candidate
        for candidate in candidates:
            agent_id = candidate["agent_id"]

            # Check if the agent has a schedule
            schedule = self.repository.get_agent_schedule(agent_id)

            # If no schedule or no specific start time yet, assume available
            if not schedule or not task.scheduled_start:
                return agent_id

            # Check availability at the scheduled time
            if schedule.is_available_at(task.scheduled_start):
                return agent_id

        # If no good match found, return None
        return None

    async def _find_optimal_time_slot(
        self,
        task: ScheduledTask,
        agent_schedule: Optional[AgentSchedule] = None
    ) -> Optional[TimeWindow]:
        """Find the optimal time slot for a task based on constraints and agent availability."""
        if not task.assigned_to:
            return None

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

        # Get available slots - use provided agent_schedule if available
        available_slots = await self.find_available_time_slots(
            agent_id,
            duration,
            start_after,
            count=1,
            agent_schedule=agent_schedule
        )

        # Return the first available slot, if any
        return available_slots[0] if available_slots else None

    def _sort_tasks_by_priority_and_dependencies(self, tasks: List[ScheduledTask]) -> List[ScheduledTask]:
        """Sort tasks by priority and dependencies."""
        # First, build dependency graph
        task_map = {task.task_id: task for task in tasks}
        dependency_graph = {task.task_id: set(
            task.dependencies) for task in tasks}

        # Calculate priority score (higher is more important)
        def calculate_priority_score(task):
            base_priority = task.priority or 5

            # Increase priority for tasks with deadlines
            urgency_bonus = 0
            if task.scheduled_end:
                # How soon is the deadline?
                time_until_deadline = (
                    task.scheduled_end - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                # Convert to hours
                hours_remaining = max(0, time_until_deadline / 3600)

                # More urgent as deadline approaches
                if hours_remaining < 24:
                    urgency_bonus = 5  # Very urgent: <24h
                elif hours_remaining < 48:
                    urgency_bonus = 3  # Urgent: 1-2 days
                elif hours_remaining < 72:
                    urgency_bonus = 1  # Somewhat urgent: 2-3 days

            # Increase priority for blocking tasks
            dependency_count = 0
            for other_task_id, deps in dependency_graph.items():
                if task.task_id in deps:
                    dependency_count += 1

            blocking_bonus = min(dependency_count, 5)  # Cap at +5

            return base_priority + urgency_bonus + blocking_bonus

        # Assign priority scores
        for task in tasks:
            task.priority_score = calculate_priority_score(task)

        # Sort by priority score (descending)
        sorted_tasks = sorted(
            tasks, key=lambda t: t.priority_score, reverse=True)

        # Move tasks with dependencies after their dependencies
        final_order = []
        processed = set()

        def process_task(task_id):
            if task_id in processed:
                return

            # First process all dependencies
            for dep_id in dependency_graph.get(task_id, []):
                if dep_id in task_map:  # Skip if dependency doesn't exist
                    process_task(dep_id)

            # Now add this task
            if task_id in task_map:  # Make sure task exists
                final_order.append(task_map[task_id])
                processed.add(task_id)

        # Process all tasks
        for task in sorted_tasks:
            process_task(task.task_id)

        return final_order

    def _log_scheduling_event(
        self,
        event_type: str,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        details: Dict[str, Any] = None
    ) -> None:
        """Log a scheduling event."""
        event = SchedulingEvent(
            event_type=event_type,
            task_id=task_id,
            agent_id=agent_id,
            details=details or {}
        )
        self.repository.log_scheduling_event(event)

    async def request_time_off(
        self,
        agent_id: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        reason: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Request time off for a human agent.

        Returns:
            Tuple of (success, status, request_id)
        """
        # Create the request object
        request = TimeOffRequest(
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time,
            reason=reason
        )

        # Store the request
        self.repository.create_time_off_request(request)

        # Process the request automatically
        return await self._process_time_off_request(request)

    async def _process_time_off_request(
        self,
        request: TimeOffRequest
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Process a time off request automatically.

        Returns:
            Tuple of (success, status, request_id)
        """
        # Get affected tasks during this time period
        affected_tasks = self.repository.get_agent_tasks(
            request.agent_id,
            request.start_time,
            request.end_time,
            "scheduled"
        )

        # Check if we can reassign all affected tasks
        reassignable_tasks = []
        non_reassignable_tasks = []

        for task in affected_tasks:
            # For each affected task, check if we can find another suitable agent
            alternate_agent = await self._find_optimal_agent(
                task,
                excluded_agents=[request.agent_id]
            )

            if alternate_agent:
                reassignable_tasks.append((task, alternate_agent))
            else:
                non_reassignable_tasks.append(task)

        # Make approval decision
        approval_threshold = 0.8  # We require 80% of tasks to be reassignable

        if len(affected_tasks) == 0 or (
            len(reassignable_tasks) / len(affected_tasks) >= approval_threshold
        ):
            # Approve the request
            request.status = TimeOffStatus.APPROVED
            self.repository.update_time_off_request(request)

            # Create unavailability window in agent's schedule
            agent_schedule = self.repository.get_agent_schedule(
                request.agent_id)
            if agent_schedule:
                time_off_window = TimeWindow(
                    start=request.start_time,
                    end=request.end_time
                )
                agent_schedule.availability_exceptions.append(time_off_window)
                self.repository.save_agent_schedule(agent_schedule)

            # Reassign tasks that can be reassigned
            for task, new_agent in reassignable_tasks:
                task.assigned_to = new_agent
                self.repository.update_scheduled_task(task)

                self._log_scheduling_event(
                    "task_reassigned_time_off",
                    task.task_id,
                    request.agent_id,
                    {
                        "original_agent": request.agent_id,
                        "new_agent": new_agent,
                        "time_off_request_id": request.request_id
                    }
                )

            # For tasks that can't be reassigned, mark them for review
            for task in non_reassignable_tasks:
                self._log_scheduling_event(
                    "task_needs_reassignment",
                    task.task_id,
                    request.agent_id,
                    {
                        "time_off_request_id": request.request_id,
                        "reason": "Cannot find suitable replacement agent"
                    }
                )

            return (True, "approved", request.request_id)
        else:
            # Reject the request
            request.status = TimeOffStatus.REJECTED
            request.rejection_reason = f"Cannot reassign {len(non_reassignable_tasks)} critical tasks during requested time period."
            self.repository.update_time_off_request(request)

            return (False, "rejected", request.request_id)

    async def cancel_time_off_request(
        self,
        agent_id: str,
        request_id: str
    ) -> Tuple[bool, str]:
        """
        Cancel a time off request.

        Returns:
            Tuple of (success, status)
        """
        # Get the request
        request = self.repository.get_time_off_request(request_id)

        if not request:
            return (False, "not_found")

        if request.agent_id != agent_id:
            return (False, "unauthorized")

        if request.status not in [TimeOffStatus.REQUESTED, TimeOffStatus.APPROVED]:
            return (False, "invalid_status")

        # Check if the time off has already started
        now = datetime.datetime.now(datetime.timezone.utc)
        if request.status == TimeOffStatus.APPROVED and request.start_time <= now:
            return (False, "already_started")

        # Cancel the request
        request.status = TimeOffStatus.CANCELLED
        self.repository.update_time_off_request(request)

        # If it was approved, also remove from agent's schedule
        if request.status == TimeOffStatus.APPROVED:
            agent_schedule = self.repository.get_agent_schedule(agent_id)
            if agent_schedule:
                # Remove the exception for this time off period
                agent_schedule.availability_exceptions = [
                    exception for exception in agent_schedule.availability_exceptions
                    if not (exception.start == request.start_time and
                            exception.end == request.end_time)
                ]
                self.repository.save_agent_schedule(agent_schedule)

        return (True, "cancelled")

    async def get_agent_time_off_history(
        self,
        agent_id: str
    ) -> List[Dict[str, Any]]:
        """Get an agent's time off history."""
        requests = self.repository.get_agent_time_off_requests(agent_id)

        # Format for display
        formatted_requests = []
        for request in requests:
            formatted_requests.append({
                "request_id": request.request_id,
                "start_time": request.start_time.isoformat(),
                "end_time": request.end_time.isoformat(),
                "duration_hours": (request.end_time - request.start_time).total_seconds() / 3600,
                "reason": request.reason,
                "status": request.status,
                "created_at": request.created_at.isoformat(),
                "rejection_reason": request.rejection_reason
            })

        return formatted_requests

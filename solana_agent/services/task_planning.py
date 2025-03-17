"""
Task planning service implementation.

This service manages complex task planning, breakdown into subtasks,
and resource allocation.
"""
import json
import uuid
import datetime
from typing import Dict, List, Optional, Any, Tuple

from solana_agent.interfaces import TaskPlanningService as TaskPlanningServiceInterface
from solana_agent.interfaces import ResourceService
from solana_agent.interfaces import TicketRepository
from solana_agent.interfaces import LLMProvider
from solana_agent.domains import (
    ComplexityAssessment, SubtaskModel, TaskBreakdown, TaskBreakdownWithResources, WorkCapacity, PlanStatus,
    TaskStatus
)
from solana_agent.domains import Ticket, TicketStatus
from solana_agent.domains import ScheduledTask


class TaskPlanningService(TaskPlanningServiceInterface):
    """Service for managing complex task planning and breakdown."""

    def __init__(
        self,
        ticket_repository: TicketRepository,
        llm_provider: LLMProvider,
        agent_service: Any,
        scheduling_service: Optional[Any] = None
    ):
        """Initialize the task planning service.

        Args:
            ticket_repository: Repository for ticket operations
            llm_provider: Provider for language model interactions
            agent_service: Service for agent management
            scheduling_service: Optional service for task scheduling
        """
        self.ticket_repository = ticket_repository
        self.llm_provider = llm_provider
        self.agent_service = agent_service
        self.scheduling_service = scheduling_service
        self.capacity_registry = {}  # agent_id -> WorkCapacity

    def register_agent_capacity(
        self,
        agent_id: str,
        agent_type: str,
        max_tasks: int,
        specializations: List[str],
    ) -> None:
        """Register an agent's work capacity."""
        self.capacity_registry[agent_id] = WorkCapacity(
            agent_id=agent_id,
            agent_type=agent_type,
            max_concurrent_tasks=max_tasks,
            active_tasks=0,
            specializations=specializations,
        )

    def update_agent_availability(self, agent_id: str, status: str) -> bool:
        """Update an agent's availability status."""
        if agent_id in self.capacity_registry:
            self.capacity_registry[agent_id].availability_status = status
            self.capacity_registry[agent_id].last_updated = datetime.datetime.now(
                datetime.timezone.utc
            )
            return True
        return False

    def get_agent_capacity(self, agent_id: str) -> Optional[WorkCapacity]:
        """Get an agent's capacity information."""
        return self.capacity_registry.get(agent_id)

    def get_available_agents(self, specialization: Optional[str] = None) -> List[str]:
        """Get list of available agents, optionally filtered by specialization."""
        agents = []

        for agent_id, capacity in self.capacity_registry.items():
            if capacity.availability_status != "available":
                continue

            if capacity.active_tasks >= capacity.max_concurrent_tasks:
                continue

            if specialization and specialization not in capacity.specializations:
                continue

            agents.append(agent_id)

        return agents

    async def needs_breakdown(self, task_description: str) -> Tuple[bool, str]:
        """Determine if a task needs to be broken down into subtasks."""
        complexity = await self._assess_task_complexity(task_description)

        # Tasks with high story points, large t-shirt sizes, or long estimated
        # resolution times are candidates for breakdown
        story_points = complexity.get("story_points", 3)
        t_shirt_size = complexity.get("t_shirt_size", "M")
        estimated_minutes = complexity.get("estimated_minutes", 30)

        needs_breakdown = (
            story_points >= 8
            or t_shirt_size in ["L", "XL", "XXL"]
            or estimated_minutes >= 60
        )

        reasoning = f"Task complexity: {t_shirt_size}, {story_points} story points, {estimated_minutes} minutes estimated"

        return (needs_breakdown, reasoning)

    async def generate_subtasks(
        self, ticket_id: str, task_description: str
    ) -> List[SubtaskModel]:
        """Generate subtasks for a complex task."""
        # Fetch ticket to verify it exists
        ticket = self.ticket_repository.get_by_id(ticket_id)
        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")

        # Mark parent ticket as a parent
        self.ticket_repository.update(
            ticket_id, {"is_parent": True, "status": TicketStatus.PLANNING}
        )

        # Generate subtasks using LLM with structured output
        prompt = f"""
        Break down the following complex task into logical subtasks:
        
        TASK: {task_description}
        
        For each subtask, provide:
        1. A brief title
        2. A clear description of what needs to be done
        3. An estimate of time required in minutes
        4. Any dependencies (which subtasks must be completed first)
        
        The subtasks should be in a logical sequence. Keep dependencies minimal and avoid circular dependencies.
        """

        try:
            # Use structured output parsing
            breakdown = await self.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="You are an expert project planner who breaks down complex tasks efficiently.",
                model_class=TaskBreakdown,
                temperature=0.2
            )

            subtasks_data = breakdown.subtasks

            # Create subtask objects
            subtasks = []
            for i, task_data in enumerate(subtasks_data):
                subtask = SubtaskModel(
                    id=str(uuid.uuid4()),
                    parent_id=ticket_id,
                    title=task_data.title,
                    description=task_data.description,
                    sequence=i + 1,
                    estimated_minutes=task_data.estimated_minutes,
                    dependencies=[],
                )
                subtasks.append(subtask)

            # Process dependencies (convert title references to IDs)
            title_to_id = {task.title: task.id for task in subtasks}

            for i, task_data in enumerate(subtasks_data):
                for dep_title in task_data.dependencies:
                    if dep_title in title_to_id:
                        subtasks[i].dependencies.append(title_to_id[dep_title])

            # Store subtasks in database
            for subtask in subtasks:
                new_ticket = Ticket(
                    id=subtask.id,
                    title=subtask.title,
                    description=subtask.description,
                    user_id=ticket.user_id,
                    status=TicketStatus.PLANNING,
                    assigned_to="",
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                    updated_at=datetime.datetime.now(datetime.timezone.utc),
                    is_subtask=True,
                    parent_id=ticket_id,
                    metadata={
                        "estimated_minutes": subtask.estimated_minutes,
                        "sequence": subtask.sequence,
                        "dependencies": subtask.dependencies
                    },
                )
                self.ticket_repository.create(new_ticket)

            # After creating subtasks, schedule them if scheduling_service is available
            if self.scheduling_service:
                # Schedule each subtask
                for subtask in subtasks:
                    scheduled_task = ScheduledTask(
                        task_id=subtask.id,
                        title=subtask.title,
                        description=subtask.description,
                        status=TaskStatus.PLANNING,
                        priority=5,  # Default priority
                        estimated_minutes=subtask.estimated_minutes,
                        depends_on=subtask.dependencies,
                        metadata={
                            "ticket_id": ticket_id,
                            "parent_ticket_id": ticket_id,
                            "is_subtask": True,
                            "sequence": subtask.sequence
                        }
                    )

                    # Try to schedule the task
                    await self.scheduling_service.schedule_task(scheduled_task)

            return subtasks

        except Exception as e:
            print(f"Error generating subtasks: {e}")
            return []

    async def assign_subtasks(self, parent_ticket_id: str) -> Dict[str, List[str]]:
        """Assign subtasks to available agents based on capacity."""
        # Get all subtasks for the parent
        subtasks = self.ticket_repository.find({
            "parent_id": parent_ticket_id,
            "is_subtask": True,
            "status": TicketStatus.PLANNING,
        })

        if not subtasks:
            return {}

        # Find available agents
        available_agents = self.get_available_agents()
        if not available_agents:
            return {}

        # Simple round-robin assignment
        assignments = {agent_id: [] for agent_id in available_agents}
        agent_idx = 0

        for subtask in subtasks:
            agent_id = available_agents[agent_idx]
            assignments[agent_id].append(subtask.id)

            # Update subtask with assignment
            self.ticket_repository.update(
                subtask.id, {
                    "assigned_to": agent_id,
                    "status": TicketStatus.ACTIVE,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc)
                }
            )

            # Update agent capacity
            if agent_id in self.capacity_registry:
                self.capacity_registry[agent_id].active_tasks += 1

            # Move to next agent in round-robin
            agent_idx = (agent_idx + 1) % len(available_agents)

        return assignments

    async def get_plan_status(self, parent_ticket_id: str) -> PlanStatus:
        """Get the status of a task plan."""
        # Get parent ticket
        parent = self.ticket_repository.get_by_id(parent_ticket_id)
        if not parent or not getattr(parent, "is_parent", False):
            raise ValueError(
                f"Parent ticket {parent_ticket_id} not found or is not a parent")

        # Get all subtasks
        subtasks = self.ticket_repository.find({
            "parent_id": parent_ticket_id,
            "is_subtask": True
        })

        subtask_count = len(subtasks)
        if subtask_count == 0:
            return PlanStatus(
                visualization="No subtasks found",
                progress=0,
                status="unknown",
                estimated_completion="unknown",
                subtask_count=0,
            )

        # Count completed tasks
        completed = sum(1 for task in subtasks if task.status ==
                        TicketStatus.RESOLVED)

        # Calculate progress percentage
        progress = int((completed / subtask_count) *
                       100) if subtask_count > 0 else 0

        # Determine status
        if progress == 100:
            status = "completed"
        elif progress == 0:
            status = "not started"
        else:
            status = "in progress"

        # Create visualization
        bars = "█" * (progress // 10) + "░" * (10 - (progress // 10))
        visualization = f"Progress: {progress}% [{bars}] ({completed}/{subtask_count} subtasks complete)"

        # Estimate completion time
        if status == "completed":
            estimated_completion = "Completed"
        elif status == "not started":
            estimated_completion = "Not started"
        else:
            # Simple linear projection based on progress
            if progress > 0:
                first_subtask = min(subtasks, key=lambda t: t.created_at)
                start_time = first_subtask.created_at
                time_elapsed = (datetime.datetime.now(
                    datetime.timezone.utc) - start_time).total_seconds()
                time_remaining = (time_elapsed / progress) * (100 - progress)
                completion_time = datetime.datetime.now(
                    datetime.timezone.utc) + datetime.timedelta(seconds=time_remaining)
                estimated_completion = completion_time.strftime(
                    "%Y-%m-%d %H:%M")
            else:
                estimated_completion = "Unknown"

        return PlanStatus(
            visualization=visualization,
            progress=progress,
            status=status,
            estimated_completion=estimated_completion,
            subtask_count=subtask_count,
        )

    async def generate_subtasks_with_resources(
        self, ticket_id: str, task_description: str
    ) -> List[SubtaskModel]:
        """Generate subtasks for a complex task with resource requirements."""
        # Fetch ticket to verify it exists
        ticket = self.ticket_repository.get_by_id(ticket_id)
        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")

        # Mark parent ticket as a parent
        self.ticket_repository.update(
            ticket_id, {
                "is_parent": True,
                "status": TicketStatus.PLANNING,
                "updated_at": datetime.datetime.now(datetime.timezone.utc)
            }
        )

        # Generate subtasks using LLM with structured output
        prompt = f"""
        Break down the following complex task into logical subtasks with resource requirements:
        
        TASK: {task_description}
        
        For each subtask, provide:
        1. A brief title
        2. A clear description of what needs to be done
        3. An estimate of time required in minutes
        4. Any dependencies (which subtasks must be completed first)
        5. Required resources with these details:
           - Resource type (room, equipment, etc.)
           - Quantity needed
           - Specific requirements (e.g., "room with projector", "laptop with design software")
        
        The subtasks should be in a logical sequence. Keep dependencies minimal and avoid circular dependencies.
        """

        try:
            # Use structured output parsing
            breakdown = await self.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="You are an expert project planner who breaks down complex tasks efficiently and identifies required resources.",
                model_class=TaskBreakdownWithResources,
                temperature=0.2
            )

            subtasks_data = breakdown.subtasks

            # Create subtask objects
            subtasks = []
            for i, task_data in enumerate(subtasks_data):
                subtask_id = str(uuid.uuid4())
                subtask = SubtaskModel(
                    id=subtask_id,
                    parent_id=ticket_id,
                    title=task_data.title,
                    description=task_data.description,
                    sequence=i + 1,
                    estimated_minutes=task_data.estimated_minutes,
                    dependencies=[],  # We'll fill this after all subtasks are created
                    status="planning",
                    required_resources=[resource.model_dump()
                                        for resource in task_data.required_resources],
                    is_subtask=True,
                    created_at=datetime.datetime.now(datetime.timezone.utc)
                )
                subtasks.append(subtask)

            # Process dependencies (convert title references to IDs)
            title_to_id = {task.title: task.id for task in subtasks}
            for i, task_data in enumerate(subtasks_data):
                for dep_title in task_data.dependencies:
                    if dep_title in title_to_id:
                        subtasks[i].dependencies.append(title_to_id[dep_title])

            # Store subtasks in database
            for subtask in subtasks:
                new_ticket = Ticket(
                    id=subtask.id,
                    title=subtask.title,
                    description=subtask.description,
                    user_id=ticket.user_id,
                    status=TicketStatus.PLANNING,
                    assigned_to="",
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                    updated_at=datetime.datetime.now(datetime.timezone.utc),
                    is_subtask=True,
                    parent_id=ticket_id,
                    metadata={
                        "estimated_minutes": subtask.estimated_minutes,
                        "sequence": subtask.sequence,
                        "dependencies": subtask.dependencies,
                        "required_resources": subtask.required_resources
                    },
                )
                self.ticket_repository.create(new_ticket)

            return subtasks

        except Exception as e:
            print(f"Error generating subtasks with resources: {e}")
            return []

    async def allocate_resources(
        self, subtask_id: str, resource_service: ResourceService
    ) -> Tuple[bool, str]:
        """Allocate resources to a subtask."""
        # Get the subtask
        subtask = self.ticket_repository.get_by_id(subtask_id)
        if not subtask or not getattr(subtask, "is_subtask", False):
            return False, "Subtask not found"

        # Get required resources from metadata
        required_resources = subtask.metadata.get("required_resources", [])
        if not required_resources:
            return True, "No resources required"

        # Check if subtask is scheduled
        scheduled_start = getattr(subtask, "scheduled_start", None)
        scheduled_end = getattr(subtask, "scheduled_end", None)

        if not scheduled_start or not scheduled_end:
            # Try to get from scheduling service if available
            if self.scheduling_service:
                try:
                    scheduled_task = self.scheduling_service.repository.get_scheduled_task(
                        subtask_id)
                    if scheduled_task:
                        scheduled_start = scheduled_task.scheduled_start
                        scheduled_end = scheduled_task.scheduled_end
                except Exception:
                    pass

        if not scheduled_start or not scheduled_end:
            return False, "Subtask must be scheduled before resources can be allocated"

        # For each required resource
        resource_assignments = []
        for resource_req in required_resources:
            resource_type = resource_req.get("resource_type")
            requirements = resource_req.get("requirements", "")
            quantity = resource_req.get("quantity", 1)

            # Find available resources matching the requirements
            resources = await resource_service.find_available_resources(
                start_time=scheduled_start,
                end_time=scheduled_end,
                resource_type=resource_type,
                tags=requirements.split() if requirements else None,
                capacity=None  # Could use quantity here if it represents capacity
            )

            if not resources or len(resources) < quantity:
                return False, f"Insufficient {resource_type} resources available"

            # Allocate the resources by creating bookings
            allocated_resources = []
            for i in range(quantity):
                if i >= len(resources):
                    break

                resource = resources[i]
                success, booking_id, error = await resource_service.create_booking(
                    resource_id=resource.id,
                    user_id=subtask.assigned_to or "system",
                    title=f"Task: {subtask.title}",
                    start_time=scheduled_start,
                    end_time=scheduled_end,
                    description=subtask.description
                )

                if success:
                    allocated_resources.append({
                        "resource_id": resource.id,
                        "resource_name": resource.name,
                        "booking_id": booking_id,
                        "resource_type": resource.resource_type
                    })
                else:
                    # Clean up any allocations already made
                    for alloc in allocated_resources:
                        await resource_service.cancel_booking(alloc["booking_id"], subtask.assigned_to or "system")
                    return False, f"Failed to book resource: {error}"

            resource_assignments.append({
                "requirement": resource_req,
                "allocated": allocated_resources
            })

        # Update the subtask with resource assignments
        updated_metadata = subtask.metadata.copy() if hasattr(subtask, "metadata") else {}
        updated_metadata["resource_assignments"] = resource_assignments

        self.ticket_repository.update(subtask_id, {
            "metadata": updated_metadata,
            "updated_at": datetime.datetime.now(datetime.timezone.utc)
        })

        return True, f"Successfully allocated {len(resource_assignments)} resource types"

    async def _assess_task_complexity(self, query: str) -> Dict[str, Any]:
        """Assess the complexity of a task using standardized metrics."""
        prompt = f"""
        Analyze this task and provide standardized complexity metrics:
        
        TASK: {query}
        
        Assess on these dimensions:
        1. T-shirt size (XS, S, M, L, XL, XXL)
        2. Story points (1, 2, 3, 5, 8, 13, 21)
        3. Estimated resolution time in minutes/hours
        4. Technical complexity (1-10)
        5. Domain knowledge required (1-10)
        """

        try:
            complexity = await self.llm_provider.parse_structured_output(
                prompt,
                system_prompt="You are an expert at estimating task complexity.",
                model_class=ComplexityAssessment,
                temperature=0.2,
            )
            return complexity.model_dump()
        except Exception as e:
            print(f"Error assessing complexity: {e}")
            return {
                "t_shirt_size": "M",
                "story_points": 3,
                "estimated_minutes": 30,
                "technical_complexity": 5,
                "domain_knowledge": 5,
            }

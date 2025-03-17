"""
Solana Agent System: AI-powered agent coordination system with human agent integration.

This module implements a clean architecture approach with:
- Domain models for core data structures
- Interfaces for dependency inversion
- Services for business logic
- Repositories for data access
- Adapters for external integrations
- Use cases for orchestrating application flows
"""

import asyncio
import datetime
import importlib
import json
import re
import traceback
from unittest.mock import AsyncMock
import uuid
from enum import Enum
from typing import (
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Any,
    Type,
)
from pydantic import BaseModel, Field, ValidationError
from pymongo import MongoClient
from openai import OpenAI
import pymongo
from zep_cloud.client import AsyncZep as AsyncZepCloud
from zep_python.client import AsyncZep
from zep_cloud.types import Message
from pinecone import Pinecone
from abc import ABC, abstractmethod


#############################################
# MAIN AGENT PROCESSOR
#############################################


class QueryProcessor:
    """Main service to process user queries using agents and services."""

    def __init__(
        self,
        agent_service: AgentService,
        routing_service: RoutingService,
        ticket_service: TicketService,
        handoff_service: HandoffService,
        memory_service: MemoryService,
        nps_service: NPSService,
        critic_service: Optional[CriticService] = None,
        memory_provider: Optional[MemoryProvider] = None,
        enable_critic: bool = True,
        router_model: str = "gpt-4o-mini",
        task_planning_service: Optional["TaskPlanningService"] = None,
        project_approval_service: Optional[ProjectApprovalService] = None,
        project_simulation_service: Optional[ProjectSimulationService] = None,
        require_human_approval: bool = False,
        scheduling_service: Optional[SchedulingService] = None,
        stalled_ticket_timeout: Optional[int] = 60,
    ):
        self.agent_service = agent_service
        self.routing_service = routing_service
        self.ticket_service = ticket_service
        self.handoff_service = handoff_service
        self.memory_service = memory_service
        self.nps_service = nps_service
        self.critic_service = critic_service
        self.memory_provider = memory_provider
        self.enable_critic = enable_critic
        self.router_model = router_model
        self.task_planning_service = task_planning_service
        self.project_approval_service = project_approval_service
        self.project_simulation_service = project_simulation_service
        self.require_human_approval = require_human_approval
        self._shutdown_event = asyncio.Event()
        self.scheduling_service = scheduling_service
        self.stalled_ticket_timeout = stalled_ticket_timeout

        self._stalled_ticket_task = None

        # Start background task for stalled ticket detection if not already running
        if self.stalled_ticket_timeout is not None and self._stalled_ticket_task is None:
            try:
                self._stalled_ticket_task = asyncio.create_task(
                    self._run_stalled_ticket_checks())
            except RuntimeError:
                # No running event loop - likely in test environment
                pass

    async def process(
        self, user_id: str, user_text: str, timezone: str = None
    ) -> AsyncGenerator[str, None]:
        """Process the user request with appropriate agent and handle ticket management."""
        # Start background task for stalled ticket detection if not already running
        if self.stalled_ticket_timeout is not None and self._stalled_ticket_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._stalled_ticket_task = loop.create_task(
                    self._run_stalled_ticket_checks())
            except RuntimeError:
                import logging
                logging.warning(
                    "No running event loop available for stalled ticket checker.")

        try:
            # Special case for "test" and other very simple messages
            if user_text.strip().lower() in ["test", "hello", "hi", "hey", "ping"]:
                response = f"Hello! How can I help you today?"
                yield response
                # Store this simple interaction in memory
                if self.memory_provider:
                    await self._store_conversation(user_id, user_text, response)
                return

            # Handle system commands
            command_response = await self._process_system_commands(user_id, user_text)
            if command_response is not None:
                yield command_response
                return

            # Route to appropriate agent
            agent_name = await self.routing_service.route_query(user_text)

            # Check for active ticket
            active_ticket = self.ticket_service.ticket_repository.get_active_for_user(
                user_id)

            if active_ticket:
                # Process existing ticket
                try:
                    response_buffer = ""
                    async for chunk in self._process_existing_ticket(user_id, user_text, active_ticket, timezone):
                        response_buffer += chunk
                        yield chunk

                    # Check final response for unprocessed JSON
                    if response_buffer.strip().startswith('{'):
                        agent_name = active_ticket.assigned_to or "default_agent"
                        processed_response = self.agent_service.process_json_response(
                            response_buffer, agent_name)
                        if processed_response != response_buffer:
                            yield "\n\n" + processed_response
                except ValueError as e:
                    if "Ticket" in str(e) and "not found" in str(e):
                        # Ticket no longer exists - create a new one
                        complexity = await self._assess_task_complexity(user_text)
                        async for chunk in self._process_new_ticket(user_id, user_text, complexity, timezone):
                            yield chunk
                    else:
                        yield f"I'm sorry, I encountered an error: {str(e)}"

            else:
                # Create new ticket
                try:
                    complexity = await self._assess_task_complexity(user_text)

                    # Process as new ticket
                    response_buffer = ""
                    async for chunk in self._process_new_ticket(user_id, user_text, complexity, timezone):
                        response_buffer += chunk
                        yield chunk

                    # Check final response for unprocessed JSON
                    if response_buffer.strip().startswith('{'):
                        processed_response = self.agent_service.process_json_response(
                            response_buffer, agent_name)
                        if processed_response != response_buffer:
                            yield "\n\n" + processed_response
                except Exception as e:
                    yield f"I'm sorry, I encountered an error: {str(e)}"

        except Exception as e:
            print(f"Error in request processing: {str(e)}")
            print(traceback.format_exc())
            yield "I apologize for the technical difficulty.\n\n"

    async def _is_human_agent(self, user_id: str) -> bool:
        """Check if the user is a registered human agent."""
        return user_id in self.agent_service.get_all_human_agents()

    async def shutdown(self):
        """Clean shutdown of the query processor."""
        self._shutdown_event.set()

        # Cancel the stalled ticket task if running
        if hasattr(self, '_stalled_ticket_task') and self._stalled_ticket_task is not None:
            self._stalled_ticket_task.cancel()
            try:
                await self._stalled_ticket_task
            except (asyncio.CancelledError, TypeError):
                # Either properly cancelled coroutine or a mock that can't be awaited
                pass

    async def _run_stalled_ticket_checks(self):
        """Run periodic checks for stalled tickets."""
        try:
            while not self._shutdown_event.is_set():
                await self._check_for_stalled_tickets()
                # Check every 5 minutes or half the timeout period, whichever is smaller
                check_interval = min(
                    300, self.stalled_ticket_timeout * 30) if self.stalled_ticket_timeout else 300
                await asyncio.sleep(check_interval)
        except asyncio.CancelledError:
            # Task was cancelled, clean exit
            pass
        except Exception as e:
            print(f"Error in stalled ticket check: {e}")

    async def _check_for_stalled_tickets(self):
        """Check for tickets that haven't been updated in a while and reassign them."""
        # If stalled ticket detection is disabled, exit early
        if self.stalled_ticket_timeout is None:
            return

        try:
            # Find tickets that haven't been updated in the configured time
            cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
                minutes=self.stalled_ticket_timeout
            )

            # The find_stalled_tickets method should be async
            stalled_tickets = await self.ticket_service.ticket_repository.find_stalled_tickets(
                cutoff_time, [TicketStatus.ACTIVE, TicketStatus.TRANSFERRED]
            )

            for ticket in stalled_tickets:
                print(
                    f"Found stalled ticket: {ticket.id} (last updated: {ticket.updated_at})")

                # Skip tickets without an assigned agent
                if not ticket.assigned_to:
                    continue

                # Re-route the query to see if a different agent is better
                new_agent = await self.routing_service.route_query(ticket.query)

                # Only reassign if a different agent is suggested
                if new_agent != ticket.assigned_to:
                    print(
                        f"Reassigning ticket {ticket.id} from {ticket.assigned_to} to {new_agent}")

                    try:
                        await self.handoff_service.process_handoff(
                            ticket.id,
                            ticket.assigned_to,
                            new_agent,
                            f"Automatically reassigned after {self.stalled_ticket_timeout} minutes of inactivity"
                        )
                    except Exception as e:
                        print(
                            f"Error reassigning stalled ticket {ticket.id}: {e}")

        except Exception as e:
            print(f"Error in stalled ticket check: {e}")
            import traceback
            print(traceback.format_exc())

    async def _process_system_commands(
        self, user_id: str, user_text: str
    ) -> Optional[str]:
        """Process system commands and return response if command was handled."""
        # Simple command system
        if user_text.startswith("!"):
            command_parts = user_text.split(" ", 1)
            command = command_parts[0].lower()
            args = command_parts[1] if len(command_parts) > 1 else ""

            if command == "!memory" and args:
                # Search collective memory
                results = self.memory_service.search_memory(args)

                if not results:
                    return "No relevant memory entries found."

                response = "Found in collective memory:\n\n"
                for i, entry in enumerate(results, 1):
                    response += f"{i}. {entry['fact']}\n"
                    response += f"   Relevance: {entry['relevance']}\n\n"

                return response

            if command == "!plan" and args:
                # Create a new plan from the task description
                if not self.task_planning_service:
                    return "Task planning service is not available."

                complexity = await self._assess_task_complexity(args)

                # Create a parent ticket
                ticket = await self.ticket_service.get_or_create_ticket(
                    user_id, args, complexity
                )

                # Generate subtasks with resource requirements
                subtasks = await self.task_planning_service.generate_subtasks_with_resources(
                    ticket.id, args
                )

                if not subtasks:
                    return "Failed to create task plan."

                # Create a response with subtask details
                response = f"# Task Plan Created\n\nParent task: **{args}**\n\n"
                response += f"Created {len(subtasks)} subtasks:\n\n"

                for i, subtask in enumerate(subtasks, 1):
                    response += f"{i}. **{subtask.title}**\n"
                    response += f"   - Description: {subtask.description}\n"
                    response += f"   - Estimated time: {subtask.estimated_minutes} minutes\n"

                    if subtask.required_resources:
                        response += f"   - Required resources:\n"
                        for res in subtask.required_resources:
                            res_type = res.get("resource_type", "unknown")
                            quantity = res.get("quantity", 1)
                            requirements = res.get("requirements", "")
                            response += f"     * {quantity} {res_type}" + \
                                (f" ({requirements})" if requirements else "") + "\n"

                    if subtask.dependencies:
                        response += f"   - Dependencies: {len(subtask.dependencies)} subtasks\n"

                    response += "\n"

                return response

            # Add a new command for allocating resources to tasks
            elif command == "!allocate-resources" and args:
                parts = args.split()
                if len(parts) < 1:
                    return "Usage: !allocate-resources [subtask_id]"

                subtask_id = parts[0]

                if not hasattr(self, "resource_service") or not self.resource_service:
                    return "Resource service is not available."

                success, message = await self.task_planning_service.allocate_resources(
                    subtask_id, self.resource_service
                )

                if success:
                    return f"✅ Resources allocated successfully: {message}"
                else:
                    return f"❌ Failed to allocate resources: {message}"

            elif command == "!status" and args:
                # Show status of a specific plan
                if not self.task_planning_service:
                    return "Task planning service is not available."

                try:
                    status = await self.task_planning_service.get_plan_status(args)

                    response = "# Plan Status\n\n"
                    response += f"{status.visualization}\n\n"
                    response += f"Status: {status.status}\n"
                    response += f"Subtasks: {status.subtask_count}\n"
                    response += f"Estimated completion: {status.estimated_completion}\n"

                    return response
                except ValueError as e:
                    return f"Error: {str(e)}"

            elif command == "!assign" and args:
                # Assign subtasks to agents
                if not self.task_planning_service:
                    return "Task planning service is not available."

                try:
                    assignments = await self.task_planning_service.assign_subtasks(args)

                    if not assignments:
                        return "No subtasks to assign or no agents available."

                    response = "# Subtask Assignments\n\n"

                    for agent_id, task_ids in assignments.items():
                        agent_name = agent_id
                        if agent_id in self.agent_service.get_all_human_agents():
                            agent_info = self.agent_service.get_all_human_agents()[
                                agent_id
                            ]
                            agent_name = agent_info.get("name", agent_id)

                        response += (
                            f"**{agent_name}**: {len(task_ids)} subtasks assigned\n"
                        )

                    return response
                except ValueError as e:
                    return f"Error: {str(e)}"

            elif command == "!simulate" and args:
                # Run project simulation
                if not self.project_simulation_service:
                    return "Project simulation service is not available."

                simulation = await self.project_simulation_service.simulate_project(
                    args
                )

                response = "# Project Simulation Results\n\n"
                response += f"**Project**: {args}\n\n"
                response += f"**Complexity**: {simulation['complexity']['t_shirt_size']} ({simulation['complexity']['story_points']} points)\n"
                response += f"**Timeline Estimate**: {simulation['timeline']['realistic']} days\n"
                response += f"**Risk Level**: {simulation['risks']['overall_risk']}\n\n"

                response += "## Key Risks\n\n"
                for risk in simulation["risks"]["items"][:3]:  # Top 3 risks
                    response += f"- **{risk['type']}**: {risk['description']} (P: {risk['probability']}, I: {risk['impact']})\n"

                response += f"\n## Recommendation\n\n{simulation['recommendation']}"

                return response

            elif command == "!approve" and args:
                # Format: !approve ticket_id [yes/no] [comments]
                if not self.project_approval_service:
                    return "Project approval service is not available."

                parts = args.strip().split(" ", 2)
                if len(parts) < 2:
                    return "Usage: !approve ticket_id yes/no [comments]"

                ticket_id = parts[0]
                approved = parts[1].lower() in [
                    "yes",
                    "true",
                    "approve",
                    "approved",
                    "1",
                ]
                comments = parts[2] if len(parts) > 2 else ""

                await self.project_approval_service.process_approval(
                    ticket_id, user_id, approved, comments
                )
                return f"Project {ticket_id} has been {'approved' if approved else 'rejected'}."

            elif command == "!schedule" and args and self.scheduling_service:
                # Format: !schedule task_id [agent_id] [YYYY-MM-DD HH:MM]
                parts = args.strip().split(" ", 2)
                if len(parts) < 1:
                    return "Usage: !schedule task_id [agent_id] [YYYY-MM-DD HH:MM]"

                task_id = parts[0]
                agent_id = parts[1] if len(parts) > 1 else None
                time_str = parts[2] if len(parts) > 2 else None

                # Fetch the task from ticket repository
                ticket = self.ticket_service.ticket_repository.get_by_id(
                    task_id)
                if not ticket:
                    return f"Task {task_id} not found."

                # Convert ticket to scheduled task
                scheduled_task = ScheduledTask(
                    task_id=task_id,
                    title=ticket.query[:50] +
                    "..." if len(ticket.query) > 50 else ticket.query,
                    description=ticket.query,
                    estimated_minutes=ticket.complexity.get(
                        "estimated_minutes", 30) if ticket.complexity else 30,
                    priority=5,  # Default priority
                    assigned_to=agent_id or ticket.assigned_to,
                    # Use current agent as a specialization tag
                    specialization_tags=[ticket.assigned_to],
                )

                # Set scheduled time if provided
                if time_str:
                    try:
                        scheduled_time = datetime.datetime.fromisoformat(
                            time_str)
                        scheduled_task.scheduled_start = scheduled_time
                        scheduled_task.scheduled_end = scheduled_time + datetime.timedelta(
                            minutes=scheduled_task.estimated_minutes
                        )
                    except ValueError:
                        return "Invalid date format. Use YYYY-MM-DD HH:MM."

                # Schedule the task
                result = await self.scheduling_service.schedule_task(
                    scheduled_task, preferred_agent_id=agent_id
                )

                # Update ticket with scheduling info
                self.ticket_service.update_ticket_status(
                    task_id,
                    ticket.status,
                    scheduled_start=result.scheduled_start,
                    scheduled_agent=result.assigned_to
                )

                # Format response
                response = "# Task Scheduled\n\n"
                response += f"**Task:** {scheduled_task.title}\n"
                response += f"**Assigned to:** {result.assigned_to}\n"
                response += f"**Scheduled start:** {result.scheduled_start.strftime('%Y-%m-%d %H:%M')}\n"
                response += f"**Estimated duration:** {result.estimated_minutes} minutes"

                return response

            elif command == "!timeoff" and args and self.scheduling_service:
                # Format: !timeoff request YYYY-MM-DD HH:MM YYYY-MM-DD HH:MM reason
                # or: !timeoff cancel request_id
                parts = args.strip().split(" ", 1)
                if len(parts) < 2:
                    return "Usage: \n- !timeoff request START_DATE END_DATE reason\n- !timeoff cancel request_id"

                action = parts[0].lower()
                action_args = parts[1]

                if action == "request":
                    # Parse request args
                    request_parts = action_args.split(" ", 2)
                    if len(request_parts) < 3:
                        return "Usage: !timeoff request YYYY-MM-DD YYYY-MM-DD reason"

                    start_str = request_parts[0]
                    end_str = request_parts[1]
                    reason = request_parts[2]

                    try:
                        start_time = datetime.datetime.fromisoformat(start_str)
                        end_time = datetime.datetime.fromisoformat(end_str)
                    except ValueError:
                        return "Invalid date format. Use YYYY-MM-DD HH:MM."

                    # Submit time off request
                    success, status, request_id = await self.scheduling_service.request_time_off(
                        user_id, start_time, end_time, reason
                    )

                    if success:
                        return f"Time off request submitted and automatically approved. Request ID: {request_id}"
                    else:
                        return f"Time off request {status}. Request ID: {request_id}"

                elif action == "cancel":
                    request_id = action_args.strip()
                    success, status = await self.scheduling_service.cancel_time_off_request(
                        user_id, request_id
                    )

                    if success:
                        return f"Time off request {request_id} cancelled successfully."
                    else:
                        return f"Failed to cancel request: {status}"

                return "Unknown timeoff action. Use 'request' or 'cancel'."

            elif command == "!schedule-view" and self.scheduling_service:
                # View agent's schedule for the next week
                # Default to current user if no agent specified
                agent_id = args.strip() if args else user_id

                start_time = datetime.datetime.now(datetime.timezone.utc)
                end_time = start_time + datetime.timedelta(days=7)

                # Get tasks for the specified time period
                tasks = await self.scheduling_service.get_agent_tasks(
                    agent_id, start_time, end_time, include_completed=False
                )

                if not tasks:
                    return f"No scheduled tasks found for {agent_id} in the next 7 days."

                # Sort by start time
                tasks.sort(
                    key=lambda t: t.scheduled_start or datetime.datetime.max)

                # Format response
                response = f"# Schedule for {agent_id}\n\n"

                current_day = None
                for task in tasks:
                    # Group by day
                    task_day = task.scheduled_start.strftime(
                        "%Y-%m-%d") if task.scheduled_start else "Unscheduled"

                    if task_day != current_day:
                        response += f"\n## {task_day}\n\n"
                        current_day = task_day

                    start_time = task.scheduled_start.strftime(
                        "%H:%M") if task.scheduled_start else "TBD"
                    response += f"- **{start_time}** ({task.estimated_minutes} min): {task.title}\n"

                return response

            elif command == "!resources" and self.resource_service:
                # Format: !resources [list|find|show|create|update|delete]
                parts = args.strip().split(" ", 1)
                subcommand = parts[0] if parts else "list"
                subcmd_args = parts[1] if len(parts) > 1 else ""

                if subcommand == "list":
                    # List available resources, optionally filtered by type
                    resource_type = subcmd_args if subcmd_args else None

                    query = {}
                    if resource_type:
                        query["resource_type"] = resource_type

                    resources = self.resource_service.repository.find_resources(
                        query)

                    if not resources:
                        return "No resources found."

                    response = "# Available Resources\n\n"

                    # Group by type
                    resources_by_type = {}
                    for resource in resources:
                        r_type = resource.resource_type
                        if r_type not in resources_by_type:
                            resources_by_type[r_type] = []
                        resources_by_type[r_type].append(resource)

                    for r_type, r_list in resources_by_type.items():
                        response += f"## {r_type.capitalize()}\n\n"
                        for resource in r_list:
                            status_emoji = "🟢" if resource.status == "available" else "🔴"
                            response += f"{status_emoji} **{resource.name}** (ID: {resource.id})\n"
                            if resource.description:
                                response += f"   {resource.description}\n"
                            if resource.location and resource.location.building:
                                response += f"   Location: {resource.location.building}"
                                if resource.location.room:
                                    response += f", Room {resource.location.room}"
                                response += "\n"
                            if resource.capacity:
                                response += f"   Capacity: {resource.capacity}\n"
                            response += "\n"

                    return response

                elif subcommand == "show" and subcmd_args:
                    # Show details for a specific resource
                    resource_id = subcmd_args.strip()
                    resource = await self.resource_service.get_resource(resource_id)

                    if not resource:
                        return f"Resource with ID {resource_id} not found."

                    response = f"# Resource: {resource.name}\n\n"
                    response += f"**ID**: {resource.id}\n"
                    response += f"**Type**: {resource.resource_type}\n"
                    response += f"**Status**: {resource.status}\n"

                    if resource.description:
                        response += f"\n**Description**: {resource.description}\n"

                    if resource.location:
                        response += "\n**Location**:\n"
                        if resource.location.address:
                            response += f"- Address: {resource.location.address}\n"
                        if resource.location.building:
                            response += f"- Building: {resource.location.building}\n"
                        if resource.location.floor is not None:
                            response += f"- Floor: {resource.location.floor}\n"
                        if resource.location.room:
                            response += f"- Room: {resource.location.room}\n"

                    if resource.capacity:
                        response += f"\n**Capacity**: {resource.capacity}\n"

                    if resource.tags:
                        response += f"\n**Tags**: {', '.join(resource.tags)}\n"

                    # Show availability schedule
                    if resource.availability_schedule:
                        response += "\n**Regular Availability**:\n"
                        for window in resource.availability_schedule:
                            days = "Every day"
                            if window.day_of_week:
                                day_names = [
                                    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                days = ", ".join([day_names[d]
                                                 for d in window.day_of_week])
                            response += f"- {days}: {window.start_time} - {window.end_time} ({window.timezone})\n"

                    # Show upcoming bookings
                    now = datetime.datetime.now(datetime.timezone.utc)
                    next_month = now + datetime.timedelta(days=30)
                    bookings = self.resource_service.repository.get_resource_bookings(
                        resource_id, now, next_month)

                    if bookings:
                        response += "\n**Upcoming Bookings**:\n"
                        for booking in bookings:
                            start_str = booking.start_time.strftime(
                                "%Y-%m-%d %H:%M")
                            end_str = booking.end_time.strftime("%H:%M")
                            response += f"- {start_str} - {end_str}: {booking.title}\n"

                    return response

                elif subcommand == "find":
                    # Find available resources for a time period
                    # Format: !resources find room 2023-03-15 14:00 16:00
                    parts = subcmd_args.split()
                    if len(parts) < 4:
                        return "Usage: !resources find [type] [date] [start_time] [end_time] [capacity]"

                    resource_type = parts[0]
                    date_str = parts[1]
                    start_time_str = parts[2]
                    end_time_str = parts[3]
                    capacity = int(parts[4]) if len(parts) > 4 else None

                    try:
                        # Parse date and times
                        date_obj = datetime.datetime.strptime(
                            date_str, "%Y-%m-%d").date()
                        start_time = datetime.datetime.combine(
                            date_obj,
                            datetime.datetime.strptime(
                                start_time_str, "%H:%M").time(),
                            tzinfo=datetime.timezone.utc
                        )
                        end_time = datetime.datetime.combine(
                            date_obj,
                            datetime.datetime.strptime(
                                end_time_str, "%H:%M").time(),
                            tzinfo=datetime.timezone.utc
                        )

                        # Find available resources
                        resources = await self.resource_service.find_available_resources(
                            resource_type=resource_type,
                            start_time=start_time,
                            end_time=end_time,
                            capacity=capacity
                        )

                        if not resources:
                            return f"No {resource_type}s available for the requested time period."

                        response = f"# Available {resource_type.capitalize()}s\n\n"
                        response += f"**Date**: {date_str}\n"
                        response += f"**Time**: {start_time_str} - {end_time_str}\n"
                        if capacity:
                            response += f"**Minimum Capacity**: {capacity}\n"
                        response += "\n"

                        for resource in resources:
                            response += f"- **{resource.name}** (ID: {resource.id})\n"
                            if resource.description:
                                response += f"  {resource.description}\n"
                            if resource.capacity:
                                response += f"  Capacity: {resource.capacity}\n"
                            if resource.location and resource.location.building:
                                response += f"  Location: {resource.location.building}"
                                if resource.location.room:
                                    response += f", Room {resource.location.room}"
                                response += "\n"
                            response += "\n"

                        return response

                    except ValueError as e:
                        return f"Error parsing date/time: {e}"

                elif subcommand == "book":
                    # Book a resource
                    # Format: !resources book [resource_id] [date] [start_time] [end_time] [title]
                    parts = subcmd_args.split(" ", 5)
                    if len(parts) < 5:
                        return "Usage: !resources book [resource_id] [date] [start_time] [end_time] [title]"

                    resource_id = parts[0]
                    date_str = parts[1]
                    start_time_str = parts[2]
                    end_time_str = parts[3]
                    title = parts[4] if len(parts) > 4 else "Booking"

                    try:
                        # Parse date and times
                        date_obj = datetime.datetime.strptime(
                            date_str, "%Y-%m-%d").date()
                        start_time = datetime.datetime.combine(
                            date_obj,
                            datetime.datetime.strptime(
                                start_time_str, "%H:%M").time(),
                            tzinfo=datetime.timezone.utc
                        )
                        end_time = datetime.datetime.combine(
                            date_obj,
                            datetime.datetime.strptime(
                                end_time_str, "%H:%M").time(),
                            tzinfo=datetime.timezone.utc
                        )

                        # Create booking
                        success, booking, error = await self.resource_service.create_booking(
                            resource_id=resource_id,
                            user_id=user_id,
                            title=title,
                            start_time=start_time,
                            end_time=end_time
                        )

                        if not success:
                            return f"Failed to book resource: {error}"

                        # Get resource details
                        resource = await self.resource_service.get_resource(resource_id)
                        resource_name = resource.name if resource else resource_id

                        response = "# Booking Confirmed\n\n"
                        response += f"**Resource**: {resource_name}\n"
                        response += f"**Date**: {date_str}\n"
                        response += f"**Time**: {start_time_str} - {end_time_str}\n"
                        response += f"**Title**: {title}\n"
                        response += f"**Booking ID**: {booking.id}\n\n"
                        response += "Your booking has been confirmed and added to your schedule."

                        return response

                    except ValueError as e:
                        return f"Error parsing date/time: {e}"

                elif subcommand == "bookings":
                    # View all bookings for the current user
                    include_cancelled = "all" in subcmd_args.lower()

                    bookings = await self.resource_service.get_user_bookings(user_id, include_cancelled)

                    if not bookings:
                        return "You don't have any bookings." + (
                            " (Use 'bookings all' to include cancelled bookings)" if not include_cancelled else ""
                        )

                    response = "# Your Bookings\n\n"

                    # Group bookings by date
                    bookings_by_date = {}
                    for booking_data in bookings:
                        booking = booking_data["booking"]
                        resource = booking_data["resource"]

                        date_str = booking["start_time"].strftime("%Y-%m-%d")
                        if date_str not in bookings_by_date:
                            bookings_by_date[date_str] = []

                        bookings_by_date[date_str].append((booking, resource))

                    # Sort dates
                    for date_str in sorted(bookings_by_date.keys()):
                        response += f"## {date_str}\n\n"

                        for booking, resource in bookings_by_date[date_str]:
                            start_time = booking["start_time"].strftime(
                                "%H:%M")
                            end_time = booking["end_time"].strftime("%H:%M")
                            resource_name = resource["name"] if resource else "Unknown Resource"

                            status_emoji = "🟢" if booking["status"] == "confirmed" else "🔴"
                            response += f"{status_emoji} **{start_time}-{end_time}**: {booking['title']}\n"
                            response += f"   Resource: {resource_name}\n"
                            response += f"   Booking ID: {booking['id']}\n\n"

                    return response

                elif subcommand == "cancel" and subcmd_args:
                    # Cancel a booking
                    booking_id = subcmd_args.strip()

                    success, error = await self.resource_service.cancel_booking(booking_id, user_id)

                    if success:
                        return "✅ Your booking has been successfully cancelled."
                    else:
                        return f"❌ Failed to cancel booking: {error}"

                elif subcommand == "schedule" and subcmd_args:
                    # View resource schedule
                    # Format: !resources schedule resource_id [YYYY-MM-DD] [days]
                    parts = subcmd_args.split()
                    if len(parts) < 1:
                        return "Usage: !resources schedule resource_id [YYYY-MM-DD] [days]"

                    resource_id = parts[0]

                    # Default to today and 7 days
                    start_date = datetime.datetime.now(datetime.timezone.utc)
                    days = 7

                    if len(parts) > 1:
                        try:
                            start_date = datetime.datetime.strptime(
                                parts[1], "%Y-%m-%d"
                            ).replace(tzinfo=datetime.timezone.utc)
                        except ValueError:
                            return "Invalid date format. Use YYYY-MM-DD."

                    if len(parts) > 2:
                        try:
                            days = min(int(parts[2]), 31)  # Limit to 31 days
                        except ValueError:
                            return "Days must be a number."

                    end_date = start_date + datetime.timedelta(days=days)

                    # Get the resource
                    resource = await self.resource_service.get_resource(resource_id)
                    if not resource:
                        return f"Resource with ID {resource_id} not found."

                    # Get schedule
                    schedule = await self.resource_service.get_resource_schedule(
                        resource_id, start_date, end_date
                    )

                    # Create calendar visualization
                    response = f"# Schedule for {resource.name}\n\n"
                    response += f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"

                    # Group by date
                    schedule_by_date = {}
                    current_date = start_date
                    while current_date < end_date:
                        date_str = current_date.strftime("%Y-%m-%d")
                        schedule_by_date[date_str] = []
                        current_date += datetime.timedelta(days=1)

                    # Add entries to appropriate dates
                    for entry in schedule:
                        date_str = entry["start_time"].strftime("%Y-%m-%d")
                        if date_str in schedule_by_date:
                            schedule_by_date[date_str].append(entry)

                    # Generate calendar view
                    for date_str, entries in schedule_by_date.items():
                        # Convert to datetime for day of week
                        entry_date = datetime.datetime.strptime(
                            date_str, "%Y-%m-%d")
                        day_of_week = entry_date.strftime("%A")

                        response += f"## {date_str} ({day_of_week})\n\n"

                        if not entries:
                            response += "No bookings or exceptions\n\n"
                            continue

                        # Sort by start time
                        entries.sort(key=lambda x: x["start_time"])

                        for entry in entries:
                            start_time = entry["start_time"].strftime("%H:%M")
                            end_time = entry["end_time"].strftime("%H:%M")

                            if entry["type"] == "booking":
                                response += f"- **{start_time}-{end_time}**: {entry['title']} (by {entry['user_id']})\n"
                            else:  # exception
                                response += f"- **{start_time}-{end_time}**: {entry['status']} (Unavailable)\n"

                        response += "\n"

                    return response

        return None

    async def _process_existing_ticket(
        self, user_id: str, user_text: str, ticket: Ticket, timezone: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Process a message for an existing ticket. 
        Checks for handoff data and handles it with ticket-based handoff
        unless the target agent is set to skip ticket creation.
        """
        # Get assigned agent or re-route if needed
        agent_name = ticket.assigned_to
        if not agent_name:
            agent_name = await self.routing_service.route_query(user_text)
            self.ticket_service.update_ticket_status(
                ticket.id, TicketStatus.IN_PROGRESS, assigned_to=agent_name
            )

        # Get memory context if available
        memory_context = ""
        if self.memory_provider:
            memory_context = await self.memory_provider.retrieve(user_id)

        # Try to generate response
        full_response = ""
        handoff_info = None
        handoff_detected = False

        try:
            # Generate response with streaming
            async for chunk in self.agent_service.generate_response(
                agent_name=agent_name,
                user_id=user_id,
                query=user_text,
                memory_context=memory_context,
            ):
                # Detect possible handoff signals (JSON or prefix)
                if chunk.strip().startswith("HANDOFF:") or (
                    not full_response and chunk.strip().startswith("{")
                ):
                    handoff_detected = True
                    full_response += chunk
                    continue

                full_response += chunk
                yield chunk

            # After response generation, handle handoff if needed
            if handoff_detected or (
                not full_response.strip()
                and hasattr(self.agent_service, "_last_handoff")
            ):
                if hasattr(self.agent_service, "_last_handoff") and self.agent_service._last_handoff:
                    handoff_data = {
                        "handoff": self.agent_service._last_handoff}
                    target_agent = handoff_data["handoff"].get("target_agent")
                    reason = handoff_data["handoff"].get("reason")

                    if target_agent:
                        handoff_info = {
                            "target": target_agent, "reason": reason}

                        await self.handoff_service.process_handoff(
                            ticket.id,
                            agent_name,
                            handoff_info["target"],
                            handoff_info["reason"],
                        )

                    print(
                        f"Generating response from new agent: {target_agent}")
                    new_response_buffer = ""
                    async for chunk in self.agent_service.generate_response(
                        agent_name=target_agent,
                        user_id=user_id,
                        query=user_text,
                        memory_context=memory_context,
                    ):
                        new_response_buffer += chunk
                        yield chunk

                    full_response = new_response_buffer

                self.agent_service._last_handoff = None

            # Store conversation in memory
            if self.memory_provider:
                await self.memory_provider.store(
                    user_id,
                    [
                        {"role": "user", "content": user_text},
                        {
                            "role": "assistant",
                            "content": self._truncate(full_response, 2500),
                        },
                    ],
                )

        except Exception as e:
            print(f"Error processing ticket: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield f"I'm sorry, I encountered an error processing your request: {str(e)}"

    async def _process_new_ticket(
        self,
        user_id: str,
        user_text: str,
        complexity: Dict[str, Any],
        timezone: str = None,
    ) -> AsyncGenerator[str, None]:
        """Process a message creating a new ticket."""
        if self.task_planning_service:
            (
                needs_breakdown,
                reasoning,
            ) = await self.task_planning_service.needs_breakdown(user_text)

            if needs_breakdown:
                # Create ticket with planning status
                ticket = await self.ticket_service.get_or_create_ticket(
                    user_id, user_text, complexity
                )

                # Mark as planning
                self.ticket_service.update_ticket_status(
                    ticket.id, TicketStatus.PLANNING
                )

                # Generate subtasks
                subtasks = await self.task_planning_service.generate_subtasks(
                    ticket.id, user_text
                )

                # Generate response about the plan
                yield "I've analyzed your request and determined it's a complex task that should be broken down.\n\n"
                yield f"Task complexity assessment: {reasoning}\n\n"
                yield f"I've created a plan with {len(subtasks)} subtasks:\n\n"

                for i, subtask in enumerate(subtasks, 1):
                    yield f"{i}. {subtask.title}: {subtask.description}\n"

                yield f"\nEstimated total time: {sum(s.estimated_minutes for s in subtasks)} minutes\n"
                yield f"\nYou can check the plan status with !status {ticket.id}"
                return

        # Check if human approval is required
        is_simple_query = (
            complexity.get("t_shirt_size") in ["XS", "S"]
            and complexity.get("story_points", 3) <= 3
        )

        if self.require_human_approval and not is_simple_query:
            # Create ticket first
            ticket = await self.ticket_service.get_or_create_ticket(
                user_id, user_text, complexity
            )

            # Simulate project if service is available
            if self.project_simulation_service:
                simulation = await self.project_simulation_service.simulate_project(
                    user_text
                )
                yield "Analyzing project feasibility...\n\n"
                yield "## Project Simulation Results\n\n"
                yield f"**Complexity**: {simulation['complexity']['t_shirt_size']}\n"
                yield f"**Timeline**: {simulation['timeline']['realistic']} days\n"
                yield f"**Risk Level**: {simulation['risks']['overall_risk']}\n"
                yield f"**Recommendation**: {simulation['recommendation']}\n\n"

            # Submit for approval
            if self.project_approval_service:
                await self.project_approval_service.submit_for_approval(ticket)
                yield "\nThis project has been submitted for approval. You'll be notified once it's reviewed."
                return

        # Route query to appropriate agent
        agent_name = await self.routing_service.route_query(user_text)

        # Get memory context if available
        memory_context = ""
        if self.memory_provider:
            memory_context = await self.memory_provider.retrieve(user_id)

        # Create ticket
        ticket = await self.ticket_service.get_or_create_ticket(
            user_id, user_text, complexity
        )

        # Update with routing decision
        self.ticket_service.update_ticket_status(
            ticket.id, TicketStatus.ACTIVE, assigned_to=agent_name
        )

        # Generate initial response with streaming
        full_response = ""
        handoff_detected = False

        try:
            # Generate response with streaming
            async for chunk in self.agent_service.generate_response(
                agent_name, user_id, user_text, memory_context, temperature=0.7
            ):
                # Check if this looks like a JSON handoff
                if chunk.strip().startswith("{") and not handoff_detected:
                    handoff_detected = True
                    full_response += chunk
                    continue

                # Only yield if not a JSON chunk
                if not handoff_detected:
                    yield chunk
                    full_response += chunk

            # Handle handoff if detected
            if handoff_detected or (hasattr(self.agent_service, "_last_handoff") and self.agent_service._last_handoff):
                target_agent = None
                reason = "Handoff detected"

                # Process the handoff from _last_handoff property
                if hasattr(self.agent_service, "_last_handoff") and self.agent_service._last_handoff:
                    target_agent = self.agent_service._last_handoff.get(
                        "target_agent")
                    reason = self.agent_service._last_handoff.get(
                        "reason", "No reason provided")

                    if target_agent:
                        try:
                            # Process handoff and update ticket
                            await self.handoff_service.process_handoff(
                                ticket.id,
                                agent_name,
                                target_agent,
                                reason,
                            )

                            # Generate response from new agent
                            print(
                                f"Generating response from new agent after handoff: {target_agent}")
                            new_response = ""
                            async for chunk in self.agent_service.generate_response(
                                target_agent,
                                user_id,
                                user_text,
                                memory_context,
                                temperature=0.7
                            ):
                                yield chunk
                                new_response += chunk

                            # Update full response for storage
                            full_response = new_response
                        except ValueError as e:
                            print(f"Handoff failed: {e}")
                            yield f"\n\nNote: A handoff was attempted but failed: {str(e)}"

                    # Reset handoff state
                    self.agent_service._last_handoff = None

            # Check if ticket can be considered resolved
            resolution = await self._check_ticket_resolution(
                full_response, user_text
            )

            if resolution.status == "resolved" and resolution.confidence >= 0.7:
                self.ticket_service.mark_ticket_resolved(
                    ticket.id,
                    {
                        "confidence": resolution.confidence,
                        "reasoning": resolution.reasoning,
                    },
                )

                # Create NPS survey
                self.nps_service.create_survey(
                    user_id, ticket.id, agent_name)

            # Store in memory provider
            if self.memory_provider:
                await self.memory_provider.store(
                    user_id,
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": self._truncate(
                            full_response, 2500)},
                    ],
                )

            # Extract and store insights in background
            if full_response:
                asyncio.create_task(
                    self._extract_and_store_insights(
                        user_id, {"message": user_text,
                                  "response": full_response}
                    )
                )

        except Exception as e:
            print(f"Error in _process_new_ticket: {str(e)}")
            print(traceback.format_exc())
            yield f"I'm sorry, I encountered an error processing your request: {str(e)}"

    async def _process_human_agent_message(
        self, user_id: str, user_text: str
    ) -> AsyncGenerator[str, None]:
        """Process messages from human agents."""
        # Parse for target agent specification if available
        target_agent = None
        message = user_text

        # Check if message starts with @agent_name to target specific agent
        if user_text.startswith("@"):
            parts = user_text.split(" ", 1)
            potential_target = parts[0][1:]  # Remove the @ symbol
            if potential_target in self.agent_service.get_all_ai_agents():
                target_agent = potential_target
                message = parts[1] if len(parts) > 1 else ""

        # Handle specific commands
        if message.lower() == "!agents":
            yield self._get_agent_directory()
            return

        if message.lower().startswith("!status"):
            yield await self._get_system_status()
            return

        # If no target and no command, provide help
        if not target_agent and not message.strip().startswith("!"):
            yield "Please specify a target AI agent with @agent_name or use a command. Available commands:\n"
            yield "- !agents: List available agents\n"
            yield "- !status: Show system status"
            return

        # Process with target agent
        if target_agent:
            memory_context = ""
            if self.memory_provider:
                memory_context = await self.memory_provider.retrieve(target_agent)

            async for chunk in self.agent_service.generate_response(
                target_agent, user_id, message, memory_context, temperature=0.7
            ):
                yield chunk

    def _get_agent_directory(self) -> str:
        """Get formatted list of all registered agents."""
        ai_agents = self.agent_service.get_all_ai_agents()
        human_agents = self.agent_service.get_all_human_agents()
        specializations = self.agent_service.get_specializations()

        result = "# Registered Agents\n\n"

        # AI Agents
        result += "## AI Agents\n\n"
        for name in ai_agents:
            result += (
                f"- **{name}**: {specializations.get(name, 'No specialization')}\n"
            )

        # Human Agents
        if human_agents:
            result += "\n## Human Agents\n\n"
            for agent_id, agent in human_agents.items():
                status = agent.get("availability_status", "unknown")
                name = agent.get("name", agent_id)
                status_emoji = "🟢" if status == "available" else "🔴"
                result += f"- {status_emoji} **{name}**: {agent.get('specialization', 'No specialization')}\n"

        return result

    async def _get_system_status(self) -> str:
        """Get system status summary."""
        # Get ticket metrics
        open_tickets = self.ticket_service.ticket_repository.count(
            {"status": {"$ne": TicketStatus.RESOLVED}}
        )
        resolved_today = self.ticket_service.ticket_repository.count(
            {
                "status": TicketStatus.RESOLVED,
                "resolved_at": {
                    "$gte": datetime.datetime.now(datetime.timezone.utc)
                    - datetime.timedelta(days=1)
                },
            }
        )

        # Get memory metrics
        memory_count = 0
        try:
            memory_count = self.memory_service.memory_repository.db.count_documents(
                "collective_memory", {}
            )
        except Exception:
            pass

        result = "# System Status\n\n"
        result += f"- Open tickets: {open_tickets}\n"
        result += f"- Resolved in last 24h: {resolved_today}\n"
        result += f"- Collective memory entries: {memory_count}\n"

        return result

    async def _check_ticket_resolution(
        self, response: str, query: str
    ) -> TicketResolution:
        """Determine if a ticket can be considered resolved based on the response."""
        # Get first AI agent for analysis
        first_agent = next(iter(self.agent_service.get_all_ai_agents().keys()))

        prompt = f"""
        Analyze this conversation and determine if the user query has been fully resolved.
        
        USER QUERY: {query}
        
        ASSISTANT RESPONSE: {response}
        
        Determine if this query is:
        1. "resolved" - The user's question/request has been fully addressed
        2. "needs_followup" - The assistant couldn't fully address the issue or more information is needed
        3. "cannot_determine" - Cannot tell if the issue is resolved
        
        Return a structured output with:
        - "status": One of the above values
        - "confidence": A score from 0.0 to 1.0 indicating confidence in this assessment
        - "reasoning": Brief explanation for your decision
        - "suggested_actions": Array of recommended next steps (if any)
        """

        try:
            # Use structured output parsing with the Pydantic model directly
            resolution = await self.agent_service.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="You are a resolution analysis system. Analyze conversations and determine if queries have been resolved.",
                model_class=TicketResolution,
                model=self.agent_service.ai_agents[first_agent].get(
                    "model", "gpt-4o-mini"),
                temperature=0.2,
            )
            return resolution
        except Exception as e:
            print(f"Exception in resolution check: {e}")

        # Default fallback if anything fails
        return TicketResolution(
            status="cannot_determine",
            confidence=0.2,
            reasoning="Failed to analyze resolution status",
            suggested_actions=["Review conversation manually"]
        )

    async def _assess_task_complexity(self, query: str) -> Dict[str, Any]:
        """Assess the complexity of a task using standardized metrics."""
        # Special handling for very simple messages
        if len(query.strip()) <= 10 and query.lower().strip() in ["test", "hello", "hi", "hey", "ping", "thanks"]:
            print(f"Using pre-defined complexity for simple message: {query}")
            return {
                "t_shirt_size": "XS",
                "story_points": 1,
                "estimated_minutes": 5,
                "technical_complexity": 1,
                "domain_knowledge": 1,
            }

        # Get first AI agent for analysis
        first_agent = next(iter(self.agent_service.get_all_ai_agents().keys()))

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
            response_text = ""
            async for chunk in self.agent_service.generate_response(
                first_agent,
                "complexity_assessor",
                prompt,
                "",  # No memory context needed
                stream=False,
                temperature=0.2,
                response_format={"type": "json_object"},
            ):
                response_text += chunk

            if not response_text.strip():
                print("Empty response from complexity assessment")
                return {
                    "t_shirt_size": "S",
                    "story_points": 2,
                    "estimated_minutes": 15,
                    "technical_complexity": 3,
                    "domain_knowledge": 2,
                }

            complexity_data = json.loads(response_text)
            print(f"Successfully parsed complexity: {complexity_data}")
            return complexity_data
        except Exception as e:
            print(f"Error assessing complexity: {e}")
            print(f"Failed response text: '{response_text}'")
            return {
                "t_shirt_size": "S",
                "story_points": 2,
                "estimated_minutes": 15,
                "technical_complexity": 3,
                "domain_knowledge": 2,
            }

    async def _extract_and_store_insights(
        self, user_id: str, conversation: Dict[str, str]
    ) -> None:
        """Extract insights from conversation and store in collective memory."""
        try:
            # Extract insights
            insights = await self.memory_service.extract_insights(conversation)

            # Store them if any found
            if insights:
                await self.memory_service.store_insights(user_id, insights)

            return len(insights)
        except Exception as e:
            print(f"Error extracting insights: {e}")
            return 0

    def _truncate(self, text: str, limit: int = 2500) -> str:
        """Truncate text to be within limits."""
        if len(text) <= limit:
            return text

        # Try to truncate at a sentence boundary
        truncated = text[:limit]
        last_period = truncated.rfind(".")
        if (
            last_period > limit * 0.8
        ):  # Only use period if it's reasonably close to the end
            return truncated[: last_period + 1]

        return truncated + "..."

    async def _store_conversation(self, user_id: str, user_text: str, response_text: str) -> None:
        """Store conversation history in memory provider."""
        if self.memory_provider:
            try:
                await self.memory_provider.store(
                    user_id,
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": response_text},
                    ],
                )
            except Exception as e:
                print(f"Error storing conversation: {e}")
                # Don't let memory storage errors affect the user experience

#############################################
# FACTORY AND DEPENDENCY INJECTION
#############################################


class SolanaAgentFactory:
    """Factory for creating and wiring components of the Solana Agent system."""

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> QueryProcessor:
        """Create the agent system from configuration."""
        # Create adapters
        db_adapter = MongoDBAdapter(
            connection_string=config["mongo"]["connection_string"],
            database_name=config["mongo"]["database"],
        )

        llm_adapter = OpenAIAdapter(
            api_key=config["openai"]["api_key"],
            model=config.get("openai", {}).get("default_model", "gpt-4o-mini"),
        )

        mongo_memory = MongoMemoryProvider(db_adapter)

        zep_memory = None
        if "zep" in config:
            zep_memory = ZepMemoryAdapter(
                api_key=config["zep"].get("api_key"),
                base_url=config["zep"].get("base_url"),
            )

        memory_provider = DualMemoryProvider(mongo_memory, zep_memory)

        # Create vector store provider if configured
        vector_provider = None
        if "qdrant" in config:
            vector_provider = QdrantAdapter(
                url=config["qdrant"].get("url", "http://localhost:6333"),
                api_key=config["qdrant"].get("api_key"),
                collection_name=config["qdrant"].get(
                    "collection", "solana_agent"),
                embedding_model=config["qdrant"].get(
                    "embedding_model", "text-embedding-3-small"
                ),
            )
        if "pinecone" in config:
            vector_provider = PineconeAdapter(
                api_key=config["pinecone"]["api_key"],
                index_name=config["pinecone"]["index"],
                embedding_model=config["pinecone"].get(
                    "embedding_model", "text-embedding-3-small"
                ),
            )

        # Create organization mission if specified in config
        organization_mission = None
        if "organization" in config:
            org_config = config["organization"]
            organization_mission = OrganizationMission(
                mission_statement=org_config.get("mission_statement", ""),
                values=[{"name": k, "description": v}
                        for k, v in org_config.get("values", {}).items()],
                goals=org_config.get("goals", []),
                guidance=org_config.get("guidance", "")
            )

        # Create repositories
        ticket_repo = MongoTicketRepository(db_adapter)
        handoff_repo = MongoHandoffRepository(db_adapter)
        nps_repo = MongoNPSSurveyRepository(db_adapter)
        memory_repo = MongoMemoryRepository(db_adapter, vector_provider)
        human_agent_repo = MongoHumanAgentRegistry(db_adapter)
        ai_agent_repo = MongoAIAgentRegistry(db_adapter)

        # Create services
        agent_service = AgentService(
            llm_adapter, human_agent_repo, ai_agent_repo, organization_mission, config)

        # Debug the agent service tool registry to confirm tools were registered
        print(
            f"Agent service tools after initialization: {agent_service.tool_registry.list_all_tools()}")

        routing_service = RoutingService(
            llm_adapter,
            agent_service,
            router_model=config.get("router_model", "gpt-4o-mini"),
        )

        ticket_service = TicketService(ticket_repo)

        handoff_service = HandoffService(
            handoff_repo, ticket_repo, agent_service)

        memory_service = MemoryService(memory_repo, llm_adapter)

        nps_service = NPSService(nps_repo, ticket_repo)

        # Create critic service if enabled
        critic_service = None
        if config.get("enable_critic", True):
            critic_service = CriticService(llm_adapter)

        # Create task planning service
        task_planning_service = TaskPlanningService(
            ticket_repo, llm_adapter, agent_service
        )

        notification_service = NotificationService(
            human_agent_registry=human_agent_repo,
            tool_registry=agent_service.tool_registry
        )

        project_approval_service = ProjectApprovalService(
            ticket_repo, human_agent_repo, notification_service
        )
        project_simulation_service = ProjectSimulationService(
            llm_adapter, task_planning_service
        )

        # Create scheduling repository and service
        scheduling_repository = SchedulingRepository(db_adapter)

        scheduling_service = SchedulingService(
            scheduling_repository=scheduling_repository,
            task_planning_service=task_planning_service,
            agent_service=agent_service
        )

        # Update task_planning_service with scheduling_service if needed
        if task_planning_service:
            task_planning_service.scheduling_service = scheduling_service

        # Initialize plugin system if plugins directory is configured)
        agent_service.plugin_manager = PluginManager()
        loaded_plugins = agent_service.plugin_manager.load_all_plugins()
        print(f"Loaded {loaded_plugins} plugins")

        # Get list of all agents defined in config
        config_defined_agents = [agent["name"]
                                 for agent in config.get("ai_agents", [])]

        # Sync MongoDB with config-defined agents (delete any agents not in config)
        all_db_agents = ai_agent_repo.db.find(ai_agent_repo.collection, {})
        db_agent_names = [agent["name"] for agent in all_db_agents]

        # Find agents that exist in DB but not in config
        agents_to_delete = [
            name for name in db_agent_names if name not in config_defined_agents]

        # Delete those agents
        for agent_name in agents_to_delete:
            print(
                f"Deleting agent '{agent_name}' from MongoDB - no longer defined in config")
            ai_agent_repo.db.delete_one(
                ai_agent_repo.collection, {"name": agent_name})
            if agent_name in ai_agent_repo.ai_agents_cache:
                del ai_agent_repo.ai_agents_cache[agent_name]

        # Register predefined agents if any
        for agent_config in config.get("ai_agents", []):
            agent_service.register_ai_agent(
                name=agent_config["name"],
                instructions=agent_config["instructions"],
                specialization=agent_config["specialization"],
                model=agent_config.get("model", "gpt-4o-mini"),
            )

            # Register tools for this agent if specified
            if "tools" in agent_config:
                for tool_name in agent_config["tools"]:
                    # Print available tools before registering
                    print(
                        f"Available tools before registering {tool_name}: {agent_service.tool_registry.list_all_tools()}")
                    try:
                        agent_service.register_tool_for_agent(
                            agent_config["name"], tool_name
                        )
                        print(
                            f"Successfully registered {tool_name} for agent {agent_config['name']}")
                    except ValueError as e:
                        print(
                            f"Error registering tool {tool_name} for agent {agent_config['name']}: {e}"
                        )

        # Also support global tool registrations
        if "agent_tools" in config:
            for agent_name, tools in config["agent_tools"].items():
                for tool_name in tools:
                    try:
                        agent_service.register_tool_for_agent(
                            agent_name, tool_name)
                    except ValueError as e:
                        print(f"Error registering tool: {e}")

        # Create main processor
        query_processor = QueryProcessor(
            agent_service=agent_service,
            routing_service=routing_service,
            ticket_service=ticket_service,
            handoff_service=handoff_service,
            memory_service=memory_service,
            nps_service=nps_service,
            critic_service=critic_service,
            memory_provider=memory_provider,
            enable_critic=config.get("enable_critic", True),
            router_model=config.get("router_model", "gpt-4o-mini"),
            task_planning_service=task_planning_service,
            project_approval_service=project_approval_service,
            project_simulation_service=project_simulation_service,
            require_human_approval=config.get("require_human_approval", False),
            scheduling_service=scheduling_service,
            stalled_ticket_timeout=config.get("stalled_ticket_timeout", 60),
        )

        return query_processor


#############################################
# MULTI-TENANT SUPPORT
#############################################


class TenantContext:
    """Manages tenant-specific context and configuration."""

    def __init__(self, tenant_id: str, tenant_config: Dict[str, Any] = None):
        self.tenant_id = tenant_id
        self.config = tenant_config or {}
        self.metadata = {}

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get tenant-specific configuration value."""
        return self.config.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set tenant metadata."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get tenant metadata."""
        return self.metadata.get(key, default)


class TenantManager:
    """Manager for handling multiple tenants in a multi-tenant environment."""

    def __init__(self, default_config: Dict[str, Any] = None):
        self.tenants = {}
        self.default_config = default_config or {}
        self._repositories = {}  # Cache for tenant repositories
        self._services = {}  # Cache for tenant services

    def register_tenant(
        self, tenant_id: str, config: Dict[str, Any] = None
    ) -> TenantContext:
        """Register a new tenant with optional custom config."""
        tenant_config = self.default_config.copy()
        if config:
            # Deep merge configs
            self._deep_merge(tenant_config, config)

        context = TenantContext(tenant_id, tenant_config)
        self.tenants[tenant_id] = context
        return context

    def get_tenant(self, tenant_id: str) -> Optional[TenantContext]:
        """Get tenant context by ID."""
        return self.tenants.get(tenant_id)

    def get_repository(self, tenant_id: str, repo_type: str) -> Any:
        """Get or create a repository for a specific tenant."""
        cache_key = f"{tenant_id}:{repo_type}"

        if cache_key in self._repositories:
            return self._repositories[cache_key]

        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        # Create repository with tenant-specific DB connection
        if repo_type == "ticket":
            repo = self._create_tenant_ticket_repo(tenant)
        elif repo_type == "memory":
            repo = self._create_tenant_memory_repo(tenant)
        elif repo_type == "human_agent":
            repo = self._create_tenant_human_agent_repo(tenant)
        # Add other repository types as needed
        else:
            raise ValueError(f"Unknown repository type: {repo_type}")

        self._repositories[cache_key] = repo
        return repo

    def get_service(self, tenant_id: str, service_type: str) -> Any:
        """Get or create a service for a specific tenant."""
        cache_key = f"{tenant_id}:{service_type}"

        if cache_key in self._services:
            return self._services[cache_key]

        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        # Create service with tenant-specific dependencies
        if service_type == "agent":
            service = self._create_tenant_agent_service(tenant)
        elif service_type == "query_processor":
            service = self._create_tenant_query_processor(tenant)
        # Add other service types as needed
        else:
            raise ValueError(f"Unknown service type: {service_type}")

        self._services[cache_key] = service
        return service

    def _create_tenant_db_adapter(self, tenant: TenantContext) -> DataStorageProvider:
        """Create a tenant-specific database adapter."""
        # Get tenant-specific connection info
        connection_string = tenant.get_config_value("mongo", {}).get(
            "connection_string",
            self.default_config.get("mongo", {}).get("connection_string"),
        )

        # You can either use different connection strings per tenant
        # or append tenant ID to database name for simpler isolation
        db_name = f"{self.default_config.get('mongo', {}).get('database', 'solana_agent')}_{tenant.tenant_id}"

        return MongoDBAdapter(
            connection_string=connection_string, database_name=db_name
        )

    def _create_tenant_ticket_repo(self, tenant: TenantContext) -> TicketRepository:
        """Create a tenant-specific ticket repository."""
        db_adapter = self._create_tenant_db_adapter(tenant)
        return MongoTicketRepository(db_adapter)

    def _create_tenant_memory_repo(self, tenant: TenantContext) -> MemoryRepository:
        """Create a tenant-specific memory repository."""
        db_adapter = self._create_tenant_db_adapter(tenant)

        # Get tenant-specific vector store if available
        vector_provider = None
        if "pinecone" in tenant.config or "qdrant" in tenant.config:
            vector_provider = self._create_tenant_vector_provider(tenant)

        return MongoMemoryRepository(db_adapter, vector_provider)

    def _create_tenant_human_agent_repo(self, tenant: TenantContext) -> AgentRegistry:
        """Create a tenant-specific human agent registry."""
        db_adapter = self._create_tenant_db_adapter(tenant)
        return MongoHumanAgentRegistry(db_adapter)

    def _create_tenant_vector_provider(
        self, tenant: TenantContext
    ) -> VectorStoreProvider:
        """Create a tenant-specific vector store provider."""
        # Check which vector provider to use based on tenant config
        if "qdrant" in tenant.config:
            return self._create_tenant_qdrant_adapter(tenant)
        elif "pinecone" in tenant.config:
            return self._create_tenant_pinecone_adapter(tenant)
        else:
            return None

    def _create_tenant_pinecone_adapter(self, tenant: TenantContext) -> PineconeAdapter:
        """Create a tenant-specific Pinecone adapter."""
        config = tenant.config.get("pinecone", {})

        # Use tenant-specific index or namespace
        index_name = config.get(
            "index",
            self.default_config.get("pinecone", {}).get(
                "index", "solana_agent"),
        )

        return PineconeAdapter(
            api_key=config.get(
                "api_key", self.default_config.get(
                    "pinecone", {}).get("api_key")
            ),
            index_name=index_name,
            embedding_model=config.get(
                "embedding_model", "text-embedding-3-small"),
        )

    def _create_tenant_qdrant_adapter(self, tenant: TenantContext) -> "QdrantAdapter":
        """Create a tenant-specific Qdrant adapter."""
        config = tenant.config.get("qdrant", {})

        # Use tenant-specific collection
        collection_name = (
            f"tenant_{tenant.tenant_id}_{config.get('collection', 'solana_agent')}"
        )

        return QdrantAdapter(
            url=config.get(
                "url",
                self.default_config.get("qdrant", {}).get(
                    "url", "http://localhost:6333"
                ),
            ),
            api_key=config.get(
                "api_key", self.default_config.get("qdrant", {}).get("api_key")
            ),
            collection_name=collection_name,
            embedding_model=config.get(
                "embedding_model", "text-embedding-3-small"),
        )

    def _create_tenant_agent_service(self, tenant: TenantContext) -> AgentService:
        """Create a tenant-specific agent service."""
        # Get or create LLM provider for the tenant
        llm_provider = self._create_tenant_llm_provider(tenant)

        # Get human agent registry
        human_agent_registry = self.get_repository(
            tenant.tenant_id, "human_agent")

        return AgentService(llm_provider, human_agent_registry)

    def _create_tenant_llm_provider(self, tenant: TenantContext) -> LLMProvider:
        """Create a tenant-specific LLM provider."""
        config = tenant.config.get("openai", {})

        return OpenAIAdapter(
            api_key=config.get(
                "api_key", self.default_config.get("openai", {}).get("api_key")
            ),
            model=config.get(
                "default_model",
                self.default_config.get("openai", {}).get(
                    "default_model", "gpt-4o-mini"
                ),
            ),
        )

    def _create_tenant_query_processor(self, tenant: TenantContext) -> QueryProcessor:
        """Create a tenant-specific query processor with all services."""
        # Get repositories
        ticket_repo = self.get_repository(tenant.tenant_id, "ticket")
        memory_repo = self.get_repository(tenant.tenant_id, "memory")
        human_agent_repo = self.get_repository(tenant.tenant_id, "human_agent")

        # Create or get required services
        agent_service = self.get_service(tenant.tenant_id, "agent")

        # Get LLM provider
        llm_provider = self._create_tenant_llm_provider(tenant)

        # Create other required services
        routing_service = RoutingService(
            llm_provider,
            agent_service,
            router_model=tenant.get_config_value(
                "router_model", "gpt-4o-mini"),
        )

        ticket_service = TicketService(ticket_repo)
        handoff_service = HandoffService(
            MongoHandoffRepository(self._create_tenant_db_adapter(tenant)),
            ticket_repo,
            agent_service,
        )
        memory_service = MemoryService(memory_repo, llm_provider)
        nps_service = NPSService(
            MongoNPSSurveyRepository(self._create_tenant_db_adapter(tenant)),
            ticket_repo,
        )

        # Create optional services
        critic_service = None
        if tenant.get_config_value("enable_critic", True):
            critic_service = CriticService(llm_provider)

        # Create memory provider if configured
        memory_provider = None
        if "zep" in tenant.config:
            memory_provider = ZepMemoryAdapter(
                api_key=tenant.get_config_value("zep", {}).get("api_key"),
                base_url=tenant.get_config_value("zep", {}).get("base_url"),
            )

        # Create task planning service
        task_planning_service = TaskPlanningService(
            ticket_repo, llm_provider, agent_service
        )

        # Create notification and approval services
        notification_service = NotificationService(human_agent_repo)
        project_approval_service = ProjectApprovalService(
            ticket_repo, human_agent_repo, notification_service
        )
        project_simulation_service = ProjectSimulationService(
            llm_provider, task_planning_service, ticket_repo
        )

        # Create scheduling repository and service
        tenant_db_adapter = self._create_tenant_db_adapter(
            tenant)  # Get the DB adapter properly
        scheduling_repository = SchedulingRepository(
            tenant_db_adapter)  # Use the correct adapter

        scheduling_service = SchedulingService(
            scheduling_repository=scheduling_repository,
            task_planning_service=task_planning_service,
            agent_service=agent_service
        )

        # Update task_planning_service with scheduling_service if needed
        if task_planning_service:
            task_planning_service.scheduling_service = scheduling_service

        # Create query processor
        return QueryProcessor(
            agent_service=agent_service,
            routing_service=routing_service,
            ticket_service=ticket_service,
            handoff_service=handoff_service,
            memory_service=memory_service,
            nps_service=nps_service,
            critic_service=critic_service,
            memory_provider=memory_provider,
            enable_critic=tenant.get_config_value("enable_critic", True),
            router_model=tenant.get_config_value(
                "router_model", "gpt-4o-mini"),
            task_planning_service=task_planning_service,
            project_approval_service=project_approval_service,
            project_simulation_service=project_simulation_service,
            require_human_approval=tenant.get_config_value(
                "require_human_approval", False
            ),
            scheduling_service=scheduling_service,
            stalled_ticket_timeout=tenant.get_config_value(
                "stalled_ticket_timeout"),
        )

    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """Deep merge source dict into target dict."""
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(target[key], value)
            else:
                target[key] = value


class MultitenantSolanaAgentFactory:
    """Factory for creating multi-tenant Solana Agent systems."""

    def __init__(self, global_config: Dict[str, Any]):
        """Initialize the factory with global configuration."""
        self.tenant_manager = TenantManager(global_config)

    def register_tenant(
        self, tenant_id: str, tenant_config: Dict[str, Any] = None
    ) -> None:
        """Register a new tenant with optional configuration overrides."""
        self.tenant_manager.register_tenant(tenant_id, tenant_config)

    def get_processor(self, tenant_id: str) -> QueryProcessor:
        """Get a query processor for a specific tenant."""
        return self.tenant_manager.get_service(tenant_id, "query_processor")

    def get_agent_service(self, tenant_id: str) -> AgentService:
        """Get an agent service for a specific tenant."""
        return self.tenant_manager.get_service(tenant_id, "agent")

    def register_ai_agent(
        self,
        tenant_id: str,
        name: str,
        instructions: str,
        specialization: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Register an AI agent for a specific tenant."""
        agent_service = self.get_agent_service(tenant_id)
        agent_service.register_ai_agent(
            name, instructions, specialization, model)


class MultitenantSolanaAgent:
    """Multi-tenant client interface for Solana Agent."""

    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """Initialize the multi-tenant agent system from config."""
        if (
            config is None and config_path is None
        ):  # Check for None specifically, not falsy values
            raise ValueError("Either config or config_path must be provided")

        if config_path:
            with open(config_path, "r") as f:
                if config_path.endswith(".json"):
                    config = json.load(f)
                else:
                    # Assume it's a Python file
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    config = config_module.config

        # Initialize with the config (may be empty dict, but that's still valid)
        self.factory = MultitenantSolanaAgentFactory(config or {})

    def register_tenant(
        self, tenant_id: str, tenant_config: Dict[str, Any] = None
    ) -> None:
        """Register a new tenant."""
        self.factory.register_tenant(tenant_id, tenant_config)

    async def process(
        self, tenant_id: str, user_id: str, message: str
    ) -> AsyncGenerator[str, None]:
        """Process a user message for a specific tenant."""
        processor = self.factory.get_processor(tenant_id)
        async for chunk in processor.process(user_id, message):
            yield chunk

    def register_agent(
        self,
        tenant_id: str,
        name: str,
        instructions: str,
        specialization: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Register an AI agent for a specific tenant."""
        self.factory.register_ai_agent(
            tenant_id, name, instructions, specialization, model
        )

    def register_human_agent(
        self,
        tenant_id: str,
        agent_id: str,
        name: str,
        specialization: str,
        notification_handler=None,
    ) -> None:
        """Register a human agent for a specific tenant."""
        agent_service = self.factory.get_agent_service(tenant_id)
        agent_service.register_human_agent(
            agent_id=agent_id,
            name=name,
            specialization=specialization,
            notification_handler=notification_handler,
        )


#############################################
# SIMPLIFIED CLIENT INTERFACE
#############################################


class SolanaAgent:
    """Simplified client interface for interacting with the agent system."""

    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """Initialize the agent system from config file or dictionary."""
        if not config and not config_path:
            raise ValueError("Either config or config_path must be provided")

        if config_path:
            with open(config_path, "r") as f:
                if config_path.endswith(".json"):
                    config = json.load(f)
                else:
                    # Assume it's a Python file
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    config = config_module.config

        self.processor = SolanaAgentFactory.create_from_config(config)

    async def process(self, user_id: str, message: str) -> AsyncGenerator[str, None]:
        """Process a user message and return the response stream."""
        async for chunk in self.processor.process(user_id, message):
            yield chunk

    def register_agent(
        self,
        name: str,
        instructions: str,
        specialization: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Register a new AI agent."""
        self.processor.agent_service.register_ai_agent(
            name=name,
            instructions=instructions,
            specialization=specialization,
            model=model,
        )

    def register_human_agent(
        self, agent_id: str, name: str, specialization: str, notification_handler=None
    ) -> None:
        """Register a human agent."""
        self.processor.agent_service.register_human_agent(
            agent_id=agent_id,
            name=name,
            specialization=specialization,
            notification_handler=notification_handler,
        )

    async def get_pending_surveys(self, user_id: str) -> List[Dict[str, Any]]:
        """Get pending surveys for a user."""
        if not self.processor or not hasattr(self.processor, "nps_service"):
            return []

        # Query for pending surveys from the NPS service
        surveys = self.processor.nps_service.nps_repository.db.find(
            "nps_surveys",
            {
                "user_id": user_id,
                "status": "pending",
                "created_at": {"$gte": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)}
            }
        )

        return surveys

    async def submit_survey_response(self, survey_id: str, score: int, feedback: str = "") -> bool:
        """Submit a response to an NPS survey."""
        if not self.processor or not hasattr(self.processor, "nps_service"):
            return False

        # Process the survey response
        return self.processor.nps_service.process_response(survey_id, score, feedback)

    async def get_paginated_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "asc"  # "asc" for oldest-first, "desc" for newest-first
    ) -> Dict[str, Any]:
        """
        Get paginated message history for a user, with user messages and assistant responses grouped together.

        Args:
            user_id: User ID to retrieve history for
            page_num: Page number (starting from 1)
            page_size: Number of messages per page (number of conversation turns)
            sort_order: "asc" for chronological order, "desc" for reverse chronological

        Returns:
            Dictionary containing paginated results and metadata
        """
        # Access the MongoDB adapter through the processor
        db_adapter = None

        # Find the MongoDB adapter - it could be in different locations depending on setup
        if hasattr(self.processor, "ticket_service") and hasattr(self.processor.ticket_service, "ticket_repository"):
            if hasattr(self.processor.ticket_service.ticket_repository, "db"):
                db_adapter = self.processor.ticket_service.ticket_repository.db

        if not db_adapter:
            return {
                "data": [],
                "total": 0,
                "page": page_num,
                "page_size": page_size,
                "total_pages": 0,
                "error": "Database adapter not found"
            }

        try:
            # Set the sort direction
            sort_direction = pymongo.ASCENDING if sort_order.lower() == "asc" else pymongo.DESCENDING

            # Get total count of user messages (each user message represents one conversation turn)
            total_user_messages = db_adapter.count_documents(
                "messages", {"user_id": user_id, "role": "user"}
            )

            # We'll determine total conversation turns based on user messages
            total_turns = total_user_messages

            # Calculate skip amount for pagination (in terms of user messages)
            skip = (page_num - 1) * page_size

            # Get all messages for this user, sorted by timestamp
            all_messages = db_adapter.find(
                "messages",
                {"user_id": user_id},
                sort=[("timestamp", sort_direction)],
                limit=0  # No limit initially, we'll filter after grouping
            )

            # Group messages into conversation turns
            conversation_turns = []
            current_turn = None

            for message in all_messages:
                if message["role"] == "user":
                    # Start a new conversation turn
                    if current_turn:
                        conversation_turns.append(current_turn)

                    current_turn = {
                        "user_message": message["content"],
                        "assistant_message": None,
                        "timestamp": message["timestamp"].isoformat() if isinstance(message["timestamp"], datetime.datetime) else message["timestamp"],
                    }
                elif message["role"] == "assistant" and current_turn and current_turn["assistant_message"] is None:
                    # Add this as the response to the current turn
                    current_turn["assistant_message"] = message["content"]
                    current_turn["response_timestamp"] = message["timestamp"].isoformat() if isinstance(
                        message["timestamp"], datetime.datetime) else message["timestamp"]

            # Add the last turn if it exists
            if current_turn:
                conversation_turns.append(current_turn)

            # Apply pagination to conversation turns
            paginated_turns = conversation_turns[skip:skip + page_size]

            # Format response with pagination metadata
            return {
                "data": paginated_turns,
                "total": total_turns,
                "page": page_num,
                "page_size": page_size,
                "total_pages": (total_turns // page_size) + (1 if total_turns % page_size > 0 else 0)
            }

        except Exception as e:
            print(f"Error retrieving message history: {e}")
            return {
                "data": [],
                "total": 0,
                "page": page_num,
                "page_size": page_size,
                "total_pages": 0,
                "error": str(e)
            }

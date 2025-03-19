"""
Agent service implementation.

This service manages AI and human agents, their registration, tool assignments,
and response generation.
"""
import datetime as main_datetime
from datetime import datetime
import json
from typing import AsyncGenerator, Dict, List, Optional, Any

from solana_agent.interfaces import AgentService as AgentServiceInterface
from solana_agent.interfaces import LLMProvider
from solana_agent.interfaces import AgentRepository
from solana_agent.interfaces import ToolRegistry
from solana_agent.domains import AIAgent, HumanAgent, OrganizationMission
from solana_agent.interfaces.services import HandoffObserver


class AgentService(AgentServiceInterface):
    """Service for managing agents and generating responses."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        agent_repository: AgentRepository,
        organization_mission: Optional[OrganizationMission] = None,
        config: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """Initialize the agent service.

        Args:
            llm_provider: Provider for language model interactions
            agent_repository: Repository for agent data
            organization_mission: Optional organization mission and values
            config: Optional service configuration
        """
        self.llm_provider = llm_provider
        self.agent_repository = agent_repository
        self.organization_mission = organization_mission
        self.config = config or {}

        # Initialize tool registry with concrete implementation
        if tool_registry:
            self.tool_registry = tool_registry
        else:
            # Import the concrete implementation
            from solana_agent.plugins.registry import ToolRegistry as ConcreteToolRegistry
            self.tool_registry = ConcreteToolRegistry()

        # Will be set by factory if plugin system is enabled
        self.plugin_manager = None

        self._handoff_observers: List[HandoffObserver] = []

    def add_handoff_observer(self, observer: HandoffObserver) -> None:
        """Add a handoff observer."""
        self._handoff_observers.append(observer)

    def _notify_handoff(self, handoff_data: Dict[str, Any]) -> None:
        """Notify observers of a handoff."""
        for observer in self._handoff_observers:
            observer.on_handoff(handoff_data)

    def register_ai_agent(
        self, name: str, instructions: str, specialization: str, model: str = "gpt-4o-mini"
    ) -> None:
        """Register an AI agent with its specialization.

        Args:
            name: Agent name
            instructions: Agent instructions
            specialization: Agent specialization
            model: LLM model to use
        """
        agent = AIAgent(
            name=name,
            instructions=instructions,
            specialization=specialization,
            model=model
        )
        self.agent_repository.save_ai_agent(agent)

    def register_human_agent(
        self, agent_id: str, name: str, specialization: str, notification_handler=None
    ) -> str:
        """Register a human agent and return its ID.

        Args:
            agent_id: Agent ID
            name: Agent name
            specialization: Agent specialization
            notification_handler: Optional handler for notifications

        Returns:
            Agent ID
        """
        agent = HumanAgent(
            id=agent_id,
            name=name,
            specializations=[specialization],
            availability=True,
            notification_handler=notification_handler
        )
        self.agent_repository.save_human_agent(agent)
        return agent_id

    def get_agent_by_name(self, name: str) -> Optional[Any]:
        """Get an agent by name.

        Args:
            name: Agent name

        Returns:
            AI or human agent
        """
        # Try AI agents first
        agent = self.agent_repository.get_ai_agent_by_name(name)
        if agent:
            return agent

        # Check human agents by name
        human_agents = self.agent_repository.get_all_human_agents()
        for agent in human_agents:
            if agent.name == name:
                return agent

        return None

    def get_agent_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for an agent.

        Args:
            agent_name: Agent name

        Returns:
            System prompt
        """
        agent = self.agent_repository.get_ai_agent_by_name(agent_name)
        if not agent:
            return ""

        # Build system prompt
        system_prompt = f"You are {agent.name}, an AI assistant with the following instructions:\n\n"
        system_prompt += agent.instructions

        # add current time
        system_prompt += f"\n\nThe current time is {datetime.now(tz=main_datetime.timezone.utc)}\n\n."

        # Add mission and values if available
        if self.organization_mission:
            system_prompt += f"\n\nORGANIZATION MISSION:\n{self.organization_mission.mission_statement}"

            if self.organization_mission.values:
                values_text = "\n".join([
                    f"- {value.get('name', '')}: {value.get('description', '')}"
                    for value in self.organization_mission.values
                ])
                system_prompt += f"\n\nORGANIZATION VALUES:\n{values_text}"

        # Add organization goals if available
        if self.organization_mission and self.organization_mission.goals:
            goals_text = "\n".join(
                [f"- {goal}" for goal in self.organization_mission.goals])
            system_prompt += f"\n\nORGANIZATION GOALS:\n{goals_text}"

        return system_prompt

    def get_all_ai_agents(self) -> Dict[str, AIAgent]:
        """Get all registered AI agents.

        Returns:
            Dictionary mapping agent names to agents
        """
        agents = self.agent_repository.get_all_ai_agents()
        return {agent.name: agent for agent in agents}

    def get_all_human_agents(self) -> Dict[str, HumanAgent]:
        """Get all registered human agents.

        Returns:
            Dictionary mapping agent IDs to agents
        """
        agents = self.agent_repository.get_all_human_agents()
        return {agent.id: agent for agent in agents}

    def get_specializations(self) -> Dict[str, str]:
        """Get all registered specializations.

        Returns:
            Dictionary mapping specialization names to descriptions
        """
        specializations = {}

        # Gather from AI agents
        ai_agents = self.agent_repository.get_all_ai_agents()
        for agent in ai_agents:
            if agent.specialization:
                specializations[agent.specialization] = f"AI expertise in {agent.specialization}"

        # Gather from human agents
        human_agents = self.agent_repository.get_all_human_agents()
        for agent in human_agents:
            for spec in agent.specializations:
                if spec not in specializations:
                    specializations[spec] = f"Human expertise in {spec}"

        return specializations

    def find_agents_by_specialization(self, specialization: str) -> List[str]:
        """Find agents that have a specific specialization.

        Args:
            specialization: Specialization to search for

        Returns:
            List of agent names/IDs
        """
        agent_ids = []

        # Check AI agents
        ai_agents = self.agent_repository.get_all_ai_agents()
        for agent in ai_agents:
            if agent.specialization.lower() == specialization.lower():
                agent_ids.append(agent.name)

        # Check human agents
        human_agents = self.agent_repository.get_all_human_agents()
        for agent in human_agents:
            if any(spec.lower() == specialization.lower() for spec in agent.specializations):
                agent_ids.append(agent.id)

        return agent_ids

    def assign_tool_for_agent(self, agent_name: str, tool_name: str) -> bool:
        """Assign a tool to an agent.

        Args:
            agent_name: Agent name
            tool_name: Tool name

        Returns:
            True if assignment was successful
        """
        if not self.tool_registry:
            return False

        # Check if agent exists
        agent = self.agent_repository.get_ai_agent_by_name(agent_name)
        if not agent:
            return False

        return self.tool_registry.assign_tool_to_agent(agent_name, tool_name)

    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get tools available to an agent.

        Args:
            agent_name: Agent name

        Returns:
            List of tool configurations
        """
        if not self.tool_registry:
            return []

        return self.tool_registry.get_agent_tools(agent_name)

    def execute_tool(self, agent_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on behalf of an agent.

        Args:
            agent_name: Agent name
            tool_name: Tool name
            parameters: Tool parameters

        Returns:
            Tool execution result
        """
        if not self.tool_registry:
            return {"status": "error", "message": "Tool registry not available"}

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return {"status": "error", "message": f"Tool '{tool_name}' not found"}

        # Check if agent has access to this tool
        agent_tools = self.tool_registry.get_agent_tools(agent_name)
        if not any(t.get("name") == tool_name for t in agent_tools):
            return {
                "status": "error",
                "message": f"Agent '{agent_name}' doesn't have access to tool '{tool_name}'"
            }

        try:
            return tool.execute(**parameters)
        except Exception as e:
            return {"status": "error", "message": f"Error executing tool: {str(e)}"}

    def agent_exists(self, agent_name_or_id: str) -> bool:
        """Check if agent exists.

        Args:
            agent_name_or_id: Agent name or ID

        Returns:
            True if agent exists
        """
        # Check AI agents
        ai_agent = self.agent_repository.get_ai_agent_by_name(agent_name_or_id)
        if ai_agent:
            return True

        # Check human agents
        human_agents = self.agent_repository.get_all_human_agents()
        for agent in human_agents:
            if agent.id == agent_name_or_id or agent.name == agent_name_or_id:
                return True

        # This likely was missing or had different logic
        return False

    def has_specialization(self, agent_id: str, specialization: str) -> bool:
        """Check if agent has the specified specialization.

        Args:
            agent_id: Agent ID or name
            specialization: Specialization to check for

        Returns:
            True if the agent has the specialization
        """
        # Check AI agents
        ai_agent = self.agent_repository.get_ai_agent_by_name(agent_id)
        if ai_agent:
            # This was likely the bug - comparing with specific specialization
            return ai_agent.specialization.lower() == specialization.lower()

        # Check human agents
        human_agent = None
        for agent in self.agent_repository.get_all_human_agents():
            if agent.id == agent_id or agent.name == agent_id:
                human_agent = agent
                break

        if human_agent:
            return any(spec.lower() == specialization.lower() for spec in human_agent.specializations)

        return False

    def list_active_agents(self) -> List[str]:
        """List all active agent IDs.

        Returns:
            List of active agent IDs/names
        """
        active_agents = []

        # Get AI agents (all are considered active)
        ai_agents = self.agent_repository.get_all_ai_agents()
        active_agents.extend([agent.name for agent in ai_agents])

        # Get human agents with availability=True
        human_agents = self.agent_repository.get_all_human_agents()
        active_agents.extend(
            [agent.id for agent in human_agents if agent.availability])

        return active_agents

    def has_pending_handoff(self, agent_name: str) -> bool:
        """Check if an agent has any pending handoffs.

        Args:
            agent_name: Name or ID of the agent to check

        Returns:
            True if the agent has pending handoffs, False otherwise
        """
        # We need to check if we have a handoff_service available
        if hasattr(self, 'handoff_service') and self.handoff_service:
            # Delegate to handoff service to check for pending handoffs
            return self.handoff_service.has_pending_handoffs_for_agent(agent_name)

        # If no handoff service is available, assume no pending handoffs
        return False

    def clear_pending_handoff(self, agent_name: str) -> bool:
        """Clear any pending handoffs for an agent.

        Args:
            agent_name: Name or ID of the agent whose handoffs should be cleared

        Returns:
            bool: True if handoffs were cleared successfully, False otherwise
        """
        try:
            # Check if handoff service is available
            if hasattr(self, 'handoff_service') and self.handoff_service:
                # Delegate to handoff service to clear pending handoffs
                return self.handoff_service.clear_pending_handoffs_for_agent(agent_name)

            # If we get here with no handoff service, log warning and return True
            # since there can't be pending handoffs without a service
            print(
                f"Warning: No handoff service available when clearing handoffs for agent: {agent_name}")
            return True

        except Exception as e:
            print(
                f"Error clearing pending handoffs for agent {agent_name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    async def generate_response(
        self,
        agent_name: str,
        user_id: str,
        query: str,
        memory_context: str = "",
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a response with handoff support."""
        agent = self.agent_repository.get_ai_agent_by_name(agent_name)
        if not agent:
            yield f"Agent '{agent_name}' not found."
            return

        # Get system prompt with tools and handoff instructions
        system_prompt = self.get_agent_system_prompt(agent_name)

        # Add memory context
        if memory_context:
            system_prompt += f"\n\nContext: {memory_context}"

        try:
            # Generate initial response
            full_response = ""
            json_response = ""
            is_json = False

            async for chunk in self.llm_provider.generate_text(
                user_id=user_id,
                prompt=query,
                system_prompt=system_prompt,
                model=agent.model,
                needs_search=True,  # Enable web search by default
                **kwargs
            ):
                # Check for JSON start
                if chunk.strip().startswith("{"):
                    is_json = True
                    json_response = chunk
                    continue

                # Collect JSON or yield normal text
                if is_json:
                    json_response += chunk
                    try:
                        # Try to parse complete JSON
                        data = json.loads(json_response)

                        # Handle handoff
                        if "handoff" in data:
                            handoff_data = data["handoff"]
                            # Store handoff data for query service
                            self._last_handoff = {
                                "target_agent": handoff_data.get("target_agent"),
                                "reason": handoff_data.get("reason", "No reason provided"),
                                "ticket_id": kwargs.get("ticket_id")
                            }
                            return

                        # If valid JSON but not handoff, yield it
                        yield json_response
                        break
                    except json.JSONDecodeError:
                        # Not complete JSON yet, keep collecting
                        continue
                else:
                    full_response += chunk
                    yield chunk

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield f"I apologize, but I encountered an error: {str(e)}"

    def _get_tool_usage_prompt(self, agent_name: str) -> str:
        """Generate JSON-based instructions for tool usage and handoffs."""
        # Get tools assigned to this agent
        tools = self.get_agent_tools(agent_name)

        # Get available agents for handoff
        available_agents = [agent.name for agent in self.agent_repository.get_all_ai_agents()
                            if agent.name != agent_name]

        # Build the prompt with both tool and handoff instructions
        tools_section = ""
        if tools:
            # Get actual tool names
            available_tool_names = [tool.get("name", "") for tool in tools]
            tools_json = json.dumps(tools, indent=2)

            # Only show example with actually available tools
            tool_example = ""
            if "search_internet" in available_tool_names:
                tool_example = """
        For latest news query:
        {
            "tool_call": {
                "name": "search_internet",
                "parameters": {
                    "query": "latest Solana blockchain news March 2025"
                }
            }
        }"""

            tools_section = f"""
        AVAILABLE TOOLS:
        {tools_json}
        
        TOOL USAGE FORMAT:
        {{
            "tool_call": {{
                "name": "<one_of:{', '.join(available_tool_names)}>",
                "parameters": {{
                    // parameters as specified in tool definition above
                }}
            }}
        }}
        
        {tool_example if tool_example else ''}"""

        handoff_section = f"""
        HANDOFF FORMAT:
        {{
            "handoff": {{
                "target_agent": "<one_of:{', '.join(available_agents)}>",
                "reason": "detailed reason for handoff"
            }}
        }}
        
        AVAILABLE AGENTS FOR HANDOFF:
        {', '.join(available_agents)}"""

        return f"""
        {tools_section}
        
        {handoff_section}
        
        RESPONSE RULES:
        1. For specialized queries beyond your expertise:
           - Use the handoff format to transfer to appropriate agent
           - Only use agents from the available agents list
           - Provide clear handoff reason
        
        2. For tool usage:
           - Only use tools from the AVAILABLE TOOLS list above
           - Follow the exact parameter format shown in the tool definition
        
        3. Format Requirements:
           - Return ONLY the JSON object
           - No explanation text before or after
           - Use exact tool names as shown in AVAILABLE TOOLS
           - Use exact agent names as shown in AVAILABLE AGENTS
        """

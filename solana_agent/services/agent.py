"""
Agent service implementation.

This service manages AI and human agents, their registration, tool assignments,
and response generation.
"""
import datetime as main_datetime
from datetime import datetime
import json
from typing import AsyncGenerator, Dict, List, Optional, Any

from solana_agent.interfaces.services.agent import AgentService as AgentServiceInterface
from solana_agent.interfaces.providers.llm import LLMProvider
from solana_agent.interfaces.repositories.agent import AgentRepository
from solana_agent.interfaces.plugins.plugins import ToolRegistry
from solana_agent.domains.agents import AIAgent, OrganizationMission


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

    def get_agent_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for an agent.

        Args:
            agent_name: Agent name

        Returns:
            System prompt
        """
        agent = self.agent_repository.get_ai_agent_by_name(agent_name)

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

        return specializations

    def assign_tool_for_agent(self, agent_name: str, tool_name: str) -> bool:
        """Assign a tool to an agent.

        Args:
            agent_name: Agent name
            tool_name: Tool name

        Returns:
            True if assignment was successful
        """
        return self.tool_registry.assign_tool_to_agent(agent_name, tool_name)

    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get tools available to an agent.

        Args:
            agent_name: Agent name

        Returns:
            List of tool configurations
        """
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

    async def generate_response(
        self,
        agent_name: str,
        user_id: str,
        query: str,
        memory_context: str = "",
        **kwargs
    ) -> AsyncGenerator[str, None]:  # pragma: no cover
        """Generate a response with tool execution support."""
        agent = self.agent_repository.get_ai_agent_by_name(agent_name)
        if not agent:
            yield f"Agent '{agent_name}' not found."
            return

        # Get system prompt and add tool instructions
        system_prompt = self.get_agent_system_prompt(agent_name)
        if self.tool_registry:
            tool_usage_prompt = self._get_tool_usage_prompt(agent_name)
            if tool_usage_prompt:
                system_prompt = f"{system_prompt}\n\n{tool_usage_prompt}"

        # Add User ID context
        system_prompt += f"\n\n User ID: {user_id}"

        # Add memory context
        if memory_context:
            system_prompt += f"\n\n Memory Context: {memory_context}"

        try:
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

                        # Handle tool call
                        if "tool_call" in data:
                            tool_data = data["tool_call"]
                            tool_name = tool_data.get("name")
                            parameters = tool_data.get("parameters", {})

                            if tool_name:
                                result = self.execute_tool(
                                    agent_name, tool_name, parameters)
                                if result.get("status") == "success":
                                    yield result.get("result", "")
                                else:
                                    yield f"I apologize, but I encountered an issue: {result.get('message', 'Unknown error')}"
                                break
                        else:
                            # If JSON but not a tool call, yield as text
                            yield json_response
                            break
                    except json.JSONDecodeError:
                        # Not complete JSON yet, keep collecting
                        continue
                else:
                    yield chunk

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield f"I apologize, but I encountered an error: {str(e)}"

    def _get_tool_usage_prompt(self, agent_name: str) -> str:
        """Generate JSON-based instructions for tool usage."""
        # Get tools assigned to this agent
        tools = self.get_agent_tools(agent_name)
        if not tools:
            return ""

        # Get actual tool names
        available_tool_names = [tool.get("name", "") for tool in tools]
        tools_json = json.dumps(tools, indent=2)

        # Create tool example if search is available
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

        return f"""
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
    
    {tool_example if tool_example else ''}
    
    RESPONSE RULES:
    1. For tool usage:
       - Only use tools from the AVAILABLE TOOLS list above
       - Follow the exact parameter format shown in the tool definition
       - Include "March 2025" in any search queries for current information
    
    2. Format Requirements:
       - Return ONLY the JSON object for tool calls
       - No explanation text before or after
       - Use exact tool names as shown in AVAILABLE TOOLS
    """

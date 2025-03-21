"""
Agent service implementation.

This service manages AI and human agents, their registration, tool assignments,
and response generation.
"""
import datetime as main_datetime
from datetime import datetime
import json
from pathlib import Path
from typing import AsyncGenerator, BinaryIO, Dict, List, Literal, Optional, Any, Union

from solana_agent.interfaces.services.agent import AgentService as AgentServiceInterface
from solana_agent.interfaces.providers.llm import LLMProvider
from solana_agent.interfaces.repositories.agent import AgentRepository
from solana_agent.interfaces.plugins.plugins import ToolRegistry
from solana_agent.domains.agent import AIAgent, OrganizationMission


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
        self, name: str, instructions: str, specialization: str,
    ) -> None:
        """Register an AI agent with its specialization.

        Args:
            name: Agent name
            instructions: Agent instructions
            specialization: Agent specialization
        """
        agent = AIAgent(
            name=name,
            instructions=instructions,
            specialization=specialization,
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
        query: Union[str, Path, BinaryIO],
        memory_context: str = "",
        output_format: Literal["text", "audio"] = "text",
        voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                       "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
        audio_instructions: Optional[str] = None,
    ) -> AsyncGenerator[Union[str, bytes], None]:  # pragma: no cover
        """Generate a response with support for text/audio input/output.

        Args:
            agent_name: Agent name
            user_id: User ID
            query: Text query or audio file input
            memory_context: Optional conversation context
            output_format: Response format ("text" or "audio")
            voice: Voice to use for audio output
            audio_instructions: Optional instructions for audio synthesis

        Yields:
            Text chunks or audio bytes depending on output_format
        """
        agent = self.agent_repository.get_ai_agent_by_name(agent_name)
        if not agent:
            error_msg = f"Agent '{agent_name}' not found."
            if output_format == "audio":
                async for chunk in self.llm_provider.tts(error_msg, voice=voice):
                    yield chunk
            else:
                yield error_msg
            return

        try:
            # Handle audio input if provided
            query_text = ""
            if not isinstance(query, str):
                async for transcript in self.llm_provider.transcribe_audio(query):
                    query_text += transcript
            else:
                query_text = query

            # Get system prompt and add tool instructions
            system_prompt = self.get_agent_system_prompt(agent_name)
            if self.tool_registry:
                tool_usage_prompt = self._get_tool_usage_prompt(agent_name)
                if tool_usage_prompt:
                    system_prompt = f"{system_prompt}\n\n{tool_usage_prompt}"

            # Add User ID and memory context
            system_prompt += f"\n\nUser ID: {user_id}"
            if memory_context:
                system_prompt += f"\n\nMemory Context: {memory_context}"

            # Buffer for collecting text when generating audio
            text_buffer = ""

            # Generate and stream response
            async for chunk in self.llm_provider.generate_text(
                prompt=query_text,
                system_prompt=system_prompt,
            ):
                if chunk.strip().startswith("{"):
                    # Handle tool calls
                    result = await self._handle_tool_call(
                        agent_name, chunk, output_format, voice
                    )
                    if output_format == "audio":
                        async for audio_chunk in self.llm_provider.tts(result, instructions=audio_instructions, voice=voice):
                            yield audio_chunk
                    else:
                        yield result
                else:
                    if output_format == "audio":
                        # Buffer text until we have a complete sentence
                        text_buffer += chunk
                        if any(punct in chunk for punct in ".!?"):
                            async for audio_chunk in self.llm_provider.tts(
                                text_buffer, instructions=audio_instructions, voice=voice
                            ):
                                yield audio_chunk
                            text_buffer = ""
                    else:
                        yield chunk

            # Handle any remaining text in buffer
            if output_format == "audio" and text_buffer:
                async for audio_chunk in self.llm_provider.tts(
                    text_buffer, instructions=audio_instructions, voice=voice
                ):
                    yield audio_chunk

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            if output_format == "audio":
                async for chunk in self.llm_provider.tts(error_msg, instructions=audio_instructions, voice=voice):
                    yield chunk
            else:
                yield error_msg

            print(f"Error in generate_response: {str(e)}")
            import traceback
            print(traceback.format_exc())

    async def _handle_tool_call(
        self,
        agent_name: str,
        json_chunk: str,
    ) -> str:
        """Handle tool calls and return formatted response."""
        try:
            data = json.loads(json_chunk)
            if "tool_call" in data:
                tool_data = data["tool_call"]
                tool_name = tool_data.get("name")
                parameters = tool_data.get("parameters", {})

                if tool_name:
                    result = self.execute_tool(
                        agent_name, tool_name, parameters)
                    if result.get("status") == "success":
                        return result.get("result", "")
                    else:
                        return f"I apologize, but I encountered an issue: {result.get('message', 'Unknown error')}"
            return json_chunk
        except json.JSONDecodeError:
            return json_chunk

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

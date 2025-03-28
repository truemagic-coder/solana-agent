"""
Agent service implementation.

This service manages AI and human agents, their registration, tool assignments,
and response generation.
"""
import asyncio
import datetime as main_datetime
from datetime import datetime
import json
from typing import AsyncGenerator, Dict, List, Literal, Optional, Any, Union

from solana_agent.interfaces.services.agent import AgentService as AgentServiceInterface
from solana_agent.interfaces.providers.llm import LLMProvider
from solana_agent.interfaces.plugins.plugins import ToolRegistry as ToolRegistryInterface
from solana_agent.plugins.registry import ToolRegistry
from solana_agent.domains.agent import AIAgent, BusinessMission


class AgentService(AgentServiceInterface):
    """Service for managing agents and generating responses."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        business_mission: Optional[BusinessMission] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the agent service.

        Args:
            llm_provider: Provider for language model interactions
            business_mission: Optional business mission and values
            config: Optional service configuration
        """
        self.llm_provider = llm_provider
        self.business_mission = business_mission
        self.config = config or {}
        self.last_text_response = ""
        self.tool_registry = ToolRegistry(config=self.config)
        self.agents: List[AIAgent] = []

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
        self.agents.append(agent)

    def get_agent_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for an agent.

        Args:
            agent_name: Agent name

        Returns:
            System prompt
        """

        # Get agent by name
        agent = next((a for a in self.agents if a.name == agent_name), None)

        # Build system prompt
        system_prompt = f"You are {agent.name}, an AI assistant with the following instructions:\n\n"
        system_prompt += agent.instructions

        # add current time
        system_prompt += f"\n\nThe current time is {datetime.now(tz=main_datetime.timezone.utc)}\n\n."

        # Add mission and values if available
        if self.business_mission:
            system_prompt += f"\n\nBUSINESS MISSION:\n{self.business_mission.mission}"
            system_prompt += f"\n\nVOICE OF THE BRAND:\n{self.business_mission.voice}"

            if self.business_mission.values:
                values_text = "\n".join([
                    f"- {value.get('name', '')}: {value.get('description', '')}"
                    for value in self.business_mission.values
                ])
                system_prompt += f"\n\nBUSINESS VALUES:\n{values_text}"

            # Add goals if available
            if self.business_mission.goals:
                goals_text = "\n".join(
                    [f"- {goal}" for goal in self.business_mission.goals])
                system_prompt += f"\n\nBUSINESS GOALS:\n{goals_text}"

        return system_prompt

    def get_all_ai_agents(self) -> Dict[str, AIAgent]:
        """Get all registered AI agents.

        Returns:
            Dictionary mapping agent names to agents
        """
        return {agent.name: agent for agent in self.agents}

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

    async def execute_tool(self, agent_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on behalf of an agent."""

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
            result = await tool.execute(**parameters)
            return result
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {"status": "error", "message": f"Error executing tool: {str(e)}"}

    async def generate_response(
        self,
        agent_name: str,
        user_id: str,
        query: Union[str, bytes],
        memory_context: str = "",
        output_format: Literal["text", "audio"] = "text",
        audio_voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                             "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
        audio_instructions: Optional[str] = None,
        audio_output_format: Literal['mp3', 'opus',
                                     'aac', 'flac', 'wav', 'pcm'] = "aac",
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
        prompt: Optional[str] = None,
    ) -> AsyncGenerator[Union[str, bytes], None]:  # pragma: no cover
        """Generate a response with support for text/audio input/output.

        Args:
            agent_name: Agent name
            user_id: User ID
            query: Text query or audio bytes
            memory_context: Optional conversation context
            output_format: Response format ("text" or "audio")
            audio_voice: Voice to use for audio output
            audio_instructions: Optional instructions for audio synthesis
            audio_output_format: Audio output format
            audio_input_format: Audio input format
            prompt: Optional prompt for the agent

        Yields:
            Text chunks or audio bytes depending on output_format
        """
        agent = next((a for a in self.agents if a.name == agent_name), None)
        if not agent:
            error_msg = f"Agent '{agent_name}' not found."
            if output_format == "audio":
                async for chunk in self.llm_provider.tts(error_msg, instructions=audio_instructions, response_format=audio_output_format, voice=audio_voice):
                    yield chunk
            else:
                yield error_msg
            return

        try:
            # Handle audio input if provided
            query_text = ""
            if not isinstance(query, str):
                async for transcript in self.llm_provider.transcribe_audio(query, input_format=audio_input_format):
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
                system_prompt += f"\n\nMEMORY CONTEXT: {memory_context}"
            if prompt:
                system_prompt += f"\n\nADDITIONAL PROMPT: {prompt}"

            # Keep track of the complete text response
            complete_text_response = ""
            json_buffer = ""
            is_json = False
            text_buffer = ""

            # Generate and stream response
            async for chunk in self.llm_provider.generate_text(
                prompt=query_text,
                system_prompt=system_prompt,
            ):
                # Check for JSON start
                if chunk.strip().startswith("{") and not is_json:
                    is_json = True
                    json_buffer = chunk
                    continue

                # Collect JSON or handle normal text
                if is_json:
                    json_buffer += chunk
                    try:
                        # Try to parse complete JSON
                        data = json.loads(json_buffer)

                        # Valid JSON found, handle it
                        if "tool_call" in data:
                            # Process tool call with existing method
                            response_text = await self._handle_tool_call(
                                agent_name=agent_name,
                                json_chunk=json_buffer
                            )

                            system_prompt = system_prompt + \
                                "\n DO NOT make any tool calls or return JSON."

                            user_prompt = f"\n USER QUERY: {query_text} \n"
                            user_prompt += f"\n TOOL RESPONSE: {response_text} \n"

                            # Collect all processed text first
                            processed_text = ""
                            async for processed_chunk in self.llm_provider.generate_text(
                                prompt=user_prompt,
                                system_prompt=system_prompt,
                            ):
                                processed_text += processed_chunk
                                # For text output, yield chunks as they come
                                if output_format == "text":
                                    yield processed_chunk

                            # Add to complete response
                            complete_text_response += processed_text

                            # For audio output, process the complete text
                            if output_format == "audio":
                                async for audio_chunk in self.llm_provider.tts(
                                    text=processed_text,
                                    voice=audio_voice,
                                    response_format=audio_output_format
                                ):
                                    yield audio_chunk
                        else:
                            # For non-tool JSON, still capture the text
                            complete_text_response += json_buffer

                            if output_format == "audio":
                                async for audio_chunk in self.llm_provider.tts(
                                    text=json_buffer,
                                    voice=audio_voice,
                                    response_format=audio_output_format
                                ):
                                    yield audio_chunk
                            else:
                                yield json_buffer

                        # Reset JSON handling
                        is_json = False
                        json_buffer = ""

                    except json.JSONDecodeError:
                        pass
                else:
                    # For regular text, always add to the complete response
                    complete_text_response += chunk

                    # Handle audio buffering or direct text output
                    if output_format == "audio":
                        text_buffer += chunk
                        if any(punct in chunk for punct in ".!?"):
                            async for audio_chunk in self.llm_provider.tts(
                                text=text_buffer,
                                voice=audio_voice,
                                response_format=audio_output_format
                            ):
                                yield audio_chunk
                            text_buffer = ""
                    else:
                        yield chunk

            # Handle any remaining text or incomplete JSON
            remaining_text = ""
            if text_buffer:
                remaining_text += text_buffer
            if is_json and json_buffer:
                remaining_text += json_buffer

            if remaining_text:
                # Add remaining text to complete response
                complete_text_response += remaining_text

                if output_format == "audio":
                    async for audio_chunk in self.llm_provider.tts(
                        text=remaining_text,
                        voice=audio_voice,
                        response_format=audio_output_format
                    ):
                        yield audio_chunk
                else:
                    yield remaining_text

            # Store the complete text response for the caller to access
            # This needs to be done in the query service using the self.last_text_response
            self.last_text_response = complete_text_response

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            if output_format == "audio":
                async for chunk in self.llm_provider.tts(error_msg, voice=audio_voice, response_format=audio_output_format):
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
                    result = await self.execute_tool(
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
    
    RESPONSE RULES:
    1. For tool usage:
       - Only use tools from the AVAILABLE TOOLS list above
       - Follow the exact parameter format shown in the tool definition
    
    2. Format Requirements:
       - Return ONLY the JSON object for tool calls
       - No explanation text before or after
       - Use exact tool names as shown in AVAILABLE TOOLS
    """

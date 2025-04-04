"""
Agent service implementation.

This service manages AI and human agents, their registration, tool assignments,
and response generation.
"""
import asyncio
from copy import deepcopy
import datetime as main_datetime
from datetime import datetime
import json
from typing import AsyncGenerator, Dict, List, Literal, Optional, Any, Union

from solana_agent.interfaces.services.agent import AgentService as AgentServiceInterface
from solana_agent.interfaces.providers.llm import LLMProvider
from solana_agent.plugins.manager import PluginManager
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

        self.plugin_manager = PluginManager(
            config=self.config,
            tool_registry=self.tool_registry,
        )

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
        audio_instructions: str = "You speak in a friendly and helpful manner.",
        audio_output_format: Literal['mp3', 'opus',
                                     'aac', 'flac', 'wav', 'pcm'] = "aac",
        audio_input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
        prompt: Optional[str] = None,
        internet_search: bool = True,
    ) -> AsyncGenerator[Union[str, bytes], None]:  # pragma: no cover
        """Generate a response with support for text/audio input/output."""
        agent = next((a for a in self.agents if a.name == agent_name), None)
        if not agent:
            error_msg = f"Agent '{agent_name}' not found."
            if output_format == "audio":
                async for chunk in self.llm_provider.tts(error_msg, instructions=audio_instructions,
                                                         response_format=audio_output_format, voice=audio_voice):
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

            # Get system prompt
            system_prompt = self.get_agent_system_prompt(agent_name)

            # Add User ID and memory context
            system_prompt += f"\n\nUser ID: {user_id}"
            if memory_context:
                system_prompt += f"\n\nMEMORY CONTEXT: {memory_context}"
            if prompt:
                system_prompt += f"\n\nADDITIONAL PROMPT: {prompt}"

            # make tool calling prompt
            tool_calling_system_prompt = deepcopy(system_prompt)
            if self.tool_registry:
                tool_usage_prompt = self._get_tool_usage_prompt(agent_name)
                if tool_usage_prompt:
                    tool_calling_system_prompt += f"\n\nTOOL CALLING PROMPT: {tool_usage_prompt}"

            # Variables for tracking the response
            complete_text_response = ""

            # For audio output, we'll collect everything first
            full_response_buffer = ""

            # Variables for handling JSON processing
            json_buffer = ""
            is_json = False

            # Generate and stream response
            async for chunk in self.llm_provider.generate_text(
                prompt=query_text,
                system_prompt=tool_calling_system_prompt,
                internet_search=internet_search,
            ):
                # Check if the chunk is JSON or a tool call
                if (chunk.strip().startswith("{") or "{\"tool_call\":" in chunk) and not is_json:
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
                            response_text = await self._handle_tool_call(
                                agent_name=agent_name,
                                json_chunk=json_buffer
                            )

                            # Update system prompt to prevent further tool calls
                            tool_system_prompt = system_prompt + \
                                "\n DO NOT make any tool calls or return JSON."

                            # Create prompt with tool response
                            user_prompt = f"\n USER QUERY: {query_text} \n"
                            user_prompt += f"\n TOOL RESPONSE: {response_text} \n"

                            # For text output, process chunks directly
                            if output_format == "text":
                                # Stream text response for text output
                                async for processed_chunk in self.llm_provider.generate_text(
                                    prompt=user_prompt,
                                    system_prompt=tool_system_prompt,
                                ):
                                    complete_text_response += processed_chunk
                                    yield processed_chunk
                            else:
                                # For audio output, collect the full tool response first
                                tool_response = ""
                                async for processed_chunk in self.llm_provider.generate_text(
                                    prompt=user_prompt,
                                    system_prompt=tool_system_prompt,
                                ):
                                    tool_response += processed_chunk

                                # Add to our complete text record and full audio buffer
                                tool_response = self._clean_for_audio(
                                    tool_response)
                                complete_text_response += tool_response
                                full_response_buffer += tool_response
                        else:
                            # For non-tool JSON, still capture the text
                            complete_text_response += json_buffer

                            if output_format == "text":
                                yield json_buffer
                            else:
                                # Add to full response buffer for audio
                                full_response_buffer += json_buffer

                        # Reset JSON handling
                        is_json = False
                        json_buffer = ""

                    except json.JSONDecodeError:
                        # JSON not complete yet, continue collecting
                        pass
                else:
                    # For regular text
                    complete_text_response += chunk

                    if output_format == "text":
                        # For text output, yield directly
                        yield chunk
                    else:
                        # For audio output, add to the full response buffer
                        full_response_buffer += chunk

            # Handle any leftover JSON buffer
            if json_buffer:
                complete_text_response += json_buffer
                if output_format == "text":
                    yield json_buffer
                else:
                    full_response_buffer += json_buffer

            # For audio output, now process the complete response
            if output_format == "audio" and full_response_buffer:
                # Clean text before TTS
                full_response_buffer = self._clean_for_audio(
                    full_response_buffer)

                # Process the entire response with TTS
                async for audio_chunk in self.llm_provider.tts(
                    text=full_response_buffer,
                    voice=audio_voice,
                    response_format=audio_output_format,
                    instructions=audio_instructions
                ):
                    yield audio_chunk

            # Store the complete text response
            self.last_text_response = complete_text_response

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            if output_format == "audio":
                async for chunk in self.llm_provider.tts(
                    error_msg,
                    voice=audio_voice,
                    response_format=audio_output_format,
                    instructions=audio_instructions
                ):
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
                    # Execute the tool and get the result
                    result = await self.execute_tool(agent_name, tool_name, parameters)

                    if result.get("status") == "success":
                        tool_result = result.get("result", "")
                        return tool_result
                    else:
                        error_message = f"I apologize, but I encountered an issue with the {tool_name} tool: {result.get('message', 'Unknown error')}"
                        print(f"Tool error: {error_message}")
                        return error_message
                else:
                    return "Tool name was not provided in the tool call."
            else:
                print(f"JSON received but no tool_call found: {json_chunk}")

            # If we get here, it wasn't properly handled as a tool
            return f"The following request was not processed as a valid tool call:\n{json_chunk}"
        except json.JSONDecodeError as e:
            print(f"JSON decode error in tool call: {e}")
            return json_chunk
        except Exception as e:
            print(f"Unexpected error in tool call handling: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return f"Error processing tool call: {str(e)}"

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

    def _clean_for_audio(self, text: str) -> str:
        """Remove Markdown formatting, emojis, and non-pronounceable characters from text.

        Args:
            text: Input text with potential Markdown formatting and special characters

        Returns:
            Clean text without Markdown, emojis, and special characters
        """
        import re

        if not text:
            return ""

        # Remove Markdown links - [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        # Remove inline code with backticks
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Remove bold formatting - **text** or __text__ -> text
        text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)

        # Remove italic formatting - *text* or _text_ -> text
        text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)

        # Remove headers - ## Header -> Header
        text = re.sub(r'^\s*#+\s*(.*?)$', r'\1', text, flags=re.MULTILINE)

        # Remove blockquotes - > Text -> Text
        text = re.sub(r'^\s*>\s*(.*?)$', r'\1', text, flags=re.MULTILINE)

        # Remove horizontal rules (---, ***, ___)
        text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

        # Remove list markers - * Item or - Item or 1. Item -> Item
        text = re.sub(r'^\s*[-*+]\s+(.*?)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+(.*?)$', r'\1', text, flags=re.MULTILINE)

        # Remove multiple consecutive newlines (keep just one)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove emojis and other non-pronounceable characters
        # Common emoji Unicode ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0000257F"  # Enclosed characters
            "\U00002600-\U000026FF"  # Miscellaneous Symbols
            "\U00002700-\U000027BF"  # Dingbats
            "\U0000FE00-\U0000FE0F"  # Variation Selectors
            "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub(r' ', text)

        # Replace special characters that can cause issues with TTS
        text = re.sub(r'[^\w\s\.\,\;\:\?\!\'\"\-\(\)]', ' ', text)

        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

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
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
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
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

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
        prompt: Optional[str] = None,
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
            # Get system prompt
            system_prompt = self.get_agent_system_prompt(agent_name)

            # Add User ID and memory context
            system_prompt += f"\n\nUser ID: {user_id}"
            if memory_context:
                system_prompt += f"\n\nMEMORY CONTEXT: {memory_context}"
            if prompt:
                system_prompt += f"\n\nADDITIONAL PROMPT: {prompt}"

            # Add tool usage prompt if tools are available
            tool_calling_system_prompt = deepcopy(system_prompt)
            if self.tool_registry:
                tool_usage_prompt = self._get_tool_usage_prompt(agent_name)
                if tool_usage_prompt:
                    tool_calling_system_prompt += f"\n\nTOOL CALLING PROMPT: {tool_usage_prompt}"
                    print(
                        f"Tools available to agent {agent_name}: {[t.get('name') for t in self.get_agent_tools(agent_name)]}")

            # Variables for tracking the complete response
            complete_text_response = ""
            full_response_buffer = ""

            # Variables for robust handling of tool call markers that may be split across chunks
            tool_buffer = ""
            pending_chunk = ""  # To hold text that might contain partial markers
            is_tool_call = False
            window_size = 30  # Increased window size for better detection

            # Define start and end markers
            start_marker = "[TOOL]"
            end_marker = "[/TOOL]"

            # Generate and stream response (ALWAYS use non-realtime for text generation)
            print(
                f"Generating response with {len(query)} characters of query text")
            async for chunk in self.llm_provider.generate_text(
                prompt=query,
                system_prompt=tool_calling_system_prompt,
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
            ):
                # If we have pending text from the previous chunk, combine it with this chunk
                if pending_chunk:
                    combined_chunk = pending_chunk + chunk
                    pending_chunk = ""  # Reset pending chunk
                else:
                    combined_chunk = chunk

                # STEP 1: Check for tool call start marker
                if start_marker in combined_chunk and not is_tool_call:
                    print(
                        f"Found tool start marker in chunk of length {len(combined_chunk)}")
                    is_tool_call = True

                    # Extract text before the marker and the marker itself with everything after
                    start_pos = combined_chunk.find(start_marker)
                    before_marker = combined_chunk[:start_pos]
                    after_marker = combined_chunk[start_pos:]

                    # Yield text that appeared before the marker
                    if before_marker and output_format == "text":
                        yield before_marker

                    # Start collecting the tool call
                    tool_buffer = after_marker
                    continue  # Skip to next chunk

                # STEP 2: Handle ongoing tool call collection
                if is_tool_call:
                    tool_buffer += combined_chunk

                    # Check if the tool call is complete
                    if end_marker in tool_buffer:
                        print(
                            f"Tool call complete, buffer size: {len(tool_buffer)}")

                        # Process the tool call
                        response_text = await self._handle_tool_call(
                            agent_name=agent_name,
                            tool_text=tool_buffer
                        )

                        # Clean the response to remove any markers or formatting
                        response_text = self._clean_tool_response(
                            response_text)
                        print(
                            f"Tool execution complete, result size: {len(response_text)}")

                        # Create new prompt with search/tool results
                        # Using "Search Result" instead of "TOOL RESPONSE" to avoid model repeating "TOOL"
                        user_prompt = f"{query}\n\nSearch Result: {response_text}"
                        tool_system_prompt = system_prompt + \
                            "\n DO NOT use the tool calling format again."

                        # Generate a new response with the tool results
                        print("Generating new response with tool results")
                        if output_format == "text":
                            # Stream the follow-up response for text output
                            async for processed_chunk in self.llm_provider.generate_text(
                                prompt=user_prompt,
                                system_prompt=tool_system_prompt,
                                api_key=self.api_key,
                                base_url=self.base_url,
                                model=self.model,
                            ):
                                complete_text_response += processed_chunk
                                yield processed_chunk
                        else:
                            # For audio output, collect the full response first
                            tool_response = ""
                            async for processed_chunk in self.llm_provider.generate_text(
                                prompt=user_prompt,
                                system_prompt=tool_system_prompt,
                            ):
                                tool_response += processed_chunk

                            # Clean and add to our complete text record and audio buffer
                            tool_response = self._clean_for_audio(
                                tool_response)
                            complete_text_response += tool_response
                            full_response_buffer += tool_response

                        # Reset tool handling state
                        is_tool_call = False
                        tool_buffer = ""
                        pending_chunk = ""
                        break  # Exit the original generation loop after tool processing

                    # Continue collecting tool call content without yielding
                    continue

                # STEP 3: Check for possible partial start markers at the end of the chunk
                # This helps detect markers split across chunks
                potential_marker = False
                for i in range(1, len(start_marker)):
                    if combined_chunk.endswith(start_marker[:i]):
                        # Found a partial marker at the end
                        # Save the partial marker
                        pending_chunk = combined_chunk[-i:]
                        # Everything except the partial marker
                        chunk_to_yield = combined_chunk[:-i]
                        potential_marker = True
                        print(
                            f"Potential partial marker detected: '{pending_chunk}'")
                        break

                if potential_marker:
                    # Process the safe part of the chunk
                    if chunk_to_yield and output_format == "text":
                        yield chunk_to_yield
                    if chunk_to_yield:
                        complete_text_response += chunk_to_yield
                        if output_format == "audio":
                            full_response_buffer += chunk_to_yield
                    continue

                # STEP 4: Normal text processing for non-tool call content
                if output_format == "text":
                    yield combined_chunk

                complete_text_response += combined_chunk
                if output_format == "audio":
                    full_response_buffer += combined_chunk

            # Process any incomplete tool call as regular text
            if is_tool_call and tool_buffer:
                print(
                    f"Incomplete tool call detected, returning as regular text: {len(tool_buffer)} chars")
                if output_format == "text":
                    yield tool_buffer

                complete_text_response += tool_buffer
                if output_format == "audio":
                    full_response_buffer += tool_buffer

            # For audio output, generate speech from the complete buffer
            if output_format == "audio" and full_response_buffer:
                # Clean text before TTS
                print(
                    f"Processing {len(full_response_buffer)} characters for audio output")
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
            print(
                f"Response generation complete: {len(complete_text_response)} chars")

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            print(f"Error in generate_response: {str(e)}")
            import traceback
            print(traceback.format_exc())

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

    async def _bytes_to_generator(self, data: bytes) -> AsyncGenerator[bytes, None]:
        """Convert bytes to an async generator for streaming.

        Args:
            data: Bytes of audio data

        Yields:
            Chunks of audio data
        """
        # Define a reasonable chunk size (adjust based on your needs)
        chunk_size = 4096

        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
            # Small delay to simulate streaming
            await asyncio.sleep(0.01)

    async def _handle_tool_call(self, agent_name: str, tool_text: str) -> str:
        """Handle marker-based tool calls."""
        try:
            # Extract the content between markers
            start_marker = "[TOOL]"
            end_marker = "[/TOOL]"

            start_idx = tool_text.find(start_marker) + len(start_marker)
            end_idx = tool_text.find(end_marker)

            tool_content = tool_text[start_idx:end_idx].strip()

            # Parse the lines to extract name and parameters
            tool_name = None
            parameters = {}

            for line in tool_content.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if line.startswith("name:"):
                    tool_name = line[5:].strip()
                elif line.startswith("parameters:"):
                    params_text = line[11:].strip()
                    # Parse comma-separated parameters
                    param_pairs = params_text.split(",")
                    for pair in param_pairs:
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            parameters[k.strip()] = v.strip()

            # Execute the tool
            result = await self.execute_tool(agent_name, tool_name, parameters)

            # Return the result as string
            if result.get("status") == "success":
                tool_result = str(result.get("result", ""))
                return tool_result
            else:
                error_msg = f"Error calling {tool_name}: {result.get('message', 'Unknown error')}"
                return error_msg

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Error processing tool call: {str(e)}"

    def _get_tool_usage_prompt(self, agent_name: str) -> str:
        """Generate marker-based instructions for tool usage."""
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
        
        ⚠️ CRITICAL INSTRUCTION: When using a tool, NEVER include explanatory text.
        Only output the exact tool call format shown below with NO other text.
        
        TOOL USAGE FORMAT:
        [TOOL]
        name: tool_name
        parameters: key1=value1, key2=value2
        [/TOOL]
        
        EXAMPLES:
        
        ✅ CORRECT - ONLY the tool call with NOTHING else:
        [TOOL]
        name: search_internet
        parameters: query=latest news on Solana
        [/TOOL]
        
        ❌ INCORRECT - Never add explanatory text like this:
        To get the latest news on Solana, I will search the internet.
        [TOOL]
        name: search_internet
        parameters: query=latest news on Solana
        [/TOOL]
        
        REMEMBER:
        1. Output ONLY the exact tool call format with NO additional text
        2. After seeing your tool call, I will execute it automatically
        3. You will receive the tool results and can then respond to the user
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

    def _clean_tool_response(self, text: str) -> str:
        """Remove any tool markers or formatting that might have leaked into the response."""
        if not text:
            return ""

        # Remove any tool markers that might be in the response
        text = text.replace("[TOOL]", "")
        text = text.replace("[/TOOL]", "")

        # Remove the word TOOL from start if it appears
        if text.lstrip().startswith("TOOL"):
            text = text.lstrip().replace("TOOL", "", 1)

        return text.strip()

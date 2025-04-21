"""
Agent service implementation.

This service manages AI and human agents, their registration, tool assignments,
and response generation.
"""

import asyncio
import datetime as main_datetime
from datetime import datetime
import json
import logging  # Add logging
from typing import AsyncGenerator, Dict, List, Literal, Optional, Any, Union

from solana_agent.interfaces.services.agent import AgentService as AgentServiceInterface
from solana_agent.interfaces.providers.llm import LLMProvider
from solana_agent.plugins.manager import PluginManager
from solana_agent.plugins.registry import ToolRegistry
from solana_agent.domains.agent import AIAgent, BusinessMission
from solana_agent.interfaces.guardrails.guardrails import (
    OutputGuardrail,
)

logger = logging.getLogger(__name__)  # Add logger


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
        output_guardrails: List[
            OutputGuardrail
        ] = None,  # <-- Add output_guardrails parameter
    ):
        """Initialize the agent service.

        Args:
            llm_provider: Provider for language model interactions
            business_mission: Optional business mission and values
            config: Optional service configuration
            api_key: API key for the LLM provider
            base_url: Base URL for the LLM provider
            model: Model name for the LLM provider
            output_guardrails: List of output guardrail instances
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
        self.output_guardrails = output_guardrails or []  # <-- Store guardrails

        self.plugin_manager = PluginManager(
            config=self.config,
            tool_registry=self.tool_registry,
        )

    def register_ai_agent(
        self,
        name: str,
        instructions: str,
        specialization: str,
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
        logger.info(f"Registered AI agent: {name}")

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
                values_text = "\n".join(
                    [
                        f"- {value.get('name', '')}: {value.get('description', '')}"
                        for value in self.business_mission.values
                    ]
                )
                system_prompt += f"\n\nBUSINESS VALUES:\n{values_text}"

            # Add goals if available
            if self.business_mission.goals:
                goals_text = "\n".join(
                    [f"- {goal}" for goal in self.business_mission.goals]
                )
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

    async def execute_tool(
        self, agent_name: str, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool on behalf of an agent."""

        if not self.tool_registry:
            logger.error("Tool registry not available during tool execution.")
            return {"status": "error", "message": "Tool registry not available"}

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            logger.warning(f"Tool '{tool_name}' not found for execution.")
            return {"status": "error", "message": f"Tool '{tool_name}' not found"}

        # Check if agent has access to this tool
        agent_tools = self.tool_registry.get_agent_tools(agent_name)

        if not any(t.get("name") == tool_name for t in agent_tools):
            logger.warning(
                f"Agent '{agent_name}' attempted to use unassigned tool '{tool_name}'."
            )
            return {
                "status": "error",
                "message": f"Agent '{agent_name}' doesn't have access to tool '{tool_name}'",
            }

        try:
            logger.info(
                f"Executing tool '{tool_name}' for agent '{agent_name}' with params: {parameters}"
            )
            result = await tool.execute(**parameters)
            logger.info(
                f"Tool '{tool_name}' execution result status: {result.get('status')}"
            )
            return result
        except Exception as e:
            import traceback

            logger.error(
                f"Error executing tool '{tool_name}': {e}\n{traceback.format_exc()}"
            )
            return {"status": "error", "message": f"Error executing tool: {str(e)}"}

    async def generate_response(
        self,
        agent_name: str,
        user_id: str,
        query: Union[str, bytes],
        memory_context: str = "",
        output_format: Literal["text", "audio"] = "text",
        audio_voice: Literal[
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "fable",
            "onyx",
            "nova",
            "sage",
            "shimmer",
        ] = "nova",
        audio_instructions: str = "You speak in a friendly and helpful manner.",
        audio_output_format: Literal[
            "mp3", "opus", "aac", "flac", "wav", "pcm"
        ] = "aac",
        prompt: Optional[str] = None,
    ) -> AsyncGenerator[Union[str, bytes], None]:  # pragma: no cover
        """Generate a response with support for text/audio input/output and guardrails.

        If output_format is 'text' and output_guardrails are present, the response
        will be buffered entirely before applying guardrails and yielding a single result.
        Otherwise, text responses stream chunk-by-chunk. Audio responses always buffer.
        """
        agent = next((a for a in self.agents if a.name == agent_name), None)
        if not agent:
            error_msg = f"Agent '{agent_name}' not found."
            logger.warning(error_msg)
            # Handle error output (unchanged)
            if output_format == "audio":
                async for chunk in self.llm_provider.tts(
                    error_msg,
                    instructions=audio_instructions,
                    response_format=audio_output_format,
                    voice=audio_voice,
                ):
                    yield chunk
            else:
                yield error_msg
            return

        # --- Determine Buffering Strategy ---
        # Buffer text ONLY if format is text AND guardrails are present
        should_buffer_text = bool(self.output_guardrails) and output_format == "text"
        logger.debug(
            f"Text buffering strategy: {'Buffer full response' if should_buffer_text else 'Stream chunks'}"
        )

        try:
            # --- System Prompt Assembly ---
            system_prompt_parts = [self.get_agent_system_prompt(agent_name)]

            # Add tool usage instructions if tools are available for the agent
            tool_instructions = self._get_tool_usage_prompt(agent_name)
            if tool_instructions:
                system_prompt_parts.append(tool_instructions)

            # Add user ID context
            system_prompt_parts.append(f"USER IDENTIFIER: {user_id}")

            # Add memory context if provided
            if memory_context:
                system_prompt_parts.append(f"\nCONVERSATION HISTORY:\n{memory_context}")

            # Add optional prompt if provided
            if prompt:
                system_prompt_parts.append(f"\nADDITIONAL PROMPT:\n{prompt}")

            final_system_prompt = "\n\n".join(
                filter(None, system_prompt_parts)
            )  # Join non-empty parts
            # --- End System Prompt Assembly ---

            # --- Response Generation ---
            complete_text_response = (
                ""  # Always used for final storage and potentially for buffering
            )
            full_response_buffer = ""  # Used ONLY for audio buffering

            # Tool call handling variables (unchanged)
            tool_buffer = ""
            pending_chunk = ""
            is_tool_call = False
            start_marker = "[TOOL]"
            end_marker = "[/TOOL]"

            logger.info(
                f"Generating response for agent '{agent_name}' with query length {len(str(query))}"
            )
            async for chunk in self.llm_provider.generate_text(
                prompt=str(query),
                system_prompt=final_system_prompt,
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
            ):
                # --- Chunk Processing & Tool Call Logic (Modified Yielding) ---
                if pending_chunk:
                    combined_chunk = pending_chunk + chunk
                    pending_chunk = ""
                else:
                    combined_chunk = chunk

                # STEP 1: Check for tool call start marker
                if start_marker in combined_chunk and not is_tool_call:
                    is_tool_call = True
                    start_pos = combined_chunk.find(start_marker)
                    before_marker = combined_chunk[:start_pos]
                    after_marker = combined_chunk[start_pos:]

                    if before_marker:
                        processed_before_marker = before_marker
                        # Apply guardrails ONLY if NOT buffering text
                        if not should_buffer_text:
                            for guardrail in self.output_guardrails:
                                try:
                                    processed_before_marker = await guardrail.process(
                                        processed_before_marker
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error applying output guardrail {guardrail.__class__.__name__} to pre-tool text: {e}"
                                    )

                        # Yield ONLY if NOT buffering text
                        if (
                            processed_before_marker
                            and not should_buffer_text
                            and output_format == "text"
                        ):
                            yield processed_before_marker

                        # Always accumulate for final response / audio buffer
                        if processed_before_marker:
                            complete_text_response += processed_before_marker
                            if output_format == "audio":
                                full_response_buffer += processed_before_marker

                    tool_buffer = after_marker
                    continue

                # STEP 2: Handle ongoing tool call collection
                if is_tool_call:
                    tool_buffer += combined_chunk
                    if end_marker in tool_buffer:
                        response_text = await self._handle_tool_call(
                            agent_name=agent_name, tool_text=tool_buffer
                        )
                        response_text = self._clean_tool_response(response_text)
                        user_prompt = f"{str(query)}\n\nTOOL RESULT: {response_text}"

                        # --- Rebuild system prompt for follow-up ---
                        follow_up_system_prompt_parts = [
                            self.get_agent_system_prompt(agent_name)
                        ]
                        # Re-add tool instructions if needed for follow-up context
                        if tool_instructions:
                            follow_up_system_prompt_parts.append(tool_instructions)
                        follow_up_system_prompt_parts.append(
                            f"USER IDENTIFIER: {user_id}"
                        )
                        # Include original memory + original query + tool result context
                        if memory_context:
                            follow_up_system_prompt_parts.append(
                                f"\nORIGINAL CONVERSATION HISTORY:\n{memory_context}"
                            )
                        # Add the original prompt if it was provided
                        if prompt:
                            follow_up_system_prompt_parts.append(
                                f"\nORIGINAL ADDITIONAL PROMPT:\n{prompt}"
                            )
                        # Add context about the tool call that just happened
                        follow_up_system_prompt_parts.append(
                            f"\nPREVIOUS TOOL CALL CONTEXT:\nOriginal Query: {str(query)}\nTool Used: (Inferred from result)\nTool Result: {response_text}"
                        )

                        final_follow_up_system_prompt = "\n\n".join(
                            filter(None, follow_up_system_prompt_parts)
                        )
                        # --- End Rebuild system prompt ---

                        logger.info("Generating follow-up response with tool results")
                        async for processed_chunk in self.llm_provider.generate_text(
                            prompt=user_prompt,  # Use the prompt that includes the tool result
                            system_prompt=final_follow_up_system_prompt,
                            api_key=self.api_key,
                            base_url=self.base_url,
                            model=self.model,
                        ):
                            chunk_to_yield_followup = processed_chunk
                            # Apply guardrails ONLY if NOT buffering text
                            if not should_buffer_text:
                                for guardrail in self.output_guardrails:
                                    try:
                                        chunk_to_yield_followup = (
                                            await guardrail.process(
                                                chunk_to_yield_followup
                                            )
                                        )
                                    except Exception as e:
                                        logger.error(
                                            f"Error applying output guardrail {guardrail.__class__.__name__} to follow-up chunk: {e}"
                                        )

                            # Yield ONLY if NOT buffering text
                            if (
                                chunk_to_yield_followup
                                and not should_buffer_text
                                and output_format == "text"
                            ):
                                yield chunk_to_yield_followup

                            # Always accumulate
                            if chunk_to_yield_followup:
                                complete_text_response += chunk_to_yield_followup
                                if output_format == "audio":
                                    full_response_buffer += chunk_to_yield_followup

                        is_tool_call = False
                        tool_buffer = ""
                        pending_chunk = ""
                        break  # Exit the original generation loop

                    continue  # Continue collecting tool call

                # STEP 3: Check for possible partial start markers
                potential_marker = False
                chunk_to_yield = combined_chunk
                for i in range(1, len(start_marker)):
                    if combined_chunk.endswith(start_marker[:i]):
                        pending_chunk = combined_chunk[-i:]
                        chunk_to_yield = combined_chunk[:-i]
                        potential_marker = True
                        break

                if potential_marker:
                    chunk_to_yield_safe = chunk_to_yield
                    # Apply guardrails ONLY if NOT buffering text
                    if not should_buffer_text:
                        for guardrail in self.output_guardrails:
                            try:
                                chunk_to_yield_safe = await guardrail.process(
                                    chunk_to_yield_safe
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error applying output guardrail {guardrail.__class__.__name__} to safe chunk: {e}"
                                )

                    # Yield ONLY if NOT buffering text
                    if (
                        chunk_to_yield_safe
                        and not should_buffer_text
                        and output_format == "text"
                    ):
                        yield chunk_to_yield_safe

                    # Always accumulate
                    if chunk_to_yield_safe:
                        complete_text_response += chunk_to_yield_safe
                        if output_format == "audio":
                            full_response_buffer += chunk_to_yield_safe
                    continue

                # STEP 4: Normal text processing
                chunk_to_yield_normal = combined_chunk
                # Apply guardrails ONLY if NOT buffering text
                if not should_buffer_text:
                    for guardrail in self.output_guardrails:
                        try:
                            chunk_to_yield_normal = await guardrail.process(
                                chunk_to_yield_normal
                            )
                        except Exception as e:
                            logger.error(
                                f"Error applying output guardrail {guardrail.__class__.__name__} to normal chunk: {e}"
                            )

                # Yield ONLY if NOT buffering text
                if (
                    chunk_to_yield_normal
                    and not should_buffer_text
                    and output_format == "text"
                ):
                    yield chunk_to_yield_normal

                # Always accumulate
                if chunk_to_yield_normal:
                    complete_text_response += chunk_to_yield_normal
                    if output_format == "audio":
                        full_response_buffer += chunk_to_yield_normal

            # --- Post-Loop Processing ---

            # Process any incomplete tool call
            if is_tool_call and tool_buffer:
                logger.warning(
                    f"Incomplete tool call detected, processing as regular text: {len(tool_buffer)} chars"
                )
                processed_tool_buffer = tool_buffer
                # Apply guardrails ONLY if NOT buffering text
                if not should_buffer_text:
                    for guardrail in self.output_guardrails:
                        try:
                            processed_tool_buffer = await guardrail.process(
                                processed_tool_buffer
                            )
                        except Exception as e:
                            logger.error(
                                f"Error applying output guardrail {guardrail.__class__.__name__} to incomplete tool buffer: {e}"
                            )

                # Yield ONLY if NOT buffering text
                if (
                    processed_tool_buffer
                    and not should_buffer_text
                    and output_format == "text"
                ):
                    yield processed_tool_buffer

                # Always accumulate
                if processed_tool_buffer:
                    complete_text_response += processed_tool_buffer
                    if output_format == "audio":
                        full_response_buffer += processed_tool_buffer

            # --- Final Output Generation ---

            # Case 1: Text output WITH guardrails (apply to buffered response)
            if should_buffer_text:
                logger.info(
                    f"Applying output guardrails to buffered text response (length: {len(complete_text_response)})"
                )
                processed_full_text = complete_text_response
                for guardrail in self.output_guardrails:
                    try:
                        processed_full_text = await guardrail.process(
                            processed_full_text
                        )
                    except Exception as e:
                        logger.error(
                            f"Error applying output guardrail {guardrail.__class__.__name__} to full text buffer: {e}"
                        )

                if processed_full_text:
                    yield processed_full_text
                # Update last_text_response with the final processed text
                self.last_text_response = processed_full_text

            # Case 2: Audio output (apply guardrails to buffer before TTS) - Unchanged Logic
            elif output_format == "audio" and full_response_buffer:
                original_buffer = full_response_buffer
                processed_audio_buffer = full_response_buffer
                for (
                    guardrail
                ) in self.output_guardrails:  # Apply even if empty, for consistency
                    try:
                        processed_audio_buffer = await guardrail.process(
                            processed_audio_buffer
                        )
                    except Exception as e:
                        logger.error(
                            f"Error applying output guardrail {guardrail.__class__.__name__} to audio buffer: {e}"
                        )
                if processed_audio_buffer != original_buffer:
                    logger.info(
                        f"Output guardrails modified audio buffer. Original length: {len(original_buffer)}, New length: {len(processed_audio_buffer)}"
                    )

                cleaned_audio_buffer = self._clean_for_audio(processed_audio_buffer)
                logger.info(
                    f"Processing {len(cleaned_audio_buffer)} characters for audio output"
                )
                async for audio_chunk in self.llm_provider.tts(
                    text=cleaned_audio_buffer,
                    voice=audio_voice,
                    response_format=audio_output_format,
                    instructions=audio_instructions,
                ):
                    yield audio_chunk
                # Update last_text_response with the text *before* TTS cleaning
                self.last_text_response = (
                    processed_audio_buffer  # Store the guardrail-processed text
                )

            # Case 3: Text output WITHOUT guardrails (already streamed)
            elif output_format == "text" and not should_buffer_text:
                # Store the complete text response (accumulated from non-processed chunks)
                self.last_text_response = complete_text_response
                logger.info(
                    "Text streaming complete (no guardrails applied post-stream)."
                )

            logger.info(
                f"Response generation complete for agent '{agent_name}': {len(self.last_text_response)} final chars"
            )

        except Exception as e:
            # --- Error Handling (unchanged) ---
            import traceback

            error_msg = (
                "I apologize, but I encountered an error processing your request."
            )
            logger.error(
                f"Error in generate_response for agent '{agent_name}': {e}\n{traceback.format_exc()}"
            )
            if output_format == "audio":
                async for chunk in self.llm_provider.tts(
                    error_msg,
                    voice=audio_voice,
                    response_format=audio_output_format,
                    instructions=audio_instructions,
                ):
                    yield chunk
            else:
                yield error_msg

    async def _bytes_to_generator(self, data: bytes) -> AsyncGenerator[bytes, None]:
        """Convert bytes to an async generator for streaming."""
        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]
            await asyncio.sleep(0.01)

    async def _handle_tool_call(self, agent_name: str, tool_text: str) -> str:
        """Handle marker-based tool calls."""
        try:
            start_marker = "[TOOL]"
            end_marker = "[/TOOL]"
            start_idx = tool_text.find(start_marker) + len(start_marker)
            end_idx = tool_text.find(end_marker)
            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                logger.error(f"Malformed tool call text received: {tool_text}")
                return "Error: Malformed tool call format."

            tool_content = tool_text[start_idx:end_idx].strip()
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
                    try:
                        # Attempt to parse as JSON first for robustness
                        parameters = json.loads(params_text)
                    except json.JSONDecodeError:
                        # Fallback to comma-separated key=value pairs
                        param_pairs = params_text.split(",")
                        for pair in param_pairs:
                            if "=" in pair:
                                k, v = pair.split("=", 1)
                                parameters[k.strip()] = v.strip()
                        logger.warning(
                            f"Parsed tool parameters using fallback method: {params_text}"
                        )

            if not tool_name:
                logger.error(f"Tool name missing in tool call: {tool_content}")
                return "Error: Tool name missing in call."

            result = await self.execute_tool(agent_name, tool_name, parameters)

            if result.get("status") == "success":
                tool_result = str(result.get("result", ""))
                return tool_result
            else:
                error_msg = f"Error calling {tool_name}: {result.get('message', 'Unknown error')}"
                logger.error(error_msg)
                return error_msg

        except Exception as e:
            import traceback

            logger.error(f"Error processing tool call: {e}\n{traceback.format_exc()}")
            return f"Error processing tool call: {str(e)}"

    def _get_tool_usage_prompt(self, agent_name: str) -> str:
        """Generate marker-based instructions for tool usage."""
        tools = self.get_agent_tools(agent_name)
        if not tools:
            return ""

        # Simplify tool representation for the prompt
        simplified_tools = []
        for tool in tools:
            simplified_tool = {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters", {}).get("properties", {}),
            }
            simplified_tools.append(simplified_tool)

        tools_json = json.dumps(simplified_tools, indent=2)

        return f"""
        AVAILABLE TOOLS:
        {tools_json}

        ⚠️ CRITICAL INSTRUCTION: When using a tool, NEVER include explanatory text.
        Only output the exact tool call format shown below with NO other text.
        Always call the necessary tool to give the latest information.

        TOOL USAGE FORMAT:
        [TOOL]
        name: tool_name
        parameters: {{"key1": "value1", "key2": "value2"}}
        [/TOOL]

        EXAMPLES:
        ✅ CORRECT - ONLY the tool call with NOTHING else:
        [TOOL]
        name: search_internet
        parameters: {{"query": "latest news on Solana"}}
        [/TOOL]

        ❌ INCORRECT - Never add explanatory text like this:
        To get the latest news on Solana, I will search the internet.
        [TOOL]
        name: search_internet
        parameters: {{"query": "latest news on Solana"}}
        [/TOOL]

        REMEMBER:
        1. Output ONLY the exact tool call format with NO additional text
        2. If the query is time-sensitive (latest news, current status, etc.), ALWAYS use the tool.
        3. After seeing your tool call, I will execute it automatically
        4. You will receive the tool results and can then respond to the user
        """

    def _clean_for_audio(self, text: str) -> str:
        """Remove Markdown formatting, emojis, and non-pronounceable characters from text."""
        import re

        if not text:
            return ""
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
        text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
        text = re.sub(r"^\s*#+\s*(.*?)$", r"\1", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*>\s*(.*?)$", r"\1", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*+]\s+(.*?)$", r"\1", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+(.*?)$", r"\1", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f700-\U0001f77f"  # alchemical symbols
            "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
            "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
            "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
            "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027b0"  # Dingbats
            "\U000024c2-\U0001f251"
            "\U00002600-\U000026ff"  # Miscellaneous Symbols
            "\U00002700-\U000027bf"  # Dingbats
            "\U0000fe00-\U0000fe0f"  # Variation Selectors
            "\U0001f1e0-\U0001f1ff"  # Flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r" ", text)
        text = re.sub(
            r"[^\w\s\.\,\;\:\?\!\'\"\-\(\)]", " ", text
        )  # Keep basic punctuation
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _clean_tool_response(self, text: str) -> str:
        """Remove any tool markers or formatting that might have leaked into the response."""
        if not text:
            return ""
        text = text.replace("[TOOL]", "").replace("[/TOOL]", "")
        if text.lstrip().startswith("TOOL"):
            text = text.lstrip()[4:].lstrip()  # Remove "TOOL" and leading space
        return text.strip()

    # --- Add methods from factory logic ---
    def load_and_register_plugins(self):
        """Loads plugins using the PluginManager."""
        try:
            self.plugin_manager.load_plugins()
            logger.info("Plugins loaded successfully via PluginManager.")
        except Exception as e:
            logger.error(f"Error loading plugins: {e}", exc_info=True)

    def register_agents_from_config(self):
        """Registers agents defined in the main configuration."""
        agents_config = self.config.get("agents", [])
        if not agents_config:
            logger.warning("No agents defined in the configuration.")
            return

        for agent_config in agents_config:
            name = agent_config.get("name")
            instructions = agent_config.get("instructions")
            specialization = agent_config.get("specialization")
            tools = agent_config.get("tools", [])

            if not name or not instructions or not specialization:
                logger.warning(
                    f"Skipping agent due to missing name, instructions, or specialization: {agent_config}"
                )
                continue

            self.register_ai_agent(name, instructions, specialization)
            # logger.info(f"Registered agent: {name}") # Logging done in register_ai_agent

            # Assign tools to the agent
            for tool_name in tools:
                if self.assign_tool_for_agent(name, tool_name):
                    logger.info(f"Assigned tool '{tool_name}' to agent '{name}'.")
                else:
                    logger.warning(
                        f"Failed to assign tool '{tool_name}' to agent '{name}' (Tool might not be registered)."
                    )

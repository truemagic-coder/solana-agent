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
import re
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

    # --- Helper function to recursively substitute placeholders ---
    def _substitute_placeholders(self, data: Any, results_map: Dict[str, str]) -> Any:
        """Recursively substitutes placeholders like {{tool_name.result}} or {output_of_tool_name} in strings."""
        if isinstance(data, str):
            # Regex to find placeholders like {{tool_name.result}} or {output_of_tool_name}
            placeholder_pattern = re.compile(
                r"\{\{(?P<name1>[a-zA-Z0-9_]+)\.result\}\}|\{output_of_(?P<name2>[a-zA-Z0-9_]+)\}"
            )

            def replace_match(match):
                tool_name = match.group("name1") or match.group("name2")
                if tool_name and tool_name in results_map:
                    logger.debug(f"Substituting placeholder for '{tool_name}'")
                    return results_map[tool_name]
                else:
                    # If placeholder not found, leave it as is but log warning
                    logger.warning(
                        f"Could not find result for placeholder tool '{tool_name}'. Leaving placeholder."
                    )
                    return match.group(0)  # Return original placeholder

            # Use re.sub with the replacement function
            return placeholder_pattern.sub(replace_match, data)
        elif isinstance(data, dict):
            # Recursively process dictionary values
            return {
                k: self._substitute_placeholders(v, results_map)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            # Recursively process list items
            return [self._substitute_placeholders(item, results_map) for item in data]
        else:
            # Return non-string/dict/list types as is
            return data

    # --- Helper to parse tool calls ---
    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parses all [TOOL]...[/TOOL] blocks in the text."""
        tool_calls = []
        # Regex to find all tool blocks, non-greedy match for content
        pattern = re.compile(r"\[TOOL\](.*?)\[/TOOL\]", re.DOTALL | re.IGNORECASE)
        matches = pattern.finditer(text)

        for match in matches:
            tool_content = match.group(1).strip()
            tool_name = None
            parameters = {}
            try:
                for line in tool_content.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if line.lower().startswith("name:"):
                        tool_name = line[5:].strip()
                    elif line.lower().startswith("parameters:"):
                        params_text = line[11:].strip()
                        try:
                            # Prefer JSON parsing
                            parameters = json.loads(params_text)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse parameters as JSON, falling back: {params_text}"
                            )
                            # Fallback: Treat as simple key=value (less robust)
                            try:
                                # Basic eval might work for {"key": "value"} but is risky
                                # parameters = eval(params_text) # Avoid eval if possible
                                # Safer fallback: Assume simple string if not JSON-like
                                if not params_text.startswith("{"):
                                    # Try splitting key=value pairs? Very brittle.
                                    # For now, log warning and skip complex fallback parsing
                                    logger.error(
                                        f"Cannot parse non-JSON parameters reliably: {params_text}"
                                    )
                                    parameters = {
                                        "_raw_params": params_text
                                    }  # Store raw string
                                else:
                                    # If it looks like a dict but isn't valid JSON, log error
                                    logger.error(
                                        f"Invalid dictionary format for parameters: {params_text}"
                                    )
                                    parameters = {"_raw_params": params_text}

                            except Exception as parse_err:
                                logger.error(
                                    f"Fallback parameter parsing failed: {parse_err}"
                                )
                                parameters = {
                                    "_raw_params": params_text
                                }  # Store raw string on error

                if tool_name:
                    tool_calls.append({"name": tool_name, "parameters": parameters})
                else:
                    logger.warning(f"Parsed tool block missing name: {tool_content}")
            except Exception as e:
                logger.error(f"Error parsing tool content: {tool_content} - {e}")

        logger.info(f"Parsed {len(tool_calls)} tool calls from response.")
        return tool_calls

    # --- Helper to execute a single parsed tool call ---
    async def _execute_single_tool(
        self, agent_name: str, tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes a single tool call dictionary and returns its result."""
        tool_name = tool_call.get("name")
        parameters = tool_call.get("parameters", {})
        if not tool_name:
            return {
                "tool_name": "unknown",
                "status": "error",
                "message": "Tool name missing in parsed call",
            }
        # Ensure parameters is a dict, even if parsing failed
        if not isinstance(parameters, dict):
            logger.warning(
                f"Parameters for tool '{tool_name}' is not a dict: {parameters}. Attempting execution with empty params."
            )
            parameters = {}

        logger.debug(
            f"Preparing to execute tool '{tool_name}' with params: {parameters}"
        )
        result = await self.execute_tool(agent_name, tool_name, parameters)
        # Add tool name to result for easier aggregation
        result["tool_name"] = tool_name
        return result

    async def generate_response(
        self,
        agent_name: str,
        user_id: str,
        query: Union[str, bytes],
        images: Optional[List[Union[str, bytes]]] = None,
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
        """Generate a response, supporting multiple sequential tool calls with placeholder substitution.
        Optionally accepts images for vision-capable models.
        """
        agent = next((a for a in self.agents if a.name == agent_name), None)
        if not agent:
            error_msg = f"Agent '{agent_name}' not found."
            logger.warning(error_msg)
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

        logger.debug(
            f"Generating response for agent '{agent_name}'. Output format: {output_format}. Images provided: {bool(images)}."
        )

        try:
            # --- System Prompt Assembly ---
            system_prompt_parts = [self.get_agent_system_prompt(agent_name)]
            tool_instructions = self._get_tool_usage_prompt(agent_name)
            if tool_instructions:
                system_prompt_parts.append(tool_instructions)
            system_prompt_parts.append(f"USER IDENTIFIER: {user_id}")
            if memory_context:
                system_prompt_parts.append(f"\nCONVERSATION HISTORY:\n{memory_context}")
            if prompt:
                system_prompt_parts.append(f"\nADDITIONAL PROMPT:\n{prompt}")
            final_system_prompt = "\n\n".join(filter(None, system_prompt_parts))

            # --- Initial Response Generation (No Streaming) ---
            initial_llm_response_buffer = ""
            tool_calls_detected = False
            start_marker = "[TOOL]"

            logger.info(f"Generating initial response for agent '{agent_name}'...")

            # --- CHOOSE LLM METHOD BASED ON IMAGE PRESENCE ---
            if images:
                # Use the new vision method if images are present
                logger.info(
                    f"Using generate_text_with_images for {len(images)} images."
                )
                # Ensure query is string for the text part
                text_query = str(query) if isinstance(query, bytes) else query
                initial_llm_response_buffer = (
                    await self.llm_provider.generate_text_with_images(
                        prompt=text_query,
                        images=images,
                        system_prompt=final_system_prompt,
                    )
                )
            else:
                # Use the standard text generation method
                logger.info("Using generate_text (no images provided).")
                initial_llm_response_buffer = await self.llm_provider.generate_text(
                    prompt=str(query),
                    system_prompt=final_system_prompt,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.model,
                )
            # --- END LLM METHOD CHOICE ---

            # Check for errors returned as string by the adapter
            if isinstance(initial_llm_response_buffer, str) and (
                initial_llm_response_buffer.startswith(
                    "I apologize, but I encountered an"
                )
                or initial_llm_response_buffer.startswith("Error:")
            ):
                logger.error(
                    f"LLM provider failed during initial generation: {initial_llm_response_buffer}"
                )
                # Yield the error and exit
                if output_format == "audio":
                    async for chunk in self.llm_provider.tts(
                        initial_llm_response_buffer,
                        voice=audio_voice,
                        response_format=audio_output_format,
                        instructions=audio_instructions,
                    ):
                        yield chunk
                else:
                    yield initial_llm_response_buffer
                return

            # Check for tool markers in the complete response
            if start_marker.lower() in initial_llm_response_buffer.lower():
                tool_calls_detected = True
                logger.info("Tool call marker detected in initial response.")

            logger.debug(
                f"Full initial LLM response buffer:\n--- START ---\n{initial_llm_response_buffer}\n--- END ---"
            )
            logger.info(
                f"Initial LLM response received (length: {len(initial_llm_response_buffer)}). Tools detected: {tool_calls_detected}"
            )

            # --- Tool Execution Phase (if tools were detected) ---
            final_response_text = ""
            if tool_calls_detected:
                # NOTE: If tools need to operate on image content, this logic needs significant changes.
                # Assuming for now tools operate based on the text query or the LLM's understanding derived from images.
                parsed_calls = self._parse_tool_calls(initial_llm_response_buffer)

                if parsed_calls:
                    # ... (existing sequential tool execution with substitution) ...
                    executed_tool_results = []
                    tool_results_map: Dict[str, str] = {}
                    logger.info(
                        f"Executing {len(parsed_calls)} tools sequentially with substitution..."
                    )
                    for i, call in enumerate(parsed_calls):
                        # ... (existing substitution logic) ...
                        tool_name_to_exec = call.get("name", "unknown")
                        logger.info(
                            f"Executing tool {i + 1}/{len(parsed_calls)}: {tool_name_to_exec}"
                        )
                        try:
                            original_params = call.get("parameters", {})
                            substituted_params = self._substitute_placeholders(
                                original_params, tool_results_map
                            )
                            if substituted_params != original_params:
                                logger.info(
                                    f"Substituted parameters for tool '{tool_name_to_exec}': {substituted_params}"
                                )
                            call["parameters"] = substituted_params
                        except Exception as sub_err:
                            logger.error(
                                f"Error substituting placeholders for tool '{tool_name_to_exec}': {sub_err}",
                                exc_info=True,
                            )

                        # ... (existing tool execution call) ...
                        try:
                            result = await self._execute_single_tool(agent_name, call)
                            executed_tool_results.append(result)
                            if result.get("status") == "success":
                                tool_result_str = str(result.get("result", ""))
                                tool_results_map[tool_name_to_exec] = tool_result_str
                                logger.debug(
                                    f"Stored result for '{tool_name_to_exec}' (length: {len(tool_result_str)})"
                                )
                            else:
                                error_message = result.get("message", "Unknown error")
                                tool_results_map[tool_name_to_exec] = (
                                    f"Error: {error_message}"
                                )
                                logger.warning(
                                    f"Tool '{tool_name_to_exec}' failed, storing error message."
                                )
                        except Exception as tool_exec_err:
                            logger.error(
                                f"Exception during execution of tool {tool_name_to_exec}: {tool_exec_err}",
                                exc_info=True,
                            )
                            error_result = {
                                "tool_name": tool_name_to_exec,
                                "status": "error",
                                "message": f"Exception during execution: {str(tool_exec_err)}",
                            }
                            executed_tool_results.append(error_result)
                            tool_results_map[tool_name_to_exec] = (
                                f"Error: {str(tool_exec_err)}"
                            )

                    logger.info("Sequential tool execution with substitution complete.")

                    # ... (existing formatting of tool results) ...
                    tool_results_text_parts = []
                    for i, result in enumerate(executed_tool_results):
                        tool_name = result.get("tool_name", "unknown")
                        if (
                            isinstance(result, Exception)
                            or result.get("status") == "error"
                        ):
                            error_msg = (
                                result.get("message", str(result))
                                if isinstance(result, dict)
                                else str(result)
                            )
                            logger.error(f"Tool '{tool_name}' failed: {error_msg}")
                            tool_results_text_parts.append(
                                f"Tool {i + 1} ({tool_name}) Execution Failed:\n{error_msg}"
                            )
                        else:
                            tool_output = str(result.get("result", ""))
                            tool_results_text_parts.append(
                                f"Tool {i + 1} ({tool_name}) Result:\n{tool_output}"
                            )
                    tool_results_context = "\n\n".join(tool_results_text_parts)

                    # --- Generate Final Response using Tool Results (No Streaming) ---
                    # Include original query (text part) and mention images were provided if applicable
                    original_query_context = f"Original Query: {str(query)}"
                    if images:
                        original_query_context += f" (with {len(images)} image(s))"

                    follow_up_prompt = f"{original_query_context}\n\nRESULTS FROM TOOL CALLS:\n{tool_results_context}\n\nBased on the original query, any provided images, and the tool results, please provide the final response to the user."
                    follow_up_system_prompt_parts = [
                        self.get_agent_system_prompt(agent_name)
                    ]
                    follow_up_system_prompt_parts.append(f"USER IDENTIFIER: {user_id}")
                    if memory_context:
                        follow_up_system_prompt_parts.append(
                            f"\nORIGINAL CONVERSATION HISTORY:\n{memory_context}"
                        )
                    if prompt:
                        follow_up_system_prompt_parts.append(
                            f"\nORIGINAL ADDITIONAL PROMPT:\n{prompt}"
                        )
                    follow_up_system_prompt_parts.append(
                        f"\nCONTEXT: You previously decided to run {len(parsed_calls)} tool(s) sequentially. The results are provided above."
                    )
                    final_follow_up_system_prompt = "\n\n".join(
                        filter(None, follow_up_system_prompt_parts)
                    )

                    logger.info(
                        "Generating final response incorporating tool results..."
                    )
                    # Use standard text generation for the final synthesis
                    synthesized_response_buffer = await self.llm_provider.generate_text(
                        prompt=follow_up_prompt,
                        system_prompt=final_follow_up_system_prompt,
                        api_key=self.api_key,
                        base_url=self.base_url,
                        model=self.model
                        or self.llm_provider.text_model,  # Use text model for synthesis
                    )

                    if isinstance(synthesized_response_buffer, str) and (
                        synthesized_response_buffer.startswith(
                            "I apologize, but I encountered an"
                        )
                        or synthesized_response_buffer.startswith("Error:")
                    ):
                        logger.error(
                            f"LLM provider failed during final generation: {synthesized_response_buffer}"
                        )
                        if output_format == "audio":
                            async for chunk in self.llm_provider.tts(
                                synthesized_response_buffer,
                                voice=audio_voice,
                                response_format=audio_output_format,
                                instructions=audio_instructions,
                            ):
                                yield chunk
                        else:
                            yield synthesized_response_buffer
                        return

                    final_response_text = synthesized_response_buffer
                    logger.info(
                        f"Final synthesized response length: {len(final_response_text)}"
                    )

                else:
                    logger.warning(
                        "Tool markers detected, but no valid tool calls parsed. Treating initial response as final."
                    )
                    final_response_text = initial_llm_response_buffer
            else:
                final_response_text = initial_llm_response_buffer
                logger.info("No tools detected. Using initial response as final.")

            # --- Final Output Processing (Guardrails, TTS, Yielding) ---
            processed_final_text = final_response_text
            if self.output_guardrails:
                logger.info(
                    f"Applying output guardrails to final text response (length: {len(processed_final_text)})"
                )
                original_len = len(processed_final_text)
                for guardrail in self.output_guardrails:
                    try:
                        processed_final_text = await guardrail.process(
                            processed_final_text
                        )
                    except Exception as e:
                        logger.error(
                            f"Error applying output guardrail {guardrail.__class__.__name__}: {e}"
                        )
                if len(processed_final_text) != original_len:
                    logger.info(
                        f"Guardrails modified final text length from {original_len} to {len(processed_final_text)}"
                    )

            self.last_text_response = processed_final_text

            if output_format == "text":
                if processed_final_text:
                    yield processed_final_text
                else:
                    logger.warning("Final processed text was empty.")
                    yield ""
            elif output_format == "audio":
                text_for_tts = processed_final_text
                cleaned_audio_buffer = self._clean_for_audio(text_for_tts)
                logger.info(
                    f"Processing {len(cleaned_audio_buffer)} characters for audio output"
                )
                if cleaned_audio_buffer:
                    async for audio_chunk in self.llm_provider.tts(
                        text=cleaned_audio_buffer,
                        voice=audio_voice,
                        response_format=audio_output_format,
                        instructions=audio_instructions,
                    ):
                        yield audio_chunk
                else:
                    logger.warning("Final text for audio was empty after cleaning.")

            logger.info(
                f"Response generation complete for agent '{agent_name}': {len(self.last_text_response)} final chars"
            )

        except Exception as e:
            # --- Error Handling ---
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

        ⚠️ CRITICAL INSTRUCTIONS FOR TOOL USAGE:
        1. EXECUTION ORDER MATTERS: If multiple steps are needed (e.g., get information THEN use it), you MUST output the [TOOL] blocks in the exact sequence they need to run. Output the information-gathering tool call FIRST, then the tool call that uses the information.
        2. ONLY TOOL CALLS: When using a tool, NEVER include explanatory text before or after the tool call block. Only output the exact tool call format shown below.
        3. USE TOOLS WHEN NEEDED: Always call the necessary tool to give the latest information, especially for time-sensitive queries.

        TOOL USAGE FORMAT:
        [TOOL]
        name: tool_name
        parameters: {{"key1": "value1", "key2": "value2"}}
        [/TOOL]

        EXAMPLES:

        ✅ CORRECT - Get news THEN email (Correct Order):
        [TOOL]
        name: search_internet
        parameters: {{"query": "latest news on Canada"}}
        [/TOOL]
        [TOOL]
        name: mcp
        parameters: {{"query": "Send an email to
                             bob@bob.com with subject
                             'Friendly Reminder to Clean Your Room'
                             and body 'Hi Bob, just a friendly
                             reminder to please clean your room
                             when you get a chance.'"}}
        [/TOOL]
        (Note: The system will handle replacing placeholders like '{{output_of_search_internet}}' if possible, but the ORDER is crucial.)


        ❌ INCORRECT - Wrong Order:
        [TOOL]
        name: mcp
        parameters: {{"query": "Send an email to
                             bob@bob.com with subject
                             'Friendly Reminder to Clean Your Room'
                             and body 'Hi Bob, just a friendly
                             reminder to please clean your room
                             when you get a chance.'"}}
        [/TOOL]
        [TOOL]
        name: search_internet
        parameters: {{"query": "latest news on Canada"}}
        [/TOOL]


        ❌ INCORRECT - Explanatory Text:
        To get the news, I'll search.
        [TOOL]
        name: search_internet
        parameters: {{"query": "latest news on Solana"}}
        [/TOOL]
        Now I will email it.
        [TOOL]
        name: mcp
        parameters: {{"query": "Send an email to
                             bob@bob.com with subject
                             'Friendly Reminder to Clean Your Room'
                             and body 'Hi Bob, just a friendly
                             reminder to please clean your room
                             when you get a chance.'"}}
        [/TOOL]


        REMEMBER:
        - Output ONLY the [TOOL] blocks in the correct execution order.
        - I will execute the tools sequentially as you provide them.
        - You will receive the results of ALL tool calls before formulating the final response.
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

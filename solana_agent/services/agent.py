"""
Agent service implementation.

This service manages AI and human agents, their registration, tool assignments,
and response generation.
"""

import datetime as main_datetime
from datetime import datetime
import json
import logging  # Add logging
import re
from typing import AsyncGenerator, Dict, List, Literal, Optional, Any, Type, Union

from pydantic import BaseModel

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
        model: Optional[str] = None,
        output_guardrails: List[OutputGuardrail] = None,
    ):
        """Initialize the agent service.

        Args:
            llm_provider: Provider for language model interactions
            business_mission: Optional business mission and values
            config: Optional service configuration
            model: Model name for the LLM provider
            output_guardrails: List of output guardrail instances
        """
        self.llm_provider = llm_provider
        self.business_mission = business_mission
        self.config = config or {}
        self.last_text_response = ""
        self.tool_registry = ToolRegistry(config=self.config)
        self.agents: List[AIAgent] = []
        self.model = model
        self.output_guardrails = output_guardrails or []

        self.plugin_manager = PluginManager(
            config=self.config,
            tool_registry=self.tool_registry,
        )

    def register_ai_agent(
        self,
        name: str,
        instructions: str,
        specialization: str,
        capture_name: Optional[str] = None,
        capture_schema: Optional[Dict[str, Any]] = None,
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
            capture_name=capture_name,
            capture_schema=capture_schema,
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

        # Add capture guidance if this agent has a capture schema
        if getattr(agent, "capture_schema", None) and getattr(
            agent, "capture_name", None
        ):  # pragma: no cover
            system_prompt += (
                "\n\nSTRUCTURED DATA CAPTURE:\n"
                f"You must collect the following fields for the form '{agent.capture_name}'. "
                "Ask concise follow-up questions to fill any missing required fields one at a time. "
                "Confirm values when ambiguous, and summarize the captured data before finalizing.\n\n"
                "JSON Schema (authoritative definition of the fields):\n"
                f"{agent.capture_schema}\n\n"
                "Rules:\n"
                "- Never invent values—ask the user.\n"
                "- Validate types (emails look like emails, numbers are numbers, booleans are yes/no).\n"
                "- If the user declines to provide a required value, note it clearly.\n"
                "- When all required fields are provided, acknowledge completion.\n"
            )

        return system_prompt

    def get_agent_capture(
        self, agent_name: str
    ) -> Optional[Dict[str, Any]]:  # pragma: no cover
        """Return capture metadata for the agent, if any."""
        agent = next((a for a in self.agents if a.name == agent_name), None)
        if not agent:
            return None
        if agent.capture_name and agent.capture_schema:
            return {"name": agent.capture_name, "schema": agent.capture_schema}
        return None

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
        audio_output_format: Literal[
            "mp3", "opus", "aac", "flac", "wav", "pcm"
        ] = "aac",
        prompt: Optional[str] = None,
        output_model: Optional[Type[BaseModel]] = None,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:  # pragma: no cover
        """Generate a response using tool-calling with full streaming support."""

        try:
            # Validate agent
            agent = next((a for a in self.agents if a.name == agent_name), None)
            if not agent:
                error_msg = f"Agent '{agent_name}' not found."
                logger.warning(error_msg)
                if output_format == "audio":
                    async for chunk in self.llm_provider.tts(
                        error_msg,
                        response_format=audio_output_format,
                        voice=audio_voice,
                    ):
                        yield chunk
                else:
                    yield error_msg
                return

            # Build system prompt and messages
            system_prompt = self.get_agent_system_prompt(agent_name)
            user_content = str(query)
            if images:
                user_content += "\n\n[Images attached]"

            # Compose the prompt for generate_text
            full_prompt = ""
            if memory_context:
                full_prompt += f"CONVERSATION HISTORY:\n{memory_context}\n\n Always use your tools to perform actions and don't rely on your memory!\n\n"
            if prompt:
                full_prompt += f"ADDITIONAL PROMPT:\n{prompt}\n\n"
            full_prompt += user_content
            full_prompt += f"USER IDENTIFIER: {user_id}"

            # Get OpenAI function schemas for this agent's tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                        "strict": True,
                    },
                }
                for tool in self.get_agent_tools(agent_name)
            ]

            # Structured output path
            if output_model is not None:
                model_instance = await self.llm_provider.parse_structured_output(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    model_class=output_model,
                    model=self.model,
                    tools=tools if tools else None,
                )
                yield model_instance
                return

            # Vision fallback (non-streaming for now)
            if images:
                vision_text = await self.llm_provider.generate_text_with_images(
                    prompt=full_prompt, images=images, system_prompt=system_prompt
                )
                if output_format == "audio":
                    cleaned_audio_buffer = self._clean_for_audio(vision_text)
                    async for audio_chunk in self.llm_provider.tts(
                        text=cleaned_audio_buffer,
                        voice=audio_voice,
                        response_format=audio_output_format,
                    ):
                        yield audio_chunk
                else:
                    yield vision_text
                return

            # Build initial messages for chat streaming
            messages: List[Dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": full_prompt})

            accumulated_text = ""

            # Loop to handle tool calls in streaming mode
            while True:
                # Aggregate tool calls by index and merge late IDs
                tool_calls: Dict[int, Dict[str, Any]] = {}

                async for event in self.llm_provider.chat_stream(
                    messages=messages,
                    model=self.model,
                    tools=tools if tools else None,
                ):
                    etype = event.get("type")
                    if etype == "content":
                        delta = event.get("delta", "")
                        accumulated_text += delta
                        if output_format == "text":
                            yield delta
                    elif etype == "tool_call_delta":
                        tc_id = event.get("id")
                        index_raw = event.get("index")
                        try:
                            index = int(index_raw) if index_raw is not None else 0
                        except Exception:
                            index = 0
                        name = event.get("name")
                        args_piece = event.get("arguments_delta", "")
                        entry = tool_calls.setdefault(
                            index, {"id": None, "name": None, "arguments": ""}
                        )
                        if tc_id and not entry.get("id"):
                            entry["id"] = tc_id
                        if name and not entry.get("name"):
                            entry["name"] = name
                        entry["arguments"] += args_piece
                    elif etype == "message_end":
                        _ = event.get("finish_reason")

                # If tool calls were requested, execute them and continue the loop
                if tool_calls:
                    assistant_tool_calls: List[Dict[str, Any]] = []
                    call_id_map: Dict[int, str] = {}
                    for idx, tc in tool_calls.items():
                        name = (tc.get("name") or "").strip()
                        if not name:
                            logger.warning(
                                f"Skipping unnamed tool call at index {idx}; cannot send empty function name."
                            )
                            continue
                        norm_id = tc.get("id") or f"call_{idx}"
                        call_id_map[idx] = norm_id
                        assistant_tool_calls.append(
                            {
                                "id": norm_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": tc.get("arguments") or "{}",
                                },
                            }
                        )

                    if assistant_tool_calls:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": assistant_tool_calls,
                            }
                        )

                    # Execute each tool and append the tool result messages
                    for idx, tc in tool_calls.items():
                        func_name = (tc.get("name") or "").strip()
                        if not func_name:
                            continue
                        try:
                            args = json.loads(tc.get("arguments") or "{}")
                        except Exception:
                            args = {}
                        logger.info(
                            f"Streaming: executing tool '{func_name}' with args: {args}"
                        )
                        tool_result = await self.execute_tool(
                            agent_name, func_name, args
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id_map.get(idx, f"call_{idx}"),
                                "content": json.dumps(tool_result),
                            }
                        )

                    accumulated_text = ""
                    continue

                # No tool calls: we've streamed the final answer
                final_text = accumulated_text
                if output_format == "audio":
                    cleaned_audio_buffer = self._clean_for_audio(final_text)
                    async for audio_chunk in self.llm_provider.tts(
                        text=cleaned_audio_buffer,
                        voice=audio_voice,
                        response_format=audio_output_format,
                    ):
                        yield audio_chunk
                else:
                    if not final_text:
                        yield ""
                self.last_text_response = final_text
                break
        except Exception as e:
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
                ):
                    yield chunk
            else:
                yield error_msg

    def _clean_for_audio(self, text: str) -> str:
        """Remove Markdown formatting, emojis, and non-pronounceable characters from text."""

        if not text:
            return ""
        text = text.replace("’", "'").replace("‘", "'")
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
        text = re.sub(r"[^\w\s\.\,\;\:\?\!\'\"\-\(\)]", " ", text)
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

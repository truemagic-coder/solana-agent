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
        output_model: Optional[Type[BaseModel]] = None,
    ) -> AsyncGenerator[Union[str, bytes, BaseModel], None]:  # pragma: no cover
        """Generate a response using OpenAI function calling (tools API) or structured output."""

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

        # Build system prompt and messages
        system_prompt = self.get_agent_system_prompt(agent_name)
        user_content = str(query)
        if images:
            user_content += "\n\n[Images attached]"

        # Compose the prompt for generate_text
        full_prompt = ""
        if memory_context:
            full_prompt += f"CONVERSATION HISTORY:\n{memory_context}\n\n"
        if prompt:
            full_prompt += f"ADDITIONAL PROMPT:\n{prompt}\n\n"
        full_prompt += user_content
        full_prompt += f"USER IDENTIFIER: {user_id}"

        # Get OpenAI function schemas for this agent's tools
        functions = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
            for tool in self.get_agent_tools(agent_name)
        ]

        try:
            if output_model is not None:
                # --- Structured output with tool support ---
                model_instance = await self.llm_provider.parse_structured_output(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    model_class=output_model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.model,
                    functions=functions if functions else None,
                    function_call="auto" if functions else None,
                )
                yield model_instance
                return

            # --- Streaming text/audio with tool support (as before) ---
            response_text = ""
            while True:
                response = await self.llm_provider.generate_text(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    functions=functions if functions else None,
                    function_call="auto" if functions else None,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.model,
                )
                if (
                    not response
                    or not hasattr(response, "choices")
                    or not response.choices
                ):
                    logger.error("No response or choices from LLM provider.")
                    response_text = "I apologize, but I could not generate a response."
                    break

                choice = response.choices[0]
                message = getattr(choice, "message", choice)

                # If the model wants to call a function/tool
                if hasattr(message, "function_call") and message.function_call:
                    function_name = message.function_call.name
                    arguments = json.loads(message.function_call.arguments)
                    logger.info(
                        f"Model requested tool '{function_name}' with args: {arguments}"
                    )

                    # Execute the tool (async)
                    tool_result = await self.execute_tool(
                        agent_name, function_name, arguments
                    )

                    # Add the tool result to the prompt for the next round
                    full_prompt += (
                        f"\n\nTool '{function_name}' was called with arguments {arguments}.\n"
                        f"Result: {tool_result}\n"
                    )
                    continue  # Loop again, LLM will see tool result and may call another tool or finish

                # Otherwise, it's a normal message (final answer)
                response_text = message.content
                break

            # Apply output guardrails if any
            processed_final_text = response_text
            if self.output_guardrails:
                for guardrail in self.output_guardrails:
                    try:
                        processed_final_text = await guardrail.process(
                            processed_final_text
                        )
                    except Exception as e:
                        logger.error(
                            f"Error applying output guardrail {guardrail.__class__.__name__}: {e}"
                        )

            self.last_text_response = processed_final_text

            if output_format == "text":
                yield processed_final_text or ""
            elif output_format == "audio":
                cleaned_audio_buffer = self._clean_for_audio(processed_final_text)
                if cleaned_audio_buffer:
                    async for audio_chunk in self.llm_provider.tts(
                        text=cleaned_audio_buffer,
                        voice=audio_voice,
                        response_format=audio_output_format,
                        instructions=audio_instructions,
                    ):
                        yield audio_chunk
                else:
                    yield ""
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
                    instructions=audio_instructions,
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

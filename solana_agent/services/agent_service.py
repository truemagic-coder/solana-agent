"""
Agent service implementation.

This service manages AI and human agents, their registration, tool assignments,
and response generation.
"""
import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Any

from solana_agent.interfaces.services import AgentService as AgentServiceInterface
from solana_agent.interfaces.providers import LLMProvider
from solana_agent.domain.agents import AIAgent, HumanAgent
from solana_agent.interfaces.plugins import ToolRegistry


class AgentService(AgentServiceInterface):
    """Service for managing agents and generating responses."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_registry: Optional[ToolRegistry] = None,
        mission_values: Optional[str] = None,
        organization_guidelines: Optional[List[str]] = None
    ):
        """Initialize the agent service.

        Args:
            llm_provider: Provider for language model interactions
            tool_registry: Optional registry for agent tools
            mission_values: Optional organization mission and values
            organization_guidelines: Optional guidelines for agents
        """
        self.llm_provider = llm_provider
        self.tool_registry = tool_registry
        self.mission_values = mission_values or ""
        self.organization_guidelines = organization_guidelines or []

        # Agent storage
        self._ai_agents: Dict[str, AIAgent] = {}
        self._human_agents: Dict[str, HumanAgent] = {}
        # specialization -> description
        self._specializations: Dict[str, str] = {}

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
        self._ai_agents[name] = agent
        self._specializations[specialization] = instructions

    def register_human_agent(self, agent: HumanAgent) -> str:
        """Register a human agent and return its ID.

        Args:
            agent: Human agent to register

        Returns:
            Agent ID
        """
        self._human_agents[agent.id] = agent

        # Add specializations
        for spec in agent.specializations:
            if spec not in self._specializations:
                self._specializations[spec] = f"Expertise in {spec}"

        return agent.id

    def get_agent_by_name(self, name: str) -> Optional[Any]:
        """Get an agent by name.

        Args:
            name: Agent name

        Returns:
            AI or human agent
        """
        if name in self._ai_agents:
            return self._ai_agents[name]

        # Check human agents
        for agent_id, agent in self._human_agents.items():
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
        agent = self._ai_agents.get(agent_name)
        if not agent:
            return ""

        # Build system prompt
        system_prompt = f"You are {agent.name}, an AI assistant with the following instructions:\n\n"
        system_prompt += agent.instructions

        # Add mission and values if available
        if self.mission_values:
            system_prompt += f"\n\nORGANIZATION VALUES:\n{self.mission_values}"

        # Add guidelines if available
        if self.organization_guidelines:
            guidelines_text = "\n".join(
                [f"- {guideline}" for guideline in self.organization_guidelines])
            system_prompt += f"\n\nGUIDELINES:\n{guidelines_text}"

        return system_prompt

    def get_specializations(self) -> Dict[str, str]:
        """Get all registered specializations.

        Returns:
            Dictionary mapping specialization names to descriptions
        """
        return self._specializations

    async def generate_response(
        self,
        agent_name: str,
        user_id: str,
        query: str,
        memory_context: str = "",
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a response from an agent.

        Args:
            agent_name: Agent name
            user_id: User ID
            query: User query
            memory_context: Optional memory context
            **kwargs: Additional parameters

        Yields:
            Response text chunks
        """
        agent = self._ai_agents.get(agent_name)
        if not agent:
            yield f"Agent '{agent_name}' not found."
            return

        # Get system prompt
        system_prompt = self.get_agent_system_prompt(agent_name)

        # Add memory context if available
        if memory_context:
            prompt = f"Memory context:\n{memory_context}\n\nUser query: {query}"
        else:
            prompt = query

        # Get agent tools if available
        tools = None
        if self.tool_registry:
            agent_tools = self.tool_registry.get_agent_tools(agent_name)
            if agent_tools:
                tools = agent_tools

        # Generate response using LLM provider
        model = agent.model

        # Pass tools if available
        if tools:
            kwargs["tools"] = tools

        async for chunk in self.llm_provider.generate_text(
            user_id=user_id,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            **kwargs
        ):
            yield chunk

    def assign_tool_to_agent(self, agent_name: str, tool_name: str) -> bool:
        """Assign a tool to an agent.

        Args:
            agent_name: Agent name
            tool_name: Tool name

        Returns:
            True if assignment was successful
        """
        if not self.tool_registry:
            return False

        if agent_name not in self._ai_agents:
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

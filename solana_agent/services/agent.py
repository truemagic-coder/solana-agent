"""
Agent service implementation.

This service manages AI and human agents, their registration, tool assignments,
and response generation.
"""
import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Any

from solana_agent.interfaces import AgentService as AgentServiceInterface
from solana_agent.interfaces import LLMProvider
from solana_agent.interfaces import AgentRepository
from solana_agent.interfaces import ToolRegistry
from solana_agent.domains import AIAgent, HumanAgent, OrganizationMission


class AgentService(AgentServiceInterface):
    """Service for managing agents and generating responses."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        agent_repository: AgentRepository,
        organization_mission: Optional[OrganizationMission] = None,
        config: Optional[Dict[str, Any]] = None
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

        # Initialize tool registry
        self.tool_registry = ToolRegistry()

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

    def register_human_agent(
        self, agent_id: str, name: str, specialization: str, notification_handler=None
    ) -> str:
        """Register a human agent and return its ID.

        Args:
            agent_id: Agent ID
            name: Agent name
            specialization: Agent specialization
            notification_handler: Optional handler for notifications

        Returns:
            Agent ID
        """
        agent = HumanAgent(
            id=agent_id,
            name=name,
            specializations=[specialization],
            availability=True,
            notification_handler=notification_handler
        )
        self.agent_repository.save_human_agent(agent)
        return agent_id

    def get_agent_by_name(self, name: str) -> Optional[Any]:
        """Get an agent by name.

        Args:
            name: Agent name

        Returns:
            AI or human agent
        """
        # Try AI agents first
        agent = self.agent_repository.get_ai_agent_by_name(name)
        if agent:
            return agent

        # Check human agents by name
        human_agents = self.agent_repository.get_all_human_agents()
        for agent in human_agents:
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
        agent = self.agent_repository.get_ai_agent_by_name(agent_name)
        if not agent:
            return ""

        # Build system prompt
        system_prompt = f"You are {agent.name}, an AI assistant with the following instructions:\n\n"
        system_prompt += agent.instructions

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

    def get_all_human_agents(self) -> Dict[str, HumanAgent]:
        """Get all registered human agents.

        Returns:
            Dictionary mapping agent IDs to agents
        """
        agents = self.agent_repository.get_all_human_agents()
        return {agent.id: agent for agent in agents}

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

        # Gather from human agents
        human_agents = self.agent_repository.get_all_human_agents()
        for agent in human_agents:
            for spec in agent.specializations:
                if spec not in specializations:
                    specializations[spec] = f"Human expertise in {spec}"

        return specializations

    def find_agents_by_specialization(self, specialization: str) -> List[str]:
        """Find agents that have a specific specialization.

        Args:
            specialization: Specialization to search for

        Returns:
            List of agent names/IDs
        """
        agent_ids = []

        # Check AI agents
        ai_agents = self.agent_repository.get_all_ai_agents()
        for agent in ai_agents:
            if agent.specialization.lower() == specialization.lower():
                agent_ids.append(agent.name)

        # Check human agents
        human_agents = self.agent_repository.get_all_human_agents()
        for agent in human_agents:
            if any(spec.lower() == specialization.lower() for spec in agent.specializations):
                agent_ids.append(agent.id)

        return agent_ids

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
        agent = self.agent_repository.get_ai_agent_by_name(agent_name)
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

    def register_tool_for_agent(self, agent_name: str, tool_name: str) -> bool:
        """Assign a tool to an agent.

        Args:
            agent_name: Agent name
            tool_name: Tool name

        Returns:
            True if assignment was successful
        """
        if not self.tool_registry:
            return False

        # Check if agent exists
        agent = self.agent_repository.get_ai_agent_by_name(agent_name)
        if not agent:
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

    def agent_exists(self, agent_id: str) -> bool:
        """Check if an agent exists.

        Args:
            agent_id: Agent ID or name

        Returns:
            True if the agent exists
        """
        # Try as AI agent
        if self.agent_repository.get_ai_agent_by_name(agent_id):
            return True

        # Try as human agent
        if self.agent_repository.get_human_agent_by_id(agent_id):
            return True

        return False

    def has_specialization(self, agent_id: str, specialization: str) -> bool:
        """Check if an agent has a specific specialization.

        Args:
            agent_id: Agent ID or name
            specialization: Specialization to check

        Returns:
            True if the agent has the specialization
        """
        # Try as AI agent
        ai_agent = self.agent_repository.get_ai_agent_by_name(agent_id)
        if ai_agent and ai_agent.specialization.lower() == specialization.lower():
            return True

        # Try as human agent
        human_agent = self.agent_repository.get_human_agent_by_id(agent_id)
        if human_agent:
            return any(spec.lower() == specialization.lower() for spec in human_agent.specializations)

        return False

    def list_active_agents(self) -> List[str]:
        """List all active agent IDs.

        Returns:
            List of active agent IDs/names
        """
        active_agents = []

        # Get AI agents (all are considered active)
        ai_agents = self.agent_repository.get_all_ai_agents()
        active_agents.extend([agent.name for agent in ai_agents])

        # Get human agents with availability=True
        human_agents = self.agent_repository.get_all_human_agents()
        active_agents.extend(
            [agent.id for agent in human_agents if agent.availability])

        return active_agents

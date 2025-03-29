"""
Routing service implementation.

This service manages query routing to appropriate agents based on
specializations and query analysis.
"""
from typing import Dict, List, Optional, Any
from solana_agent.interfaces.services.routing import RoutingService as RoutingServiceInterface
from solana_agent.interfaces.services.agent import AgentService
from solana_agent.interfaces.providers.llm import LLMProvider
from solana_agent.domains.routing import QueryAnalysis


class RoutingService(RoutingServiceInterface):
    """Service for routing queries to appropriate agents."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        agent_service: AgentService,
    ):
        """Initialize the routing service.

        Args:
            llm_provider: Provider for language model interactions
            agent_service: Service for agent management
        """
        self.llm_provider = llm_provider
        self.agent_service = agent_service

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query to determine routing information.

        Args:
            query: User query to analyze

        Returns:
            Analysis results including specializations and complexity
        """
        # Get all available agents and their specializations
        agents = self.agent_service.get_all_ai_agents()
        available_specializations = []

        for agent_id, agent in agents.items():
            available_specializations.append({
                "agent_name": agent_id,
                "specialization": agent.specialization,
            })

        specializations_text = "\n".join([
            f"- {spec['agent_name']}: {spec['specialization']}"
            for spec in available_specializations
        ])

        prompt = f"""
        Analyze this user query and determine which agent would be best suited to answer it.
        
        AVAILABLE AGENTS AND THEIR SPECIALIZATIONS:
        {specializations_text}
        
        USER QUERY: {query}
        
        Please determine:
        1. Which agent is the primary best match for this query (must be one of the listed agents)
        2. Any secondary agents that might be helpful (must be from the listed agents)
        3. The complexity level (1-5, where 5 is most complex)
        4. Any key topics or technologies mentioned
        
        Think carefully about whether the query is more technical/development-focused or more 
        financial/market-focused to match with the appropriate agent.
        """

        try:
            analysis = await self.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="Match user queries to the most appropriate agent based on specializations.",
                model_class=QueryAnalysis,
            )

            return {
                "primary_specialization": analysis.primary_specialization,
                "secondary_specializations": analysis.secondary_specializations,
                "complexity_level": analysis.complexity_level,
                "topics": analysis.topics,
                "confidence": analysis.confidence
            }
        except Exception as e:
            print(f"Error analyzing query: {e}")
            # Return default analysis on error
            return {
                "primary_specialization": list(agents.keys())[0] if agents else "general",
                "secondary_specializations": [],
                "complexity_level": 1,
                "topics": [],
                "confidence": 0.0
            }

    async def route_query(self, query: str) -> str:  # pragma: no cover
        """Route a query to the appropriate agent.

        Args:
            query: The query text

        Returns:
            Name of the best agent
        """
        # If only one agent - use that agent
        agents = self.agent_service.get_all_ai_agents()
        if len(agents) == 1:
            print(f"Only one agent available: {next(iter(agents.keys()))}")
            return next(iter(agents.keys()))

        # Analyze query
        analysis = await self._analyze_query(query)

        # Find best agent based on analysis
        best_agent = await self._find_best_ai_agent(
            analysis["primary_specialization"],
            analysis["secondary_specializations"]
        )

        # Return best agent
        return best_agent

    async def _find_best_ai_agent(
        self,
        primary_specialization: str,
        secondary_specializations: List[str],
    ) -> Optional[str]:
        """Find the best AI agent for a query.

        Args:
            primary_specialization: Primary agent name or specialization
            secondary_specializations: Secondary agent names or specializations

        Returns:
            Name of the best matching agent, or None if no match
        """
        # Get all AI agents
        ai_agents = self.agent_service.get_all_ai_agents()
        if not ai_agents:
            return None

        # First, check if primary_specialization is directly an agent name
        if primary_specialization in ai_agents:
            return primary_specialization

        # If not a direct agent name match, use specialization matching
        agent_scores = []

        for agent_id, agent in ai_agents.items():
            score = 0

            # Check for specialization match
            if agent.specialization.lower() in primary_specialization.lower() or \
                    primary_specialization.lower() in agent.specialization.lower():
                score += 10

            # Check secondary specializations
            for sec_spec in secondary_specializations:
                if sec_spec in ai_agents:  # Direct agent name match
                    if sec_spec == agent_id:
                        score += 5
                elif agent.specialization.lower() in sec_spec.lower() or \
                        sec_spec.lower() in agent.specialization.lower():
                    score += 3

            agent_scores.append((agent_id, score))

        # Sort by score
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"Agent scores: {agent_scores}")

        # Return the highest scoring agent, if any
        if agent_scores and agent_scores[0][1] > 0:
            return agent_scores[0][0]

        # If no match found, return first agent as fallback
        if ai_agents:
            return next(iter(ai_agents.keys()))

        return None

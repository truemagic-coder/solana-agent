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
        prompt = f"""
        Analyze this user query and determine:
        1. The primary specialization needed to address it
        2. Any secondary specializations that might be helpful
        3. The complexity level (1-5, where 5 is most complex)
        4. Any key topics or technologies mentioned

        User Query: {query}

        Be objective and thorough in your analysis.
        """

        try:
            analysis = await self.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="Analyze user queries to determine appropriate routing.",
                model_class=QueryAnalysis,
                temperature=0.2
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
                "primary_specialization": "general",
                "secondary_specializations": [],
                "complexity_level": 1,
                "topics": [],
                "confidence": 0.0
            }

    async def route_query(self, user_id: str, query: str) -> str:
        """Route a query to the appropriate agent.

        Args:
            user_id: ID of the user making the query
            query: The query text

        Returns:
            Name of the selected agent
        """
        # Analyze query
        analysis = await self._analyze_query(query)

        # Find best agent based on analysis
        selected_agent = await self._find_best_ai_agent(
            analysis["primary_specialization"],
            analysis["secondary_specializations"]
        )

        # Return default agent if none found
        return selected_agent or "general_ai"

    async def _find_best_ai_agent(
        self,
        primary_specialization: str,
        secondary_specializations: List[str],
    ) -> Optional[str]:
        """Find the best AI agent for a query.

        Args:
            primary_specialization: Primary specialization needed
            secondary_specializations: Secondary specializations

        Returns:
            Name of the best matching agent, or None if no match
        """
        # Get all AI agents
        ai_agents = self.agent_service.get_all_ai_agents()
        if not ai_agents:
            return None

        # Create a list to score agents
        agent_scores = []

        for agent_id, agent in ai_agents.items():
            # Base score
            score = 0

            # Check primary specialization
            if agent.specialization.lower() == primary_specialization.lower():
                score += 10

            # Check secondary specializations
            if hasattr(agent, 'secondary_specializations'):
                for sec_spec in secondary_specializations:
                    if sec_spec.lower() in [s.lower() for s in agent.secondary_specializations]:
                        score += 3

            agent_scores.append((agent_id, score))

        # Sort by score
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        # Return the highest scoring agent, if any
        if agent_scores and agent_scores[0][1] > 0:
            return agent_scores[0][0]

        # If no good match, return the first AI agent as fallback
        if ai_agents:
            return next(iter(ai_agents.keys()))

        return None

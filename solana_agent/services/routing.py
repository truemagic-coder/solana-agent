"""
Routing service implementation.

This service manages query routing to appropriate agents using an LLM-based
analysis. It defaults to a small, low-cost model for routing to minimize
overhead while maintaining quality.
"""

import logging
from typing import Dict, List, Optional, Any
from solana_agent.interfaces.services.routing import (
    RoutingService as RoutingServiceInterface,
)
from solana_agent.interfaces.services.agent import AgentService
from solana_agent.interfaces.providers.llm import LLMProvider
from solana_agent.domains.routing import QueryAnalysis

# Setup logger for this module
logger = logging.getLogger(__name__)


class RoutingService(RoutingServiceInterface):
    """Service for routing queries to appropriate agents."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        agent_service: AgentService,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the routing service.

        Args:
            llm_provider: Provider for language model interactions
            agent_service: Service for agent management
        """
        self.llm_provider = llm_provider
        self.agent_service = agent_service
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or "gpt-4.1"
        # Simple sticky session: remember last routed agent in-process
        self._last_agent = None

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
            available_specializations.append(
                {
                    "agent_name": agent_id,
                    "specialization": agent.specialization,
                }
            )

        specializations_text = "\n".join(
            [
                f"- {spec['agent_name']}: {spec['specialization']}"
                for spec in available_specializations
            ]
        )

        prompt = f"""
        Analyze this user query and determine which agent would be best suited to answer it.

        AVAILABLE AGENTS AND THEIR SPECIALIZATIONS:
        {specializations_text}

        USER QUERY: {query}

        INSTRUCTIONS:
        - Look at the user query and match it to the most appropriate agent from the list above
        - If the user mentions a specific topic or need that matches an agent's specialization, choose that agent
        - Return the EXACT agent name (not the specialization description)

        Please determine:
        1. primary_agent: The exact name of the best matching agent (e.g., "onboarding", "support")
        2. secondary_agents: Names of other agents that might help (empty list if none)
        3. complexity_level: 1-5 (5 being most complex)
        4. topics: Key topics mentioned
        5. confidence: 0.0-1.0 (how confident you are in this routing decision)
        """

        try:
            analysis = await self.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="You are an expert at routing user queries to the most appropriate AI agent. Always return the exact agent name that best matches the user's needs based on the specializations provided. If the user mentions a specific topic, prioritize agents whose specialization matches that topic.",
                model_class=QueryAnalysis,
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
            )

            logger.debug(f"LLM analysis result: {analysis}")

            return {
                "primary_specialization": analysis.primary_agent,
                "secondary_specializations": analysis.secondary_agents,
                "complexity_level": analysis.complexity_level,
                "topics": analysis.topics,
                "confidence": analysis.confidence,
            }
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")  # Use logger.error
            # Return default analysis on error
            return {
                "primary_specialization": list(agents.keys())[0]
                if agents
                else "general",
                "secondary_specializations": [],
                "complexity_level": 1,
                "topics": [],
                "confidence": 0.0,
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
            agent_name = next(iter(agents.keys()))
            logger.info(f"Only one agent available: {agent_name}")  # Use logger.info
            self._last_agent = agent_name
            return agent_name

        # Short reply bypass and default stickiness
        short = query.strip().lower()
        short_replies = {"", "yes", "no", "ok", "k", "y", "n", "1", "0"}
        if short in short_replies and self._last_agent:
            return self._last_agent

        # Always analyze with a small model to select the best agent
        analysis = await self._analyze_query(query)
        logger.debug(f"Routing analysis for query '{query}': {analysis}")
        best_agent = await self._find_best_ai_agent(
            analysis["primary_specialization"], analysis["secondary_specializations"]
        )
        logger.debug(f"Selected agent: {best_agent}")
        chosen = best_agent or next(iter(agents.keys()))
        self._last_agent = chosen
        return chosen

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
            logger.debug(f"Direct agent match: {primary_specialization}")
            return primary_specialization

        # If not a direct agent name match, use specialization matching
        agent_scores = []

        for agent_id, agent in ai_agents.items():
            score = 0

            # Check for specialization match
            if (
                agent.specialization.lower() in primary_specialization.lower()
                or primary_specialization.lower() in agent.specialization.lower()
            ):
                score += 10
                logger.debug(
                    f"Specialization match for {agent_id}: '{agent.specialization}' matches '{primary_specialization}'"
                )

            # Check secondary specializations
            for sec_spec in secondary_specializations:
                if sec_spec in ai_agents:  # Direct agent name match
                    if sec_spec == agent_id:
                        score += 5
                elif (
                    agent.specialization.lower() in sec_spec.lower()
                    or sec_spec.lower() in agent.specialization.lower()
                ):
                    score += 3

            agent_scores.append((agent_id, score))

        # Sort by score
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Agent scores: {agent_scores}")  # Use logger.debug

        # Return the highest scoring agent, if any
        if agent_scores and agent_scores[0][1] > 0:
            return agent_scores[0][0]

        # If no match found, return first agent as fallback
        if ai_agents:
            fallback_agent = next(iter(ai_agents.keys()))
            logger.debug(f"No match found, using fallback agent: {fallback_agent}")
            return fallback_agent

        return None

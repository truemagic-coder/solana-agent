class RoutingService:
    """Service for routing queries to appropriate agents."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        agent_registry: AgentRegistry,
        router_model: str = "gpt-4o-mini",
    ):
        self.llm_provider = llm_provider
        self.agent_registry = agent_registry
        self.router_model = router_model

    async def route_query(self, query: str) -> str:
        """Route a query to the appropriate agent based on content."""
        # Get available agents
        agents = self.agent_registry.get_all_ai_agents()
        if not agents:
            return "default_agent"  # Fallback to default if no agents

        agent_names = list(agents.keys())

        # Format agent descriptions for prompt
        agent_descriptions = []
        specializations = self.agent_registry.get_specializations()

        for name in agent_names:
            spec = specializations.get(name, "General assistant")
            agent_descriptions.append(f"- {name}: {spec}")

        agent_info = "\n".join(agent_descriptions)

        # Create prompt for routing
        prompt = f"""
        You are a router that determines which AI agent should handle a user query.
        
        User query: "{query}"
        
        Available agents:
        {agent_info}
        
        Select the most appropriate agent based on the query and agent specializations.
        Respond with ONLY the agent name, nothing else.
        """

        response_text = ""
        try:
            async for chunk in self.llm_provider.generate_text(
                "system",
                prompt,
                system_prompt="You are a routing system. Only respond with the name of the most appropriate agent.",
                model=self.router_model,
                temperature=0.1,
            ):
                response_text += chunk

            # Clean up the response text to handle different formats
            response_text = response_text.strip()

            # First try to parse as JSON (old behavior)
            try:
                data = json.loads(response_text)
                if isinstance(data, dict) and "agent" in data:
                    return self._match_agent_name(data["agent"], agent_names)
            except json.JSONDecodeError:
                # Not JSON, try to parse as plain text
                pass

            # Treat as plain text - just match the agent name directly
            return self._match_agent_name(response_text, agent_names)

        except Exception as e:
            print(f"Error in routing: {e}")
            # Default to the first agent if there's an error
            return agent_names[0]

    def _match_agent_name(self, response: str, agent_names: List[str]) -> str:
        """Match the response to a valid agent name."""
        # Clean up the response
        if isinstance(response, dict) and "name" in response:
            response = response["name"]  # Handle {"name": "agent_name"} format

        # Convert to string and clean it up
        clean_response = str(response).strip().lower()

        # Direct match first
        for name in agent_names:
            if name.lower() == clean_response:
                return name

        # Check for partial matches
        for name in agent_names:
            if name.lower() in clean_response or clean_response in name.lower():
                return name

        # If no match, return first agent as default
        print(
            f"No matching agent found for: '{response}'. Using {agent_names[0]}")
        return agent_names[0]

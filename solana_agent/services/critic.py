

class CriticService:
    """Service for providing critique and feedback on agent responses."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.feedback_collection = []

    async def critique_response(
        self, user_query: str, agent_response: str, agent_name: str
    ) -> CritiqueFeedback:
        """Analyze and critique an agent's response."""
        prompt = f"""
        Analyze this agent's response to the user query and provide detailed feedback:
        
        USER QUERY: {user_query}
        
        AGENT RESPONSE: {agent_response}
        
        Provide a structured critique with:
        1. Strengths of the response
        2. Areas for improvement with specific issues and recommendations
        3. Overall quality score (0.0-1.0)
        4. Priority level for improvements (low/medium/high)
        
        Format as JSON matching the CritiqueFeedback schema.
        """

        response = ""
        async for chunk in self.llm_provider.generate_text(
            "critic",
            prompt,
            system_prompt="You are an expert evaluator of AI responses. Provide objective, specific feedback.",
            stream=False,
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
        ):
            response += chunk

        try:
            data = json.loads(response)
            feedback = CritiqueFeedback(**data)

            # Store feedback for analytics
            self.feedback_collection.append(
                {
                    "agent_name": agent_name,
                    "strengths_count": len(feedback.strengths),
                    "issues_count": len(feedback.improvement_areas),
                    "overall_score": feedback.overall_score,
                    "priority": feedback.priority,
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                }
            )

            return feedback
        except Exception as e:
            print(f"Error parsing critique feedback: {e}")
            return CritiqueFeedback(
                strengths=["Unable to analyze response"],
                improvement_areas=[],
                overall_score=0.5,
                priority="medium",
            )

    def get_agent_feedback(self, agent_name: str, limit: int = 50) -> List[Dict]:
        """Get historical feedback for a specific agent."""
        return [
            fb for fb in self.feedback_collection if fb["agent_name"] == agent_name
        ][-limit:]

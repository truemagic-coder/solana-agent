"""
Critic service implementation.

This service provides quality assessment of responses and feedback.
"""
from typing import Dict, Optional, Any

from solana_agent.interfaces.services import CriticService as CriticServiceInterface
from solana_agent.interfaces.providers import LLMProvider
from solana_agent.domain.models import ResponseEvaluation


class CriticService(CriticServiceInterface):
    """Service for evaluating response quality and providing feedback."""

    def __init__(self, llm_provider: LLMProvider, model: str = "gpt-4o-mini"):
        """Initialize the critic service.

        Args:
            llm_provider: Provider for language model interactions
            model: Model to use for evaluations
        """
        self.llm_provider = llm_provider
        self.model = model

    async def evaluate_response(self, query: str, response: str) -> Dict[str, Any]:
        """Evaluate the quality of a response to a user query.

        Args:
            query: User query
            response: Agent response to evaluate

        Returns:
            Evaluation results including scores and feedback
        """
        prompt = f"""
        Please evaluate this AI assistant response to the user query.
        
        User query: {query}
        
        AI response: {response}
        
        Evaluate on:
        1. Accuracy (Is the information correct?)
        2. Relevance (Does it address the query?)
        3. Completeness (Is anything important missing?)
        4. Clarity (Is it easy to understand?)
        5. Helpfulness (Does it solve the user's problem?)
        """

        try:
            evaluation = await self.llm_provider.parse_structured_output(
                prompt=prompt,
                system_prompt="You are an expert evaluator of AI responses. Provide fair and objective ratings.",
                model_class=ResponseEvaluation,
                model=self.model,
                temperature=0.3
            )

            # Convert to dictionary
            return {
                "scores": {
                    "accuracy": evaluation.accuracy,
                    "relevance": evaluation.relevance,
                    "completeness": evaluation.completeness,
                    "clarity": evaluation.clarity,
                    "helpfulness": evaluation.helpfulness,
                    "overall": evaluation.overall_score
                },
                "feedback": evaluation.feedback,
                "improvement_suggestions": evaluation.improvement_suggestions,
                "action_needed": evaluation.overall_score < 7.0
            }
        except Exception as e:
            print(f"Error in response evaluation: {e}")
            return {
                "scores": {
                    "accuracy": 0, "relevance": 0, "completeness": 0,
                    "clarity": 0, "helpfulness": 0, "overall": 0
                },
                "feedback": f"Evaluation failed: {str(e)}",
                "improvement_suggestions": [],
                "action_needed": True
            }

    async def needs_human_intervention(
        self, query: str, response: str, threshold: float = 5.0
    ) -> bool:
        """Determine if a response requires human intervention.

        Args:
            query: User query
            response: Agent response
            threshold: Quality threshold below which human help is needed

        Returns:
            True if human intervention is recommended
        """
        evaluation = await self.evaluate_response(query, response)
        overall_score = evaluation.get("scores", {}).get("overall", 0)

        return overall_score < threshold or evaluation.get("action_needed", False)

    async def suggest_improvements(self, query: str, response: str) -> str:
        """Suggest improvements for a response.

        Args:
            query: User query
            response: Agent response to improve

        Returns:
            Improvement suggestions
        """
        prompt = f"""
        Please suggest specific improvements for this AI assistant response:
        
        User query: {query}
        
        AI response: {response}
        
        Focus on concrete ways to make the response more accurate, relevant, complete,
        clear, and helpful. Provide specific suggestions, not general feedback.
        """

        try:
            async for chunk in self.llm_provider.generate_text(
                user_id="system",
                prompt=prompt,
                system_prompt="You are an expert at improving AI responses. Provide specific, actionable suggestions.",
                model=self.model,
                stream=False,
                temperature=0.4
            ):
                return chunk

            return "No suggestions available."
        except Exception as e:
            print(f"Error generating improvement suggestions: {e}")
            return f"Error: {str(e)}"

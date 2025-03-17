
class MemoryService:
    """Service for managing collective memory and insights."""

    def __init__(self, memory_repository: MemoryRepository, llm_provider: LLMProvider):
        self.memory_repository = memory_repository
        self.llm_provider = llm_provider

    async def extract_insights(
        self, conversation: Dict[str, str]
    ) -> List[MemoryInsight]:
        """Extract insights from a conversation."""
        prompt = f"""
        Extract factual, generalizable insights from this conversation that would be valuable to remember.
        
        User: {conversation.get('message', '')}
        Assistant: {conversation.get('response', '')}
        
        Extract only factual information that would be useful for future similar conversations.
        Ignore subjective opinions, preferences, or greeting messages.
        Only extract high-quality insights worth remembering.
        If no valuable insights exist, return an empty array.
        """

        try:
            # Use the new parse method
            result = await self.llm_provider.parse_structured_output(
                prompt,
                system_prompt="Extract factual insights from conversations.",
                model_class=MemoryInsightsResponse,
                temperature=0.2,
            )

            # Convert to domain model instances
            return [MemoryInsight(**insight.model_dump()) for insight in result.insights]
        except Exception as e:
            print(f"Error extracting insights: {e}")
            return []

    async def store_insights(self, user_id: str, insights: List[MemoryInsight]) -> None:
        """Store multiple insights in memory."""
        for insight in insights:
            self.memory_repository.store_insight(user_id, insight)

    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """Search collective memory for relevant insights."""
        return self.memory_repository.search(query, limit)

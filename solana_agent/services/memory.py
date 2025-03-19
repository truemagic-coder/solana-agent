"""
Memory service implementation.

This service manages collective memory, insights extraction, and memory search.
"""
from typing import Dict, List, Optional, Any

from solana_agent.interfaces import MemoryService as MemoryServiceInterface
from solana_agent.interfaces import LLMProvider
from solana_agent.interfaces import MemoryRepository
from solana_agent.domains import MemoryInsight
from solana_agent.domains import MemoryInsightsResponse


class MemoryService(MemoryServiceInterface):
    """Service for managing collective memory and insights."""

    def __init__(self, memory_repository: MemoryRepository, llm_provider: LLMProvider):
        """Initialize the memory service.

        Args:
            memory_repository: Repository for storing and retrieving memory
            llm_provider: Provider for language model interactions
        """
        self.memory_repository = memory_repository
        self.llm_provider = llm_provider

    async def extract_insights(self, conversation: Dict[str, str]) -> List[MemoryInsight]:
        """Extract insights from a conversation.

        Args:
            conversation: Dictionary with 'message' and 'response' keys

        Returns:
            List of extracted memory insights
        """
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
            # Use the structured output parsing
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
        """Store multiple insights in memory.

        Args:
            user_id: ID of the user these insights relate to
            insights: List of insights to store
        """
        for insight in insights:
            self.memory_repository.store_insight(user_id, insight)

    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """Search collective memory for relevant insights.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of matching memory items
        """
        return self.memory_repository.search(query, limit)

    async def summarize_user_history(self, user_id: str) -> str:
        """Summarize a user's conversation history.

        Args:
            user_id: User ID to summarize history for

        Returns:
            Summary text
        """
        history = self.memory_repository.get_user_history(user_id, limit=20)
        if not history:
            return "No conversation history available."

        # Format history for the LLM
        formatted_history = "\n".join([
            f"User: {entry.get('user_message', '')}\n"
            f"Assistant: {entry.get('assistant_message', '')}\n"
            for entry in history
        ])

        prompt = f"""
        Please provide a concise summary of this conversation history.
        Focus on key topics discussed, user preferences, and important information.
        
        {formatted_history}
        """

        try:
            async for chunk in self.llm_provider.generate_text(
                user_id=user_id,
                prompt=prompt,
                system_prompt="You summarize conversation history accurately and concisely.",
                stream=False,
                temperature=0.3
            ):
                return chunk

            return "Unable to generate summary."
        except Exception as e:
            print(f"Error summarizing history: {e}")
            return "Error generating conversation summary."

    async def get_user_history(self, user_id, limit=20):
        """Get a user's conversation history.

        Args:
            user_id: User ID
            limit: Maximum number of history items to return

        Returns:
            List of conversation history items
        """
        return self.memory_repository.get_user_history(user_id, limit)

    async def get_user_history_paginated(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "desc"  # "asc" for oldest-first, "desc" for newest-first
    ) -> Dict[str, Any]:
        """Get paginated message history for a user.

        Args:
            user_id: User ID
            page_num: Page number (starting from 1)
            page_size: Number of messages per page
            sort_order: Sort order ("asc" or "desc")

        Returns:
            Dictionary with paginated results and metadata
        """
        try:
            # Validate inputs
            if page_num < 1:
                page_num = 1
            if page_size < 1:
                page_size = 20
            if sort_order not in ["asc", "desc"]:
                sort_order = "asc"

            # Get total count of messages for this user
            total_count = self.memory_repository.count_user_history(user_id)

            # Calculate pagination metadata
            total_pages = (total_count + page_size -
                           1) // page_size  # Ceiling division

            # Adjust page number if it exceeds total pages
            if page_num > total_pages and total_pages > 0:
                page_num = total_pages

            # Calculate skip/offset
            skip = (page_num - 1) * page_size

            # Get paginated history from repository
            history_items = self.memory_repository.get_user_history_paginated(
                user_id=user_id,
                skip=skip,
                limit=page_size,
                sort_order=sort_order
            )

            # Return formatted response
            return {
                "data": history_items,
                "total": total_count,
                "page": page_num,
                "page_size": page_size,
                "total_pages": total_pages
            }
        except Exception as e:
            print(f"Error retrieving paginated history: {e}")
            return {
                "data": [],
                "total": 0,
                "page": page_num,
                "page_size": page_size,
                "total_pages": 0,
                "error": str(e)
            }

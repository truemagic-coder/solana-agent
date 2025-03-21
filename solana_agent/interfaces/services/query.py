from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, BinaryIO, Dict, Literal, Optional, Union


class QueryService(ABC):
    """Interface for processing user queries."""

    @abstractmethod
    async def process(
        self,
        user_id: str,
        query: Union[str, Path, BinaryIO],
        output_format: Literal["text", "audio"] = "text",
        voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                       "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
        audio_instructions: Optional[str] = None,
    ) -> AsyncGenerator[Union[str, bytes], None]:
        """Process the user request and generate a response."""
        pass

    @abstractmethod
    async def get_user_history(
        self,
        user_id: str,
        page_num: int = 1,
        page_size: int = 20,
        sort_order: str = "desc"  # "asc" for oldest-first, "desc" for newest-first
    ) -> Dict[str, Any]:
        """Get paginated message history for a user."""
        pass

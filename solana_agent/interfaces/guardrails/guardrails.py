from abc import ABC, abstractmethod
from typing import Any, Dict


class Guardrail(ABC):
    """Base class for all guardrails."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @abstractmethod
    async def process(self, text: str) -> str:
        """Process the text and return the modified text."""
        pass


class InputGuardrail(Guardrail):
    """Interface for guardrails applied to user input."""

    pass


class OutputGuardrail(Guardrail):
    """Interface for guardrails applied to agent output."""

    pass

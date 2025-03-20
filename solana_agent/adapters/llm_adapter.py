"""
LLM provider adapters for the Solana Agent system.

These adapters implement the LLMProvider interface for different LLM services.
"""
from typing import AsyncGenerator, List, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from solana_agent.interfaces.providers.llm import LLMProvider

T = TypeVar('T', bound=BaseModel)


class OpenAIAdapter(LLMProvider):
    """OpenAI implementation of LLMProvider with web search capabilities."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        # Add search-enabled model variants
        self.search_models = {
            "gpt-4o": "gpt-4o-search-preview",
            "gpt-4o-mini": "gpt-4o-mini-search-preview"
        }

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        needs_search: bool = False,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate text from OpenAI models with web search capability."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Prepare request parameters
        request_params = {
            "messages": messages,
            "stream": kwargs.get("stream", True),
            "response_format": kwargs.get("response_format", None),
        }

        # If search is needed, update model and add search options
        if needs_search:
            base_model = kwargs.get("model", self.model)
            request_params["model"] = self.search_models.get(
                base_model, "gpt-4o-mini-search-preview")
            request_params["web_search_options"] = {
                "search_context_size": "medium",
                "user_location": {
                    "type": "approximate",
                    "approximate": {
                        "country": "US",
                        "timezone": "America/Los_Angeles"
                    }
                }
            }
        else:
            request_params["model"] = kwargs.get("model", self.model)

        try:
            response = self.client.chat.completions.create(**request_params)
            current_text = ""

            for chunk in response:
                if chunk.choices:
                    # Handle content
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        current_text += text
                        yield text

        except Exception as e:
            print(f"Error in generate_text: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield f"I apologize, but I encountered an error: {str(e)}"

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for a given text using OpenAI's embedding model."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a zero vector as fallback (not ideal but prevents crashing)
            return [0.0] * 1536  # Standard size for text-embedding-3-small

    async def parse_structured_output(
        self,
        prompt: str,
        system_prompt: str,
        model_class: Type[T],
        **kwargs
    ) -> T:
        """Generate structured output using Pydantic model parsing."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            # First try the beta parsing API
            completion = self.client.beta.chat.completions.parse(
                model=kwargs.get("model", self.model),
                messages=messages,
                response_format=model_class,
                temperature=kwargs.get("temperature", 0.2),
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"Error with beta.parse method: {e}")

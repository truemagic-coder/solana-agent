"""
LLM provider adapters for the Solana Agent system.

These adapters implement the LLMProvider interface for different LLM services.
"""
from pathlib import Path
from typing import AsyncGenerator, BinaryIO, List, Literal, Type, TypeVar, Union

from openai import OpenAI
from pydantic import BaseModel

from solana_agent.interfaces.providers.llm import LLMProvider

T = TypeVar('T', bound=BaseModel)


class OpenAIAdapter(LLMProvider):
    """OpenAI implementation of LLMProvider with web search capabilities."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.parse_model = "gpt-4o-mini"
        self.search_model = "gpt-4o-mini-search-preview"
        self.transcription_model = "gpt-4o-mini-transcribe"
        self.tts_model = "gpt-4o-mini-tts"

    async def tts(
        self,
        text: str,
        instructions: str = "",
        voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                       "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
    ) -> AsyncGenerator[bytes, None]:  # pragma: no cover
        """Stream text-to-speech audio from OpenAI models.

        Args:
            text: Text to convert to speech
            instructions: Optional instructions for speech generation
            voice: Voice to use for synthesis

        Yields:
            Audio bytes as they become available
        """
        try:
            stream = self.client.audio.speech.create(
                model=self.tts_model,
                voice=voice,
                input=text,
                instructions=instructions,
            )

            # Stream the bytes in chunks
            for chunk in stream.iter_bytes(chunk_size=1024 * 16):  # 16KB chunks
                yield chunk

        except Exception as e:
            print(f"Error in text_to_speech: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield b""  # Return empty bytes on error

        except Exception as e:
            print(f"Error in text_to_speech: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield f"I apologize, but I encountered an error converting text to speech: {str(e)}"

    async def transcribe_audio(
        self,
        audio_file: Union[str, Path, BinaryIO],
    ) -> AsyncGenerator[str, None]:  # pragma: no cover
        """Stream transcription of an audio file.

        Args:
            audio_file: Path to audio file or file-like object

        Yields:
            Transcript text chunks as they become available
        """
        try:
            # Handle file path vs file object
            if isinstance(audio_file, (str, Path)):
                audio_file = open(audio_file, "rb")

            stream = self.client.audio.transcriptions.create(
                model=self.transcription_model,
                file=audio_file,
                response_format="text",
                stream=True
            )

            for event in stream:
                if hasattr(event, 'text') and event.text:
                    yield event.text

        except Exception as e:
            print(f"Error in transcribe_audio: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield f"I apologize, but I encountered an error transcribing the audio: {str(e)}"

        finally:
            # Close file if we opened it
            if isinstance(audio_file, (str, Path)):
                audio_file.close()

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> AsyncGenerator[str, None]:  # pragma: no cover
        """Generate text from OpenAI models with web search capability."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Prepare request parameters
        request_params = {
            "messages": messages,
            "stream": True,
            "model": self.search_model,
        }
        try:
            response = self.client.chat.completions.create(**request_params)

            for chunk in response:
                if chunk.choices:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        yield text

        except Exception as e:
            print(f"Error in generate_text: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield f"I apologize, but I encountered an error: {str(e)}"

    def generate_embedding(self, text: str) -> List[float]:  # pragma: no cover
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
    ) -> T:  # pragma: no cover
        """Generate structured output using Pydantic model parsing."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            # First try the beta parsing API
            completion = self.client.beta.chat.completions.parse(
                model=self.parse_model,
                messages=messages,
                response_format=model_class,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"Error with beta.parse method: {e}")

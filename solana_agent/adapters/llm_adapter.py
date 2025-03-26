"""
LLM provider adapters for the Solana Agent system.

These adapters implement the LLMProvider interface for different LLM services.
"""
from typing import AsyncGenerator, List, Literal, Type, TypeVar, Union

from openai import OpenAI
from pydantic import BaseModel

from solana_agent.interfaces.providers.llm import LLMProvider

T = TypeVar('T', bound=BaseModel)


class OpenAIAdapter(LLMProvider):
    """OpenAI implementation of LLMProvider with web search capabilities."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.parse_model = "gpt-4o-mini"
        self.text_model = "gpt-4o-mini"
        self.transcription_model = "gpt-4o-mini-transcribe"
        self.tts_model = "tts-1"

    async def tts(
        self,
        text: str,
        instructions: str = "",
        voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                       "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
        response_format: Literal['mp3', 'opus',
                                 'aac', 'flac', 'wav', 'pcm'] = "aac",
    ) -> AsyncGenerator[bytes, None]:  # pragma: no cover
        """Stream text-to-speech audio from OpenAI models.

        Args:
            text: Text to convert to speech
            instructions: Optional instructions for speech generation
            voice: Voice to use for synthesis
            response_format: Audio format

        Yields:
            Audio bytes as they become available
        """
        try:
            stream = self.client.audio.speech.create(
                model=self.tts_model,
                voice=voice,
                input=text,
                response_format=response_format
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
        audio_bytes: bytes,
        input_format: Literal[
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        ] = "mp4",
    ) -> AsyncGenerator[str, None]:  # pragma: no cover
        """Stream transcription of an audio file.

        Args:
            audio_bytes: Audio file bytes
            input_format: Format of the input audio file

        Yields:
            Transcript text chunks as they become available
        """
        try:
            stream = self.client.audio.transcriptions.create(
                model=self.transcription_model,
                file=(f"file.{input_format}", audio_bytes),
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

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> AsyncGenerator[str, None]:  # pragma: no cover
        """Generate text from OpenAI models."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Prepare request parameters
        request_params = {
            "messages": messages,
            "stream": True,
            "model": self.text_model,
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

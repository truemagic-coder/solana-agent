"""
LLM provider adapters for the Solana Agent system.

These adapters implement the LLMProvider interface for different LLM services.
"""
from typing import AsyncGenerator, Literal, Optional, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel
import instructor
from instructor import Mode

from solana_agent.interfaces.providers.llm import LLMProvider

T = TypeVar('T', bound=BaseModel)


class OpenAIAdapter(LLMProvider):
    """OpenAI implementation of LLMProvider with web search capabilities."""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.parse_model = "gpt-4o-mini"
        self.text_model = "gpt-4o-mini"
        self.transcription_model = "gpt-4o-mini-transcribe"
        self.tts_model = "tts-1"

    async def tts(
        self,
        text: str,
        instructions: str = "You speak in a friendly and helpful manner.",
        voice: Literal["alloy", "ash", "ballad", "coral", "echo",
                       "fable", "onyx", "nova", "sage", "shimmer"] = "nova",
        response_format: Literal['mp3', 'opus',
                                 'aac', 'flac', 'wav', 'pcm'] = "aac",
    ) -> AsyncGenerator[bytes, None]:  # pragma: no cover
        """Stream text-to-speech audio from OpenAI models.

        Args:
            text: Text to convert to speech
            instructions: Not used in this implementation
            voice: Voice to use for synthesis
            response_format: Audio format

        Yields:
            Audio bytes as they become available
        """
        try:
            async with self.client.audio.speech.with_streaming_response.create(
                model=self.tts_model,
                voice=voice,
                input=text,
                response_format=response_format
            ) as stream:
                # Stream the bytes in 16KB chunks
                async for chunk in stream.iter_bytes(chunk_size=1024 * 16):
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
            yield b""  # Return empty bytes on error

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
            async with self.client.audio.transcriptions.with_streaming_response.create(
                model=self.transcription_model,
                file=(f"file.{input_format}", audio_bytes),
                response_format="text",
            ) as stream:
                # Stream the text in 16KB chunks
                async for chunk in stream.iter_text(chunk_size=1024 * 16):
                    yield chunk

        except Exception as e:
            print(f"Error in transcribe_audio: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield f"I apologize, but I encountered an error transcribing the audio: {str(e)}"

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
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

        if api_key and base_url:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            client = self.client

        if model:
            request_params["model"] = model

        try:
            response = await client.chat.completions.create(**request_params)

            async for chunk in response:
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
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> T:  # pragma: no cover
        """Generate structured output using Pydantic model parsing with Instructor."""

        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            if api_key and base_url:
                client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            else:
                client = self.client

            if model:
                self.parse_model = model

            # Create a patched client with TOOLS_STRICT mode
            patched_client = instructor.from_openai(
                client, mode=Mode.TOOLS_STRICT)

            # Use instructor's structured generation with function calling
            response = await patched_client.chat.completions.create(
                model=self.parse_model,
                messages=messages,
                response_model=model_class,
                max_retries=2  # Automatically retry on validation errors
            )
            return response
        except Exception as e:
            print(
                f"Error with instructor parsing (TOOLS_STRICT mode): {e}")

            try:
                # First fallback: Try regular JSON mode
                patched_client = instructor.from_openai(
                    self.client, mode=Mode.JSON)
                response = await patched_client.chat.completions.create(
                    model=self.parse_model,
                    messages=messages,
                    response_model=model_class,
                    max_retries=1
                )
                return response
            except Exception as json_error:
                print(f"JSON mode fallback also failed: {json_error}")

                try:
                    # Final fallback: Manual extraction with a detailed prompt
                    fallback_system_prompt = f"""
                    {system_prompt}

                    You must respond with valid JSON that can be parsed as the following Pydantic model:
                    {model_class.model_json_schema()}

                    Ensure the response contains ONLY the JSON object and nothing else.
                    """

                    # Regular completion without instructor
                    completion = await self.client.chat.completions.create(
                        model=self.parse_model,
                        messages=[
                            {"role": "system", "content": fallback_system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"}
                    )

                    # Extract and parse the JSON response
                    json_str = completion.choices[0].message.content

                    # Use Pydantic to parse and validate
                    return model_class.model_validate_json(json_str)

                except Exception as fallback_error:
                    print(f"All fallback methods failed: {fallback_error}")
                    raise ValueError(
                        f"Failed to generate structured output: {e}. All fallbacks failed."
                    ) from e

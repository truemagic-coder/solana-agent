"""
LLM provider adapters for the Solana Agent system.

These adapters implement the LLMProvider interface for different LLM services.
"""

import logging
from typing import AsyncGenerator, List, Literal, Optional, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel
import instructor
from instructor import Mode
import logfire

from solana_agent.interfaces.providers.llm import LLMProvider

# Setup logger for this module
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_CHAT_MODEL = "gpt-4.1"
DEFAULT_PARSE_MODEL = "gpt-4.1-nano"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 3072
DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_TTS_MODEL = "tts-1"


class OpenAIAdapter(LLMProvider):
    """OpenAI implementation of LLMProvider with web search capabilities."""

    def __init__(self, api_key: str, logfire_api_key: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key)

        self.logfire = False
        if logfire_api_key:
            try:
                logfire.configure(token=logfire_api_key)
                self.logfire = True
                logger.info("Logfire configured successfully.")  # Use logger.info
            except Exception as e:
                logger.error(f"Failed to configure Logfire: {e}")  # Use logger.error
                self.logfire = False

        self.parse_model = DEFAULT_PARSE_MODEL
        self.text_model = DEFAULT_CHAT_MODEL
        self.transcription_model = DEFAULT_TRANSCRIPTION_MODEL
        self.tts_model = DEFAULT_TTS_MODEL
        self.embedding_model = DEFAULT_EMBEDDING_MODEL
        self.embedding_dimensions = DEFAULT_EMBEDDING_DIMENSIONS

    async def tts(
        self,
        text: str,
        instructions: str = "You speak in a friendly and helpful manner.",
        voice: Literal[
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "fable",
            "onyx",
            "nova",
            "sage",
            "shimmer",
        ] = "nova",
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "aac",
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
            if self.logfire:  # Instrument only if logfire is enabled
                logfire.instrument_openai(self.client)
            async with self.client.audio.speech.with_streaming_response.create(
                model=self.tts_model,
                voice=voice,
                input=text,
                response_format=response_format,
            ) as stream:
                # Stream the bytes in 16KB chunks
                async for chunk in stream.iter_bytes(chunk_size=1024 * 16):
                    yield chunk

        except Exception as e:
            # Log the exception with traceback
            logger.exception(f"Error in text_to_speech: {e}")
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
            if self.logfire:  # Instrument only if logfire is enabled
                logfire.instrument_openai(self.client)
            async with self.client.audio.transcriptions.with_streaming_response.create(
                model=self.transcription_model,
                file=(f"file.{input_format}", audio_bytes),
                response_format="text",
            ) as stream:
                # Stream the text in 16KB chunks
                async for chunk in stream.iter_text(chunk_size=1024 * 16):
                    yield chunk

        except Exception as e:
            # Log the exception with traceback
            logger.exception(f"Error in transcribe_audio: {e}")
            yield f"I apologize, but I encountered an error transcribing the audio: {str(e)}"

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:  # pragma: no cover
        """Generate text from OpenAI models as a single string."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare request parameters - stream is always False now
        request_params = {
            "messages": messages,
            "stream": False,  # Hardcoded to False
            "model": model or self.text_model,
        }

        # Determine client based on provided api_key/base_url
        if api_key and base_url:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            client = self.client

        if self.logfire:
            logfire.instrument_openai(client)

        try:
            # Make the non-streaming API call
            response = await client.chat.completions.create(**request_params)

            # Handle non-streaming response
            if response.choices and response.choices[0].message.content:
                full_text = response.choices[0].message.content
                return full_text  # Return the complete string
            else:
                logger.warning(
                    "Received non-streaming response with no content."
                )  # Use logger.warning
                return ""  # Return empty string if no content

        except Exception as e:
            # Log the exception and return an error message string
            logger.exception(f"Error in generate_text: {e}")
            # Consider returning a more informative error string or raising
            return f"Error generating text: {e}"

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

            if self.logfire:
                logfire.instrument_openai(client)

            # Use the provided model or the default parse model
            current_parse_model = model or self.parse_model

            patched_client = instructor.from_openai(client, mode=Mode.TOOLS_STRICT)

            # Use instructor's structured generation with function calling
            response = await patched_client.chat.completions.create(
                model=current_parse_model,  # Use the determined model
                messages=messages,
                response_model=model_class,
                max_retries=2,  # Automatically retry on validation errors
            )
            return response
        except Exception as e:
            logger.warning(
                f"Instructor parsing (TOOLS_STRICT mode) failed: {e}"
            )  # Log warning

            try:
                # Determine client again for fallback
                if api_key and base_url:
                    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                else:
                    client = self.client

                if self.logfire:  # Instrument again if needed
                    logfire.instrument_openai(client)

                # Use the provided model or the default parse model
                current_parse_model = model or self.parse_model

                # First fallback: Try regular JSON mode
                logger.info("Falling back to instructor JSON mode.")  # Log info
                patched_client = instructor.from_openai(client, mode=Mode.JSON)
                response = await patched_client.chat.completions.create(
                    model=current_parse_model,  # Use the determined model
                    messages=messages,
                    response_model=model_class,
                    max_retries=1,
                )
                return response
            except Exception as json_error:
                logger.warning(
                    f"Instructor JSON mode fallback also failed: {json_error}"
                )  # Log warning

                try:
                    # Determine client again for final fallback
                    if api_key and base_url:
                        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                    else:
                        client = self.client

                    if self.logfire:  # Instrument again if needed
                        logfire.instrument_openai(client)

                    # Use the provided model or the default parse model
                    current_parse_model = model or self.parse_model

                    # Final fallback: Manual extraction with a detailed prompt
                    logger.info("Falling back to manual JSON extraction.")  # Log info
                    fallback_system_prompt = f"""
                    {system_prompt}

                    You must respond with valid JSON that can be parsed as the following Pydantic model:
                    {model_class.model_json_schema()}

                    Ensure the response contains ONLY the JSON object and nothing else.
                    """

                    # Regular completion without instructor
                    completion = await client.chat.completions.create(
                        model=current_parse_model,  # Use the determined model
                        messages=[
                            {"role": "system", "content": fallback_system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                    )

                    # Extract and parse the JSON response
                    json_str = completion.choices[0].message.content

                    # Use Pydantic to parse and validate
                    return model_class.model_validate_json(json_str)

                except Exception as fallback_error:
                    # Log the final exception with traceback
                    logger.exception(
                        f"All structured output fallback methods failed: {fallback_error}"
                    )
                    raise ValueError(
                        f"Failed to generate structured output: {e}. All fallbacks failed."
                    ) from e

    async def embed_text(
        self, text: str, model: Optional[str] = None, dimensions: Optional[int] = None
    ) -> List[float]:  # pragma: no cover
        """Generate an embedding for the given text using OpenAI.

        Args:
            text: The text to embed.
            model: The embedding model to use (defaults to text-embedding-3-large).
            dimensions: Desired output dimensions for the embedding.

        Returns:
            A list of floats representing the embedding vector.
        """
        if not text:
            # Log error instead of raising immediately, let caller handle empty input if needed
            logger.error("Attempted to embed empty text.")
            raise ValueError("Text cannot be empty")

        try:
            # Use provided model/dimensions or fall back to defaults
            embedding_model = model or self.embedding_model
            embedding_dimensions = dimensions or self.embedding_dimensions

            # Replace newlines with spaces as recommended by OpenAI
            text = text.replace("\n", " ")

            if self.logfire:  # Instrument only if logfire is enabled
                logfire.instrument_openai(self.client)

            response = await self.client.embeddings.create(
                input=[text], model=embedding_model, dimensions=embedding_dimensions
            )

            if response.data and response.data[0].embedding:
                return response.data[0].embedding
            else:
                # Log warning about unexpected response structure
                logger.warning(
                    "Failed to retrieve embedding from OpenAI response structure."
                )
                raise ValueError("Failed to retrieve embedding from OpenAI response")

        except Exception as e:
            # Log the exception with traceback before raising
            logger.exception(f"Error generating embedding: {e}")
            raise  # Re-raise the original exception

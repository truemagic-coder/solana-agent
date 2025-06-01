"""
LLM provider adapters for the Solana Agent system.

These adapters implement the LLMProvider interface for different LLM services.
"""

import logging
import base64
import io
import math
from typing import (
    AsyncGenerator,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Dict,
    Any,
    Union,
)
from PIL import Image
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel
import instructor
from instructor import Mode
import logfire

from solana_agent.interfaces.providers.llm import LLMProvider

# Setup logger for this module
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_CHAT_MODEL = "gpt-4.1"
DEFAULT_VISION_MODEL = "gpt-4.1"
DEFAULT_PARSE_MODEL = "gpt-4.1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 3072
DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_TTS_MODEL = "tts-1"

# Image constants
SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}
MAX_IMAGE_SIZE_MB = 20
MAX_TOTAL_IMAGE_SIZE_MB = 50
MAX_IMAGE_COUNT = 500
GPT41_PATCH_SIZE = 32
GPT41_MAX_PATCHES = 1536
GPT41_MINI_MULTIPLIER = 1.62
GPT41_NANO_MULTIPLIER = 2.46


class OpenAIAdapter(LLMProvider):
    """OpenAI implementation of LLMProvider with web search capabilities."""

    def __init__(self, api_key: str, logfire_api_key: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key)

        self.logfire = False
        if logfire_api_key:
            try:
                logfire.configure(token=logfire_api_key)
                self.logfire = True
                logger.info("Logfire configured successfully.")
            except Exception as e:
                logger.error(f"Failed to configure Logfire: {e}")
                self.logfire = False

        self.parse_model = DEFAULT_PARSE_MODEL
        self.text_model = DEFAULT_CHAT_MODEL
        self.vision_model = DEFAULT_VISION_MODEL  # Add vision model attribute
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
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:  # pragma: no cover
        """Generate text or function call from OpenAI models."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_params = {
            "messages": messages,
            "model": model or self.text_model,
        }
        if tools:
            request_params["tools"] = tools

        if api_key and base_url:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            client = self.client

        if self.logfire:
            logfire.instrument_openai(client)

        try:
            response = await client.chat.completions.create(**request_params)
            return response
        except OpenAIError as e:
            logger.error(f"OpenAI API error during text generation: {e}")
            return None
        except Exception as e:
            logger.exception(f"Error in generate_text: {e}")
            return None

    def _calculate_gpt41_image_cost(self, width: int, height: int, model: str) -> int:
        """Calculates the token cost for an image with GPT-4.1 models."""
        patches_wide = math.ceil(width / GPT41_PATCH_SIZE)
        patches_high = math.ceil(height / GPT41_PATCH_SIZE)
        total_patches_needed = patches_wide * patches_high

        if total_patches_needed > GPT41_MAX_PATCHES:
            scale_factor = math.sqrt(GPT41_MAX_PATCHES / total_patches_needed)
            new_width = math.floor(width * scale_factor)
            new_height = math.floor(height * scale_factor)

            final_patches_wide_scaled = math.ceil(new_width / GPT41_PATCH_SIZE)
            final_patches_high_scaled = math.ceil(new_height / GPT41_PATCH_SIZE)
            image_tokens = final_patches_wide_scaled * final_patches_high_scaled

            # Ensure it doesn't exceed the cap due to ceiling operations after scaling
            image_tokens = min(image_tokens, GPT41_MAX_PATCHES)

            logger.debug(
                f"Image scaled down. Original patches: {total_patches_needed}, New dims: ~{new_width}x{new_height}, Final patches: {image_tokens}"
            )

        else:
            image_tokens = total_patches_needed
            logger.debug(f"Image fits within patch limit. Patches: {image_tokens}")

        # Apply model-specific multiplier
        if "mini" in model:
            total_tokens = math.ceil(image_tokens * GPT41_MINI_MULTIPLIER)
        elif "nano" in model:
            total_tokens = math.ceil(image_tokens * GPT41_NANO_MULTIPLIER)
        else:  # Assume base gpt-4.1
            total_tokens = image_tokens

        logger.info(
            f"Calculated token cost for image ({width}x{height}) with model '{model}': {total_tokens} tokens (base image tokens: {image_tokens})"
        )
        return total_tokens

    async def generate_text_with_images(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        system_prompt: str = "",
        detail: Literal["low", "high", "auto"] = "auto",
    ) -> str:  # pragma: no cover
        """Generate text from OpenAI models using text and image inputs."""
        if not images:
            logger.warning(
                "generate_text_with_images called with no images. Falling back to generate_text."
            )
            return await self.generate_text(prompt, system_prompt)

        target_model = self.vision_model
        if "gpt-4.1" not in target_model:  # Basic check for vision model
            logger.warning(
                f"Model '{target_model}' might not support vision. Using it anyway."
            )

        content_list: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        total_image_bytes = 0
        total_image_tokens = 0

        if len(images) > MAX_IMAGE_COUNT:
            logger.error(
                f"Too many images provided ({len(images)}). Maximum is {MAX_IMAGE_COUNT}."
            )
            return f"Error: Too many images provided ({len(images)}). Maximum is {MAX_IMAGE_COUNT}."

        for i, image_input in enumerate(images):
            image_url_data: Dict[str, Any] = {"detail": detail}
            image_bytes: Optional[bytes] = None
            image_format: Optional[str] = None
            width: Optional[int] = None
            height: Optional[int] = None

            try:
                if isinstance(image_input, str):  # It's a URL
                    logger.debug(f"Processing image URL: {image_input[:50]}...")
                    image_url_data["url"] = image_input
                    # Cannot easily validate size/format/dimensions or calculate cost for URLs
                    logger.warning(
                        "Cannot validate size/format or calculate token cost for image URLs."
                    )

                elif isinstance(image_input, bytes):  # It's image bytes
                    logger.debug(
                        f"Processing image bytes (size: {len(image_input)})..."
                    )
                    image_bytes = image_input
                    size_mb = len(image_bytes) / (1024 * 1024)
                    if size_mb > MAX_IMAGE_SIZE_MB:
                        logger.error(
                            f"Image {i + 1} size ({size_mb:.2f}MB) exceeds limit ({MAX_IMAGE_SIZE_MB}MB)."
                        )
                        return f"Error: Image {i + 1} size ({size_mb:.2f}MB) exceeds limit ({MAX_IMAGE_SIZE_MB}MB)."
                    total_image_bytes += len(image_bytes)

                    # Use Pillow to validate format and get dimensions
                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        image_format = img.format
                        width, height = img.size
                        img.verify()  # Verify integrity
                        # Re-open after verify
                        img = Image.open(io.BytesIO(image_bytes))
                        width, height = img.size  # Get dimensions again

                        if image_format not in SUPPORTED_IMAGE_FORMATS:
                            logger.error(
                                f"Unsupported image format '{image_format}' for image {i + 1}."
                            )
                            return f"Error: Unsupported image format '{image_format}'. Supported formats: {SUPPORTED_IMAGE_FORMATS}."

                        logger.debug(
                            f"Image {i + 1}: Format={image_format}, Dimensions={width}x{height}"
                        )

                        # Calculate cost only if dimensions are available
                        if width and height and "gpt-4.1" in target_model:
                            total_image_tokens += self._calculate_gpt41_image_cost(
                                width, height, target_model
                            )

                    except (IOError, SyntaxError) as img_err:
                        logger.error(
                            f"Invalid or corrupted image data for image {i + 1}: {img_err}"
                        )
                        return f"Error: Invalid or corrupted image data provided for image {i + 1}."
                    except Exception as pillow_err:
                        logger.error(
                            f"Pillow error processing image {i + 1}: {pillow_err}"
                        )
                        return f"Error: Could not process image data for image {i + 1}."

                    # Encode to Base64 Data URL
                    mime_type = Image.MIME.get(image_format)
                    if not mime_type:
                        logger.warning(
                            f"Could not determine MIME type for format {image_format}. Defaulting to image/jpeg."
                        )
                        mime_type = "image/jpeg"
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    image_url_data["url"] = f"data:{mime_type};base64,{base64_image}"

                else:
                    logger.error(
                        f"Invalid image input type for image {i + 1}: {type(image_input)}"
                    )
                    return f"Error: Invalid image input type for image {i + 1}. Must be URL (str) or bytes."

                content_list.append({"type": "image_url", "image_url": image_url_data})

            except Exception as proc_err:
                logger.error(
                    f"Error processing image {i + 1}: {proc_err}", exc_info=True
                )
                return f"Error: Failed to process image {i + 1}."

        total_size_mb = total_image_bytes / (1024 * 1024)
        if total_size_mb > MAX_TOTAL_IMAGE_SIZE_MB:
            logger.error(
                f"Total image size ({total_size_mb:.2f}MB) exceeds limit ({MAX_TOTAL_IMAGE_SIZE_MB}MB)."
            )
            return f"Error: Total image size ({total_size_mb:.2f}MB) exceeds limit ({MAX_TOTAL_IMAGE_SIZE_MB}MB)."

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content_list})

        request_params = {
            "messages": messages,
            "model": target_model,
            # "max_tokens": 300 # Optional: Add max_tokens if needed
        }

        if self.logfire:
            logfire.instrument_openai(self.client)

        logger.info(
            f"Sending request to '{target_model}' with {len(images)} images. Total calculated image tokens (approx): {total_image_tokens}"
        )

        try:
            response = await self.client.chat.completions.create(**request_params)
            if response.choices and response.choices[0].message.content:
                # Log actual usage if available
                if response.usage:
                    logger.info(
                        f"OpenAI API Usage: Prompt={response.usage.prompt_tokens}, Completion={response.usage.completion_tokens}, Total={response.usage.total_tokens}"
                    )
                return response.choices[0].message.content
            else:
                logger.warning("Received vision response with no content.")
                return ""
        except OpenAIError as e:  # Catch specific OpenAI errors
            logger.error(f"OpenAI API error during vision request: {e}")
            return f"I apologize, but I encountered an API error: {e}"
        except Exception as e:
            logger.exception(f"Error in generate_text_with_images: {e}")
            return f"I apologize, but I encountered an unexpected error: {e}"

    async def parse_structured_output(
        self,
        prompt: str,
        system_prompt: str,
        model_class: Type[T],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
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

            create_args = {
                "model": current_parse_model,
                "messages": messages,
                "response_model": model_class,
                "max_retries": 2,  # Automatically retry on validation errors
            }
            if tools:
                create_args["tools"] = tools

            response = await patched_client.chat.completions.create(**create_args)
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

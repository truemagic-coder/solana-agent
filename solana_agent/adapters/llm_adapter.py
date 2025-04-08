"""
LLM provider adapters for the Solana Agent system.

These adapters implement the LLMProvider interface for different LLM services.
"""
import asyncio
import json
from typing import Any, AsyncGenerator, Callable, Dict, Literal, Optional, Type, TypeVar

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
import websockets

from solana_agent.interfaces.providers.llm import LLMProvider

T = TypeVar('T', bound=BaseModel)


class OpenAIAdapter(LLMProvider):
    """OpenAI implementation of LLMProvider with web search capabilities."""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.parse_model = "gpt-4o-mini"
        self.text_model = "gpt-4o-mini"
        self.transcription_model = "gpt-4o-mini-transcribe"
        self.tts_model = "gpt-4o-mini-tts"

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
            instructions: Optional instructions for speech generation
            voice: Voice to use for synthesis
            response_format: Audio format

        Yields:
            Audio bytes as they become available
        """
        try:
            async with self.client.audio.speech.with_streaming_response.create(
                model=self.tts_model,
                voice=voice,
                instructions=instructions,
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
            response = await self.client.chat.completions.create(**request_params)

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
    ) -> T:  # pragma: no cover
        """Generate structured output using Pydantic model parsing."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            # First try the beta parsing API
            completion = await self.client.beta.chat.completions.parse(
                model=self.parse_model,
                messages=messages,
                response_format=model_class,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"Error with beta.parse method: {e}")

    async def create_realtime_session(
        self,
        model: str = "gpt-4o-mini-realtime-preview",
        modalities: list = ["audio", "text"],
        instructions: str = "You are a helpful assistant.",
        voice: str = "alloy",
        input_audio_format: str = "pcm16",
        output_audio_format: str = "pcm16",
    ) -> Dict[str, Any]:  # pragma: no cover
        """Create a realtime session token for WebSocket communication."""
        try:
            # Get the API key from the AsyncOpenAI client
            api_key = self.client.api_key

            # Create an async HTTP client
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/realtime/sessions",
                    json={
                        "model": model,
                        "modalities": modalities,
                        "instructions": instructions,
                        "voice": voice,
                        "input_audio_format": input_audio_format,
                        "output_audio_format": output_audio_format,
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "OpenAI-Beta": "realtime=v1"
                    }
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(
                        f"Failed to create realtime session: {response.text}")
        except Exception as e:
            print(f"Error creating realtime session: {str(e)}")
            raise

    async def realtime_audio_transcription(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        transcription_config: Optional[Dict[str, Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> AsyncGenerator[str, None]:  # pragma: no cover
        """Stream real-time audio transcription using the Realtime API.

        Args:
            audio_generator: Async generator that yields audio chunks
            transcription_config: Optional custom configuration for transcription
            on_event: Optional callback function for handling raw events

        Yields:
            Transcription text as it becomes available
        """
        # Create default transcription config if none provided
        if transcription_config is None:
            transcription_config = {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "gpt-4o-mini-transcribe"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                }
            }

        try:
            # Get the API key from the AsyncOpenAI client
            api_key = self.client.api_key

            # Create transcription session
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/realtime/transcription_sessions",
                    json=transcription_config,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "OpenAI-Beta": "realtime=v1"
                    }
                )

                if response.status_code != 200:
                    raise Exception(
                        f"Failed to create transcription session: {response.text}")

                session = response.json()
                client_secret = session["client_secret"]["value"]

            # Connect to WebSocket with proper headers as dictionary
            url = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-transcribe"
            headers = {
                "Authorization": f"Bearer {client_secret}",
                "OpenAI-Beta": "realtime=v1"
            }

            async with websockets.connect(url, additional_headers=headers) as websocket:
                # Handle WebSocket communication in the background
                audio_task = None

                async def send_audio():
                    try:
                        async for audio_chunk in audio_generator:
                            # Base64 encode the audio
                            import base64
                            encoded_audio = base64.b64encode(
                                audio_chunk).decode('utf-8')

                            # Send audio chunk
                            await websocket.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": encoded_audio
                            }))

                            # Small delay to prevent flooding
                            await asyncio.sleep(0.05)

                        # Commit the audio buffer when done
                        await websocket.send(json.dumps({
                            "type": "input_audio_buffer.commit"
                        }))
                    except Exception as e:
                        print(f"Error sending audio: {str(e)}")

                # Start sending audio in the background
                audio_task = asyncio.create_task(send_audio())

                # Process transcription events
                try:
                    while True:
                        message = await websocket.recv()
                        event = json.loads(message)

                        if on_event:
                            # Check if on_event is a coroutine function and await it if needed
                            if asyncio.iscoroutinefunction(on_event):
                                await on_event(event)
                            else:
                                on_event(event)

                        # Extract transcription deltas
                        if event["type"] == "conversation.item.input_audio_transcription.delta":
                            yield event["delta"]

                        # Also handle completed transcriptions
                        elif event["type"] == "conversation.item.input_audio_transcription.completed":
                            yield event["transcript"]
                            break
                finally:
                    # Clean up audio task if it's still running
                    if audio_task and not audio_task.done():
                        audio_task.cancel()
                        try:
                            await audio_task
                        except asyncio.CancelledError:
                            pass

        except Exception as e:
            print(f"Error in realtime audio transcription: {str(e)}")
            import traceback
            print(traceback.format_exc())
            yield f"I apologize, but I encountered an error transcribing the audio: {str(e)}"

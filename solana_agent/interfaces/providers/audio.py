from __future__ import annotations

from abc import ABC, abstractmethod


class AudioTranscoder(ABC):
    """Abstract audio transcoder for converting between compressed audio and PCM16.

    Implementations may rely on external tools (e.g., ffmpeg).
    """

    @abstractmethod
    async def to_pcm16(
        self, audio_bytes: bytes, input_mime: str, rate_hz: int
    ) -> bytes:
        """Transcode arbitrary audio to mono PCM16LE at rate_hz.

        Args:
            audio_bytes: Source audio bytes (e.g., MP4/AAC)
            input_mime: Source mime-type (e.g., 'audio/mp4', 'audio/aac')
            rate_hz: Target sample rate
        Returns:
            Raw PCM16LE mono bytes at the given rate
        """
        raise NotImplementedError

    @abstractmethod
    async def from_pcm16(
        self, pcm16_bytes: bytes, output_mime: str, rate_hz: int
    ) -> bytes:
        """Transcode mono PCM16LE at rate_hz to the desired output mime.

        Args:
            pcm16_bytes: Raw PCM16LE bytes
            output_mime: Target mime-type (e.g., 'audio/aac')
            rate_hz: Sample rate of the PCM
        Returns:
            Encoded audio bytes
        """
        raise NotImplementedError

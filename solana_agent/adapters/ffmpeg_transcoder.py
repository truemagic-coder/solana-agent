from __future__ import annotations

import asyncio
from typing import List

from solana_agent.interfaces.providers.audio import AudioTranscoder


class FFmpegTranscoder(AudioTranscoder):
    """FFmpeg-based transcoder. Requires 'ffmpeg' binary in PATH.

    This uses subprocess to stream bytes through ffmpeg for encode/decode.
    """

    async def _run_ffmpeg(self, args: List[str], data: bytes) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate(input=data)
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg failed to transcode audio")
        return stdout

    async def to_pcm16(
        self, audio_bytes: bytes, input_mime: str, rate_hz: int
    ) -> bytes:
        """Decode compressed audio to mono PCM16LE at rate_hz."""
        # Prefer to hint format for MP4/AAC; ffmpeg can still autodetect if hint is wrong.
        hinted_format = None
        if input_mime in ("audio/mp4", "audio/aac", "audio/m4a"):
            hinted_format = "mp4"
        elif input_mime in ("audio/ogg", "audio/webm"):
            hinted_format = None  # container detection is decent here
        elif input_mime in ("audio/wav", "audio/x-wav"):
            hinted_format = "wav"

        args = [
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if hinted_format:
            args += ["-f", hinted_format]
        args += [
            "-i",
            "pipe:0",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(rate_hz),
            "-f",
            "s16le",
            "pipe:1",
        ]
        return await self._run_ffmpeg(args, audio_bytes)

    async def from_pcm16(
        self, pcm16_bytes: bytes, output_mime: str, rate_hz: int
    ) -> bytes:
        """Encode PCM16LE to desired format (currently AAC ADTS for mobile streaming)."""
        if output_mime in ("audio/aac", "audio/mp4", "audio/m4a"):
            # Encode to AAC in ADTS stream; clients can play it as AAC.
            args = [
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                str(rate_hz),
                "-i",
                "pipe:0",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-f",
                "adts",
                "pipe:1",
            ]
            return await self._run_ffmpeg(args, pcm16_bytes)
        # Default: passthrough
        return pcm16_bytes

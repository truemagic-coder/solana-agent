from __future__ import annotations

import asyncio
import logging
from typing import List

from solana_agent.interfaces.providers.audio import AudioTranscoder

logger = logging.getLogger(__name__)


class FFmpegTranscoder(AudioTranscoder):
    """FFmpeg-based transcoder. Requires 'ffmpeg' binary in PATH.

    This uses subprocess to stream bytes through ffmpeg for encode/decode.
    """

    async def _run_ffmpeg(
        self, args: List[str], data: bytes
    ) -> bytes:  # pragma: no cover
        logger.info("FFmpeg: starting process args=%s, input_len=%d", args, len(data))
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=data)
        if proc.returncode != 0:
            err = (stderr or b"").decode("utf-8", errors="ignore")
            logger.error("FFmpeg failed (code=%s): %s", proc.returncode, err[:2000])
            raise RuntimeError("ffmpeg failed to transcode audio")
        logger.info("FFmpeg: finished successfully, output_len=%d", len(stdout or b""))
        if stderr:
            logger.debug(
                "FFmpeg stderr: %s", stderr.decode("utf-8", errors="ignore")[:2000]
            )
        return stdout

    async def to_pcm16(  # pragma: no cover
        self, audio_bytes: bytes, input_mime: str, rate_hz: int
    ) -> bytes:
        """Decode compressed audio to mono PCM16LE at rate_hz."""
        logger.info(
            "Transcode to PCM16: input_mime=%s, rate_hz=%d, input_len=%d",
            input_mime,
            rate_hz,
            len(audio_bytes),
        )
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
        out = await self._run_ffmpeg(args, audio_bytes)
        logger.info("Transcoded to PCM16: output_len=%d", len(out))
        return out

    async def from_pcm16(  # pragma: no cover
        self, pcm16_bytes: bytes, output_mime: str, rate_hz: int
    ) -> bytes:
        """Encode PCM16LE to desired format (currently AAC ADTS for mobile streaming)."""
        logger.info(
            "Encode from PCM16: output_mime=%s, rate_hz=%d, input_len=%d",
            output_mime,
            rate_hz,
            len(pcm16_bytes),
        )
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
            out = await self._run_ffmpeg(args, pcm16_bytes)
            logger.info(
                "Encoded from PCM16 to %s: output_len=%d", output_mime, len(out)
            )
            return out
        # Default: passthrough
        logger.info("Encode passthrough (no change), output_len=%d", len(pcm16_bytes))
        return pcm16_bytes

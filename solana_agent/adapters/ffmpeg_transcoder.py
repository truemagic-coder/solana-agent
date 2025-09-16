from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import List, AsyncGenerator
import tempfile
import os

from solana_agent.interfaces.providers.audio import AudioTranscoder

logger = logging.getLogger(__name__)


class FFmpegTranscoder(AudioTranscoder):
    """FFmpeg-based transcoder. Requires 'ffmpeg' binary in PATH.

    This uses subprocess to stream bytes through ffmpeg for encode/decode.
    """

    async def _run_ffmpeg(
        self, args: List[str], data: bytes
    ) -> bytes:  # pragma: no cover
        logger.debug("FFmpeg: starting process args=%s, input_len=%d", args, len(data))
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
        logger.debug("FFmpeg: finished successfully, output_len=%d", len(stdout or b""))
        if stderr:
            logger.debug(
                "FFmpeg stderr: %s", stderr.decode("utf-8", errors="ignore")[:2000]
            )
        return stdout

    async def to_pcm16(  # pragma: no cover
        self, audio_bytes: bytes, input_mime: str, rate_hz: int
    ) -> bytes:
        """Decode compressed audio to mono PCM16LE at rate_hz."""
        logger.debug(
            "Transcode to PCM16: input_mime=%s, rate_hz=%d, input_len=%d",
            input_mime,
            rate_hz,
            len(audio_bytes),
        )
        # Fast-path: raw PCM payloads (already PCM16LE mono). Some callers may still
        # set accept_compressed_input=True with client_input_mime=audio/pcm; in that
        # case ffmpeg autodetection would fail (no container). Provide explicit demux args.
        try:
            im = (input_mime or "").lower()
        except Exception:
            im = input_mime
        if im in {"audio/pcm", "audio/pcm16", "audio/x-pcm", "pcm16"}:
            # If the requested output rate matches, just return bytes (identity).
            # Otherwise perform a raw resample.
            # Use resample_pcm16 for consistency.
            logger.debug("FFmpegTranscoder.to_pcm16: raw PCM fast-path (mime=%s)", im)
            if rate_hz in (0, None):  # Defensive: ensure positive rate
                rate_hz = 24000
            # We don't know the input sample rate explicitly here; assume caller already
            # provides audio at the desired rate. (If not, higher-level logic should invoke resample.)
            return audio_bytes
        # iOS-recorded MP4/M4A often requires a seekable input for reliable demuxing.
        # Decode from a temporary file instead of stdin for MP4/M4A.
        if input_mime in ("audio/mp4", "audio/m4a"):
            suffix = ".m4a" if input_mime == "audio/m4a" else ".mp4"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                    tmp_path = tf.name
                    tf.write(audio_bytes)
                args = [
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    tmp_path,
                    "-vn",  # ignore any video tracks
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
                out = await self._run_ffmpeg(args, b"")
                logger.debug(
                    "Transcoded (MP4/M4A temp-file) to PCM16: output_len=%d", len(out)
                )
                return out
            finally:
                if tmp_path:
                    with contextlib.suppress(Exception):
                        os.remove(tmp_path)

        # For other formats, prefer a format hint when helpful and decode from stdin.
        hinted_format = None
        if input_mime in ("audio/aac",):
            # Raw AAC is typically in ADTS stream format
            hinted_format = "adts"
        elif input_mime in ("audio/ogg", "audio/webm"):
            hinted_format = None  # container detection is decent here
        elif input_mime in ("audio/wav", "audio/x-wav"):
            hinted_format = "wav"
        elif input_mime in ("audio/pcm", "audio/pcm16", "audio/x-pcm"):
            hinted_format = "s16le"  # Raw PCM requires explicit format

        args = [
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if hinted_format:
            # For raw PCM we also need channels/sample rate BEFORE -i
            if hinted_format == "s16le":
                args += [
                    "-f",
                    "s16le",
                    "-ac",
                    "1",
                    "-ar",
                    str(rate_hz),
                ]
            else:
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
        logger.debug("Transcoded to PCM16: output_len=%d", len(out))
        return out

    async def from_pcm16(  # pragma: no cover
        self, pcm16_bytes: bytes, output_mime: str, rate_hz: int
    ) -> bytes:
        """Encode PCM16LE to desired format (AAC ADTS, fragmented MP4, or MP3)."""
        logger.debug(
            "Encode from PCM16: output_mime=%s, rate_hz=%d, input_len=%d",
            output_mime,
            rate_hz,
            len(pcm16_bytes),
        )

        if output_mime in ("audio/mpeg", "audio/mp3"):
            # Encode to MP3 (often better streaming compatibility on mobile)
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
                "libmp3lame",
                "-b:a",
                "128k",
                "-f",
                "mp3",
                "pipe:1",
            ]
            out = await self._run_ffmpeg(args, pcm16_bytes)
            logger.debug(
                "Encoded from PCM16 to %s: output_len=%d", output_mime, len(out)
            )
            return out

        if output_mime in ("audio/aac",):
            # Encode to AAC in ADTS stream; good for streaming over sockets/HTTP chunked
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
            logger.debug(
                "Encoded from PCM16 to %s: output_len=%d", output_mime, len(out)
            )
            return out

        if output_mime in ("audio/mp4", "audio/m4a"):
            # Encode to fragmented MP4 (fMP4) with AAC for better iOS compatibility
            # For streaming, write an initial moov and fragment over stdout.
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
                "-movflags",
                "+frag_keyframe+empty_moov",
                "-f",
                "mp4",
                "pipe:1",
            ]
            out = await self._run_ffmpeg(args, pcm16_bytes)
            logger.debug(
                "Encoded from PCM16 to %s (fMP4): output_len=%d", output_mime, len(out)
            )
            return out

        # Default: passthrough
        logger.debug("Encode passthrough (no change), output_len=%d", len(pcm16_bytes))
        return pcm16_bytes

    async def resample_pcm16(
        self,
        pcm16_bytes: bytes,
        in_rate_hz: int,
        out_rate_hz: int,
        output_container: str = "raw",
    ) -> bytes:  # pragma: no cover
        """Resample mono PCM16LE from in_rate_hz to out_rate_hz.

        Args:
            pcm16_bytes: Raw little-endian PCM16 mono audio.
            in_rate_hz: Input sample rate.
            out_rate_hz: Desired output sample rate.
            output_container: "raw" for s16le stream, "wav" for RIFF WAV.
        """
        if in_rate_hz == out_rate_hz and output_container == "raw":
            return pcm16_bytes
        fmt = "s16le"
        args = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            fmt,
            "-ac",
            "1",
            "-ar",
            str(in_rate_hz),
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            str(out_rate_hz),
        ]
        if output_container == "wav":
            args += [
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                "pipe:1",
            ]
        else:  # raw
            args += [
                "-f",
                "s16le",
                "pipe:1",
            ]
        return await self._run_ffmpeg(args, pcm16_bytes)

    async def stream_resample_pcm16(
        self,
        pcm_iter: AsyncGenerator[bytes, None],
        in_rate_hz: int,
        out_rate_hz: int,
        output_container: str = "wav",
        read_chunk_size: int = 4096,
    ) -> AsyncGenerator[bytes, None]:  # pragma: no cover
        """Streaming resample for PCM16 mono.

        Launches a single ffmpeg process to convert sample rate on-the-fly.
        If output_container == "wav" a single RIFF header is emitted at start.
        """
        if in_rate_hz == out_rate_hz and output_container == "raw":
            # Passthrough
            async for c in pcm_iter:
                yield c
            return
        args = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(in_rate_hz),
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            str(out_rate_hz),
        ]
        if output_container == "wav":
            args += ["-c:a", "pcm_s16le", "-f", "wav", "pipe:1"]
        else:
            args += ["-f", "s16le", "pipe:1"]
        logger.debug("FFmpeg(resample stream): starting args=%s", args)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin and proc.stdout

        async def _writer():
            try:
                async for p in pcm_iter:
                    if not p:
                        continue
                    proc.stdin.write(p)
                    await proc.stdin.drain()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug("FFmpeg(resample stream) writer error: %s", e)
            finally:
                with contextlib.suppress(Exception):
                    proc.stdin.close()

        writer_task = asyncio.create_task(_writer())
        try:
            while True:
                data = await proc.stdout.read(read_chunk_size)
                if not data:
                    break
                yield data
        finally:
            if not writer_task.done():
                with contextlib.suppress(Exception):
                    writer_task.cancel()
                try:
                    await writer_task
                except Exception:
                    pass
            with contextlib.suppress(Exception):
                stderr = await proc.stderr.read() if proc.stderr else b""
                code = await proc.wait()
                if code != 0:
                    err = (stderr or b"").decode("utf-8", errors="ignore")
                    logger.error(
                        "FFmpeg(resample stream) failed (code=%s): %s", code, err[:1000]
                    )

    async def stream_from_pcm16(  # pragma: no cover
        self,
        pcm_iter: AsyncGenerator[bytes, None],
        output_mime: str,
        rate_hz: int,
        read_chunk_size: int = 4096,
    ) -> AsyncGenerator[bytes, None]:
        """Start a single continuous encoder and stream encoded audio chunks.

        - Launches one ffmpeg subprocess for the entire response.
        - Feeds PCM16LE mono bytes from pcm_iter into stdin.
        - Yields encoded bytes from stdout as they become available.
        """
        if output_mime in ("audio/mpeg", "audio/mp3"):
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
                "libmp3lame",
                "-b:a",
                "128k",
                "-f",
                "mp3",
                "pipe:1",
            ]
        elif output_mime in ("audio/aac",):
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
        elif output_mime in ("audio/mp4", "audio/m4a"):
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
                "-movflags",
                "+frag_keyframe+empty_moov",
                "-f",
                "mp4",
                "pipe:1",
            ]
        else:
            # Passthrough streaming: just yield input
            async for chunk in pcm_iter:
                yield chunk
            return

        logger.debug("FFmpeg(stream): starting args=%s", args)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert proc.stdin is not None and proc.stdout is not None

        async def _writer():
            try:
                async for pcm in pcm_iter:
                    if not pcm:
                        continue
                    proc.stdin.write(pcm)
                    # Backpressure
                    await proc.stdin.drain()
            except asyncio.CancelledError:
                # Swallow cancellation; stdin will be closed below.
                pass
            except Exception as e:
                logger.debug("FFmpeg(stream) writer error: %s", str(e))
            finally:
                with contextlib.suppress(Exception):
                    proc.stdin.close()

        writer_task = asyncio.create_task(_writer())

        buf = bytearray()
        try:
            while True:
                data = await proc.stdout.read(read_chunk_size)
                if not data:
                    break
                buf.extend(data)
                # Emit fixed-size chunks even if read returns a larger blob
                while len(buf) >= read_chunk_size:
                    yield bytes(buf[:read_chunk_size])
                    del buf[:read_chunk_size]
            # Flush any remainder
            if buf:
                yield bytes(buf)
        finally:
            # Ensure writer is done
            if not writer_task.done():
                with contextlib.suppress(Exception):
                    writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            # Drain remaining stderr and check return code
            try:
                stderr = await proc.stderr.read() if proc.stderr else b""
                code = await proc.wait()
                if code != 0:
                    err = (stderr or b"").decode("utf-8", errors="ignore")
                    logger.error(
                        "FFmpeg(stream) failed (code=%s): %s", code, err[:2000]
                    )
            except Exception:
                pass

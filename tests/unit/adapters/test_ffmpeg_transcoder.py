import asyncio
import types
import pytest

from solana_agent.adapters.ffmpeg_transcoder import FFmpegTranscoder


class FakeProc:
    def __init__(self, stdout=b"", stderr=b"", rc=0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = rc
        self.stdin = types.SimpleNamespace(
            write=lambda b: None, drain=lambda: asyncio.sleep(0), close=lambda: None
        )
        self.stdout = types.SimpleNamespace(read=lambda n=-1: self._read_all())
        self.stderr = types.SimpleNamespace(read=lambda n=-1: self._read_err())

    async def communicate(self, input=b""):
        return self._stdout, self._stderr

    async def wait(self):
        return self.returncode

    async def _read_all(self):
        out = self._stdout
        self._stdout = b""
        return out

    async def _read_err(self):
        err = self._stderr
        self._stderr = b""
        return err


@pytest.mark.asyncio
async def test_to_pcm16_success(monkeypatch):
    tx = FFmpegTranscoder()

    async def fake_create_subprocess_exec(*cmd, **kw):
        # Return PCM s16le as stdout
        return FakeProc(stdout=b"\x00\x01\x02\x03", stderr=b"", rc=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    out = await tx.to_pcm16(b"data", "audio/mp4", 24000)
    assert out == b"\x00\x01\x02\x03"


@pytest.mark.asyncio
async def test_from_pcm16_encode_mp3(monkeypatch):
    tx = FFmpegTranscoder()

    async def fake_create_subprocess_exec(*cmd, **kw):
        return FakeProc(stdout=b"MP3DATA", stderr=b"", rc=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    out = await tx.from_pcm16(b"PCM", "audio/mpeg", 24000)
    assert out == b"MP3DATA"


@pytest.mark.asyncio
async def test_stream_from_pcm16_aac(monkeypatch):
    tx = FFmpegTranscoder()

    async def fake_create_subprocess_exec(*cmd, **kw):
        # Simulate streaming encoder that outputs in two reads
        return FakeProc(stdout=b"AAC1AAC2", stderr=b"", rc=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def pcm_iter():
        yield b"abcd"
        yield b"efgh"

    out = []
    async for chunk in tx.stream_from_pcm16(
        pcm_iter(), "audio/aac", 24000, read_chunk_size=4
    ):
        out.append(chunk)

    assert out == [b"AAC1", b"AAC2"]

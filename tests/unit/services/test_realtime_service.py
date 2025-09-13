import pytest

from solana_agent.services.realtime import RealtimeService


class FakeSession:
    def __init__(self):
        self.appended = []
        self._pending_tool_checks = 0
        self._pending_tool_limit = 0

    # Session API used by RealtimeService
    async def connect(self):
        return

    async def close(self):
        return

    async def update_session(self, patch):
        self._last_patch = patch

    async def append_audio(self, pcm):
        self.appended.append(pcm)

    async def commit_input(self):
        return

    async def clear_input(self):
        return

    async def create_response(self, rp=None):
        return

    # Streams
    def iter_events(self):
        async def _gen():
            if False:
                yield {}

        return _gen()

    def iter_output_audio(self):
        # Default: no audio
        async def _gen():
            if False:
                yield b""

        return _gen()

    def iter_input_transcript(self):
        async def _gen():
            if False:
                yield ""

        return _gen()

    def iter_output_transcript(self):
        async def _gen():
            if False:
                yield ""

        return _gen()

    # Optional hook queried by the service
    def has_pending_tool_call(self):
        if self._pending_tool_checks < self._pending_tool_limit:
            self._pending_tool_checks += 1
            return True
        return False


class FakeTranscoder:
    def __init__(self, to_pcm_out=b"PCM", encode_prefix=b"E:"):
        self.to_pcm_calls = []
        self.stream_calls = []
        self._to_pcm_out = to_pcm_out
        self._prefix = encode_prefix

    async def to_pcm16(self, data: bytes, input_mime: str, rate_hz: int) -> bytes:
        self.to_pcm_calls.append((len(data), input_mime, rate_hz))
        return self._to_pcm_out

    async def stream_from_pcm16(self, pcm_iter, output_mime: str, rate_hz: int):
        self.stream_calls.append((output_mime, rate_hz))
        async for chunk in pcm_iter:
            if not chunk:
                continue
            yield self._prefix + chunk


@pytest.mark.asyncio
async def test_append_audio_passthrough():
    sess = FakeSession()
    svc = RealtimeService(session=sess, accept_compressed_input=False)

    data = b"\x01\x02\x03\x04"
    await svc.append_audio(data)

    assert sess.appended == [data]


@pytest.mark.asyncio
async def test_append_audio_with_transcode():
    sess = FakeSession()
    transcoder = FakeTranscoder(to_pcm_out=b"PCM16")
    svc = RealtimeService(
        session=sess,
        transcoder=transcoder,
        accept_compressed_input=True,
        client_input_mime="audio/mp4",
    )

    src = b"compressed-audio"
    await svc.append_audio(src)

    # Transcoder used and PCM forwarded to session
    assert transcoder.to_pcm_calls and transcoder.to_pcm_calls[0][0] == len(src)
    assert sess.appended == [b"PCM16"]


@pytest.mark.asyncio
async def test_iter_output_audio_encoded_passthrough_waits_on_pending_tool(monkeypatch):
    sess = FakeSession()
    # First, have pending tool for a few checks, with no PCM produced
    sess._pending_tool_limit = 3

    # Then provide some PCM chunks once pending clears
    async def pcm_gen():
        # Simulate no chunks while pending; once service retries, yield chunks
        # Yield two small PCM chunks and then end
        yield b"A" * 4
        yield b"B" * 4

    # Swap iter_output_audio to our generator after a short delay
    def make_iter_output_audio():
        called = {"n": 0}

        def _iter():
            async def _gen():
                called["n"] += 1
                # First few attempts: end immediately (no PCM)
                if called["n"] <= 2:
                    if False:
                        yield b""
                    return
                # Afterwards, produce chunks
                async for c in pcm_gen():
                    yield c

            return _gen()

        return _iter

    sess.iter_output_audio = make_iter_output_audio()

    svc = RealtimeService(session=sess, encode_output=False)

    out = bytearray()
    async for chunk in svc.iter_output_audio_encoded():
        if hasattr(chunk, "audio_data") and chunk.audio_data:
            out.extend(chunk.audio_data)
        else:
            # Fallback for raw bytes (backward compatibility)
            out.extend(chunk)
        if len(out) >= 8:
            break

    assert bytes(out) == (b"A" * 4 + b"B" * 4)


@pytest.mark.asyncio
async def test_iter_output_audio_encoded_with_encoding():
    sess = FakeSession()

    async def pcm_gen():
        yield b"1234"
        yield b"5678"

    def make_iter_output_audio():
        def _iter():
            async def _gen():
                async for c in pcm_gen():
                    yield c

            return _gen()

        return _iter

    sess.iter_output_audio = make_iter_output_audio()

    transcoder = FakeTranscoder(encode_prefix=b"E:")
    svc = RealtimeService(session=sess, transcoder=transcoder, encode_output=True)

    chunks = []
    async for c in svc.iter_output_audio_encoded():
        if hasattr(c, "data"):
            chunks.append(c.data)
        else:
            # Fallback for raw bytes (backward compatibility)
            chunks.append(c)
        if len(chunks) >= 2:
            break

    assert chunks == [b"E:1234", b"E:5678"]

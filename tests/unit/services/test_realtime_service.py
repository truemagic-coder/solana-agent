import pytest
from unittest.mock import Mock, AsyncMock

from solana_agent.services.realtime import RealtimeService
from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions


class FakeSession:
    def __init__(self):
        self.appended = []
        self._pending_tool_checks = 0
        self._pending_tool_limit = 0
        self._last_patch = None
        self.connect = AsyncMock()
        self.close = AsyncMock()
        self.update_session = AsyncMock()
        self.update_session.side_effect = self._store_patch
        self.append_audio = AsyncMock()
        self.commit_input = AsyncMock()
        self.clear_input = AsyncMock()
        self.create_response = AsyncMock()
        self.iter_events = Mock(return_value=self._async_gen([]))
        self.iter_output_audio = Mock(
            return_value=self._async_gen([b"test1", b"test2"])
        )
        self.iter_input_transcript = Mock(return_value=self._async_gen(["test"]))
        self.iter_output_transcript = Mock(return_value=self._async_gen(["test"]))

    def _store_patch(self, patch):
        self._last_patch = patch

    def _async_gen(self, items):
        async def _gen():
            for item in items:
                yield item

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

    sess.append_audio.assert_called_once_with(data)


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
    sess.append_audio.assert_called_once_with(b"PCM16")


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


@pytest.mark.asyncio
async def test_start_and_stop():
    """Test connection start and stop methods."""
    sess = FakeSession()
    svc = RealtimeService(session=sess)

    # Test start
    await svc.start()
    assert svc._connected is True
    sess.connect.assert_called_once()

    # Test stop
    await svc.stop()
    assert svc._connected is False
    sess.close.assert_called_once()


@pytest.mark.asyncio
async def test_configure():
    """Test session configuration method."""
    sess = FakeSession()
    svc = RealtimeService(session=sess)

    # Test configure with various parameters
    await svc.configure(
        voice="alloy",
        vad_enabled=True,
        instructions="Test instructions",
        input_rate_hz=16000,
        output_rate_hz=16000,
        input_mime="audio/pcm",
        output_mime="audio/pcm",
        tools=[{"type": "function", "function": {"name": "test"}}],
        tool_choice="auto",
    )

    # Verify session update was called
    sess.update_session.assert_called_once()
    patch = sess._last_patch

    # Verify patch contains expected fields
    assert "audio" in patch
    assert "instructions" in patch
    assert "tools" in patch
    assert "tool_choice" in patch

    # Verify local options were updated
    assert svc._options.voice == "alloy"
    assert svc._options.vad_enabled is True
    assert svc._options.instructions == "Test instructions"
    assert svc._options.input_rate_hz == 16000
    assert svc._options.output_rate_hz == 16000
    assert svc._options.input_mime == "audio/pcm"
    assert svc._options.output_mime == "audio/pcm"
    assert svc._options.tools == [{"type": "function", "function": {"name": "test"}}]
    assert svc._options.tool_choice == "auto"


@pytest.mark.asyncio
async def test_commit_input():
    """Test input commit method."""
    sess = FakeSession()
    svc = RealtimeService(session=sess)

    await svc.commit_input()
    sess.commit_input.assert_called_once()


@pytest.mark.asyncio
async def test_clear_input():
    """Test input clear method."""
    sess = FakeSession()
    svc = RealtimeService(session=sess)

    await svc.clear_input()
    sess.clear_input.assert_called_once()


@pytest.mark.asyncio
async def test_create_response():
    """Test response creation method."""
    sess = FakeSession()
    svc = RealtimeService(session=sess)

    response_patch = {"modalities": ["audio"]}
    await svc.create_response(response_patch)
    sess.create_response.assert_called_once_with(response_patch)


@pytest.mark.asyncio
async def test_iter_events():
    """Test events iterator."""
    sess = FakeSession()
    svc = RealtimeService(session=sess)

    # Test that iter_events returns the session's iterator
    events_iter = svc.iter_events()
    assert events_iter == sess.iter_events.return_value


@pytest.mark.asyncio
async def test_iter_output_audio():
    """Test output audio iterator."""
    sess = FakeSession()
    svc = RealtimeService(session=sess)

    # Test that iter_output_audio returns the session's iterator
    audio_iter = svc.iter_output_audio()
    assert audio_iter == sess.iter_output_audio.return_value


@pytest.mark.asyncio
async def test_iter_input_transcript():
    """Test input transcript iterator."""
    sess = FakeSession()
    svc = RealtimeService(session=sess)

    # Test that iter_input_transcript returns the session's iterator
    transcript_iter = svc.iter_input_transcript()
    assert transcript_iter == sess.iter_input_transcript.return_value


@pytest.mark.asyncio
async def test_iter_output_transcript():
    """Test output transcript iterator."""
    sess = FakeSession()
    svc = RealtimeService(session=sess)

    # Test that iter_output_transcript returns the session's iterator
    transcript_iter = svc.iter_output_transcript()
    assert transcript_iter == sess.iter_output_transcript.return_value


@pytest.mark.asyncio
async def test_reset_output_stream():
    """Test output stream reset method."""
    sess = FakeSession()
    svc = RealtimeService(session=sess)

    # Test reset with session that has reset_output_stream
    sess.reset_output_stream = Mock()
    svc.reset_output_stream()
    sess.reset_output_stream.assert_called_once()

    # Test reset with session that doesn't have reset_output_stream
    delattr(sess, "reset_output_stream")
    svc.reset_output_stream()
    # Should not raise an exception


@pytest.mark.asyncio
async def test_iter_output_combined_audio_only():
    """Test combined iterator with audio-only modalities."""
    sess = FakeSession()
    options = RealtimeSessionOptions(output_modalities=["audio"])
    svc = RealtimeService(session=sess, options=options)

    chunks = []
    async for chunk in svc.iter_output_combined():
        chunks.append(chunk)
        if len(chunks) >= 2:
            break

    # Should get RealtimeChunk objects with audio modality
    assert len(chunks) == 2
    for chunk in chunks:
        assert hasattr(chunk, "modality")
        assert chunk.modality == "audio"
        assert hasattr(chunk, "data")


@pytest.mark.asyncio
async def test_iter_output_combined_text_only():
    """Test combined iterator with text-only modalities."""
    sess = FakeSession()
    options = RealtimeSessionOptions(output_modalities=["text"])
    svc = RealtimeService(session=sess, options=options)

    chunks = []
    async for chunk in svc.iter_output_combined():
        chunks.append(chunk)
        if len(chunks) >= 1:
            break

    # Should get RealtimeChunk objects with text modality
    assert len(chunks) == 1
    chunk = chunks[0]
    assert hasattr(chunk, "modality")
    assert chunk.modality == "text"
    assert hasattr(chunk, "data")


@pytest.mark.asyncio
async def test_iter_output_combined_both_modalities():
    """Test combined iterator with both audio and text modalities."""
    sess = FakeSession()
    options = RealtimeSessionOptions(output_modalities=["audio", "text"])
    svc = RealtimeService(session=sess, options=options)

    chunks = []
    async for chunk in svc.iter_output_combined():
        chunks.append(chunk)
        if len(chunks) >= 3:  # Get a few chunks to test both modalities
            break

    # Should get RealtimeChunk objects with both modalities
    assert len(chunks) >= 2
    modalities = {chunk.modality for chunk in chunks}
    assert "audio" in modalities
    assert "text" in modalities


@pytest.mark.asyncio
async def test_iter_output_combined_no_modalities():
    """Test combined iterator with no modalities specified."""
    sess = FakeSession()
    options = RealtimeSessionOptions(output_modalities=[])
    svc = RealtimeService(session=sess, options=options)

    chunks = []
    async for chunk in svc.iter_output_combined():
        chunks.append(chunk)

    # Should get no chunks when no modalities are specified
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_iter_output_combined_default_modalities():
    """Test combined iterator with default modalities (None)."""
    sess = FakeSession()
    svc = RealtimeService(session=sess)  # No options specified

    chunks = []
    async for chunk in svc.iter_output_combined():
        chunks.append(chunk)
        if len(chunks) >= 2:
            break

    # Should get RealtimeChunk objects (default behavior)
    assert len(chunks) == 2
    for chunk in chunks:
        assert hasattr(chunk, "modality")
        assert hasattr(chunk, "data")


# Tests for TwinRealtimeService
@pytest.mark.asyncio
async def test_twin_realtime_service_init():
    """Test TwinRealtimeService initialization."""
    from solana_agent.services.realtime import TwinRealtimeService

    conv_sess = FakeSession()
    trans_sess = FakeSession()

    svc = TwinRealtimeService(conversation=conv_sess, transcription=trans_sess)

    assert svc._conv == conv_sess
    assert svc._trans == trans_sess
    assert svc._connected is False


@pytest.mark.asyncio
async def test_twin_realtime_service_start_stop():
    """Test TwinRealtimeService start and stop methods."""
    from solana_agent.services.realtime import TwinRealtimeService

    conv_sess = FakeSession()
    trans_sess = FakeSession()

    svc = TwinRealtimeService(conversation=conv_sess, transcription=trans_sess)

    # Test start
    await svc.start()
    assert svc._connected is True
    conv_sess.connect.assert_called_once()
    trans_sess.connect.assert_called_once()

    # Test stop
    await svc.stop()
    assert svc._connected is False
    conv_sess.close.assert_called_once()
    trans_sess.close.assert_called_once()


@pytest.mark.asyncio
async def test_twin_realtime_service_configure():
    """Test TwinRealtimeService configure method."""
    from solana_agent.services.realtime import TwinRealtimeService

    conv_sess = FakeSession()
    trans_sess = FakeSession()

    svc = TwinRealtimeService(conversation=conv_sess, transcription=trans_sess)

    await svc.configure(voice="alloy", vad_enabled=True)

    # Verify conversation session was updated (transcription session doesn't need voice/tools)
    conv_sess.update_session.assert_called_once()
    trans_sess.update_session.assert_not_called()


@pytest.mark.asyncio
async def test_twin_realtime_service_iter_output_audio_encoded():
    """Test TwinRealtimeService iter_output_audio_encoded method."""
    from solana_agent.services.realtime import TwinRealtimeService

    conv_sess = FakeSession()
    trans_sess = FakeSession()

    svc = TwinRealtimeService(conversation=conv_sess, transcription=trans_sess)

    chunks = []
    async for chunk in svc.iter_output_audio_encoded():
        chunks.append(chunk)
        if len(chunks) >= 1:
            break

    # Should get RealtimeChunk objects
    assert len(chunks) == 1
    chunk = chunks[0]
    assert hasattr(chunk, "modality")
    assert chunk.modality == "audio"
    assert hasattr(chunk, "data")


@pytest.mark.asyncio
async def test_twin_realtime_service_iter_input_transcript():
    """Test TwinRealtimeService iter_input_transcript method."""
    from solana_agent.services.realtime import TwinRealtimeService

    conv_sess = FakeSession()
    trans_sess = FakeSession()

    svc = TwinRealtimeService(conversation=conv_sess, transcription=trans_sess)

    # Should return transcription session's iterator
    transcript_iter = svc.iter_input_transcript()
    assert transcript_iter == trans_sess.iter_input_transcript.return_value

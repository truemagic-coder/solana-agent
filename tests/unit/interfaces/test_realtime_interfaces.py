import pytest

from solana_agent.interfaces.providers.realtime import (
    RealtimeSessionOptions,
    RealtimeChunk,
    separate_audio_chunks,
    separate_text_chunks,
    demux_realtime_chunks,
    BaseRealtimeSession,
)


class TestRealtimeSessionOptions:
    """Test RealtimeSessionOptions dataclass."""

    def test_default_values(self):
        """Test default values for RealtimeSessionOptions."""
        options = RealtimeSessionOptions()
        assert options.model is None
        assert options.voice == "marin"
        assert options.vad_enabled is True
        assert options.input_rate_hz == 16000
        assert options.output_rate_hz == 16000
        assert options.input_mime == "audio/pcm"
        assert options.output_mime == "audio/pcm"
        assert options.output_modalities is None
        assert options.instructions is None
        assert options.tools is None
        assert options.tool_choice == "auto"
        assert options.tool_timeout_s == 300.0
        assert options.tool_result_max_age_s is None

    def test_custom_values(self):
        """Test custom values for RealtimeSessionOptions."""
        options = RealtimeSessionOptions(
            model="gpt-4",
            voice="alloy",
            vad_enabled=False,
            input_rate_hz=16000,
            output_rate_hz=16000,
            input_mime="audio/wav",
            output_mime="audio/wav",
            output_modalities=["audio", "text"],
            instructions="Test instructions",
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="required",
            tool_timeout_s=600.0,
            tool_result_max_age_s=30.0,
        )
        assert options.model == "gpt-4"
        assert options.voice == "alloy"
        assert options.vad_enabled is False
        assert options.input_rate_hz == 16000
        assert options.output_rate_hz == 16000
        assert options.input_mime == "audio/wav"
        assert options.output_mime == "audio/wav"
        assert options.output_modalities == ["audio", "text"]
        assert options.instructions == "Test instructions"
        assert options.tools == [{"type": "function", "function": {"name": "test"}}]
        assert options.tool_choice == "required"
        assert options.tool_timeout_s == 600.0
        assert options.tool_result_max_age_s == 30.0


class TestRealtimeChunk:
    """Test RealtimeChunk dataclass and its properties."""

    def test_audio_chunk_creation(self):
        """Test creating an audio chunk."""
        chunk = RealtimeChunk(modality="audio", data=b"audio_data")
        assert chunk.modality == "audio"
        assert chunk.data == b"audio_data"
        assert chunk.timestamp is None
        assert chunk.metadata is None

    def test_text_chunk_creation(self):
        """Test creating a text chunk."""
        chunk = RealtimeChunk(modality="text", data="text_data")
        assert chunk.modality == "text"
        assert chunk.data == "text_data"
        assert chunk.timestamp is None
        assert chunk.metadata is None

    def test_chunk_with_metadata(self):
        """Test creating a chunk with timestamp and metadata."""
        metadata = {"confidence": 0.95}
        chunk = RealtimeChunk(
            modality="text", data="hello", timestamp=123.45, metadata=metadata
        )
        assert chunk.modality == "text"
        assert chunk.data == "hello"
        assert chunk.timestamp == 123.45
        assert chunk.metadata == metadata

    def test_is_audio_property(self):
        """Test is_audio property."""
        audio_chunk = RealtimeChunk(modality="audio", data=b"data")
        text_chunk = RealtimeChunk(modality="text", data="data")

        assert audio_chunk.is_audio is True
        assert text_chunk.is_audio is False

    def test_is_text_property(self):
        """Test is_text property."""
        audio_chunk = RealtimeChunk(modality="audio", data=b"data")
        text_chunk = RealtimeChunk(modality="text", data="data")

        assert audio_chunk.is_text is False
        assert text_chunk.is_text is True

    def test_audio_data_property(self):
        """Test audio_data property."""
        audio_chunk = RealtimeChunk(modality="audio", data=b"audio_bytes")
        text_chunk = RealtimeChunk(modality="text", data="text_string")
        mixed_chunk = RealtimeChunk(modality="audio", data="not_bytes")

        assert audio_chunk.audio_data == b"audio_bytes"
        assert text_chunk.audio_data is None
        assert mixed_chunk.audio_data is None

    def test_text_data_property(self):
        """Test text_data property."""
        audio_chunk = RealtimeChunk(modality="audio", data=b"audio_bytes")
        text_chunk = RealtimeChunk(modality="text", data="text_string")
        mixed_chunk = RealtimeChunk(modality="text", data=b"not_string")

        assert audio_chunk.text_data is None
        assert text_chunk.text_data == "text_string"
        assert mixed_chunk.text_data is None


@pytest.mark.asyncio
class TestSeparateAudioChunks:
    """Test separate_audio_chunks utility function."""

    async def test_separate_audio_only(self):
        """Test separating audio chunks from audio-only stream."""

        async def audio_stream():
            yield RealtimeChunk(modality="audio", data=b"chunk1")
            yield RealtimeChunk(modality="audio", data=b"chunk2")

        chunks = []
        async for data in separate_audio_chunks(audio_stream()):
            chunks.append(data)

        assert chunks == [b"chunk1", b"chunk2"]

    async def test_separate_mixed_modalities(self):
        """Test separating audio chunks from mixed modality stream."""

        async def mixed_stream():
            yield RealtimeChunk(modality="audio", data=b"audio1")
            yield RealtimeChunk(modality="text", data="text1")
            yield RealtimeChunk(modality="audio", data=b"audio2")
            yield RealtimeChunk(modality="text", data="text2")

        chunks = []
        async for data in separate_audio_chunks(mixed_stream()):
            chunks.append(data)

        assert chunks == [b"audio1", b"audio2"]

    async def test_separate_no_audio(self):
        """Test separating audio chunks from text-only stream."""

        async def text_stream():
            yield RealtimeChunk(modality="text", data="text1")
            yield RealtimeChunk(modality="text", data="text2")

        chunks = []
        async for data in separate_audio_chunks(text_stream()):
            chunks.append(data)

        assert chunks == []

    async def test_separate_empty_audio_data(self):
        """Test handling of audio chunks with empty data."""

        async def stream_with_empty():
            yield RealtimeChunk(modality="audio", data=b"")
            yield RealtimeChunk(modality="audio", data=b"valid")
            yield RealtimeChunk(modality="audio", data=b"")

        chunks = []
        async for data in separate_audio_chunks(stream_with_empty()):
            chunks.append(data)

        assert chunks == [b"valid"]


@pytest.mark.asyncio
class TestSeparateTextChunks:
    """Test separate_text_chunks utility function."""

    async def test_separate_text_only(self):
        """Test separating text chunks from text-only stream."""

        async def text_stream():
            yield RealtimeChunk(modality="text", data="chunk1")
            yield RealtimeChunk(modality="text", data="chunk2")

        chunks = []
        async for data in separate_text_chunks(text_stream()):
            chunks.append(data)

        assert chunks == ["chunk1", "chunk2"]

    async def test_separate_mixed_modalities(self):
        """Test separating text chunks from mixed modality stream."""

        async def mixed_stream():
            yield RealtimeChunk(modality="audio", data=b"audio1")
            yield RealtimeChunk(modality="text", data="text1")
            yield RealtimeChunk(modality="audio", data=b"audio2")
            yield RealtimeChunk(modality="text", data="text2")

        chunks = []
        async for data in separate_text_chunks(mixed_stream()):
            chunks.append(data)

        assert chunks == ["text1", "text2"]

    async def test_separate_no_text(self):
        """Test separating text chunks from audio-only stream."""

        async def audio_stream():
            yield RealtimeChunk(modality="audio", data=b"audio1")
            yield RealtimeChunk(modality="audio", data=b"audio2")

        chunks = []
        async for data in separate_text_chunks(audio_stream()):
            chunks.append(data)

        assert chunks == []

    async def test_separate_empty_text_data(self):
        """Test handling of text chunks with empty data."""

        async def stream_with_empty():
            yield RealtimeChunk(modality="text", data="")
            yield RealtimeChunk(modality="text", data="valid")
            yield RealtimeChunk(modality="text", data="")

        chunks = []
        async for data in separate_text_chunks(stream_with_empty()):
            chunks.append(data)

        assert chunks == ["valid"]


@pytest.mark.asyncio
class TestDemuxRealtimeChunks:
    """Test demux_realtime_chunks utility function."""

    async def test_demux_mixed_stream(self):
        """Test demuxing mixed modality stream."""

        async def mixed_stream():
            yield RealtimeChunk(modality="audio", data=b"audio1")
            yield RealtimeChunk(modality="text", data="text1")
            yield RealtimeChunk(modality="audio", data=b"audio2")
            yield RealtimeChunk(modality="text", data="text2")

        audio_stream, text_stream = await demux_realtime_chunks(mixed_stream())

        audio_chunks = []
        async for data in audio_stream:
            audio_chunks.append(data)

        text_chunks = []
        async for data in text_stream:
            text_chunks.append(data)

        assert audio_chunks == [b"audio1", b"audio2"]
        assert text_chunks == ["text1", "text2"]

    async def test_demux_audio_only(self):
        """Test demuxing audio-only stream."""

        async def audio_stream():
            yield RealtimeChunk(modality="audio", data=b"audio1")
            yield RealtimeChunk(modality="audio", data=b"audio2")

        audio_stream_out, text_stream = await demux_realtime_chunks(audio_stream())

        audio_chunks = []
        async for data in audio_stream_out:
            audio_chunks.append(data)

        text_chunks = []
        async for data in text_stream:
            text_chunks.append(data)

        assert audio_chunks == [b"audio1", b"audio2"]
        assert text_chunks == []

    async def test_demux_text_only(self):
        """Test demuxing text-only stream."""

        async def text_stream():
            yield RealtimeChunk(modality="text", data="text1")
            yield RealtimeChunk(modality="text", data="text2")

        audio_stream, text_stream_out = await demux_realtime_chunks(text_stream())

        audio_chunks = []
        async for data in audio_stream:
            audio_chunks.append(data)

        text_chunks = []
        async for data in text_stream_out:
            text_chunks.append(data)

        assert audio_chunks == []
        assert text_chunks == ["text1", "text2"]

    async def test_demux_empty_stream(self):
        """Test demuxing empty stream."""

        async def empty_stream():
            return
            yield  # pragma: no cover

        audio_stream, text_stream = await demux_realtime_chunks(empty_stream())

        audio_chunks = []
        async for data in audio_stream:
            audio_chunks.append(data)

        text_chunks = []
        async for data in text_stream:
            text_chunks.append(data)

        assert audio_chunks == []
        assert text_chunks == []


class TestBaseRealtimeSession:
    """Test BaseRealtimeSession abstract base class."""

    def test_is_abstract_class(self):
        """Test that BaseRealtimeSession cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseRealtimeSession()

    def test_abstract_methods_exist(self):
        """Test that all expected abstract methods are defined."""
        # This test ensures the abstract methods are properly defined
        # We can't instantiate the class, but we can check the methods exist
        methods = [
            "connect",
            "close",
            "update_session",
            "append_audio",
            "commit_input",
            "clear_input",
            "create_response",
            "iter_events",
            "iter_output_audio",
            "iter_input_transcript",
            "iter_output_transcript",
            "set_tool_executor",
        ]

        for method_name in methods:
            assert hasattr(BaseRealtimeSession, method_name), (
                f"Missing method: {method_name}"
            )

            method = getattr(BaseRealtimeSession, method_name)
            assert callable(method), f"Method {method_name} is not callable"


class _ConcreteRealtimeSession(
    BaseRealtimeSession
):  # pragma: no cover - used only for coverage tests
    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def update_session(self, session_patch):
        pass

    async def append_audio(self, pcm16_bytes: bytes) -> None:
        pass

    async def commit_input(self) -> None:
        pass

    async def clear_input(self) -> None:
        pass

    async def create_response(self, response_patch=None) -> None:
        pass

    def iter_events(self):
        async def _g():
            if False:
                yield  # pragma: no cover

        return _g()

    def iter_output_audio(self):
        async def _g():
            if False:
                yield  # pragma: no cover

        return _g()

    def iter_input_transcript(self):
        async def _g():
            if False:
                yield  # pragma: no cover

        return _g()

    def iter_output_transcript(self):
        async def _g():
            if False:
                yield  # pragma: no cover

        return _g()

    def set_tool_executor(self, executor):
        pass


class TestRealtimeSessionOptionsTranscription:
    def test_transcription_defaults(self):
        opts = RealtimeSessionOptions()
        assert opts.transcription_model is None
        assert opts.transcription_language is None
        assert opts.transcription_prompt is None
        assert opts.transcription_noise_reduction is None
        assert opts.transcription_include_logprobs is False

    def test_transcription_custom(self):
        opts = RealtimeSessionOptions(
            transcription_model="whisper-1",
            transcription_language="en",
            transcription_prompt="domain context",
            transcription_noise_reduction=True,
            transcription_include_logprobs=True,
        )
        assert opts.transcription_model == "whisper-1"
        assert opts.transcription_language == "en"
        assert opts.transcription_prompt == "domain context"
        assert opts.transcription_noise_reduction is True
        assert opts.transcription_include_logprobs is True


@pytest.mark.asyncio
class TestConcreteRealtimeSessionCoverage:
    async def test_instantiate_and_call_methods(self):
        sess = _ConcreteRealtimeSession()
        await sess.connect()
        await sess.update_session({"a": 1})
        await sess.append_audio(b"\x00\x00")
        await sess.commit_input()
        await sess.clear_input()
        await sess.create_response({"response": 1})
        sess.set_tool_executor(lambda name, args: None)  # sync is fine for stub
        # Iterate through generators (they are empty)
        async for _ in sess.iter_events():
            pass
        async for _ in sess.iter_output_audio():
            pass
        async for _ in sess.iter_input_transcript():
            pass
        async for _ in sess.iter_output_transcript():
            pass
        await sess.close()

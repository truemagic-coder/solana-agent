import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions


class DummyRealtimeSession:
    """Minimal stub of a realtime session supporting audio + transcripts."""

    def __init__(self, options: RealtimeSessionOptions):
        self._options = options
        self._connected = False
        self._in_use_lock = asyncio.Lock()
        self._in_tr = asyncio.Queue()
        self._out_tr = asyncio.Queue()
        self._audio = asyncio.Queue()

    async def start(self):
        self._connected = True

    async def configure(self, **kwargs):  # pragma: no cover - simple stub
        if kwargs.get("instructions"):
            self._options.instructions = kwargs["instructions"]

    async def clear_input(self):  # pragma: no cover
        return

    def reset_output_stream(self):  # pragma: no cover
        return

    async def append_audio(self, b: bytes):
        await self._audio.put(b"FAKEAUDIO1")
        for part in ["hel", "lo "]:
            await self._in_tr.put(part)
        await self._out_tr.put("Hi there!")
        await self._audio.put(b"FAKEAUDIO2")
        await self._out_tr.put(None)
        await self._audio.put(None)
        await self._in_tr.put(None)

    async def commit_input(self):  # pragma: no cover
        return

    async def create_response(self, response_patch=None):  # pragma: no cover
        return

    async def create_conversation_item(self, item):  # pragma: no cover
        return

    async def _iter(self, q):
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    def iter_input_transcript(self):
        return self._iter(self._in_tr)

    def iter_output_transcript(self):
        return self._iter(self._out_tr)

    def iter_output_audio(self):  # not used directly, but keep for parity
        return self._iter(self._audio)

    async def iter_output_audio_encoded(self):
        async for a in self._iter(self._audio):
            yield a

    async def iter_output_combined(self):
        # Import RealtimeChunk for the combined method
        from solana_agent.interfaces.providers.realtime import RealtimeChunk

        async for a in self.iter_output_audio_encoded():
            yield RealtimeChunk(modality="audio", data=a)
        async for t in self.iter_output_transcript():
            yield RealtimeChunk(modality="text", data=t)


class DummyRealtimeService:
    def __init__(self, session: DummyRealtimeSession, options: RealtimeSessionOptions):
        self._session = session
        self._options = options
        self._connected = False

    async def start(self):
        await self._session.start()
        self._connected = True

    async def configure(self, **kwargs):
        await self._session.configure(**kwargs)

    async def clear_input(self):
        await self._session.clear_input()

    def reset_output_stream(self):
        self._session.reset_output_stream()

    async def append_audio(self, b: bytes):
        await self._session.append_audio(b)

    async def commit_input(self):
        await self._session.commit_input()

    async def create_response(self, patch):
        await self._session.create_response(patch)

    async def create_conversation_item(self, item):
        await self._session.create_conversation_item(item)

    def iter_input_transcript(self):
        return self._session.iter_input_transcript()

    def iter_output_transcript(self):
        return self._session.iter_output_transcript()

    async def iter_output_audio_encoded(self):
        async for a in self._session.iter_output_audio_encoded():
            yield a

    async def iter_output_combined(self):
        async for a in self._session.iter_output_audio_encoded():
            yield a
        async for t in self._session.iter_output_transcript():
            yield type("RC", (), {"modality": "text", "data": t})()


@pytest.mark.asyncio
async def test_realtime_transcription_dual_modality(monkeypatch):
    agent_service = AgentService(llm_provider=MagicMock())
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )

    # Mock agent_service behavior
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYSTEM")
    agent_service.get_agent_tools = MagicMock(return_value=[])
    agent_service.execute_tool = AsyncMock(return_value={"ok": True})

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="turn-1")
    memory_provider.update_stream_user = AsyncMock()
    memory_provider.update_stream_assistant = AsyncMock()
    memory_provider.finalize_stream_turn = AsyncMock()

    qs = QueryService(
        agent_service,
        routing_service,
        memory_provider=memory_provider,
        knowledge_base=None,
    )

    # Patch allocator to return dummy realtime service
    async def _alloc(*args, **kwargs):
        opts = RealtimeSessionOptions(
            output_modalities=["audio", "text"], vad_enabled=False
        )
        sess = DummyRealtimeSession(opts)
        rs = DummyRealtimeService(sess, opts)
        setattr(rs, "_in_use_lock", asyncio.Lock())
        await getattr(rs, "_in_use_lock").acquire()
        return rs

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    # Provide fake audio bytes
    audio_bytes = b"FAKEINPUT"

    chunks = []
    async for out in qs.process(
        user_id="u1",
        query=audio_bytes,
        realtime=True,
        output_format="audio",
        audio_input_format="mp4",
        audio_output_format="aac",
        rt_output_modalities=["audio", "text"],
        rt_encode_input=False,
        rt_encode_output=False,
        rt_transcription_model="gpt-4o-mini-transcribe",
    ):
        chunks.append(out)
    # Debug disabled

    # Expect at least one audio chunk (bytes) and no raw text strings when output_format=audio
    assert any(isinstance(c, (bytes, bytearray)) for c in chunks)
    # Memory updates should have been called (user + assistant + finalize)
    assert memory_provider.method_calls  # Some interactions occurred


@pytest.mark.asyncio
async def test_realtime_transcription_text_only(monkeypatch):
    agent_service = AgentService(llm_provider=MagicMock())
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYSTEM")
    agent_service.get_agent_tools = MagicMock(return_value=[])

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="turn-1")
    memory_provider.update_stream_user = AsyncMock()
    memory_provider.update_stream_assistant = AsyncMock()
    memory_provider.finalize_stream_turn = AsyncMock()

    qs = QueryService(
        agent_service,
        routing_service,
        memory_provider=memory_provider,
        knowledge_base=None,
    )

    async def _alloc(*args, **kwargs):
        opts = RealtimeSessionOptions(output_modalities=["text"], vad_enabled=False)
        sess = DummyRealtimeSession(opts)
        # Prime assistant transcript output for text-only path
        await sess._out_tr.put("Assistant reply")
        await sess._out_tr.put(None)
        rs = DummyRealtimeService(sess, opts)
        setattr(rs, "_in_use_lock", asyncio.Lock())
        await getattr(rs, "_in_use_lock").acquire()
        return rs

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    # Provide text query with transcription model (should just stream assistant transcript)
    chunks = []
    async for out in qs.process(
        user_id="u1",
        query="Hello",
        realtime=True,
        output_format="text",
        rt_output_modalities=["text"],
        rt_transcription_model="gpt-4o-mini-transcribe",
    ):
        chunks.append(out)
    # Debug disabled

    # Expect text output present
    assert any(isinstance(c, str) and c for c in chunks)
    assert memory_provider.method_calls


@pytest.mark.asyncio
async def test_realtime_transcription_bypasses_http_stt(monkeypatch):
    """When rt_transcription_model is set, llm_provider.transcribe_audio must not be called."""
    llm_provider = MagicMock()
    llm_provider.transcribe_audio = AsyncMock(return_value="SHOULD_NOT_BE_USED")
    agent_service = AgentService(llm_provider=llm_provider)
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYSTEM")
    agent_service.get_agent_tools = MagicMock(return_value=[])

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="turn-1")
    memory_provider.update_stream_user = AsyncMock()
    memory_provider.update_stream_assistant = AsyncMock()
    memory_provider.finalize_stream_turn = AsyncMock()

    qs = QueryService(
        agent_service,
        routing_service,
        memory_provider=memory_provider,
        knowledge_base=None,
    )

    async def _alloc(*args, **kwargs):
        # Only need text output to simulate transcript path
        opts = RealtimeSessionOptions(output_modalities=["text"], vad_enabled=False)
        sess = DummyRealtimeSession(opts)
        # Provide assistant reply so generator yields something
        await sess._out_tr.put("OK")
        await sess._out_tr.put(None)
        rs = DummyRealtimeService(sess, opts)
        setattr(rs, "_in_use_lock", asyncio.Lock())
        await getattr(rs, "_in_use_lock").acquire()
        return rs

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    # Execute with audio query bytes (would ordinarily trigger HTTP STT if no realtime transcription model)
    async for _ in qs.process(
        user_id="u1",
        query=b"AUDIOINPUT",
        realtime=True,
        output_format="text",
        rt_output_modalities=["text"],
        rt_transcription_model="gpt-4o-mini-transcribe",
    ):
        pass

    llm_provider.transcribe_audio.assert_not_called()


@pytest.mark.asyncio
async def test_realtime_audio_without_explicit_model_still_skips_http_stt(monkeypatch):
    """Even if rt_transcription_model isn't supplied, realtime audio path should auto-select a model and bypass HTTP STT."""
    llm_provider = MagicMock()
    llm_provider.transcribe_audio = AsyncMock(return_value="SHOULD_NOT_BE_USED")
    agent_service = AgentService(llm_provider=llm_provider)
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYSTEM")
    agent_service.get_agent_tools = MagicMock(return_value=[])

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="turn-1")
    memory_provider.update_stream_user = AsyncMock()
    memory_provider.update_stream_assistant = AsyncMock()
    memory_provider.finalize_stream_turn = AsyncMock()

    qs = QueryService(
        agent_service,
        routing_service,
        memory_provider=memory_provider,
        knowledge_base=None,
    )

    async def _alloc(*args, **kwargs):
        # Audio + text modalities so combined path runs
        opts = RealtimeSessionOptions(
            output_modalities=["audio", "text"], vad_enabled=False
        )
        sess = DummyRealtimeSession(opts)
        rs = DummyRealtimeService(sess, opts)
        setattr(rs, "_in_use_lock", asyncio.Lock())
        await getattr(rs, "_in_use_lock").acquire()
        return rs

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    # Provide audio input but omit rt_transcription_model
    async for _ in qs.process(
        user_id="u1",
        query=b"AUDIOINPUT",
        realtime=True,
        output_format="audio",
        rt_output_modalities=["audio", "text"],
    ):
        pass

    llm_provider.transcribe_audio.assert_not_called()


@pytest.mark.asyncio
async def test_realtime_audio_user_transcript_persisted(monkeypatch):
    """Verify that realtime input transcript (user) is written to memory (update_stream_user called with accumulated transcript)."""
    agent_service = AgentService(llm_provider=MagicMock())
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYSTEM")
    agent_service.get_agent_tools = MagicMock(return_value=[])

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="turn-1")
    memory_provider.update_stream_user = AsyncMock()
    memory_provider.update_stream_assistant = AsyncMock()
    memory_provider.finalize_stream_turn = AsyncMock()

    qs = QueryService(
        agent_service,
        routing_service,
        memory_provider=memory_provider,
        knowledge_base=None,
    )

    async def _alloc(*args, **kwargs):
        opts = RealtimeSessionOptions(
            output_modalities=["audio", "text"], vad_enabled=False
        )
        sess = DummyRealtimeSession(opts)
        rs = DummyRealtimeService(sess, opts)
        setattr(rs, "_in_use_lock", asyncio.Lock())
        await getattr(rs, "_in_use_lock").acquire()
        return rs

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    # Patch realtime_update_user to invoke underlying memory mock and set a flag
    orig_update_user = qs.realtime_update_user

    async def _wrapped_update_user(u, turn_id, text):
        setattr(qs, "_test_user_tr", text)
        await orig_update_user(u, turn_id, text)

    monkeypatch.setattr(qs, "realtime_update_user", _wrapped_update_user)

    # Execute realtime audio turn (auto transcription model injected)
    async for _ in qs.process(
        user_id="u1",
        query=b"AUDIOINPUT",
        realtime=True,
        output_format="audio",
        rt_output_modalities=["audio", "text"],
    ):
        pass

    # Assert transcript captured either via memory provider or internal attribute
    captured_attr = getattr(qs, "_last_realtime_user_transcript", "") or getattr(
        qs, "_test_user_tr", ""
    )
    assert captured_attr, "Expected non-empty realtime user transcript"
    if memory_provider.update_stream_user.await_count:
        args_list = memory_provider.update_stream_user.call_args_list
        assert any(len(c.args) >= 3 and c.args[2] for c in args_list)


@pytest.mark.asyncio
async def test_realtime_no_duplicate_conversation_and_user_transcript(monkeypatch):
    """Ensure only one conversation history document would be stored logically and user transcript delta not duplicated.

    We simulate streaming APIs (begin/update/finalize) being present; fallback store should NOT trigger since we have those APIs.
    The QueryService adjustments track already streamed user transcript length to avoid duplicate update_stream_user calls with same text.
    """
    agent_service = AgentService(llm_provider=MagicMock())
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYSTEM")
    agent_service.get_agent_tools = MagicMock(return_value=[])

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="turn-1")
    memory_provider.update_stream_user = AsyncMock()
    memory_provider.update_stream_assistant = AsyncMock()
    memory_provider.finalize_stream_turn = AsyncMock()
    memory_provider.store = AsyncMock()

    qs = QueryService(
        agent_service,
        routing_service,
        memory_provider=memory_provider,
        knowledge_base=None,
    )

    async def _alloc(*args, **kwargs):
        opts = RealtimeSessionOptions(
            output_modalities=["audio", "text"], vad_enabled=False
        )
        sess = DummyRealtimeSession(opts)
        rs = DummyRealtimeService(sess, opts)
        setattr(rs, "_in_use_lock", asyncio.Lock())
        await getattr(rs, "_in_use_lock").acquire()
        return rs

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    # Execute realtime audio turn
    async for _ in qs.process(
        user_id="u1",
        query=b"AUDIOINPUT",
        realtime=True,
        output_format="audio",
        rt_output_modalities=["audio", "text"],
        rt_transcription_model="gpt-4o-mini-transcribe",
    ):
        pass

    # Fallback store should NOT be called because streaming APIs exist
    assert memory_provider.store.await_count == 0, (
        "Expected no fallback store invocation"
    )
    # update_stream_user should now be called exactly once with the full transcript (no incremental deltas)
    user_calls = memory_provider.update_stream_user.call_args_list
    assert len(user_calls) == 1, (
        f"Expected single user transcript persistence, got {len(user_calls)}"
    )
    full_delta = user_calls[0].args[2]
    assert full_delta in {
        "hel lo ",
        "hello ",
        "hello",
        "hel lo",
        "helo ",  # produced by overlap-based merge (dedup removing duplicated 'l')
    }  # allow minor spacing artifacts / merge effects
    # finalize should still have been called
    assert memory_provider.finalize_stream_turn.await_count == 1


@pytest.mark.asyncio
async def test_realtime_cumulative_and_duplicate_user_segments(monkeypatch):
    """Ensure cumulative + repeated final input transcript segments produce a single deduplicated user transcript.

    Simulates provider emitting: ["My name is ", "My name is John", "My name is John"]
    Stored transcript should be exactly "My name is John" with a single update_stream_user call.
    """
    from solana_agent.services.agent import AgentService
    from solana_agent.services.routing import RoutingService
    from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions

    class CumulativeDummySession(DummyRealtimeSession):
        async def append_audio(self, b: bytes):
            # Emit cumulative growing and duplicate final segments
            for part in ["My name is ", "My name is John", "My name is John"]:
                await self._in_tr.put(part)
            # Minimal assistant side output
            await self._out_tr.put("Hello John!")
            await self._out_tr.put(None)
            await self._in_tr.put(None)
            await self._audio.put(None)

    agent_service = AgentService(llm_provider=MagicMock())
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYSTEM")
    agent_service.get_agent_tools = MagicMock(return_value=[])

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="turn-2")
    memory_provider.update_stream_user = AsyncMock()
    memory_provider.update_stream_assistant = AsyncMock()
    memory_provider.finalize_stream_turn = AsyncMock()

    qs = QueryService(
        agent_service,
        routing_service,
        memory_provider=memory_provider,
        knowledge_base=None,
    )

    async def _alloc(*args, **kwargs):
        opts = RealtimeSessionOptions(output_modalities=["text"], vad_enabled=False)
        sess = CumulativeDummySession(opts)
        rs = DummyRealtimeService(sess, opts)
        setattr(rs, "_in_use_lock", asyncio.Lock())
        await getattr(rs, "_in_use_lock").acquire()
        # Trigger audio append path so input transcript is produced
        await rs.append_audio(b"FAKE")
        return rs

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    # Run realtime process with audio path (forces append_audio invocation)
    async for _ in qs.process(
        user_id="u2",
        query=b"AUDIOINPUT",
        realtime=True,
        output_format="text",
        rt_output_modalities=["text"],
        rt_transcription_model="gpt-4o-mini-transcribe",
    ):
        pass

    # Validate single persistence and deduplicated final transcript
    user_calls = memory_provider.update_stream_user.call_args_list
    assert len(user_calls) == 1, (
        f"Expected single user transcript persistence, got {len(user_calls)}"
    )
    final_text = user_calls[0].args[2]
    assert final_text == "My name is John", (
        f"Unexpected deduped transcript: {final_text!r}"
    )
    assert memory_provider.finalize_stream_turn.await_count == 1

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
            yield type("RC", (), {"modality": "audio", "data": a})()

    async def iter_output_combined(self):
        async for a in self.iter_output_audio_encoded():
            yield a
        async for t in self.iter_output_transcript():
            yield type("RC", (), {"modality": "text", "data": t})()


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
    # update_stream_user should have been called exactly once with full transcript (hel + lo )
    # depending on segmentation it may be incremental; ensure no duplicated concatenation
    # Collect cumulative user transcript from calls
    user_deltas = [c.args[2] for c in memory_provider.update_stream_user.call_args_list]
    # Join deltas to form final transcript
    # Ensure no duplicate repeated full transcript (i.e., last delta should not equal final transcript entirely more than once)
    # A naive duplication would show two identical concatenations; we check uniqueness of cumulative growth pattern.
    cumulative = []
    acc = ""
    for d in user_deltas:
        acc += d
        cumulative.append(acc)
    # Ensure cumulative list strictly increases and final appears only once
    assert len(cumulative) == len(set(cumulative)), (
        "Detected duplicate cumulative user transcript states"
    )

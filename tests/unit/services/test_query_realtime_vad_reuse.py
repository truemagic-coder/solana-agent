import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions


# Minimal fake session capturing last update payloads
class FakeRealtimeSession:
    def __init__(self, options: RealtimeSessionOptions):
        self._options = options
        self._audio = asyncio.Queue()
        self._updates = []
        self._connected = False

    async def connect(self):
        self._connected = True

    async def close(self):
        self._connected = False

    async def update_session(self, patch):
        self._updates.append(patch)

    async def append_audio(self, b: bytes):
        return

    async def commit_input(self):
        # Simulate one chunk produced after commit (auto response due to VAD)
        await self._audio.put(b"A" * 4800)
        await self._audio.put(None)

    async def create_response(self, patch):
        # Not called when VAD is active
        return

    def iter_output_audio(self):
        async def _gen():
            while True:
                c = await self._audio.get()
                if c is None:
                    break
                yield c

        return _gen()

    def iter_input_transcript(self):
        async def _empty():
            if False:
                yield None

        return _empty()

    def iter_output_transcript(self):
        async def _empty():
            if False:
                yield None

        return _empty()


class Wrapper:
    def __init__(self, sess: FakeRealtimeSession, opts: RealtimeSessionOptions):
        self._session = sess
        self._options = opts
        self._transcoder = None
        self._encode_output = False
        self._client_output_mime = "audio/pcm"
        self._client_input_mime = "audio/pcm"
        self._in_use_lock = asyncio.Lock()
        self._connected = False

    async def start(self):
        await self._session.connect()
        self._connected = True

    async def configure(self, **kw):
        # Mirror minimal behavior: push update_session
        if "vad_enabled" in kw:
            self._options.vad_enabled = kw["vad_enabled"]
        await self._session.update_session(
            {"audio": {"input": {"turn_detection": kw.get("vad_enabled")}}}
        )

    async def clear_input(self):
        return

    async def append_audio(self, b: bytes):
        await self._session.append_audio(b)

    async def commit_input(self):
        await self._session.commit_input()

    async def create_response(self, patch):
        await self._session.create_response(patch)

    def iter_output_audio(self):
        return self._session.iter_output_audio()

    def reset_output_stream(self):
        return

    def iter_output_audio_encoded(self):
        async def _gen():
            async for c in self.iter_output_audio():
                yield type("RC", (), {"modality": "audio", "data": c})()

        return _gen()

    def iter_input_transcript(self):
        return self._session.iter_input_transcript()

    def iter_output_transcript(self):
        return self._session.iter_output_transcript()


@pytest.mark.asyncio
async def test_vad_persists_across_realtime_reuse(monkeypatch):
    agent_service = AgentService(llm_provider=MagicMock())
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYS")
    agent_service.get_agent_tools = MagicMock(return_value=[])

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="t1")
    memory_provider.update_stream_user = AsyncMock(return_value="t1")
    memory_provider.update_stream_assistant = AsyncMock()
    memory_provider.finalize_stream_turn = AsyncMock()

    qs = QueryService(
        agent_service,
        routing_service,
        memory_provider=memory_provider,
        knowledge_base=None,
    )

    sessions = []

    async def _alloc(*args, **kwargs):
        if sessions:
            rt = sessions[0]
            if rt._in_use_lock.locked():
                rt._in_use_lock.release()
            await rt._in_use_lock.acquire()
            return rt
        opts = RealtimeSessionOptions(output_modalities=["audio"], vad_enabled=True)
        sess = FakeRealtimeSession(opts)
        rt = Wrapper(sess, opts)
        await rt._in_use_lock.acquire()
        sessions.append(rt)
        return rt

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    # First turn with VAD enabled
    async for _ in qs.process(
        user_id="u-vad",
        query=b"\x00" * 320,
        realtime=True,
        output_format="audio",
        audio_input_format="pcm",
        audio_output_format="pcm",
        audio_preset="expo_pcm16",
        rt_output_modalities=["audio"],
        vad=True,
    ):
        break

    rt = sessions[0]
    first_updates = len(rt._session._updates)
    assert first_updates > 0

    # Second turn should cause another VAD reassert configure call
    async for _ in qs.process(
        user_id="u-vad",
        query=b"\x01" * 320,
        realtime=True,
        output_format="audio",
        audio_input_format="pcm",
        audio_output_format="pcm",
        audio_preset="expo_pcm16",
        rt_output_modalities=["audio"],
        vad=True,
    ):
        break

    assert len(rt._session._updates) > first_updates, (
        "Expected additional session.update to reassert VAD on reuse"
    )

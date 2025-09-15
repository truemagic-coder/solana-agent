import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions


class CaptureRealtimeSession:
    def __init__(self, options: RealtimeSessionOptions):
        self._options = options
        self._audio = asyncio.Queue()
        self._in_use_lock = asyncio.Lock()
        self._connected = False

    async def connect(self):
        self._connected = True

    async def close(self):
        self._connected = False

    async def update_session(self, patch):
        return

    async def append_audio(self, b: bytes):
        return

    async def commit_input(self):
        # Simulate model generating one PCM chunk
        await self._audio.put(b"X" * 4800)  # 2400 samples @24k ~0.1s
        await self._audio.put(None)

    async def create_response(self, patch):
        # Response generation is coupled with commit in tests
        return

    def iter_output_audio(self):
        async def _gen():
            while True:
                item = await self._audio.get()
                if item is None:
                    break
                yield item

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


class DummyRealtimeServiceWrapper:
    def __init__(self, sess: CaptureRealtimeSession, opts: RealtimeSessionOptions):
        self._session = sess
        self._options = opts
        self._transcoder = None
        self._encode_output = False
        self._client_output_mime = "audio/pcm"
        self._client_input_mime = "audio/pcm"
        self._in_use_lock = asyncio.Lock()

    async def start(self):
        await self._session.connect()

    async def append_audio(self, b: bytes):
        await self._session.append_audio(b)

    async def commit_input(self):
        await self._session.commit_input()

    async def create_response(self, patch):
        await self._session.create_response(patch)

    def iter_output_audio(self):
        return self._session.iter_output_audio()

    def iter_output_audio_encoded(self):  # mimic RealtimeService passthrough
        async def _gen():
            async for c in self.iter_output_audio():
                yield type("RC", (), {"modality": "audio", "data": c})()

        return _gen()

    def reset_output_stream(self):
        return

    def iter_input_transcript(self):
        return self._session.iter_input_transcript()

    def iter_output_transcript(self):
        return self._session.iter_output_transcript()


@pytest.mark.asyncio
async def test_mp3_output_encoding_reuse(monkeypatch):
    agent_service = AgentService(llm_provider=MagicMock())
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYS")
    agent_service.get_agent_tools = MagicMock(return_value=[])

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="turn-a")
    memory_provider.update_stream_user = AsyncMock(return_value="turn-a")
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
        # Always create a new wrapper once; subsequent calls reuse list entry
        if sessions:
            rt = sessions[0]
            # Simulate free lock
            if rt._in_use_lock.locked():
                rt._in_use_lock.release()
            await rt._in_use_lock.acquire()
            return rt
        opts = RealtimeSessionOptions(output_modalities=["audio"], vad_enabled=False)
        sess = CaptureRealtimeSession(opts)
        rt = DummyRealtimeServiceWrapper(sess, opts)
        await rt._in_use_lock.acquire()
        sessions.append(rt)
        return rt

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    # First call: PCM output (no encoding)
    async for _ in qs.process(
        user_id="u-mp3",
        query=b"" + b"\x00" * 160,
        realtime=True,
        output_format="audio",
        audio_input_format="pcm",
        audio_output_format="pcm",
        audio_preset="expo_pcm16",
        rt_output_modalities=["audio"],
    ):
        break

    rt = sessions[0]
    assert rt._encode_output is False
    assert rt._client_output_mime == "audio/pcm"

    # Second call: request mp3 output; allocator should upgrade wrapper to encode_output True and mime audio/mpeg
    async for _ in qs.process(
        user_id="u-mp3",
        query=b"" + b"\x00" * 160,
        realtime=True,
        output_format="audio",
        audio_input_format="pcm",
        audio_output_format="mp3",
        audio_preset="expo_pcm16",
        rt_output_modalities=["audio"],
    ):
        break

    assert rt._client_output_mime == "audio/mpeg"
    assert rt._encode_output is True
    # Transcoder should now be attached
    assert rt._transcoder is not None

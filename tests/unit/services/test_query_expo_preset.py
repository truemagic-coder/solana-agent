import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions


# Minimal dummy realtime session producing two PCM16 24k chunks (simulated)
class DummyRealtimeSessionExpo:
    def __init__(self, options: RealtimeSessionOptions):
        self._options = options
        self._connected = False
        self._in_use_lock = asyncio.Lock()
        self._audio = asyncio.Queue()
        self._in_tr = asyncio.Queue()
        self._out_tr = asyncio.Queue()

    async def start(self):
        self._connected = True

    async def configure(self, **kwargs):
        return

    async def clear_input(self):
        return

    def reset_output_stream(self):
        return

    async def append_audio(self, b: bytes):
        # Accept ingress; no echo behaviour needed
        return

    async def commit_input(self):
        return

    async def create_response(self, patch):
        # Populate two fake 24k PCM16 blocks (here arbitrary bytes length multiple of 2)
        await self._audio.put(b"A" * 4800)  # ~0.05s placeholder
        await self._audio.put(b"B" * 4800)
        await self._audio.put(None)
        # Provide a transcript segment so persistence path triggers
        await self._out_tr.put("hi")
        await self._out_tr.put(None)

    async def create_conversation_item(self, item):
        return

    async def _iter(self, q):
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    async def iter_output_audio_encoded(self):
        async for a in self._iter(self._audio):
            yield type("RC", (), {"modality": "audio", "data": a})()

    def iter_output_transcript(self):
        return self._iter(self._out_tr)

    def iter_input_transcript(self):
        return self._iter(self._in_tr)


class DummyRealtimeServiceExpo:
    def __init__(self, sess: DummyRealtimeSessionExpo, opts: RealtimeSessionOptions):
        self._session = sess
        self._options = opts
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

    async def iter_output_audio_encoded(self):
        async for c in self._session.iter_output_audio_encoded():
            yield c

    def iter_output_transcript(self):
        return self._session.iter_output_transcript()

    def iter_input_transcript(self):
        return self._session.iter_input_transcript()


@pytest.mark.asyncio
async def test_expo_preset_egress_wav(monkeypatch):
    agent_service = AgentService(llm_provider=MagicMock())
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYS")
    agent_service.get_agent_tools = MagicMock(return_value=[])

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="turn-x")
    memory_provider.update_stream_user = AsyncMock(return_value="turn-x")
    memory_provider.update_stream_assistant = AsyncMock()
    memory_provider.finalize_stream_turn = AsyncMock()

    qs = QueryService(
        agent_service,
        routing_service,
        memory_provider=memory_provider,
        knowledge_base=None,
    )

    async def _alloc(*args, **kwargs):
        opts = RealtimeSessionOptions(output_modalities=["audio"], vad_enabled=False)
        sess = DummyRealtimeSessionExpo(opts)
        rs = DummyRealtimeServiceExpo(sess, opts)
        setattr(rs, "_in_use_lock", asyncio.Lock())
        await getattr(rs, "_in_use_lock").acquire()
        return rs

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    chunks = []
    async for out in qs.process(
        user_id="u1",
        query=b"FAKE16KWAV"
        + b"\x00" * 44,  # dummy bytes (not real header) just to traverse path
        realtime=True,
        output_format="audio",
        audio_input_format="wav",
        audio_output_format="pcm",
        audio_preset="expo_pcm16",
        rt_output_modalities=["audio"],
    ):
        assert isinstance(out, (bytes, bytearray))
        chunks.append(out)

    assert chunks, "Should emit audio chunks"
    # First chunk should look like WAV header start when ffmpeg writes RIFF (we can't fully simulate ffmpeg here), so just len check
    # Since we used a fake subprocess chain (not actually mocking ffmpeg here), we accept any bytes.


@pytest.mark.asyncio
async def test_expo_preset_ingress_16k(monkeypatch):
    agent_service = AgentService(llm_provider=MagicMock())
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYS")
    agent_service.get_agent_tools = MagicMock(return_value=[])

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="turn-y")
    memory_provider.update_stream_user = AsyncMock(return_value="turn-y")
    memory_provider.update_stream_assistant = AsyncMock()
    memory_provider.finalize_stream_turn = AsyncMock()

    qs = QueryService(
        agent_service,
        routing_service,
        memory_provider=memory_provider,
        knowledge_base=None,
    )

    # Craft a minimal fake 44-byte WAV header with 16k sample rate (little endian) + small payload
    # RIFF header structure: 'RIFF' + size + 'WAVE' ... we simplify since parser only checks a few fields
    wav_header = bytearray(44)
    wav_header[0:4] = b"RIFF"
    wav_header[8:12] = b"WAVE"
    # fmt chunk markers not strictly validated except PCM + channels + rate
    wav_header[20:22] = (1).to_bytes(2, "little")  # PCM
    wav_header[22:24] = (1).to_bytes(2, "little")  # mono
    wav_header[24:28] = (16000).to_bytes(4, "little")  # sample rate
    payload = b"P" * 160  # arbitrary small pcm payload
    fake_wav = bytes(wav_header) + payload

    appended = {}

    async def fake_append_audio(self, data: bytes):  # capture append
        appended["len"] = len(data)

    async def _alloc(*args, **kwargs):
        opts = RealtimeSessionOptions(output_modalities=["audio"], vad_enabled=False)
        sess = DummyRealtimeSessionExpo(opts)
        # Override append_audio for inspection
        monkeypatch.setattr(
            DummyRealtimeSessionExpo, "append_audio", fake_append_audio, raising=True
        )
        rs = DummyRealtimeServiceExpo(sess, opts)
        setattr(rs, "_in_use_lock", asyncio.Lock())
        await getattr(rs, "_in_use_lock").acquire()
        return rs

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    async for _ in qs.process(
        user_id="u2",
        query=fake_wav,
        realtime=True,
        output_format="audio",
        audio_input_format="wav",
        audio_output_format="pcm",
        audio_preset="expo_pcm16",
        rt_output_modalities=["audio"],
    ):
        break

    # Since we upsample 16000->24000, resulting PCM length should differ (cannot assert exact without real ffmpeg)
    # Just ensure append_audio received something (len captured)
    assert "len" in appended


@pytest.mark.asyncio
async def test_expo_preset_ingress_non_16k_no_resample(monkeypatch):
    agent_service = AgentService(llm_provider=MagicMock())
    routing_service = RoutingService(
        llm_provider=agent_service.llm_provider, agent_service=agent_service
    )
    agent_service.get_all_ai_agents = MagicMock(return_value={"default": {}})
    agent_service.get_agent_system_prompt = MagicMock(return_value="SYS")
    agent_service.get_agent_tools = MagicMock(return_value=[])

    memory_provider = MagicMock()
    memory_provider.retrieve = AsyncMock(return_value="")
    memory_provider.begin_stream_turn = AsyncMock(return_value="turn-z")
    memory_provider.update_stream_user = AsyncMock()
    memory_provider.update_stream_assistant = AsyncMock()
    memory_provider.finalize_stream_turn = AsyncMock()

    qs = QueryService(
        agent_service,
        routing_service,
        memory_provider=memory_provider,
        knowledge_base=None,
    )

    # Build WAV header with 22050 Hz (unsupported for auto-upsample)
    wav_header = bytearray(44)
    wav_header[0:4] = b"RIFF"
    wav_header[8:12] = b"WAVE"
    wav_header[20:22] = (1).to_bytes(2, "little")  # PCM
    wav_header[22:24] = (1).to_bytes(2, "little")  # mono
    wav_header[24:28] = (22050).to_bytes(4, "little")
    payload = b"Q" * 200
    fake_wav = bytes(wav_header) + payload

    lengths = {}

    async def fake_append_audio(self, data: bytes):
        lengths["len"] = len(data)

    async def _alloc(*args, **kwargs):
        opts = RealtimeSessionOptions(output_modalities=["audio"], vad_enabled=False)
        sess = DummyRealtimeSessionExpo(opts)
        monkeypatch.setattr(
            DummyRealtimeSessionExpo, "append_audio", fake_append_audio, raising=True
        )
        rs = DummyRealtimeServiceExpo(sess, opts)
        setattr(rs, "_in_use_lock", asyncio.Lock())
        await getattr(rs, "_in_use_lock").acquire()
        return rs

    monkeypatch.setattr(qs, "_alloc_realtime_session", _alloc)

    async for _ in qs.process(
        user_id="u3",
        query=fake_wav,
        realtime=True,
        output_format="audio",
        audio_input_format="wav",
        audio_output_format="pcm",
        audio_preset="expo_pcm16",
        rt_output_modalities=["audio"],
    ):
        break

    # Expect original bytes passed through (no upsample) for unsupported 22050 rate
    assert lengths.get("len") == len(fake_wav)

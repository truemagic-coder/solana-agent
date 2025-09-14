import pytest
from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService


class DummyRoutingService:
    def process(self, *args, **kwargs):  # pragma: no cover
        pass


class DummyAgentService(AgentService):  # pragma: no cover
    def __init__(self):
        # minimum needed attributes
        self.llm_provider = type("L", (), {"get_api_key": lambda self: "sk-test"})()
        self.agents = {"default": object()}
        self.default_agent_name = "default"
        self.tool_registry = type("R", (), {"register_tool": lambda self, t: True})()

    def assign_tool_for_agent(self, agent, tool):
        pass


@pytest.mark.asyncio
async def test_pcm_input_no_ffmpeg(monkeypatch, caplog):
    """Ensure PCM->PCM realtime request does not invoke FFmpeg transcoding and yields audio bytes.

    We stub _alloc_realtime_session to inject a minimal realtime session that produces one audio chunk.
    """
    svc = QueryService(
        agent_service=DummyAgentService(),
        routing_service=DummyRoutingService(),
        memory_provider=None,
        knowledge_base=None,
    )
    pcm_bytes = b"\x00\x01" * 100

    import solana_agent.services.query as query_mod

    orig_alloc = query_mod.QueryService._alloc_realtime_session

    class _AudioChunk:
        def __init__(self, data: bytes):
            self.modality = "audio"
            self.data = data

    class DummyRT:
        def __init__(self):
            self._in_use_lock = type("L", (), {"release": lambda self: None})()
            # Simulate options indicating only audio modality
            self._options = type("O", (), {"output_modalities": ["audio"]})()
            self._connected = False

        async def start(self):  # pragma: no cover
            self._connected = True

        async def configure(self, **kwargs):  # pragma: no cover
            pass

        async def clear_input(self):  # pragma: no cover
            pass

        async def append_audio(self, b: bytes):  # pragma: no cover
            # No-op; we don't transcode in test
            pass

        async def commit_input(self):  # pragma: no cover
            pass

        async def create_response(self, patch):  # pragma: no cover
            pass

        async def iter_output_transcript(self):  # pragma: no cover
            if False:
                yield ""  # pragma: no cover

        async def iter_output_audio_encoded(self):
            # Yield a single audio chunk representing assistant audio
            yield _AudioChunk(b"ok")

        async def clear_input_buffers(self):  # not used, defensive
            pass

    async def fake_alloc(self, *a, **kw):  # pragma: no cover
        # encode_in should be False for pcm input; assert for safety
        assert kw.get("encode_in") is False
        return DummyRT()

    query_mod.QueryService._alloc_realtime_session = fake_alloc

    results = []
    async for c in svc.process(
        user_id="u1",
        query=pcm_bytes,
        images=None,
        output_format="audio",
        realtime=True,
        vad=False,
        rt_encode_input=False,
        rt_encode_output=False,
        rt_output_modalities=["audio"],
        rt_voice="marin",
        audio_input_format="pcm",
        audio_output_format="pcm",
        prompt=None,
        router=None,
        output_model=None,
        capture_schema=None,
        capture_name=None,
    ):
        results.append(c)

    # Expect one raw audio chunk (bytes) containing b"ok"
    assert results == [b"ok"]
    for rec in caplog.records:
        assert "FFmpeg:" not in rec.message

    query_mod.QueryService._alloc_realtime_session = orig_alloc

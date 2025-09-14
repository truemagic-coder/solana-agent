import pytest
from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService


class DummyRoutingService:  # pragma: no cover
    pass


class DummyAgentService(AgentService):  # pragma: no cover
    def __init__(self):
        self.llm_provider = type("L", (), {"get_api_key": lambda self: "sk-test"})()
        self.agents = {"default": object()}
        self.default_agent_name = "default"
        self.tool_registry = type("R", (), {"register_tool": lambda self, t: True})()

    def assign_tool_for_agent(self, agent, tool):
        pass


@pytest.mark.asyncio
async def test_realtime_text_only_pcm_no_ffmpeg(monkeypatch, caplog):
    """Realtime text-only with PCM input should not transcode and should stream text chunks.

    We supply audio bytes (pretend PCM) but request only text modality; verify encode_in False and
    we receive a text chunk 'hello'.
    """
    svc = QueryService(
        agent_service=DummyAgentService(),
        routing_service=DummyRoutingService(),
        memory_provider=None,
        knowledge_base=None,
    )
    pcm_bytes = b"\x00\x02" * 50

    import solana_agent.services.query as query_mod

    orig_alloc = query_mod.QueryService._alloc_realtime_session

    class DummyRT:
        def __init__(self):
            self._in_use_lock = type(
                "L", (), {"release": lambda self: None, "locked": lambda self: True}
            )()
            # Force only text modality
            self._options = type("O", (), {"output_modalities": ["text"]})()
            self._connected = False
            self._cleared = False

        async def start(self):  # pragma: no cover
            self._connected = True

        async def configure(self, **kwargs):  # pragma: no cover
            pass

        async def clear_input(self):  # pragma: no cover
            self._cleared = True

        async def append_audio(self, b: bytes):  # pragma: no cover
            # No-op
            pass

        async def commit_input(self):  # pragma: no cover
            pass

        async def create_response(self, patch):  # pragma: no cover
            # Simulate model preparing a response
            pass

        async def create_conversation_item(self, item):  # pragma: no cover
            # Accept text message item
            pass

        def reset_output_stream(self):  # pragma: no cover
            pass

        async def iter_output_transcript(self):
            # yield one token then finish
            yield "hello"

    async def fake_alloc(self, *a, **kw):  # pragma: no cover
        assert kw.get("encode_in") is False  # pcm passthrough expected
        return DummyRT()

    query_mod.QueryService._alloc_realtime_session = fake_alloc

    results = []
    async for c in svc.process(
        user_id="u1",
        query=pcm_bytes,
        images=None,
        output_format="text",  # caller wants text output
        realtime=True,
        vad=False,
        rt_encode_input=False,
        rt_encode_output=False,
        rt_output_modalities=["text"],  # text-only
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

    assert results == ["hello"]
    for rec in caplog.records:
        assert "FFmpeg:" not in rec.message

    query_mod.QueryService._alloc_realtime_session = orig_alloc

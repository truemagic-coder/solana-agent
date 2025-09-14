import pytest
import asyncio
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

    def assign_tool_for_agent(self, agent, tool):  # pragma: no cover
        pass


@pytest.mark.asyncio
async def test_audio_only_no_vad_delayed_create_response(monkeypatch):
    """Audio-only realtime with VAD disabled should yield audio even if first create_response attempt fails.

    We simulate a failing first create_response followed by a successful second call triggered by the delayed retry.
    """
    svc = QueryService(
        agent_service=DummyAgentService(),
        routing_service=DummyRoutingService(),
        memory_provider=None,
        knowledge_base=None,
    )
    pcm_bytes = (
        b"\x00\x00" * 4800
    )  # ~100ms of 24kHz mono 16-bit (enough to pass commit size check)

    import solana_agent.services.query as query_mod

    orig_alloc = query_mod.QueryService._alloc_realtime_session

    class _AudioChunk:
        def __init__(self, data: bytes):
            self.modality = "audio"
            self.data = data

    class DummyRT:
        def __init__(self):
            self._in_use_lock = type(
                "L", (), {"release": lambda self: None, "locked": lambda self: False}
            )()
            self._options = type("O", (), {"output_modalities": ["audio"]})()
            self._connected = False
            self._create_calls = 0

        async def start(self):  # pragma: no cover
            self._connected = True

        async def configure(self, **kwargs):  # pragma: no cover
            pass

        async def clear_input(self):  # pragma: no cover
            pass

        async def append_audio(self, b: bytes):  # pragma: no cover
            pass

        async def commit_input(self):  # pragma: no cover
            pass

        async def create_response(self, patch):  # pragma: no cover
            self._create_calls += 1
            # First invocation raises to trigger retry path
            if self._create_calls == 1:
                raise RuntimeError("simulated early create_response failure")
            # Second invocation is treated as success (no-op)

        async def iter_output_audio_encoded(self):  # pragma: no cover
            # Wait a bit to allow delayed retry to fire
            await asyncio.sleep(1.0)
            yield _AudioChunk(b"resp")

    async def fake_alloc(self, *a, **kw):  # pragma: no cover
        return DummyRT()

    query_mod.QueryService._alloc_realtime_session = fake_alloc

    results = []
    async for c in svc.process(
        user_id="u1",
        query=pcm_bytes,
        images=None,
        output_format="audio",
        realtime=True,
        vad=False,  # explicitly disabled
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

    # Should yield exactly one audio chunk produced after delayed retry
    assert results == [b"resp"]

    query_mod.QueryService._alloc_realtime_session = orig_alloc

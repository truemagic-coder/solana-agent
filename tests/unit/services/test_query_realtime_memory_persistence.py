import asyncio
from typing import List, Any

import pytest
from unittest.mock import AsyncMock, Mock

from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions


class TextOnlyRealtimeStub:
    def __init__(self, assistant_chunks: List[str]):
        self._connected = False
        self._assistant_chunks = assistant_chunks
        self._options = RealtimeSessionOptions(output_modalities=["text"])  # type: ignore[arg-type]
        self._in_use_lock = asyncio.Lock()

    async def start(self):  # pragma: no cover
        if not self._in_use_lock.locked():
            await self._in_use_lock.acquire()
        self._connected = True

    async def configure(self, **kwargs):  # pragma: no cover
        return

    async def clear_input(self):  # pragma: no cover
        return

    def reset_output_stream(self):  # pragma: no cover
        return

    async def create_conversation_item(self, item):  # pragma: no cover
        return

    async def create_response(self, response_patch=None):  # pragma: no cover
        return

    def iter_input_transcript(self):
        async def _gen():
            if False:
                yield ""

        return _gen()

    def iter_output_transcript(self):
        async def _gen():
            for c in self._assistant_chunks:
                await asyncio.sleep(0)
                yield c

        return _gen()


class CombinedRealtimeStub(TextOnlyRealtimeStub):
    def __init__(self, assistant_chunks: List[str], audio_chunks: List[bytes]):
        super().__init__(assistant_chunks)
        self._audio_chunks = audio_chunks
        self._options = RealtimeSessionOptions(output_modalities=["audio", "text"])  # type: ignore[arg-type]

    async def iter_output_audio_encoded(self):
        for a in self._audio_chunks:
            await asyncio.sleep(0)
            yield type("RC", (), {"modality": "audio", "data": a})()

    async def iter_output_combined(self):
        # Interleave one audio then final text
        for a in self._audio_chunks:
            await asyncio.sleep(0)
            yield type("RC", (), {"modality": "audio", "data": a})()
        for t in self._assistant_chunks:
            await asyncio.sleep(0)
            yield type("RC", (), {"modality": "text", "data": t})()

    def iter_output_transcript(self):  # reuse parent text
        return super().iter_output_transcript()


def make_service(memory_provider) -> QueryService:
    agent = AsyncMock(spec=AgentService)
    agent.get_all_ai_agents = Mock(return_value={"default": {}})
    agent.get_agent_tools = Mock(return_value=[])
    agent.get_agent_system_prompt = Mock(return_value="You are helpful.")
    agent.execute_tool = AsyncMock(return_value={"ok": True})
    agent.llm_provider = AsyncMock()
    agent.llm_provider.get_api_key = Mock(return_value="test-key")

    # Provide an async generator for transcribe_audio (returns no text quickly)
    async def _empty_transcribe(data, fmt):
        if False:  # pragma: no cover
            yield ""
        return
        yield  # unreachable

    async def _gen():
        if False:  # pragma: no cover
            yield ""
        return
        yield  # unreachable

    class _TranscribeGen:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    agent.llm_provider.transcribe_audio = AsyncMock(return_value=_TranscribeGen())
    routing = AsyncMock(spec=RoutingService)
    routing.route_query = AsyncMock(return_value="default")
    svc = QueryService(
        agent_service=agent,
        routing_service=routing,
        memory_provider=memory_provider,
        knowledge_base=None,
        input_guardrails=[],
        kb_results_count=0,
    )
    return svc


@pytest.mark.asyncio
async def test_text_only_realtime_persists():
    mem = AsyncMock()
    # Streaming hooks
    mem.begin_stream_turn = AsyncMock(return_value="turn-1")
    mem.update_stream_user = AsyncMock(return_value=None)
    mem.update_stream_assistant = AsyncMock(return_value=None)
    mem.finalize_stream_turn = AsyncMock(return_value=None)

    svc = make_service(mem)
    stub = TextOnlyRealtimeStub(["Hello", " world"])

    async def alloc(uid: str, **kwargs: Any):  # pragma: no cover
        return stub

    setattr(svc, "_alloc_realtime_session", alloc)

    # Run realtime turn
    out = []
    async for c in svc.process(
        user_id="u1",
        query="Hi",
        realtime=True,
        rt_output_modalities=["text"],
        output_format="text",
    ):
        out.append(c)

    assert "".join(out) == "Hello world"
    mem.begin_stream_turn.assert_awaited_once_with("u1")
    mem.update_stream_user.assert_awaited()  # at least once
    mem.update_stream_assistant.assert_awaited()  # at least once
    mem.finalize_stream_turn.assert_awaited_once_with("u1", "turn-1")


@pytest.mark.asyncio
async def test_combined_realtime_persists():
    mem = AsyncMock()
    mem.begin_stream_turn = AsyncMock(return_value="turn-9")
    mem.update_stream_user = AsyncMock(return_value=None)
    mem.update_stream_assistant = AsyncMock(return_value=None)
    mem.finalize_stream_turn = AsyncMock(return_value=None)

    svc = make_service(mem)
    stub = CombinedRealtimeStub(["Answer"], [b"AUDIO"])

    async def alloc(uid: str, **kwargs: Any):  # pragma: no cover
        return stub

    setattr(svc, "_alloc_realtime_session", alloc)

    # Collect as audio (to exercise combined branch adaptation) but still expect persistence
    out = bytearray()
    async for c in svc.process(
        user_id="u9",
        query="text question",  # treat as text but combined modalities still include audio
        realtime=True,
        rt_output_modalities=["audio", "text"],
        output_format="audio",
    ):
        # audio bytes yielded
        out.extend(c if isinstance(c, (bytes, bytearray)) else b"")

    assert bytes(out) == b"AUDIO"
    mem.begin_stream_turn.assert_awaited_once_with("u9")
    mem.update_stream_user.assert_awaited()
    mem.update_stream_assistant.assert_awaited()
    mem.finalize_stream_turn.assert_awaited_once_with("u9", "turn-9")

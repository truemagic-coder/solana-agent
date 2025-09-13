import asyncio
from typing import AsyncGenerator, List, Dict, Any

import pytest
from unittest.mock import AsyncMock, Mock

from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions


class TextOnlyRealtimeServiceStub:
    """Stub that emits provided transcript chunks once per response.

    After response.create it pushes chunks then terminates by placing None in queue.
    Subsequent use should still work (clears internal state on reset_output_stream/clear_input)
    to mimic a persistent session reused across turns.
    """

    def __init__(self, responses: List[List[str]]):
        self._connected = False
        self._responses = responses  # list of list of chunks per turn
        self._turn = 0
        self._options = RealtimeSessionOptions(output_modalities=["text"])  # type: ignore[arg-type]
        self._out_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._in_use_lock = asyncio.Lock()

    async def start(self) -> None:
        self._connected = True
        if not self._in_use_lock.locked():
            await self._in_use_lock.acquire()

    async def configure(self, **kwargs) -> None:
        return

    async def clear_input(self) -> None:
        return

    def reset_output_stream(self) -> None:
        # Drain any remaining queued items
        try:
            while True:
                self._out_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

    async def append_audio(self, data: bytes) -> None:  # should never be called
        raise AssertionError("append_audio called in text-only stub")

    async def commit_input(self) -> None:
        return

    async def create_conversation_item(self, item: Dict[str, Any]) -> None:
        return

    async def create_response(
        self, response_patch: Dict[str, Any] | None = None
    ) -> None:
        # Enqueue the chunks for current turn
        if self._turn >= len(self._responses):
            await self._out_queue.put(None)
            return
        for c in self._responses[self._turn]:
            await self._out_queue.put(c)
        await self._out_queue.put(None)  # end marker
        self._turn += 1

    def iter_input_transcript(self) -> AsyncGenerator[str, None]:
        async def _gen():
            if False:
                yield ""

        return _gen()

    def iter_output_transcript(self) -> AsyncGenerator[str, None]:
        async def _gen():
            while True:
                item = await self._out_queue.get()
                if item is None:
                    break
                yield item

        return _gen()

    async def iter_output_audio_encoded(self):  # pragma: no cover
        raise AssertionError("audio iterator used in text-only test")


def make_query_service(stub: TextOnlyRealtimeServiceStub) -> QueryService:
    agent = AsyncMock(spec=AgentService)
    agent.get_all_ai_agents = Mock(return_value={"default": {}})
    agent.get_agent_tools = Mock(return_value=[])
    agent.get_agent_system_prompt = Mock(return_value="You are helpful.")
    agent.execute_tool = AsyncMock(return_value={"ok": True})
    agent.llm_provider = AsyncMock()
    agent.llm_provider.get_api_key = Mock(return_value="test-key")
    routing = AsyncMock(spec=RoutingService)
    routing.route_query = AsyncMock(return_value="default")
    svc = QueryService(
        agent_service=agent,
        routing_service=routing,
        memory_provider=None,
        knowledge_base=None,
        input_guardrails=[],
        kb_results_count=0,
    )
    svc.realtime_begin_turn = AsyncMock(return_value="turn-1")
    svc.realtime_update_user = AsyncMock(return_value=None)
    svc.realtime_update_assistant = AsyncMock(return_value=None)
    svc.realtime_finalize_turn = AsyncMock(return_value=None)

    async def alloc(user_id: str, **kwargs):
        return stub

    # monkeypatch via attribute assignment
    setattr(svc, "_alloc_realtime_session", alloc)
    return svc


@pytest.mark.asyncio
async def test_two_consecutive_text_only_realtime_turns():
    stub = TextOnlyRealtimeServiceStub(
        [
            ["First answer."],
            ["Second answer."],
        ]
    )
    qs = make_query_service(stub)

    # First turn
    first_chunks = []
    async for c in qs.process(
        user_id="u1",
        query="Hi",
        realtime=True,
        rt_output_modalities=["text"],
        output_format="text",
    ):
        first_chunks.append(c)
    assert first_chunks == ["First answer."]

    # Second turn should not hang
    second_chunks = []
    async for c in qs.process(
        user_id="u1",
        query="Another question",
        realtime=True,
        rt_output_modalities=["text"],
        output_format="text",
    ):
        second_chunks.append(c)
    assert second_chunks == ["Second answer."]

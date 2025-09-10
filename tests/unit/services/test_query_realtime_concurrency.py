import asyncio
from typing import Any, AsyncGenerator, List

import pytest
from unittest.mock import AsyncMock, Mock

from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService


class FakeRealtimeService:
    """Minimal realtime service stub matching the interface used by QueryService.process.

    Each instance produces its own distinct audio stream to validate isolation.
    """

    def __init__(self, label: str, chunks: List[bytes]):
        self.label = label
        self._chunks = list(chunks)
        self._connected = False

    async def start(self) -> None:
        self._connected = True

    async def configure(self, **kwargs) -> None:
        return

    async def clear_input(self) -> None:
        return

    def reset_output_stream(self) -> None:
        return

    async def append_audio(self, data: bytes) -> None:
        return

    async def commit_input(self) -> None:
        return

    async def create_response(self, response_patch: dict | None = None) -> None:
        # No-op: we predefine chunks to stream
        return

    def iter_input_transcript(self) -> AsyncGenerator[str, None]:
        async def _gen():
            if False:
                yield ""

        return _gen()

    def iter_output_transcript(self) -> AsyncGenerator[str, None]:
        async def _gen():
            if False:
                yield ""

        return _gen()

    async def iter_output_audio_encoded(self) -> AsyncGenerator[bytes, None]:
        # Yield labeled chunks with tiny delays to simulate streaming
        for c in self._chunks:
            await asyncio.sleep(0)
            yield c


class FakeSharedRealtimeService(FakeRealtimeService):
    """Simulates a problematic shared session: a second consumer raises an error.

    Includes a barrier so tests can ensure the first consumer is active before starting the second.
    """

    def __init__(self, label: str, chunks: List[bytes]):
        super().__init__(label, chunks)
        self._audio_stream_in_use = False
        self.first_consumer_started = asyncio.Event()
        self.release_first_consumer = asyncio.Event()
        self.second_attempted = asyncio.Event()

    async def iter_output_audio_encoded(self) -> AsyncGenerator[bytes, None]:
        if self._audio_stream_in_use:
            self.second_attempted.set()
            raise RuntimeError("Output audio is already being consumed")
        self._audio_stream_in_use = True
        # Signal that a consumer has started and wait until test releases us
        self.first_consumer_started.set()
        await self.release_first_consumer.wait()
        try:
            async for c in super().iter_output_audio_encoded():
                yield c
        finally:
            self._audio_stream_in_use = False


def make_query_service() -> QueryService:
    agent = AsyncMock(spec=AgentService)
    # Agent metadata helpers used by realtime path
    agent.get_all_ai_agents = Mock(return_value={"default": {}})
    agent.get_agent_tools = Mock(return_value=[])
    agent.get_agent_system_prompt = Mock(return_value="You are helpful.")
    agent.execute_tool = AsyncMock(return_value={"ok": True})

    # LLM provider for API key + stubs
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

    # No-op realtime persistence hooks
    svc.realtime_begin_turn = AsyncMock(return_value="turn-1")
    svc.realtime_update_user = AsyncMock(return_value=None)
    svc.realtime_update_assistant = AsyncMock(return_value=None)
    svc.realtime_finalize_turn = AsyncMock(return_value=None)
    return svc


@pytest.mark.asyncio
async def test_realtime_two_users_shared_session_fails(monkeypatch):
    """Demonstrate the failure mode when two users (incorrectly) share one realtime session.

    We monkeypatch the allocator to return the same FakeSharedRealtimeService for both users.
    The second consumer should raise a RuntimeError, illustrating why sessions must not be shared.
    """

    service = make_query_service()

    shared = FakeSharedRealtimeService(
        label="SHARED",
        chunks=[b"S1", b"S2", b"S3"],
    )

    async def alloc(user_id: str, **kwargs: Any):
        return shared

    monkeypatch.setattr(service, "_alloc_realtime_session", alloc)

    async def run_client(uid: str):
        # Let exceptions propagate to the caller
        chunks = bytearray()
        async for out in service.process(
            user_id=uid,
            query="hello",
            realtime=True,
            output_format="audio",
        ):
            chunks.extend(out)
        return bytes(chunks)

    # Start first user, wait until the shared stream is active
    t1 = asyncio.create_task(run_client("user-A"))
    await shared.first_consumer_started.wait()
    # Now start second user which should error due to shared single-consumer constraint
    t2 = asyncio.create_task(run_client("user-B"))
    # Let the second attempt to attach
    await shared.second_attempted.wait()
    # Allow the first consumer to proceed and finish
    shared.release_first_consumer.set()
    r1, r2 = await asyncio.gather(t1, t2, return_exceptions=True)

    # Some implementations swallow the exception in a finally block and end the stream early.
    # Accept either: exactly one RuntimeError OR exactly one empty stream.
    errs = [r for r in (r1, r2) if isinstance(r, Exception)]
    if errs:
        assert len(errs) == 1 and isinstance(errs[0], RuntimeError), (
            f"Expected one RuntimeError due to shared session, got: {errs}"
        )
        oks = [r for r in (r1, r2) if not isinstance(r, Exception)]
        assert oks and oks[0] in (b"S1S2S3", b"S1S2", b"S1")
    else:
        datas = [r for r in (r1, r2) if not isinstance(r, Exception)]
        assert len(datas) == 2
        empties = sum(1 for d in datas if d == b"")
        assert empties == 1, (
            f"Expected one empty stream due to shared session, got: {[len(d) for d in datas]}"
        )
        non_empty = [d for d in datas if d != b""][0]
        assert non_empty in (b"S1S2S3", b"S1S2", b"S1")


@pytest.mark.asyncio
async def test_realtime_two_users_isolated_success(monkeypatch):
    """Verify two different users get distinct realtime sessions and both streams succeed."""

    service = make_query_service()

    # Create two isolated realtime services with distinct outputs
    uA = FakeRealtimeService(label="A", chunks=[b"A1", b"A2"])
    uB = FakeRealtimeService(label="B", chunks=[b"B1", b"B2", b"B3"])

    async def alloc(user_id: str, **kwargs: Any):
        return uA if user_id == "user-A" else uB

    monkeypatch.setattr(service, "_alloc_realtime_session", alloc)

    async def collect(uid: str) -> bytes:
        buf = bytearray()
        async for out in service.process(
            user_id=uid,
            query="hello",
            realtime=True,
            output_format="audio",
        ):
            buf.extend(out)
        return bytes(buf)

    a_data, b_data = await asyncio.gather(collect("user-A"), collect("user-B"))

    assert a_data == b"A1A2"
    assert b_data == b"B1B2B3"

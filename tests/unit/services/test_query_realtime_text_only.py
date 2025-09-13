import asyncio
from typing import AsyncGenerator, List, Dict, Any

import pytest
from unittest.mock import AsyncMock, Mock

from solana_agent.services.query import QueryService
from solana_agent.services.agent import AgentService
from solana_agent.services.routing import RoutingService
from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions


class TextOnlyRealtimeServiceStub:
    """Stub realtime service exposing the minimal interface QueryService.process expects.

    Configured for text-only output_modalities. Any attempt to access audio output methods
    (iter_output_audio_encoded / append_audio) will raise, causing the test to fail.
    """

    def __init__(self, transcript_chunks: List[str]):
        self._connected = False
        self._transcript_chunks = transcript_chunks
        # Options mimic real session; no audio modality
        self._options = RealtimeSessionOptions(output_modalities=["text"])  # type: ignore[arg-type]
        # Lock to satisfy QueryService finalizer which releases it
        self._in_use_lock = asyncio.Lock()
        self.audio_append_called = False
        self.audio_iter_called = False
        # We'll lazily acquire the lock in start() to avoid un-awaited coroutine warning

    async def start(self) -> None:  # pragma: no cover - trivial
        self._connected = True
        # Acquire the lock here so QueryService finalizer can release it
        await self._in_use_lock.acquire()

    async def configure(self, **kwargs) -> None:  # pragma: no cover - no-op
        return

    async def clear_input(self) -> None:  # pragma: no cover - no-op
        return

    def reset_output_stream(self) -> None:  # pragma: no cover - no-op
        return

    async def append_audio(self, data: bytes) -> None:
        self.audio_append_called = True
        raise AssertionError(
            "append_audio should not be called for text-only realtime session"
        )

    async def commit_input(self) -> None:  # pragma: no cover - no-op
        return

    async def create_response(
        self, response_patch: Dict[str, Any] | None = None
    ) -> None:  # pragma: no cover - no-op
        return

    async def create_conversation_item(
        self, item: Dict[str, Any]
    ) -> None:  # pragma: no cover - no-op
        return

    def iter_input_transcript(self) -> AsyncGenerator[str, None]:
        async def _gen():
            if False:
                yield ""  # pragma: no cover

        return _gen()

    def iter_output_transcript(self) -> AsyncGenerator[str, None]:
        async def _gen():
            for t in self._transcript_chunks:
                await asyncio.sleep(0)
                yield t

        return _gen()

    async def iter_output_audio_encoded(self) -> AsyncGenerator[bytes, None]:
        self.audio_iter_called = True
        raise AssertionError(
            "iter_output_audio_encoded should not be used for text-only realtime session"
        )


def make_query_service() -> QueryService:
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

    # Persistence hooks mocked as no-ops
    svc.realtime_begin_turn = AsyncMock(return_value="turn-1")
    svc.realtime_update_user = AsyncMock(return_value=None)
    svc.realtime_update_assistant = AsyncMock(return_value=None)
    svc.realtime_finalize_turn = AsyncMock(return_value=None)
    return svc


@pytest.mark.asyncio
async def test_realtime_text_only_skips_audio(monkeypatch):
    """Ensure that when rt_output_modalities=['text'] no audio pipeline is invoked.

    Verifies:
    - append_audio is never called
    - iter_output_audio_encoded is never called
    - yielded chunks are the text transcript pieces
    - lock is released cleanly without errors
    """

    service = make_query_service()

    stub = TextOnlyRealtimeServiceStub(["First part ", "second part."])

    async def alloc(user_id: str, **kwargs):  # pragma: no cover - simple passthrough
        return stub

    monkeypatch.setattr(service, "_alloc_realtime_session", alloc)

    outputs: List[str] = []
    async for out in service.process(
        user_id="user-1",
        query="Hello world",
        realtime=True,
        rt_output_modalities=["text"],
        output_format="text",
    ):
        assert isinstance(out, str), (
            "Expected only text chunks in text-only realtime mode"
        )
        outputs.append(out)

    # Ensure transcript pieces streamed
    assert outputs == ["First part ", "second part."], outputs
    # Confirm no audio path usage
    assert not stub.audio_append_called, "append_audio unexpectedly called"
    assert not stub.audio_iter_called, "iter_output_audio_encoded unexpectedly iterated"
    # Lock should have been released
    assert not stub._in_use_lock.locked(), "Session lock not released"

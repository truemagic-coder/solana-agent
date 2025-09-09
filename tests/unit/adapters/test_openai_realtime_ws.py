import asyncio
import json
import types
import pytest

from solana_agent.adapters.openai_realtime_ws import OpenAIRealtimeWebSocketSession
from solana_agent.interfaces.providers.realtime import RealtimeSessionOptions


class FakeWebSocket:
    def __init__(self, incoming):
        # incoming: list of dicts -> will be JSON-dumped when iterated
        self._incoming = list(incoming)
        self.sent = []  # list of parsed payload dicts
        self._closed = False

    async def send(self, message: str):
        # Record payloads as parsed JSON for easy assertions
        try:
            self.sent.append(json.loads(message))
        except Exception:
            self.sent.append({"raw": message})

    def __aiter__(self):
        self._iter = iter(self._incoming)
        return self

    async def __anext__(self):
        await asyncio.sleep(0)  # allow scheduling
        try:
            item = next(self._iter)
        except StopIteration:
            raise StopAsyncIteration
        return json.dumps(item)

    async def close(self):
        self._closed = True


@pytest.mark.asyncio
async def test_function_call_flow_conversation_item_create(monkeypatch):
    # Arrange: fake server events - session.created, session.updated, then a response.done with a function_call
    response_id = "resp_123"
    call_id = "call_abc"

    incoming = [
        {"type": "session.created"},
        {
            "type": "session.updated",
            "session": {
                "audio": {"output": {"voice": "marin"}},
                "instructions": "You are a helpful assistant.",
            },
        },
        {
            "type": "response.done",
            "response": {
                "id": response_id,
                "output": [
                    {
                        "type": "function_call",
                        "id": "item_fc_1",
                        "call_id": call_id,
                        "name": "get_time",
                        "arguments": "{}",
                        "status": "completed",
                    }
                ],
            },
        },
    ]

    fake_ws = FakeWebSocket(incoming)

    async def fake_connect(uri, additional_headers=None, max_size=None):
        return fake_ws

    # Patch websockets.connect used by the adapter
    import solana_agent.adapters.openai_realtime_ws as mod

    monkeypatch.setattr(mod, "websockets", types.SimpleNamespace(connect=fake_connect))

    # Create session with deterministic options
    opts = RealtimeSessionOptions(
        model="gpt-realtime",
        voice="marin",
        vad_enabled=False,  # disable auto-create to avoid unexpected responses
        instructions="You are a helpful assistant.",
        tools=[
            {
                "type": "function",
                "name": "get_time",
                "description": "Get the current time",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        tool_choice="auto",
        tool_timeout_s=2.0,
    )

    session = OpenAIRealtimeWebSocketSession(api_key="sk-test", options=opts)

    # Tool executor
    async def tool_exec(name, args):
        assert name == "get_time"
        assert isinstance(args, dict)
        return {"now": "2025-01-01T00:00:00Z"}

    session.set_tool_executor(tool_exec)

    # Act: connect which starts recv loop and processes simulated events
    await session.connect()

    # Allow the background task to process the function call and send outputs
    # Backoff loop to await expected sends
    for _ in range(50):
        if any(
            p.get("type") == "conversation.item.create" for p in fake_ws.sent
        ) and any(p.get("type") == "response.create" for p in fake_ws.sent):
            break
        await asyncio.sleep(0.02)

    # Assert: conversation.item.create(function_call_output) and response.create were sent
    types_sent = [p.get("type") for p in fake_ws.sent]
    assert "session.update" in types_sent, "session.update should be sent on connect"
    assert "conversation.item.create" in types_sent, (
        f"Expected conversation.item.create in sent payloads, got {types_sent}"
    )
    assert "response.create" in types_sent, (
        f"Expected response.create after tool output, got {types_sent}"
    )

    # Verify the function_call_output payload structure and that event_id is attached via _send_tracked
    fco_msgs = [p for p in fake_ws.sent if p.get("type") == "conversation.item.create"]
    assert any(
        (m.get("item") or {}).get("type") == "function_call_output"
        and (m.get("item") or {}).get("call_id") == call_id
        for m in fco_msgs
    ), f"function_call_output not found or call_id mismatch: {fco_msgs}"

    # Ensure event_ids are present for tracked sends
    tracked = [
        p
        for p in fake_ws.sent
        if p.get("type") in ("conversation.item.create", "response.create")
    ]
    assert all(
        isinstance(p.get("event_id"), str) and len(p.get("event_id")) > 0
        for p in tracked
    ), f"Missing event_id on tracked sends: {tracked}"

    # Ensure the deprecated response.function_call_output path is NOT used
    assert all(
        p.get("type") != "response.function_call_output" for p in fake_ws.sent
    ), f"Deprecated response.function_call_output was used: {fake_ws.sent}"

    await session.close()

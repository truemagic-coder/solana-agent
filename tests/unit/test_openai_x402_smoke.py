import pytest

from scripts import openai_x402_smoke as smoke


def test_sdk_config_targets_runtime_v1_endpoint() -> None:
    config = smoke._sdk_config(
        base_url="http://127.0.0.1:8000",
        private_key="test-private-key",
        model="stateless",
        max_output_tokens=64,
        rpc_url="http://localhost:8899",
    )

    assert config["ai"] == {
        "auth_mode": "x402_private_key",
        "private_key": "test-private-key",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "stateless",
        "max_output_tokens": 64,
        "x402_rpc_url": "http://localhost:8899",
    }
    assert config["agents"][0]["name"] == "default"


async def _fake_agent_process():
    yield "hello"
    yield " world"


class _FakeAgent:
    async def process(self, user_id, message, runtime_context=None):
        assert user_id == "user-123"
        assert message == "hello"
        assert runtime_context == {"conversation_id": "conv-123"}
        async for chunk in _fake_agent_process():
            yield chunk

@pytest.mark.asyncio
async def test_collect_agent_text_response_joins_streamed_chunks() -> None:
    content, elapsed_ms = await smoke._collect_agent_text_response(
        _FakeAgent(),
        user_id="user-123",
        message="hello",
        runtime_context={"conversation_id": "conv-123"},
    )

    assert content == "hello world"
    assert elapsed_ms >= 0
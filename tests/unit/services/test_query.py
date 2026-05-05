from unittest.mock import MagicMock

import pytest

from solana_agent.services.query import QueryService


@pytest.fixture
def mock_agent_service():
    service = MagicMock()
    service.get_all_ai_agents.return_value = {"assistant": MagicMock()}

    async def generate_response(**kwargs):
        yield kwargs["query"]

    async def transcribe_audio(_payload, _audio_format):
        yield "hello from audio"

    service.generate_response = MagicMock(side_effect=generate_response)
    service.llm_provider = MagicMock()
    service.llm_provider.transcribe_audio = MagicMock(side_effect=transcribe_audio)
    return service


@pytest.mark.asyncio
async def test_process_uses_single_registered_agent(mock_agent_service):
    service = QueryService(agent_service=mock_agent_service)

    chunks = []
    async for chunk in service.process(
        user_id="user-123",
        query="hello",
        runtime_context={"conversation_id": "conv-123"},
        prompt="Be concise",
    ):
        chunks.append(chunk)

    assert chunks == ["hello"]
    assert mock_agent_service.generate_response.call_args.kwargs == {
        "agent_name": "assistant",
        "user_id": "user-123",
        "query": "hello",
        "runtime_context": {"conversation_id": "conv-123"},
        "images": None,
        "output_format": "text",
        "audio_voice": "nova",
        "audio_output_format": "aac",
        "prompt": "Be concise",
        "output_model": None,
    }


@pytest.mark.asyncio
async def test_process_transcribes_audio_before_generating(mock_agent_service):
    service = QueryService(agent_service=mock_agent_service)

    chunks = []
    async for chunk in service.process(user_id="user-123", query=b"audio-bytes"):
        chunks.append(chunk)

    assert chunks == ["hello from audio"]
    mock_agent_service.llm_provider.transcribe_audio.assert_called_once_with(
        b"audio-bytes",
        "mp4",
    )


@pytest.mark.asyncio
async def test_process_raises_for_empty_audio_transcription(mock_agent_service):
    async def empty_transcription(_payload, _audio_format):
        if False:
            yield ""

    mock_agent_service.llm_provider.transcribe_audio = MagicMock(
        side_effect=empty_transcription
    )
    service = QueryService(agent_service=mock_agent_service)

    with pytest.raises(ValueError, match="Audio transcription returned no text"):
        async for _chunk in service.process(user_id="user-123", query=b"audio"):
            pass

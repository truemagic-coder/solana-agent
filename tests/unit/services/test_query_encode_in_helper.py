import pytest
from solana_agent.services.query import QueryService


class DummyAgentService:
    def __init__(self):
        self.llm_provider = type("L", (), {"get_api_key": lambda self: "sk"})()
        self.agents = {"default": object()}
        self.default_agent_name = "default"
        self.tool_registry = type("R", (), {"register_tool": lambda self, t: True})()

    def assign_tool_for_agent(self, agent, tool):
        pass


class DummyRoutingService:
    pass


@pytest.fixture
def qs():
    return QueryService(
        agent_service=DummyAgentService(),
        routing_service=DummyRoutingService(),
        memory_provider=None,
        knowledge_base=None,
    )


@pytest.mark.parametrize(
    "is_audio,fmt,rt_flag,expected",
    [
        (True, "pcm", False, False),
        (True, "pcm", True, False),  # force flag ignored for pcm
        (True, "mp3", False, True),  # compressed always needs transcode
        (True, "mp3", True, True),
        (
            True,
            "wav",
            False,
            True,
        ),  # container (could be pcm but treat as needs decode to ensure mono/rate)
        (False, "mp3", True, False),  # not audio bytes
    ],
)
def test_compute_encode_in(qs, is_audio, fmt, rt_flag, expected):
    assert (
        qs._compute_encode_in(
            is_audio_bytes=is_audio,
            audio_input_format=fmt,
            rt_encode_input=rt_flag,
        )
        == expected
    )

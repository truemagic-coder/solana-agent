import pytest


@pytest.fixture(autouse=True)
def isolated_solana_agent_state_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SOLANA_AGENT_STATE_DIR",
        str(tmp_path / "solana-agent-state"),
    )

import pytest

import solana_agent.local_state as local_state


def test_local_state_file_path_uses_default_state_dir_when_override_missing(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(local_state.STATE_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(local_state, "_default_state_dir", lambda: tmp_path / "state")

    assert local_state.local_state_file_path() == (
        tmp_path / "state" / local_state.STATE_FILE_NAME
    )


def test_load_saved_privy_user_id_returns_none_when_profiles_not_mapping(monkeypatch):
    monkeypatch.setattr(
        local_state,
        "_read_state_payload",
        lambda: {"version": local_state.STATE_VERSION, "profiles": []},
    )

    assert local_state.load_saved_privy_user_id() is None


def test_load_saved_wallet_id_returns_none_when_profiles_not_mapping(monkeypatch):
    monkeypatch.setattr(
        local_state,
        "_read_state_payload",
        lambda: {"version": local_state.STATE_VERSION, "profiles": []},
    )

    assert local_state.load_saved_wallet_id() is None


def test_save_privy_user_id_requires_non_empty_value() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        local_state.save_privy_user_id(" ")


def test_save_wallet_id_requires_non_empty_value() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        local_state.save_wallet_id(" ")


def test_save_privy_user_id_initializes_profiles_and_normalizes_base_url(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        local_state,
        "_read_state_payload",
        lambda: {"version": local_state.STATE_VERSION, "profiles": []},
    )
    monkeypatch.setattr(
        local_state,
        "_write_state_payload",
        lambda payload: captured.update(payload),
    )

    local_state.save_privy_user_id(
        " did:privy:test-user ",
        base_url=" HTTPS://AI.SOLANA-AGENT.COM/V1/ ",
    )

    profile = captured["profiles"]["https://ai.solana-agent.com/v1"]
    assert profile["privy_user_id"] == "did:privy:test-user"
    assert profile["base_url"] == "https://ai.solana-agent.com/v1"
    assert "updated_at" in profile


def test_save_wallet_id_updates_existing_profile(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        local_state,
        "_read_state_payload",
        lambda: {
            "version": local_state.STATE_VERSION,
            "profiles": {
                "default": {
                    "privy_user_id": "did:privy:test-user",
                }
            },
        },
    )
    monkeypatch.setattr(
        local_state,
        "_write_state_payload",
        lambda payload: captured.update(payload),
    )

    local_state.save_wallet_id(" wallet-123 ")

    profile = captured["profiles"]["default"]
    assert profile["privy_user_id"] == "did:privy:test-user"
    assert profile["wallet_id"] == "wallet-123"
    assert "updated_at" in profile

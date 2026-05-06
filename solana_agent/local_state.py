"""Local SDK state for cross-platform hosted identity persistence."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_DIR_ENV_VAR = "SOLANA_AGENT_STATE_DIR"
STATE_FILE_NAME = "hosted_identity.json"
STATE_VERSION = 1


def _normalized_base_url(base_url: str | None) -> str:
    return str(base_url or "").strip().rstrip("/").lower()


def _profile_key(base_url: str | None) -> str:
    normalized_base_url = _normalized_base_url(base_url)
    return normalized_base_url or "default"


def _default_state_dir() -> Path:
    if sys.platform.startswith("win"):
        app_data = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if app_data:
            return Path(app_data) / "solana-agent"
        return Path.home() / "AppData" / "Roaming" / "solana-agent"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "solana-agent"

    xdg_config_home = str(os.environ.get("XDG_CONFIG_HOME") or "").strip()
    if xdg_config_home:
        return Path(xdg_config_home) / "solana-agent"
    return Path.home() / ".config" / "solana-agent"


def local_state_file_path() -> Path:
    override_dir = str(os.environ.get(STATE_DIR_ENV_VAR) or "").strip()
    if override_dir:
        return Path(override_dir).expanduser() / STATE_FILE_NAME
    return _default_state_dir() / STATE_FILE_NAME


def _read_state_payload() -> dict[str, Any]:
    state_file = local_state_file_path()
    try:
        with state_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {"version": STATE_VERSION, "profiles": {}}
    except (OSError, json.JSONDecodeError):
        return {"version": STATE_VERSION, "profiles": {}}

    if not isinstance(payload, dict):
        return {"version": STATE_VERSION, "profiles": {}}
    return payload


def _profile_from_payload(
    payload: dict[str, Any],
    *,
    base_url: str | None = None,
) -> dict[str, Any]:
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        return {}

    profile = profiles.get(_profile_key(base_url))
    if not isinstance(profile, dict):
        return {}
    return profile


def _write_state_payload(payload: dict[str, Any]) -> None:
    state_file = local_state_file_path()
    state_file.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temp_path_str = tempfile.mkstemp(
        prefix=f"{state_file.stem}-",
        suffix=".tmp",
        dir=state_file.parent,
    )
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        try:
            os.chmod(temp_path, 0o600)
        except OSError:
            pass
        os.replace(temp_path, state_file)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _save_profile_updates(*, base_url: str | None = None, **updates: str) -> None:
    payload = _read_state_payload()
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}

    normalized_base_url = _normalized_base_url(base_url)
    profile = dict(_profile_from_payload(payload, base_url=base_url))
    for key, value in updates.items():
        normalized_value = str(value or "").strip()
        if not normalized_value:
            continue
        profile[key] = normalized_value

    profile["updated_at"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    if normalized_base_url:
        profile["base_url"] = normalized_base_url
    else:
        profile.pop("base_url", None)

    profiles[_profile_key(base_url)] = profile
    payload["version"] = STATE_VERSION
    payload["profiles"] = profiles
    _write_state_payload(payload)


def load_saved_privy_user_id(*, base_url: str | None = None) -> str | None:
    profile = _profile_from_payload(_read_state_payload(), base_url=base_url)
    privy_user_id = str(profile.get("privy_user_id") or "").strip()
    return privy_user_id or None


def load_saved_wallet_id(*, base_url: str | None = None) -> str | None:
    profile = _profile_from_payload(_read_state_payload(), base_url=base_url)
    wallet_id = str(profile.get("wallet_id") or "").strip()
    return wallet_id or None


def save_privy_user_id(privy_user_id: str, *, base_url: str | None = None) -> None:
    normalized_privy_user_id = str(privy_user_id or "").strip()
    if not normalized_privy_user_id:
        raise ValueError("privy_user_id must not be empty")

    _save_profile_updates(
        base_url=base_url,
        privy_user_id=normalized_privy_user_id,
    )


def save_wallet_id(wallet_id: str, *, base_url: str | None = None) -> None:
    normalized_wallet_id = str(wallet_id or "").strip()
    if not normalized_wallet_id:
        raise ValueError("wallet_id must not be empty")

    _save_profile_updates(
        base_url=base_url,
        wallet_id=normalized_wallet_id,
    )

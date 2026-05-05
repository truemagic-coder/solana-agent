"""Contract tests for the supported public SDK plugin surface."""

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "solana_agent" / "tools"

EXPECTED_PLUGIN_ENTRY_POINTS = {
    "mcp": "solana_agent.tools.mcp:get_plugin",
    "x402_request": "solana_agent.tools.x402_request:get_plugin",
}


def _load_plugin_table() -> dict[str, str]:
    pyproject = REPO_ROOT / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text())
    return config["tool"]["poetry"]["plugins"]["solana_agent.plugins"]


def test_poetry_plugin_table_matches_public_sdk_surface() -> None:
    assert _load_plugin_table() == EXPECTED_PLUGIN_ENTRY_POINTS


def test_supported_plugin_modules_exist_in_tools_directory() -> None:
    tool_modules = {
        path.stem for path in TOOLS_DIR.glob("*.py") if path.name != "__init__.py"
    }

    assert set(EXPECTED_PLUGIN_ENTRY_POINTS).issubset(tool_modules)


def test_registered_plugin_modules_exist() -> None:
    for target in EXPECTED_PLUGIN_ENTRY_POINTS.values():
        module_name, function_name = target.split(":", 1)
        module_path = REPO_ROOT / (module_name.replace(".", "/") + ".py")

        assert function_name == "get_plugin"
        assert module_path.exists(), module_path

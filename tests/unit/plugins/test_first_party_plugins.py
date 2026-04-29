"""Contract tests for the bundled first-party plugin surface."""

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "solana_agent" / "tools"

EXPECTED_PLUGIN_ENTRY_POINTS = {
    "birdeye": "solana_agent.tools.birdeye:get_plugin",
    "dflow_prediction": "solana_agent.tools.dflow_prediction:get_plugin",
    "image_gen": "solana_agent.tools.image_gen:get_plugin",
    "kamino": "solana_agent.tools.kamino:get_plugin",
    "jupiter_earn": "solana_agent.tools.jupiter_earn:get_plugin",
    "jupiter_holdings": "solana_agent.tools.jupiter_holdings:get_plugin",
    "jupiter_recurring": "solana_agent.tools.jupiter_recurring:get_plugin",
    "jupiter_shield": "solana_agent.tools.jupiter_shield:get_plugin",
    "jupiter_token_search": "solana_agent.tools.jupiter_token_search:get_plugin",
    "jupiter_trigger": "solana_agent.tools.jupiter_trigger:get_plugin",
    "mcp": "solana_agent.tools.mcp:get_plugin",
    "privy_create_user": "solana_agent.tools.privy_create_user:get_plugin",
    "privy_create_wallet": "solana_agent.tools.privy_create_wallet:get_plugin",
    "privy_dflow_prediction": "solana_agent.tools.privy_dflow_prediction:get_plugin",
    "privy_dflow_swap": "solana_agent.tools.privy_dflow_swap:get_plugin",
    "privy_earn": "solana_agent.tools.privy_earn:get_plugin",
    "privy_get_user_by_telegram": "solana_agent.tools.privy_get_user_by_telegram:get_plugin",
    "privy_kamino": "solana_agent.tools.privy_kamino:get_plugin",
    "privy_privacy_cash": "solana_agent.tools.privy_privacy_cash:get_plugin",
    "privy_recurring": "solana_agent.tools.privy_recurring:get_plugin",
    "privy_transfer": "solana_agent.tools.privy_transfer:get_plugin",
    "privy_trigger": "solana_agent.tools.privy_trigger:get_plugin",
    "privy_ultra": "solana_agent.tools.privy_ultra:get_plugin",
    "privy_ultra_quote": "solana_agent.tools.privy_ultra_quote:get_plugin",
    "privy_wallet_address": "solana_agent.tools.privy_wallet_address:get_plugin",
    "rugcheck": "solana_agent.tools.rugcheck:get_plugin",
    "search_internet": "solana_agent.tools.search_internet:get_plugin",
    "solana_dflow_swap": "solana_agent.tools.solana_dflow_swap:get_plugin",
    "solana_transfer": "solana_agent.tools.solana_transfer:get_plugin",
    "solana_ultra": "solana_agent.tools.solana_ultra:get_plugin",
    "solana_ultra_quote": "solana_agent.tools.solana_ultra_quote:get_plugin",
    "technical_analysis": "solana_agent.tools.technical_analysis:get_plugin",
    "token_math": "solana_agent.tools.token_math:get_plugin",
    "vybe": "solana_agent.tools.vybe:get_plugin",
    "x402_request": "solana_agent.tools.x402_request:get_plugin",
}


def _load_plugin_table() -> dict[str, str]:
    pyproject = REPO_ROOT / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text())
    return config["tool"]["poetry"]["plugins"]["solana_agent.plugins"]


def test_poetry_plugin_table_matches_first_party_surface() -> None:
    assert _load_plugin_table() == EXPECTED_PLUGIN_ENTRY_POINTS


def test_first_party_tool_modules_match_registered_plugins() -> None:
    tool_modules = {
        path.stem for path in TOOLS_DIR.glob("*.py") if path.name != "__init__.py"
    }

    assert tool_modules == set(EXPECTED_PLUGIN_ENTRY_POINTS)


def test_registered_plugin_modules_exist() -> None:
    for target in EXPECTED_PLUGIN_ENTRY_POINTS.values():
        module_name, function_name = target.split(":", 1)
        module_path = REPO_ROOT / (module_name.replace(".", "/") + ".py")

        assert function_name == "get_plugin"
        assert module_path.exists(), module_path
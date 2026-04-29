"""
Tests for OpenAI Strict Mode Compliance.

OpenAI strict mode requires:
1. All properties in the schema must be in the `required` array
2. `additionalProperties` must be `false`

This ensures the LLM always provides all expected parameters.
"""

import pytest


class TestOpenAIStrictCompliance:
    """Test all tools for OpenAI strict mode compliance."""

    def _check_schema_compliance(self, tool_class, tool_name: str):
        """Helper to check a tool's schema compliance."""
        tool = tool_class()
        schema = tool.get_schema()

        # Check additionalProperties is False
        assert schema.get("additionalProperties") is False, (
            f"{tool_name}: additionalProperties must be False for OpenAI strict mode"
        )

        # Check all properties are in required
        properties = set(schema.get("properties", {}).keys())
        required = set(schema.get("required", []))

        missing = properties - required
        assert not missing, f"{tool_name}: Properties not in required array: {missing}"

        extra = required - properties
        assert not extra, f"{tool_name}: Required fields not in properties: {extra}"

        return True

    def test_birdeye_compliance(self):
        """Test birdeye tool is OpenAI strict compliant."""
        from solana_agent.tools.birdeye import BirdeyeTool

        assert self._check_schema_compliance(BirdeyeTool, "birdeye")

    def test_dflow_prediction_compliance(self):
        """Test dflow_prediction tool is OpenAI strict compliant."""
        from solana_agent.tools.dflow_prediction import DFlowPredictionTool

        assert self._check_schema_compliance(DFlowPredictionTool, "dflow_prediction")

    def test_image_gen_compliance(self):
        """Test image_gen tool is OpenAI strict compliant."""
        from solana_agent.tools.image_gen import ImageGenTool

        assert self._check_schema_compliance(ImageGenTool, "image_gen")

    def test_jupiter_holdings_compliance(self):
        """Test jupiter_holdings tool is OpenAI strict compliant."""
        from solana_agent.tools.jupiter_holdings import JupiterHoldingsTool

        assert self._check_schema_compliance(JupiterHoldingsTool, "jupiter_holdings")

    def test_jupiter_recurring_compliance(self):
        """Test jupiter_recurring tool is OpenAI strict compliant."""
        from solana_agent.tools.jupiter_recurring import JupiterRecurringTool

        assert self._check_schema_compliance(JupiterRecurringTool, "jupiter_recurring")

    def test_jupiter_shield_compliance(self):
        """Test jupiter_shield tool is OpenAI strict compliant."""
        from solana_agent.tools.jupiter_shield import JupiterShieldTool

        assert self._check_schema_compliance(JupiterShieldTool, "jupiter_shield")

    def test_jupiter_token_search_compliance(self):
        """Test jupiter_token_search tool is OpenAI strict compliant."""
        from solana_agent.tools.jupiter_token_search import JupiterTokenSearchTool

        assert self._check_schema_compliance(
            JupiterTokenSearchTool, "jupiter_token_search"
        )

    def test_jupiter_trigger_compliance(self):
        """Test jupiter_trigger tool is OpenAI strict compliant."""
        from solana_agent.tools.jupiter_trigger import JupiterTriggerTool

        assert self._check_schema_compliance(JupiterTriggerTool, "jupiter_trigger")

    def test_mcp_compliance(self):
        """Test mcp tool is OpenAI strict compliant."""
        from solana_agent.tools.mcp import MCPTool

        assert self._check_schema_compliance(MCPTool, "mcp")

    def test_privy_create_user_compliance(self):
        """Test privy_create_user tool is OpenAI strict compliant."""
        from solana_agent.tools.privy_create_user import PrivyCreateUserTool

        assert self._check_schema_compliance(PrivyCreateUserTool, "privy_create_user")

    def test_privy_create_wallet_compliance(self):
        """Test privy_create_wallet tool is OpenAI strict compliant."""
        from solana_agent.tools.privy_create_wallet import PrivyCreateWalletTool

        assert self._check_schema_compliance(
            PrivyCreateWalletTool, "privy_create_wallet"
        )

    def test_privy_dflow_prediction_compliance(self):
        """Test privy_dflow_prediction tool is OpenAI strict compliant."""
        from solana_agent.tools.privy_dflow_prediction import PrivyDFlowPredictionTool

        assert self._check_schema_compliance(
            PrivyDFlowPredictionTool, "privy_dflow_prediction"
        )

    def test_privy_get_user_by_telegram_compliance(self):
        """Test privy_get_user_by_telegram tool is OpenAI strict compliant."""
        from solana_agent.tools.privy_get_user_by_telegram import (
            PrivyGetUserByTelegramTool,
        )

        assert self._check_schema_compliance(
            PrivyGetUserByTelegramTool, "privy_get_user_by_telegram"
        )

    def test_privy_recurring_compliance(self):
        """Test privy_recurring tool is OpenAI strict compliant."""
        from solana_agent.tools.privy_recurring import PrivyRecurringTool

        assert self._check_schema_compliance(PrivyRecurringTool, "privy_recurring")

    def test_privy_transfer_compliance(self):
        """Test privy_transfer tool is OpenAI strict compliant."""
        from solana_agent.tools.privy_transfer import PrivyTransferTool

        assert self._check_schema_compliance(PrivyTransferTool, "privy_transfer")

    def test_privy_trigger_compliance(self):
        """Test privy_trigger tool is OpenAI strict compliant."""
        from solana_agent.tools.privy_trigger import PrivyTriggerTool

        assert self._check_schema_compliance(PrivyTriggerTool, "privy_trigger")

    def test_privy_ultra_compliance(self):
        """Test privy_ultra tool is OpenAI strict compliant."""
        from solana_agent.tools.privy_ultra import PrivyUltraTool

        assert self._check_schema_compliance(PrivyUltraTool, "privy_ultra")

    def test_privy_wallet_address_compliance(self):
        """Test privy_wallet_address tool is OpenAI strict compliant."""
        from solana_agent.tools.privy_wallet_address import (
            PrivyWalletAddressCheckerTool,
        )

        assert self._check_schema_compliance(
            PrivyWalletAddressCheckerTool, "privy_wallet_address"
        )

    def test_rugcheck_compliance(self):
        """Test rugcheck tool is OpenAI strict compliant."""
        from solana_agent.tools.rugcheck import RugCheckTool

        assert self._check_schema_compliance(RugCheckTool, "rugcheck")

    def test_search_internet_compliance(self):
        """Test search_internet tool is OpenAI strict compliant."""
        from solana_agent.tools.search_internet import SearchInternetTool

        assert self._check_schema_compliance(SearchInternetTool, "search_internet")

    def test_solana_transfer_compliance(self):
        """Test solana_transfer tool is OpenAI strict compliant."""
        from solana_agent.tools.solana_transfer import SolanaTransferTool

        assert self._check_schema_compliance(SolanaTransferTool, "solana_transfer")

    def test_solana_ultra_compliance(self):
        """Test solana_ultra tool is OpenAI strict compliant."""
        from solana_agent.tools.solana_ultra import SolanaUltraTool

        assert self._check_schema_compliance(SolanaUltraTool, "solana_ultra")

    def test_technical_analysis_compliance(self):
        """Test technical_analysis tool is OpenAI strict compliant."""
        from solana_agent.tools.technical_analysis import TechnicalAnalysisTool

        assert self._check_schema_compliance(
            TechnicalAnalysisTool, "technical_analysis"
        )

    def test_vybe_compliance(self):
        """Test vybe tool is OpenAI strict compliant."""
        from solana_agent.tools.vybe import VybeTool

        assert self._check_schema_compliance(VybeTool, "vybe")


class TestSchemaDefaults:
    """Test that all required fields have sensible defaults in execute methods."""

    def test_birdeye_has_defaults_for_optional_params(self):
        """Birdeye should have defaults for optional params."""
        from solana_agent.tools.birdeye import BirdeyeTool
        import inspect

        tool = BirdeyeTool()
        sig = inspect.signature(tool.execute)

        # All params except 'self' should have defaults
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name == "action":
                # action is required and has no default
                continue
            assert param.default != inspect.Parameter.empty, (
                f"birdeye.execute param '{name}' should have a default value"
            )

    def test_vybe_has_defaults_for_optional_params(self):
        """Vybe should have defaults for optional params."""
        from solana_agent.tools.vybe import VybeTool
        import inspect

        tool = VybeTool()
        sig = inspect.signature(tool.execute)

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name == "addresses":
                # addresses is the main required param
                continue
            assert param.default != inspect.Parameter.empty, (
                f"vybe.execute param '{name}' should have a default value"
            )


class TestAllToolsHaveGetSchema:
    """Verify all tools implement get_schema method."""

    TOOL_CLASSES = [
        ("birdeye", "solana_agent.tools.birdeye", "BirdeyeTool"),
        ("image_gen", "solana_agent.tools.image_gen", "ImageGenTool"),
        (
            "jupiter_holdings",
            "solana_agent.tools.jupiter_holdings",
            "JupiterHoldingsTool",
        ),
        (
            "jupiter_recurring",
            "solana_agent.tools.jupiter_recurring",
            "JupiterRecurringTool",
        ),
        ("jupiter_shield", "solana_agent.tools.jupiter_shield", "JupiterShieldTool"),
        (
            "jupiter_token_search",
            "solana_agent.tools.jupiter_token_search",
            "JupiterTokenSearchTool",
        ),
        ("jupiter_trigger", "solana_agent.tools.jupiter_trigger", "JupiterTriggerTool"),
        ("mcp", "solana_agent.tools.mcp", "MCPTool"),
        ("privy_recurring", "solana_agent.tools.privy_recurring", "PrivyRecurringTool"),
        ("privy_transfer", "solana_agent.tools.privy_transfer", "PrivyTransferTool"),
        ("privy_trigger", "solana_agent.tools.privy_trigger", "PrivyTriggerTool"),
        ("privy_ultra", "solana_agent.tools.privy_ultra", "PrivyUltraTool"),
        (
            "privy_wallet_address",
            "solana_agent.tools.privy_wallet_address",
            "PrivyWalletAddressCheckerTool",
        ),
        ("rugcheck", "solana_agent.tools.rugcheck", "RugCheckTool"),
        ("search_internet", "solana_agent.tools.search_internet", "SearchInternetTool"),
        ("solana_transfer", "solana_agent.tools.solana_transfer", "SolanaTransferTool"),
        ("solana_ultra", "solana_agent.tools.solana_ultra", "SolanaUltraTool"),
        (
            "technical_analysis",
            "solana_agent.tools.technical_analysis",
            "TechnicalAnalysisTool",
        ),
        ("vybe", "solana_agent.tools.vybe", "VybeTool"),
    ]

    @pytest.mark.parametrize("tool_name,module_path,class_name", TOOL_CLASSES)
    def test_tool_has_get_schema(self, tool_name, module_path, class_name):
        """Each tool should have a get_schema method."""
        import importlib

        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)
        tool = tool_class()

        assert hasattr(tool, "get_schema"), f"{tool_name} should have get_schema method"
        schema = tool.get_schema()
        assert isinstance(schema, dict), (
            f"{tool_name}.get_schema() should return a dict"
        )
        assert "type" in schema, f"{tool_name} schema should have 'type' key"
        assert "properties" in schema, (
            f"{tool_name} schema should have 'properties' key"
        )

    @pytest.mark.parametrize("tool_name,module_path,class_name", TOOL_CLASSES)
    def test_tool_has_execute(self, tool_name, module_path, class_name):
        """Each tool should have an execute method."""
        import importlib

        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)
        tool = tool_class()

        assert hasattr(tool, "execute"), f"{tool_name} should have execute method"


class TestAllPluginsHaveGetPlugin:
    """Verify all plugins have get_plugin function."""

    PLUGINS = [
        "solana_agent.tools.birdeye",
        "solana_agent.tools.image_gen",
        "solana_agent.tools.jupiter_holdings",
        "solana_agent.tools.jupiter_recurring",
        "solana_agent.tools.jupiter_shield",
        "solana_agent.tools.jupiter_token_search",
        "solana_agent.tools.jupiter_trigger",
        "solana_agent.tools.mcp",
        "solana_agent.tools.privy_recurring",
        "solana_agent.tools.privy_transfer",
        "solana_agent.tools.privy_trigger",
        "solana_agent.tools.privy_ultra",
        "solana_agent.tools.privy_wallet_address",
        "solana_agent.tools.rugcheck",
        "solana_agent.tools.search_internet",
        "solana_agent.tools.solana_transfer",
        "solana_agent.tools.solana_ultra",
        "solana_agent.tools.technical_analysis",
        "solana_agent.tools.vybe",
    ]

    @pytest.mark.parametrize("module_path", PLUGINS)
    def test_plugin_has_get_plugin(self, module_path):
        """Each plugin module should have get_plugin function."""
        import importlib

        module = importlib.import_module(module_path)

        assert hasattr(module, "get_plugin"), (
            f"{module_path} should have get_plugin function"
        )
        plugin = module.get_plugin()
        assert plugin is not None, f"{module_path}.get_plugin() should not return None"

    @pytest.mark.parametrize("module_path", PLUGINS)
    def test_plugin_has_required_methods(self, module_path):
        """Each plugin should have initialize, configure, get_tools methods."""
        import importlib

        module = importlib.import_module(module_path)
        plugin = module.get_plugin()

        assert hasattr(plugin, "initialize"), (
            f"{module_path} plugin should have initialize method"
        )
        assert hasattr(plugin, "configure"), (
            f"{module_path} plugin should have configure method"
        )
        assert hasattr(plugin, "get_tools"), (
            f"{module_path} plugin should have get_tools method"
        )
        assert hasattr(plugin, "name"), (
            f"{module_path} plugin should have name attribute"
        )

"""
Tests for the AgentService class.

This module provides comprehensive test coverage for agent management,
tool execution, and response generation.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, call

from solana_agent.plugins.manager import PluginManager
from solana_agent.services.agent import AgentService
from solana_agent.domains.agent import BusinessMission
from solana_agent.interfaces.providers.llm import LLMProvider
from solana_agent.plugins.registry import ToolRegistry


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = AsyncMock(spec=LLMProvider)

    async def mock_generate():
        yield "Test response"

    async def mock_tts():
        yield b"audio data"

    async def mock_transcribe():
        yield "transcribed text"

    provider.generate_text.side_effect = mock_generate
    provider.tts.side_effect = mock_tts
    provider.transcribe_audio.side_effect = mock_transcribe
    return provider


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry."""
    registry = MagicMock(spec=ToolRegistry)
    registry.get_tool.return_value = None  # Default to no tool found
    registry.get_agent_tools.return_value = []
    registry.assign_tool_to_agent.return_value = True
    return registry


@pytest.fixture
def business_mission():
    """Create a sample business mission."""
    return BusinessMission(
        mission="Test mission",
        voice="Professional",
        values=[{"name": "Quality", "description": "High standards"}],
        goals=["Achieve excellence"],
    )


@pytest.fixture
def config():
    """Sample configuration."""
    return {"api_key": "test_key", "model": "test_model"}


class TestAgentService:
    """Test suite for AgentService."""

    def test_init_default(self, mock_llm_provider):
        """Test initialization with default values."""
        service = AgentService(llm_provider=mock_llm_provider)
        assert service.business_mission is None
        assert service.config == {}
        assert isinstance(service.tool_registry, ToolRegistry)
        assert service.agents == []

    def test_init_with_all_params(self, mock_llm_provider, business_mission, config):
        """Test initialization with all parameters."""
        service = AgentService(
            llm_provider=mock_llm_provider,
            business_mission=business_mission,
            config=config,
        )
        assert service.business_mission == business_mission
        assert service.config == config

    def test_agent_service_initialization(self):
        """Test that AgentService properly initializes all components."""
        mock_llm = AsyncMock(spec=LLMProvider)
        service = AgentService(llm_provider=mock_llm)

        assert service.llm_provider == mock_llm
        assert service.config == {}
        assert isinstance(service.tool_registry, ToolRegistry)
        assert isinstance(service.plugin_manager, PluginManager)
        assert service.plugin_manager.tool_registry == service.tool_registry
        assert service.plugin_manager.config == service.config

    def test_register_ai_agent(self, mock_llm_provider):
        """Test AI agent registration."""
        service = AgentService(llm_provider=mock_llm_provider)
        service.register_ai_agent(
            name="test_agent",
            instructions="Test instructions",
            specialization="Testing",
        )
        assert len(service.agents) == 1
        agent = service.agents[0]
        assert agent.name == "test_agent"
        assert agent.instructions == "Test instructions"
        assert agent.specialization == "Testing"

    def test_get_agent_system_prompt_basic(self, mock_llm_provider):
        """Test getting system prompt without business mission."""
        service = AgentService(llm_provider=mock_llm_provider)
        service.register_ai_agent("test_agent", "Test instructions", "Testing")
        prompt = service.get_agent_system_prompt("test_agent")
        assert "You are test_agent" in prompt
        assert "Test instructions" in prompt
        assert "current time" in prompt

    def test_get_agent_system_prompt_with_mission(
        self, mock_llm_provider, business_mission
    ):
        """Test getting system prompt with business mission."""
        service = AgentService(
            llm_provider=mock_llm_provider, business_mission=business_mission
        )
        service.register_ai_agent("test_agent", "Test instructions", "Testing")
        prompt = service.get_agent_system_prompt("test_agent")
        assert "BUSINESS MISSION" in prompt
        assert "VOICE OF THE BRAND" in prompt
        assert "BUSINESS VALUES" in prompt
        assert "BUSINESS GOALS" in prompt

    def test_get_all_ai_agents(self, mock_llm_provider):
        """Test retrieving all AI agents."""
        service = AgentService(llm_provider=mock_llm_provider)
        service.register_ai_agent("agent1", "Instructions 1", "Testing")
        service.register_ai_agent("agent2", "Instructions 2", "Development")

        agents = service.get_all_ai_agents()
        assert len(agents) == 2
        assert "agent1" in agents
        assert "agent2" in agents

    def test_assign_tool_for_agent(self, mock_llm_provider, mock_tool_registry):
        """Test tool assignment to agent."""
        # Create service with mock tool registry
        service = AgentService(llm_provider=mock_llm_provider)
        service.tool_registry = mock_tool_registry  # Explicitly set mock registry

        # Test tool assignment
        result = service.assign_tool_for_agent("test_agent", "test_tool")

        # Verify the result and interaction
        assert result is True
        mock_tool_registry.assign_tool_to_agent.assert_called_once_with(
            "test_agent", "test_tool"
        )

    def test_get_agent_tools(self, mock_llm_provider):
        """Test retrieving agent tools."""
        service = AgentService(llm_provider=mock_llm_provider)
        tools = service.get_agent_tools("test_agent")
        assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_execute_tool_no_registry(self, mock_llm_provider):
        """Test tool execution without registry."""
        service = AgentService(llm_provider=mock_llm_provider)
        service.tool_registry = None
        result = await service.execute_tool("agent", "tool", {})
        assert result["status"] == "error"
        assert "registry not available" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mock_llm_provider, mock_tool_registry):
        """Test execution of non-existent tool."""
        # Configure tool registry to return None for non-existent tool
        mock_tool_registry.get_tool.return_value = None

        service = AgentService(llm_provider=mock_llm_provider)
        service.tool_registry = mock_tool_registry

        result = await service.execute_tool("agent", "non_existent", {})

        # Verify error response
        assert result["status"] == "error"
        assert "not found" in result["message"]

        # Verify mock interactions
        mock_tool_registry.get_tool.assert_called_once_with("non_existent")

    @pytest.mark.asyncio
    async def test_execute_tool_no_access(self, mock_llm_provider, mock_tool_registry):
        """Test tool execution without access."""
        mock_tool_registry.get_tool.return_value = MagicMock()
        mock_tool_registry.get_agent_tools.return_value = []

        service = AgentService(llm_provider=mock_llm_provider)
        service.tool_registry = mock_tool_registry

        result = await service.execute_tool("agent", "tool", {})
        assert result["status"] == "error"
        assert "doesn't have access" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_llm_provider, mock_tool_registry):
        """Test successful tool execution."""
        mock_tool = AsyncMock()
        mock_tool.execute.return_value = {"status": "success", "result": "done"}
        mock_tool_registry.get_tool.return_value = mock_tool
        mock_tool_registry.get_agent_tools.return_value = [{"name": "test_tool"}]

        service = AgentService(llm_provider=mock_llm_provider)
        service.tool_registry = mock_tool_registry

        result = await service.execute_tool("agent", "test_tool", {"param": "value"})
        assert result["status"] == "success"
        assert result["result"] == "done"

    @pytest.mark.asyncio
    async def test_execute_tool_execution_error(
        self, mock_llm_provider, mock_tool_registry
    ):
        """Test handling of tool execution error."""
        # Create mock tool that raises an exception
        mock_tool = AsyncMock()
        mock_tool.execute.side_effect = Exception("Tool execution failed")

        # Configure tool registry to return the mock tool
        mock_tool_registry.get_tool.return_value = mock_tool
        mock_tool_registry.get_agent_tools.return_value = [{"name": "failing_tool"}]

        # Create service and set mock registry
        service = AgentService(llm_provider=mock_llm_provider)
        service.tool_registry = mock_tool_registry

        # Execute tool and verify error handling
        result = await service.execute_tool(
            "test_agent", "failing_tool", {"param": "value"}
        )

        # Verify error response
        assert result["status"] == "error"
        assert "Tool execution failed" in result["message"]

        # Verify mock interactions
        mock_tool.execute.assert_called_once_with(param="value")
        mock_tool_registry.get_tool.assert_called_once_with("failing_tool")
        mock_tool_registry.get_agent_tools.assert_called_once_with("test_agent")

    def test_load_and_register_plugins_success(self, mock_llm_provider):
        """Test successful loading of plugins."""
        service = AgentService(llm_provider=mock_llm_provider)
        # Mock the plugin manager directly on the instance
        service.plugin_manager = MagicMock(spec=PluginManager)

        service.load_and_register_plugins()

        service.plugin_manager.load_plugins.assert_called_once()

    def test_load_and_register_plugins_error(self, mock_llm_provider, caplog):
        """Test error handling during plugin loading."""
        service = AgentService(llm_provider=mock_llm_provider)
        # Mock the plugin manager to raise an error
        service.plugin_manager = MagicMock(spec=PluginManager)
        service.plugin_manager.load_plugins.side_effect = Exception(
            "Plugin load failed"
        )

        service.load_and_register_plugins()

        service.plugin_manager.load_plugins.assert_called_once()
        assert "Error loading plugins: Plugin load failed" in caplog.text

    # --- Tests for register_agents_from_config ---

    def test_register_agents_from_config_no_agents(self, mock_llm_provider, caplog):
        """Test registering agents when none are defined in config."""
        service = AgentService(
            llm_provider=mock_llm_provider, config={}
        )  # Empty config
        # Mock the methods that would be called if agents existed
        service.register_ai_agent = MagicMock()
        service.assign_tool_for_agent = MagicMock()

        service.register_agents_from_config()

        assert "No agents defined in the configuration" in caplog.text
        service.register_ai_agent.assert_not_called()
        service.assign_tool_for_agent.assert_not_called()

    def test_register_agents_from_config_valid_agents_no_tools(self, mock_llm_provider):
        """Test registering valid agents without tools."""
        config_with_agents = {
            "agents": [
                {"name": "agent1", "instructions": "instr1", "specialization": "spec1"},
                {"name": "agent2", "instructions": "instr2", "specialization": "spec2"},
            ]
        }
        service = AgentService(
            llm_provider=mock_llm_provider, config=config_with_agents
        )
        service.register_ai_agent = MagicMock()
        service.assign_tool_for_agent = MagicMock()

        service.register_agents_from_config()

        expected_calls = [
            call("agent1", "instr1", "spec1"),
            call("agent2", "instr2", "spec2"),
        ]
        service.register_ai_agent.assert_has_calls(expected_calls, any_order=True)
        assert service.register_ai_agent.call_count == 2
        service.assign_tool_for_agent.assert_not_called()  # No tools defined

    def test_register_agents_from_config_valid_agents_with_tools(
        self, mock_llm_provider
    ):
        """Test registering valid agents with tools."""
        config_with_agents_tools = {
            "agents": [
                {
                    "name": "agent1",
                    "instructions": "instr1",
                    "specialization": "spec1",
                    "tools": ["toolA", "toolB"],
                },
                {
                    "name": "agent2",
                    "instructions": "instr2",
                    "specialization": "spec2",
                    "tools": ["toolC"],
                },
            ]
        }
        service = AgentService(
            llm_provider=mock_llm_provider, config=config_with_agents_tools
        )
        service.register_ai_agent = MagicMock()
        # Mock assign_tool_for_agent to return True for successful assignment
        service.assign_tool_for_agent = MagicMock(return_value=True)

        service.register_agents_from_config()

        # Check agent registration calls
        expected_agent_calls = [
            call("agent1", "instr1", "spec1"),
            call("agent2", "instr2", "spec2"),
        ]
        service.register_ai_agent.assert_has_calls(expected_agent_calls, any_order=True)
        assert service.register_ai_agent.call_count == 2

        # Check tool assignment calls
        expected_tool_calls = [
            call("agent1", "toolA"),
            call("agent1", "toolB"),
            call("agent2", "toolC"),
        ]
        service.assign_tool_for_agent.assert_has_calls(
            expected_tool_calls, any_order=True
        )
        assert service.assign_tool_for_agent.call_count == 3

    def test_register_agents_from_config_invalid_agent_data(
        self, mock_llm_provider, caplog
    ):
        """Test skipping agents with missing required data."""
        config_with_invalid = {
            "agents": [
                {"name": "agent1", "instructions": "instr1"},  # Missing specialization
                {"name": "agent2", "specialization": "spec2"},  # Missing instructions
                {"instructions": "instr3", "specialization": "spec3"},  # Missing name
                {
                    "name": "agent4",
                    "instructions": "instr4",
                    "specialization": "spec4",
                },  # Valid
            ]
        }
        service = AgentService(
            llm_provider=mock_llm_provider, config=config_with_invalid
        )
        service.register_ai_agent = MagicMock()
        service.assign_tool_for_agent = MagicMock()

        service.register_agents_from_config()

        # Check that only the valid agent was registered
        service.register_ai_agent.assert_called_once_with("agent4", "instr4", "spec4")
        service.assign_tool_for_agent.assert_not_called()  # No tools on valid agent

        # Check warning logs for skipped agents
        assert (
            "Skipping agent due to missing name, instructions, or specialization"
            in caplog.text
        )
        assert caplog.text.count("Skipping agent") == 3  # Three invalid agents

    def test_register_agents_from_config_tool_assignment_failure(
        self, mock_llm_provider, caplog
    ):
        """Test logging when tool assignment fails."""
        config_with_tools = {
            "agents": [
                {
                    "name": "agent1",
                    "instructions": "instr1",
                    "specialization": "spec1",
                    "tools": ["toolA", "toolB"],
                },
            ]
        }
        service = AgentService(llm_provider=mock_llm_provider, config=config_with_tools)
        service.register_ai_agent = MagicMock()
        # Mock assign_tool_for_agent to return False for the second tool
        service.assign_tool_for_agent = MagicMock(side_effect=[True, False])

        service.register_agents_from_config()

        service.register_ai_agent.assert_called_once_with("agent1", "instr1", "spec1")

        # Check tool assignment calls
        expected_tool_calls = [
            call("agent1", "toolA"),
            call("agent1", "toolB"),
        ]
        service.assign_tool_for_agent.assert_has_calls(
            expected_tool_calls, any_order=False
        )  # Order matters for side_effect
        assert service.assign_tool_for_agent.call_count == 2

        # Check warning log for the failed assignment
        assert "Failed to assign tool 'toolB' to agent 'agent1'" in caplog.text
        assert "Tool might not be registered" in caplog.text

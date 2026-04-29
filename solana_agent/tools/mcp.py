import logging
import json
from typing import Dict, Any, List, Optional

from solana_agent import AutoTool, ToolRegistry
import urllib

try:
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    FASTMCP_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    FASTMCP_AVAILABLE = False
    logging.warning(f"fastmcp library not found: {e}")

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    OPENAI_AVAILABLE = False
    logging.warning(f"openai library not found: {e}")

logger = logging.getLogger(__name__)


class MCPTool(AutoTool):
    """
    Tool for interacting with MCP servers using fastmcp.
    Uses an OpenAI-compatible LLM (OpenAI or Grok) to select and call tools based on a natural language query.
    Supports multiple MCP servers with custom headers for each.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        # Initialize all instance attributes BEFORE calling super().__init__
        # This prevents attribute errors if the registry inspects the tool during registration
        self._servers: List[Dict[str, Any]] = []  # List of server configs
        self._llm_provider: str = "grok"  # Default to "grok", fallback to "openai"
        self._llm_api_key: Optional[str] = None
        self._llm_base_url: Optional[str] = None
        self._llm_model: Optional[str] = None

        super().__init__(
            name="mcp",
            description="Executes tasks using connected MCP servers (e.g., Zapier actions) via fastmcp. Provide a natural language query describing the task.",
            registry=registry,
        )
        logger.info("MCPTool (fastmcp) initialized.")

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language task or query to execute using connected MCP tools.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)

        # Configure MCP servers - support both single server (legacy) and multiple servers
        if "tools" in config and "mcp" in config["tools"]:
            mcp_config = config["tools"]["mcp"]

            # Check for multiple servers configuration
            if "servers" in mcp_config and isinstance(mcp_config["servers"], list):
                self._servers = []
                for server in mcp_config["servers"]:
                    server_entry = {
                        "url": server.get("url"),
                        "headers": server.get("headers", {}),
                    }
                    if server_entry["url"]:
                        self._servers.append(server_entry)
                        parsed = urllib.parse.urlparse(server_entry["url"])
                        root_domain = f"{parsed.scheme}://{parsed.hostname}"
                        logger.info(f"MCPTool: Configured server: {root_domain}")
            # Legacy single server configuration
            elif "url" in mcp_config:
                self._servers = [
                    {
                        "url": mcp_config.get("url"),
                        "headers": mcp_config.get("headers", {}),
                    }
                ]
                parsed = urllib.parse.urlparse(self._servers[0]["url"])
                root_domain = f"{parsed.scheme}://{parsed.hostname}"
                logger.info(
                    f"MCPTool: Configured with server root domain: {root_domain}"
                )

            # Get LLM provider configuration - default to grok, fallback to openai
            self._llm_provider = mcp_config.get("llm_provider", "grok")
            self._llm_model = mcp_config.get("llm_model")

            if not self._servers:
                logger.info("MCPTool: No MCP server URLs provided.")

        # Configure LLM settings - prioritize Grok, fallback to OpenAI
        # First, try to get Grok config
        grok_api_key = None
        grok_model = None
        if "grok" in config and isinstance(config["grok"], dict):
            grok_api_key = config["grok"].get("api_key")
            grok_model = config["grok"].get("model")

        # If Grok is configured (has API key), always use it
        if grok_api_key:
            self._llm_provider = "grok"
            self._llm_api_key = grok_api_key
            self._llm_base_url = "https://api.x.ai/v1"
            if not self._llm_model:
                self._llm_model = grok_model or "grok-4-1-fast-non-reasoning"
            logger.info(f"MCPTool: Using Grok with model {self._llm_model}")
        elif self._llm_provider == "grok":
            # Grok requested but no grok config - check mcp config for api_key
            if "tools" in config and "mcp" in config["tools"]:
                self._llm_api_key = config["tools"]["mcp"].get("api_key")
            self._llm_base_url = "https://api.x.ai/v1"
            if not self._llm_model:
                self._llm_model = "grok-4-1-fast"
            logger.info(f"MCPTool: Using Grok with model {self._llm_model}")
        else:  # pragma: no cover
            # Fallback to OpenAI
            self._llm_provider = "openai"
            if "openai" in config and isinstance(config["openai"], dict):
                self._llm_api_key = config["openai"].get("api_key")
                if not self._llm_model:
                    self._llm_model = config["openai"].get("model")
            elif "tools" in config and "mcp" in config["tools"]:
                self._llm_api_key = config["tools"]["mcp"].get("api_key")

            if not self._llm_model:
                self._llm_model = "gpt-4.1-mini"
            logger.info(f"MCPTool: Using OpenAI with model {self._llm_model}")

    async def execute(self, query: str) -> Dict[str, Any]:
        if not FASTMCP_AVAILABLE:  # pragma: no cover
            return {"status": "error", "message": "fastmcp library is not installed."}
        if not OPENAI_AVAILABLE:  # pragma: no cover
            return {"status": "error", "message": "openai library is not installed."}
        if not self._llm_api_key:
            return {
                "status": "error",
                "message": f"No API key configured for {self._llm_provider}.",
            }
        if not self._servers:
            return {"status": "error", "message": "No MCP servers configured."}

        # Collect all tools from all servers  # pragma: no cover
        all_tools = []
        server_clients = []

        for server_config in self._servers:
            try:
                # Create transport with custom headers if provided
                transport = StreamableHttpTransport(
                    server_config["url"], headers=server_config.get("headers", {})
                )
                client = Client(transport=transport)
                await client.__aenter__()

                tools = await client.list_tools()
                if tools:
                    all_tools.extend(tools)
                    server_clients.append(
                        {
                            "client": client,
                            "url": server_config["url"],
                            "tools": [t.name for t in tools],
                        }
                    )
                    logger.info(
                        f"Connected to MCP server: {server_config['url']} ({len(tools)} tools)"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to connect to MCP server {server_config['url']}: {e}"
                )

        if not all_tools:  # pragma: no cover
            # Clean up any opened clients
            for sc in server_clients:
                try:
                    await sc["client"].__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error closing MCP client during cleanup: {e}")
            return {
                "status": "error",
                "message": "No tools available on any MCP server.",
            }

        try:  # pragma: no cover
            # 2. Use LLM to select tool and generate parameters
            tool_descriptions = [
                {
                    "name": t.name,
                    "description": t.description,
                    "params": t.inputSchema,
                }
                for t in all_tools
            ]
            system_prompt = (
                "You are an expert AI agent. "
                "Given a user request and a list of available tools (with their parameters), "
                "choose the best tool and generate a valid parameter dictionary for it. "
                'Respond ONLY with a JSON object: {"tool": <tool_name>, "parameters": {<param_dict>}}. '
                'If you cannot find a suitable tool, respond with {"tool": null, "parameters": {}}.'
            )
            user_prompt = (
                f"User request: {query}\n\n"
                f"Available tools:\n{json.dumps(tool_descriptions, indent=2)}"
            )

            # Create OpenAI client with appropriate base URL for provider
            client_kwargs = {"api_key": self._llm_api_key}
            if self._llm_base_url:
                client_kwargs["base_url"] = self._llm_base_url

            openai_client = AsyncOpenAI(**client_kwargs)

            completion = await openai_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=512,
            )
            response = completion.choices[0].message.content.strip()
            try:
                llm_result = json.loads(response)
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {response}")
                return {
                    "status": "error",
                    "message": f"LLM output parse error: {e}",
                    "raw_llm_output": response,
                }

            tool_name = llm_result.get("tool")
            parameters = llm_result.get("parameters", {})

            if not tool_name:
                return {
                    "status": "error",
                    "message": "No suitable tool found for the query.",
                    "llm_output": response,
                }

            # 3. Find which server has this tool and call it
            target_client = None
            for sc in server_clients:
                if tool_name in sc["tools"]:
                    target_client = sc["client"]
                    break

            if not target_client:
                return {
                    "status": "error",
                    "message": f"Tool '{tool_name}' not found on any server.",
                    "tool": tool_name,
                }

            try:
                result = await target_client.call_tool(tool_name, parameters)
                # fastmcp returns a list of content objects; we assume first is main
                text_result = (
                    result[0].text
                    if result and hasattr(result[0], "text")
                    else str(result)
                )
                # Try to parse as JSON, fallback to string
                try:
                    parsed = json.loads(text_result)
                except Exception:
                    parsed = text_result

                return {
                    "status": "success",
                    "tool": tool_name,
                    "parameters": parameters,
                    "result": parsed,
                    "llm_provider": self._llm_provider,
                    "llm_model": self._llm_model,
                }
            except Exception as e:
                logger.exception(f"Error calling tool '{tool_name}': {e}")
                return {
                    "status": "error",
                    "message": f"Tool call failed: {e}",
                    "tool": tool_name,
                    "parameters": parameters,
                }
        finally:
            # Clean up all server connections
            for sc in server_clients:
                try:
                    await sc["client"].__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error closing MCP client: {e}")


class MCPPlugin:
    """Plugin for integrating MCP capabilities via fastmcp."""

    def __init__(self):
        self.name = "mcp"
        self.config = None
        self.tool_registry = None
        self._tool: Optional[MCPTool] = None
        logger.info(f"Created MCPPlugin object with name: {self.name}")

    @property
    def description(self):
        return "Plugin providing access to MCP servers (like Zapier) using fastmcp."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        if not FASTMCP_AVAILABLE:
            logger.warning("MCPPlugin: fastmcp library is not available.")
            return

        self.tool_registry = tool_registry
        logger.info("Initializing MCP plugin (fastmcp).")
        self._tool = MCPTool(registry=tool_registry)

        # Verification (only if tool_registry is provided)
        if tool_registry is not None:
            registered_tool = tool_registry.get_tool("mcp")
            if registered_tool and isinstance(registered_tool, MCPTool):
                logger.info("MCP tool registration verification: Success")
            else:
                logger.error("MCP tool registration verification: Failed or wrong type")

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        if not FASTMCP_AVAILABLE:
            logger.warning("MCPPlugin: fastmcp library is not available.")
            return

        self.config = config
        logger.info(f"Configuring {self.name} plugin")
        if self._tool:
            self._tool.configure(self.config)
            logger.info("MCP tool configured.")
        else:
            logger.warning("Warning: MCP tool instance not found during configuration.")

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        if not FASTMCP_AVAILABLE:
            return []
        if self._tool:
            return [self._tool]
        return []


def get_plugin():
    if not FASTMCP_AVAILABLE:  # pragma: no cover
        logger.warning(
            "MCPPlugin: Cannot create plugin instance, fastmcp library not available."
        )

        class DummyPlugin:
            name = "mcp (disabled)"
            description = "MCP plugin disabled (fastmcp library not found)"

            def initialize(self, *args, **kwargs):
                pass

            def configure(self, *args, **kwargs):
                pass

            def get_tools(self):
                return []

        return DummyPlugin()
    return MCPPlugin()

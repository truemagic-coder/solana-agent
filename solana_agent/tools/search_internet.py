import logging
from solana_agent import AutoTool, ToolRegistry
import httpx
from openai import AsyncOpenAI
from typing import Dict, Any, List

# --- Setup Logger ---
logger = logging.getLogger(__name__)
# Optional: Basic configuration if no other logging is set up by the framework
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SearchInternetTool(AutoTool):
    """Tool for searching the internet using Perplexity AI, OpenAI, or Grok."""

    def __init__(self, registry=None):
        """Initialize with auto-registration."""
        self._api_key = None
        self._model = ""
        self._citations = True
        self._provider = "openai"
        # Grok-specific settings
        self._grok_web_search = True
        self._grok_x_search = True
        self._grok_timeout = 90  # seconds
        super().__init__(
            name="search_internet",
            description="Search the internet for current information using Perplexity AI, OpenAI, or Grok.",
            registry=registry,
        )

        logger.debug("SearchInternetTool initialized.")  # Use debug for init

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure with all possible API key locations."""
        super().configure(config)

        model_set_from_config = False

        # Check tool-specific configuration first
        if "tools" in config and isinstance(config["tools"], dict):
            if "search_internet" in config["tools"] and isinstance(
                config["tools"]["search_internet"], dict
            ):
                tool_config = config["tools"]["search_internet"]

                # Check for API key in tool config
                if "api_key" in tool_config:
                    self._api_key = tool_config["api_key"]
                    logger.info(
                        f"Using API key from tools.search_internet.api_key: {self._api_key[:4]}..."
                    )

                # Check for provider setting in tool config
                if "provider" in tool_config:
                    self._provider = tool_config[
                        "provider"
                    ]  # Overwrite default if specified
                    logger.info(
                        f"Using provider from tools.search_internet.provider: {self._provider}"
                    )

                # Check for citations setting in tool config
                if "citations" in tool_config:
                    self._citations = bool(tool_config["citations"])
                    logger.info(f"Citations setting: {self._citations}")

                # Check for model setting in tool config
                if "model" in tool_config:
                    self._model = tool_config["model"]
                    model_set_from_config = True
                    logger.info(
                        f"Using model from tools.search_internet.model: {self._model}"
                    )

                # Grok-specific settings
                if "grok_web_search" in tool_config:
                    self._grok_web_search = bool(tool_config["grok_web_search"])
                    logger.info(f"Grok web_search: {self._grok_web_search}")
                if "grok_x_search" in tool_config:
                    self._grok_x_search = bool(tool_config["grok_x_search"])
                    logger.info(f"Grok x_search: {self._grok_x_search}")
                if "grok_timeout" in tool_config:
                    self._grok_timeout = int(tool_config["grok_timeout"])
                    logger.info(f"Grok timeout: {self._grok_timeout}s")

        # Set default model *only if* it wasn't set via config
        # This ensures self._provider exists (from __init__ or config override) before being accessed
        if not model_set_from_config:
            if self._provider == "perplexity":
                self._model = "sonar"
                logger.info(
                    f"Using default model for provider '{self._provider}': {self._model}"
                )
            elif self._provider == "openai":
                self._model = "gpt-4o-mini-search-preview"
                logger.info(
                    f"Using default model for provider '{self._provider}': {self._model}"
                )
            elif self._provider == "grok":
                self._model = "grok-4-1-fast-non-reasoning"
                logger.info(
                    f"Using default model for provider '{self._provider}': {self._model}"
                )
            else:  # pragma: no cover
                # Handle unknown provider case if necessary  # pragma: no cover
                logger.warning(
                    f"Unknown provider '{self._provider}', cannot set default model."
                )
                self._model = ""  # Fallback or raise error

    async def execute(self, query: str) -> Dict[str, Any]:  # pragma: no cover
        """Execute the search."""

        if not self._api_key:
            logger.error("API key not configured for SearchInternetTool.")
            return {"status": "error", "message": "API key not configured"}

        try:
            if self._provider == "perplexity":
                url = "https://api.perplexity.ai/chat/completions"

                # Choose appropriate prompt based on citations setting
                system_content = (
                    "You search the Internet for current information. Include detailed information with citations like [1], [2], etc."
                    if self._citations
                    else "You search the Internet for current information. Provide a comprehensive answer without citations or source references."
                )

                payload = {
                    "model": self._model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_content,
                        },
                        {"role": "user", "content": query},
                    ],
                }
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }

                # Use httpx for async requests
                async with httpx.AsyncClient(timeout=30.0) as client:
                    logger.debug(
                        f"Sending request to Perplexity: {url} with model {self._model}"
                    )
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        logger.debug("Perplexity request successful.")

                        # Only process citations if setting is enabled
                        if self._citations:
                            # Remove the existing Sources section if present
                            if "Sources:" in content:
                                content = content.split("Sources:")[0].strip()

                            # Get citations if available
                            links = []
                            if "citations" in data:
                                citations = data["citations"]
                                for i, citation in enumerate(citations, 1):
                                    url = (
                                        citation
                                        if isinstance(citation, str)
                                        else (
                                            citation["url"] if "url" in citation else ""
                                        )
                                    )
                                    if url:
                                        links.append(f"[{i}] {url}")

                            # Format the final content with properly numbered sources
                            if links:
                                formatted_content = (
                                    content + "\n\n**Sources:**\n" + "\n".join(links)
                                )
                            else:
                                formatted_content = content
                        else:
                            # No citation processing needed
                            formatted_content = content

                        return {
                            "status": "success",
                            "result": formatted_content,
                            "model_used": self._model,
                        }
                    else:
                        logger.error(
                            f"Perplexity API Error: {response.status_code} - {response.text}"
                        )
                        return {
                            "status": "error",
                            "message": f"Failed to search: {response.status_code}",
                            "details": response.text,
                        }
            elif self._provider == "grok":
                url = "https://api.x.ai/v1/responses"

                # Build the input messages
                input_messages = [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]

                # Build tools configuration based on settings
                tools = []
                if self._grok_web_search:
                    tools.append({"type": "web_search"})
                if self._grok_x_search:
                    tools.append({"type": "x_search"})

                # Ensure at least one search tool is enabled
                if not tools:
                    tools.append({"type": "web_search"})

                payload = {
                    "model": self._model,
                    "input": input_messages,
                    "tools": tools,
                }

                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }

                # Use httpx for async requests with configurable timeout
                async with httpx.AsyncClient(
                    timeout=float(self._grok_timeout)
                ) as client:
                    logger.debug(
                        f"Sending request to Grok Responses API: {url} with model {self._model}"
                    )
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        data = response.json()
                        logger.debug("Grok request successful.")

                        # Extract content from the response
                        # The output is an array - find the message item
                        content = ""
                        annotations = []
                        output_items = data.get("output", [])
                        for item in output_items:
                            if item.get("type") == "message":
                                # Get the text content from the message
                                content_list = item.get("content", [])
                                for content_item in content_list:
                                    if content_item.get("type") == "output_text":
                                        content = content_item.get("text", "")
                                        annotations = content_item.get(
                                            "annotations", []
                                        )
                                        break
                                break

                        # Only process citations if setting is enabled
                        if self._citations:
                            # Remove the existing Sources section if present
                            if "Sources:" in content:
                                content = content.split("Sources:")[0].strip()

                            # Get citations from annotations
                            links = []
                            seen_urls = set()
                            for annotation in annotations:
                                if annotation.get("type") == "url_citation":
                                    url = annotation.get("url", "")
                                    if url and url not in seen_urls:
                                        seen_urls.add(url)
                                        links.append(f"[{len(links) + 1}] {url}")

                            # Format the final content with properly numbered sources
                            if links:
                                formatted_content = (
                                    content + "\n\n**Sources:**\n" + "\n".join(links)
                                )
                            else:
                                formatted_content = content
                        else:
                            # No citation processing needed
                            formatted_content = content

                        return {
                            "status": "success",
                            "result": formatted_content,
                            "model_used": self._model,
                        }
                    else:
                        logger.error(
                            f"Grok API Error: {response.status_code} - {response.text}"
                        )
                        return {
                            "status": "error",
                            "message": f"Failed to search: {response.status_code}",
                            "details": response.text,
                        }
            elif self._provider == "openai":
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that searches the internet for current information.",
                    },
                    {"role": "user", "content": query},
                ]
                request_params = {
                    "messages": messages,
                    "model": self._model,
                }
                try:
                    client = AsyncOpenAI(api_key=self._api_key)
                    logger.debug(f"Sending request to OpenAI with model {self._model}")
                    response = await client.chat.completions.create(**request_params)
                    content = response.choices[0].message.content
                    logger.debug("OpenAI request successful.")
                    return {
                        "status": "success",
                        "result": content,
                        "model_used": self._model,
                    }
                except Exception as e:
                    logger.exception(
                        f"OpenAI API error: {str(e)}"
                    )  # Use exception for traceback
                    return {
                        "status": "error",
                        "message": "OpenAI API error",
                        "details": str(e),
                    }

        except Exception as e:
            logger.exception(
                f"Error in search execute: {str(e)}"
            )  # Use exception for traceback
            # traceback formatting is handled by logger.exception
            return {"status": "error", "message": f"Error: {str(e)}"}


class SearchInternetPlugin:
    """Plugin for Solana Agent."""

    def __init__(self):
        """Initialize the plugin."""
        self.name = "search_internet"
        self.config = None
        self.tool_registry = None
        self._tool = None
        logger.info(
            f"Created SearchInternetPlugin object with name: {self.name}"
        )  # Changed from SolanaPlugin

    @property
    def description(self):
        """Return the plugin description."""
        return "Plugin for searching the internet with Perplexity AI, OpenAI, or Grok"

    def initialize(self, tool_registry: ToolRegistry) -> None:
        """Initialize with tool registry from Solana Agent."""
        self.tool_registry = tool_registry
        logger.info("Initializing search_internet plugin")

        # Create and immediately register the tool
        self._tool = SearchInternetTool(registry=tool_registry)

        # Verification (only if tool_registry is provided)
        if tool_registry is not None:
            # AutoTool handles registration if registry is passed in __init__
            # This explicit registration is redundant but kept for backwards compatibility
            success = tool_registry.register_tool(self._tool)
            logger.info(f"Tool registration attempt result: {success}")

            # Check available tools in registry
            all_tools = tool_registry.list_all_tools()
            logger.info(f"All registered tools: {all_tools}")

            # Force a check to make sure it's registered
            registered_tool = tool_registry.get_tool("search_internet")
            logger.info(
                f"Tool registration verification: {'Success' if registered_tool else 'Failed'}"
            )

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        """Configure the plugin with the provided config."""
        self.config = config
        logger.info("Configuring search_internet plugin.")  # Added log

        # Configure the tool now that we have the config
        if self._tool:
            self._tool.configure(self.config)

            logger.info(
                f"SearchInternetTool configured with API key: {'Found' if self._tool._api_key else 'Not found'}"
            )
            logger.info(f"SearchInternetTool default model: {self._tool._model}")
        else:
            logger.warning(
                "SearchInternetTool instance not found during configuration."
            )  # Added warning

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        """Return the list of tools provided by this plugin."""
        if self._tool:
            logger.debug(f"Returning tool instance: {self._tool.name}")  # Use debug
            return [self._tool]
        logger.warning(
            "No SearchInternetTool instance to return in get_tools."
        )  # Added warning
        return []


# Entry point function
def get_plugin():  # pragma: no cover
    """Return plugin instance for registration."""
    logger.debug("get_plugin called for search_internet")  # Use debug
    return SearchInternetPlugin()

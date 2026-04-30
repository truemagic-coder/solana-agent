"""Generic x402 HTTP request tool for first-party Solana Agent usage."""

from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urlparse

from solana_agent import AutoTool, ToolRegistry

from solana_agent.tools.utils.x402 import (
    has_x402_auth_config,
    request_with_x402_private_key,
    request_with_x402_privy,
    resolve_x402_private_key,
)


class X402RequestTool(AutoTool):
    """Issue generic JSON-oriented x402 requests with Solana payment auth."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="x402_request",
            description=(
                "Make GET or POST requests to x402-protected HTTP endpoints using a Solana "
                "private key for payment settlement."
            ),
            registry=registry,
        )
        self.auth_mode = "x402_private_key"
        self.private_key = ""
        self.privy_app_id = ""
        self.privy_app_secret = ""
        self.privy_authorization_signature = ""
        self.privy_request_expiry = ""
        self.privy_api_url = ""
        self.x402_rpc_url = None
        self.allowed_hosts: list[str] = []
        self._runtime_context: Dict[str, Any] = {}

    def set_runtime_context(self, runtime_context: Optional[Dict[str, Any]]) -> None:
        self._runtime_context = dict(runtime_context or {})

    def clear_runtime_context(self) -> None:
        self._runtime_context = {}

    def _get_runtime_privy_wallet_id(self) -> str:
        return str(self._runtime_context.get("privy_wallet_id") or "").strip()

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "HTTP method.",
                },
                "url": {
                    "type": "string",
                    "description": "Absolute x402-protected URL to request.",
                },
                "query_params": {
                    "type": "object",
                    "description": "Optional query string parameters.",
                    "default": {},
                    "additionalProperties": True,
                },
                "headers": {
                    "type": "object",
                    "description": "Optional request headers.",
                    "default": {},
                    "additionalProperties": True,
                },
                "json_body": {
                    "type": "object",
                    "description": "Optional JSON body for POST requests.",
                    "default": {},
                    "additionalProperties": True,
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Request timeout in seconds.",
                    "default": 30.0,
                },
            },
            "required": ["method", "url"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        tool_config = config.get("tools", {}).get("x402_request", {})
        if isinstance(tool_config, dict):
            self.auth_mode = tool_config.get("auth_mode", "x402_private_key")
            self.private_key = tool_config.get("private_key", "")
            self.privy_app_id = tool_config.get("privy_app_id", "") or tool_config.get(
                "app_id", ""
            )
            self.privy_app_secret = tool_config.get(
                "privy_app_secret", ""
            ) or tool_config.get("app_secret", "")
            self.privy_authorization_signature = tool_config.get(
                "privy_authorization_signature", ""
            )
            self.privy_request_expiry = tool_config.get("privy_request_expiry", "")
            self.privy_api_url = tool_config.get("privy_api_url", "")
            self.x402_rpc_url = tool_config.get("x402_rpc_url")
            allowed_hosts = tool_config.get("allowed_hosts", [])
            if isinstance(allowed_hosts, list):
                self.allowed_hosts = [
                    str(host).lower() for host in allowed_hosts if host
                ]

    def _validate_request(self, method: str, url: str) -> Optional[Dict[str, Any]]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return {
                "success": False,
                "error": "url must be an absolute http or https URL",
            }

        host = parsed.hostname.lower() if parsed.hostname else ""
        if not self.allowed_hosts:
            return {
                "success": False,
                "error": "x402_request.allowed_hosts must be configured before this tool can be used.",
            }

        if host not in self.allowed_hosts:
            return {
                "success": False,
                "error": f"Host '{host}' is not permitted by x402_request.allowed_hosts.",
            }

        if method.upper() not in {"GET", "POST"}:
            return {"success": False, "error": "Only GET and POST are supported."}

        return None

    async def execute(
        self,
        method: str,
        url: str,
        query_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        validation_error = self._validate_request(method, url)
        if validation_error:
            return validation_error

        if self.auth_mode != "x402_private_key":
            if self.auth_mode != "x402_privy":
                return {
                    "success": False,
                    "error": f"Unsupported auth_mode: {self.auth_mode}",
                }

        runtime_privy_wallet_id = self._get_runtime_privy_wallet_id()

        if not has_x402_auth_config(
            auth_mode=self.auth_mode,
            private_key=self.private_key,
            privy_wallet_id=runtime_privy_wallet_id,
            privy_app_id=self.privy_app_id,
            privy_app_secret=self.privy_app_secret,
            privy_authorization_signature=self.privy_authorization_signature,
            privy_request_expiry=self.privy_request_expiry,
            privy_api_url=self.privy_api_url or None,
            rpc_url=self.x402_rpc_url,
        ):
            return {
                "success": False,
                "error": (
                    "x402 signing key not configured. Set x402_request.private_key for "
                    "x402_private_key mode or pass runtime_context.privy_wallet_id and set "
                    "x402_request.privy_app_id and x402_request.privy_app_secret for x402_privy mode."
                ),
            }

        resolved_private_key = resolve_x402_private_key(
            auth_mode=self.auth_mode,
            private_key=self.private_key,
        )

        if self.auth_mode == "x402_privy":
            response = await request_with_x402_privy(
                method=method,
                url=url,
                privy_wallet_id=runtime_privy_wallet_id,
                privy_app_id=self.privy_app_id,
                privy_app_secret=self.privy_app_secret,
                privy_authorization_signature=self.privy_authorization_signature
                or None,
                privy_request_expiry=self.privy_request_expiry or None,
                privy_api_url=self.privy_api_url or None,
                headers=headers or {},
                params=query_params or {},
                json_data=json_body,
                timeout=timeout_seconds,
                rpc_url=self.x402_rpc_url,
            )
        else:
            response = await request_with_x402_private_key(
                method=method,
                url=url,
                private_key=resolved_private_key,
                headers=headers or {},
                params=query_params or {},
                json_data=json_body,
                timeout=timeout_seconds,
                rpc_url=self.x402_rpc_url,
            )

        content_type = response.headers.get("content-type", "")
        try:
            body = response.json() if "json" in content_type else response.text
        except ValueError:
            body = response.text

        payment_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower().startswith("x-payment") or key.lower().startswith("x-402")
        }

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "data": body,
            "payment_mode": self.auth_mode,
            "payment_headers": payment_headers,
        }


class X402RequestPlugin:
    """Plugin for generic x402 HTTP requests."""

    def __init__(self):
        self.name = "x402_request"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for generic JSON-oriented x402 HTTP requests."

    def initialize(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._tool = X402RequestTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> list[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return X402RequestPlugin()

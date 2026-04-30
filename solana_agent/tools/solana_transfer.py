from typing import Dict, Any, List, Optional
from solana_agent import AutoTool, ToolRegistry
from solders.keypair import Keypair
from solana_agent.tools.utils.wallet import SolanaWalletClient
from solana_agent.tools.utils.transfer import TokenTransferManager

LAMPORTS_PER_SOL = 10**9


class SolanaTransferTool(AutoTool):
    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="solana_transfer",
            description="Transfer SOL or SPL tokens using Solana.",
            registry=registry,
        )
        self._rpc_url: Optional[str] = None
        self._private_key: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to_address": {
                    "type": "string",
                    "description": "recipient wallet address",
                },
                "amount": {
                    "type": "number",
                    "description": "amount to transfer (in SOL or token units)",
                },
                "mint": {
                    "type": "string",
                    "description": "token mint address",
                },
            },
            "required": ["to_address", "amount", "mint"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        self._rpc_url = (
            config.get("tools", {}).get("solana_transfer", {}).get("rpc_url")
        )
        self._private_key = (
            config.get("tools", {}).get("solana_transfer", {}).get("private_key")
        )

    async def execute(
        self, to_address: str, amount: float, mint: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self._rpc_url:
            return {"status": "error", "message": "RPC URL not configured."}
        if not self._private_key:
            return {"status": "error", "message": "Private key not configured."}
        keypair = Keypair.from_base58_string(self._private_key)
        wallet = SolanaWalletClient(self._rpc_url, keypair)
        provider = None
        if "helius" in self._rpc_url:  # pragma: no cover
            provider = "helius"
        try:
            transaction = await TokenTransferManager.transfer(
                wallet, to_address, amount, mint, provider
            )
            signature = await wallet.client.send_transaction(transaction)
            sig = signature.value
            return {"status": "success", "result": sig}
        except Exception as e:
            return {"status": "error", "message": str(e)}


class SolanaTransferPlugin:
    def __init__(self):
        self.name = "solana_transfer"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for transferring SOL or SPL tokens."

    def initialize(self, tool_registry: ToolRegistry) -> None:  # pragma: no cover
        self.tool_registry = tool_registry
        self._tool = SolanaTransferTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> List[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return SolanaTransferPlugin()

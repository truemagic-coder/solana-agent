import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch


def test_memory_module_uses_zep_message_type_when_dependency_present():
    repo_root = Path(__file__).resolve().parents[3]
    memory_path = repo_root / "solana_agent" / "repositories" / "memory.py"

    fake_zep_cloud = types.ModuleType("zep_cloud")
    fake_zep_client = types.ModuleType("zep_cloud.client")
    fake_zep_types = types.ModuleType("zep_cloud.types")

    class FakeAsyncZep:
        pass

    class FakeMessage:
        def __init__(self, content, role):
            self.content = content
            self.role = role

    fake_zep_cloud.client = fake_zep_client
    fake_zep_cloud.types = fake_zep_types
    fake_zep_client.AsyncZep = FakeAsyncZep
    fake_zep_types.Message = FakeMessage

    module_name = "test_memory_import_target"
    spec = importlib.util.spec_from_file_location(module_name, memory_path)
    module = importlib.util.module_from_spec(spec)

    assert spec is not None
    assert spec.loader is not None

    with patch.dict(
        sys.modules,
        {
            module_name: module,
            "zep_cloud": fake_zep_cloud,
            "zep_cloud.client": fake_zep_client,
            "zep_cloud.types": fake_zep_types,
        },
    ):
        spec.loader.exec_module(module)

    assert module.AsyncZepCloud is FakeAsyncZep
    assert module.Message is FakeMessage
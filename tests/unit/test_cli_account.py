import httpx
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from rich.console import Group
from rich.markdown import Markdown
from typer.testing import CliRunner

from solana_agent.cli import _resolve_wallet_smoke_options, app, stream_agent_response


runner = CliRunner()


class _FakeLive:
    def __init__(self, *args, **kwargs):
        del args
        del kwargs
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type
        del exc
        del tb
        return False

    def update(self, renderable, refresh=False):
        del refresh
        self.updates.append(renderable)


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.Confirm.ask", return_value=False)
@patch("solana_agent.cli.Prompt.ask", side_effect=["quit"])
@patch("solana_agent.cli.typer.prompt", return_value="You are concise.")
@patch("solana_agent.cli.SolanaAgent")
def test_chat_command_falls_back_to_hosted_defaults_when_config_missing(
    mock_solana_agent,
    mock_typer_prompt,
    mock_prompt_ask,
    mock_confirm_ask,
    mock_exists,
):
    del mock_exists
    del mock_confirm_ask
    del mock_prompt_ask
    del mock_typer_prompt
    mock_agent = MagicMock()
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(app, ["chat", "--config", "config.json"])

    assert result.exit_code == 0
    mock_solana_agent.assert_called_once_with(
        instructions="You are concise.",
        model="chat",
    )
    assert "Configuration file not found" in result.stdout
    assert "Agent initialized. Start chatting!" in result.stdout


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.Confirm.ask", return_value=False)
@patch("solana_agent.cli.Prompt.ask", side_effect=["hello", "quit"])
@patch("solana_agent.cli.typer.prompt", return_value="You are concise.")
@patch("solana_agent.cli.stream_agent_response", new_callable=AsyncMock)
@patch("solana_agent.cli.uuid4", return_value=SimpleNamespace(hex="session1234567890"))
@patch("solana_agent.cli.SolanaAgent")
def test_chat_command_runs_single_async_session_for_multiple_turns(
    mock_solana_agent,
    mock_uuid4,
    mock_stream_agent_response,
    mock_typer_prompt,
    mock_prompt_ask,
    mock_confirm_ask,
    mock_exists,
):
    del mock_prompt_ask
    del mock_exists
    del mock_confirm_ask
    del mock_typer_prompt
    del mock_uuid4
    mock_agent = MagicMock()
    mock_solana_agent.return_value = mock_agent
    original_asyncio_run = asyncio.run

    with patch(
        "solana_agent.cli.asyncio.run",
        side_effect=original_asyncio_run,
    ) as mock_asyncio_run:
        result = runner.invoke(app, ["chat", "--config", "config.json"])

    assert result.exit_code == 0
    assert mock_asyncio_run.call_count == 1
    mock_stream_agent_response.assert_awaited_once_with(
        mock_agent,
        "hello",
        None,
        search_enabled=False,
        conversation_id="cli-chat-session12345",
        model="memory",
        memory_ttl_tier="work",
    )


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.Confirm.ask", return_value=True)
@patch("solana_agent.cli.Prompt.ask", side_effect=["hello", "quit"])
@patch("solana_agent.cli.typer.prompt", return_value="You are concise.")
@patch("solana_agent.cli.stream_agent_response", new_callable=AsyncMock)
@patch("solana_agent.cli.uuid4", return_value=SimpleNamespace(hex="session1234567890"))
@patch("solana_agent.cli.SolanaAgent")
def test_chat_command_prompts_for_hosted_search_when_config_missing(
    mock_solana_agent,
    mock_uuid4,
    mock_stream_agent_response,
    mock_typer_prompt,
    mock_prompt_ask,
    mock_confirm_ask,
    mock_exists,
):
    del mock_prompt_ask
    del mock_exists
    del mock_typer_prompt
    del mock_uuid4
    mock_agent = MagicMock()
    mock_solana_agent.return_value = mock_agent
    original_asyncio_run = asyncio.run

    with patch(
        "solana_agent.cli.asyncio.run",
        side_effect=original_asyncio_run,
    ):
        result = runner.invoke(app, ["chat", "--config", "config.json"])

    assert result.exit_code == 0
    mock_confirm_ask.assert_called_once()
    mock_stream_agent_response.assert_awaited_once_with(
        mock_agent,
        "hello",
        None,
        search_enabled=True,
        conversation_id="cli-chat-session12345",
        model="memory",
        memory_ttl_tier="work",
    )


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.Confirm.ask")
@patch("solana_agent.cli.Prompt.ask", side_effect=["hello", "quit"])
@patch("solana_agent.cli.typer.prompt", return_value="You are concise.")
@patch("solana_agent.cli.stream_agent_response", new_callable=AsyncMock)
@patch("solana_agent.cli.uuid4", return_value=SimpleNamespace(hex="session1234567890"))
@patch("solana_agent.cli.SolanaAgent")
def test_chat_command_honors_explicit_search_flag_without_prompt(
    mock_solana_agent,
    mock_uuid4,
    mock_stream_agent_response,
    mock_typer_prompt,
    mock_prompt_ask,
    mock_confirm_ask,
    mock_exists,
):
    del mock_prompt_ask
    del mock_exists
    del mock_typer_prompt
    del mock_uuid4
    mock_agent = MagicMock()
    mock_solana_agent.return_value = mock_agent
    original_asyncio_run = asyncio.run

    with patch(
        "solana_agent.cli.asyncio.run",
        side_effect=original_asyncio_run,
    ):
        result = runner.invoke(
            app,
            ["chat", "--config", "config.json", "--search-enabled"],
        )

    assert result.exit_code == 0
    mock_confirm_ask.assert_not_called()
    mock_stream_agent_response.assert_awaited_once_with(
        mock_agent,
        "hello",
        None,
        search_enabled=True,
        conversation_id="cli-chat-session12345",
        model="memory",
        memory_ttl_tier="work",
    )


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.Confirm.ask", return_value=False)
@patch("solana_agent.cli.Prompt.ask", side_effect=["hello", "what is my name?", "quit"])
@patch("solana_agent.cli.typer.prompt", return_value="You are concise.")
@patch("solana_agent.cli.stream_agent_response", new_callable=AsyncMock)
@patch("solana_agent.cli.uuid4", return_value=SimpleNamespace(hex="session1234567890"))
@patch("solana_agent.cli.SolanaAgent")
def test_chat_command_reuses_same_memory_session_context_across_turns(
    mock_solana_agent,
    mock_uuid4,
    mock_stream_agent_response,
    mock_typer_prompt,
    mock_prompt_ask,
    mock_confirm_ask,
    mock_exists,
):
    del mock_prompt_ask
    del mock_exists
    del mock_confirm_ask
    del mock_typer_prompt
    del mock_uuid4
    mock_agent = MagicMock()
    mock_solana_agent.return_value = mock_agent
    original_asyncio_run = asyncio.run

    with patch(
        "solana_agent.cli.asyncio.run",
        side_effect=original_asyncio_run,
    ):
        result = runner.invoke(app, ["chat", "--config", "config.json"])

    assert result.exit_code == 0
    assert mock_stream_agent_response.await_count == 2
    first_call = mock_stream_agent_response.await_args_list[0]
    second_call = mock_stream_agent_response.await_args_list[1]
    assert first_call.args == (mock_agent, "hello", None)
    assert second_call.args == (mock_agent, "what is my name?", None)
    assert first_call.kwargs == {
        "search_enabled": False,
        "conversation_id": "cli-chat-session12345",
        "model": "memory",
        "memory_ttl_tier": "work",
    }
    assert second_call.kwargs == first_call.kwargs


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.Confirm.ask", return_value=False)
@patch(
    "solana_agent.cli.Prompt.ask",
    side_effect=["my name is Paul Hinton", "what is my name?", "quit"],
)
@patch("solana_agent.cli.typer.prompt", return_value="You are concise.")
@patch("solana_agent.cli.stream_agent_response", new_callable=AsyncMock)
@patch("solana_agent.cli.uuid4", return_value=SimpleNamespace(hex="session1234567890"))
@patch("solana_agent.cli.SolanaAgent")
def test_chat_command_fast_paths_same_session_name_recall(
    mock_solana_agent,
    mock_uuid4,
    mock_stream_agent_response,
    mock_typer_prompt,
    mock_prompt_ask,
    mock_confirm_ask,
    mock_exists,
):
    del mock_prompt_ask
    del mock_exists
    del mock_confirm_ask
    del mock_typer_prompt
    del mock_uuid4
    mock_agent = MagicMock()
    mock_solana_agent.return_value = mock_agent
    original_asyncio_run = asyncio.run

    with patch(
        "solana_agent.cli.asyncio.run",
        side_effect=original_asyncio_run,
    ):
        result = runner.invoke(app, ["chat", "--config", "config.json"])

    assert result.exit_code == 0
    mock_stream_agent_response.assert_awaited_once_with(
        mock_agent,
        "my name is Paul Hinton",
        None,
        search_enabled=False,
        conversation_id="cli-chat-session12345",
        model="memory",
        memory_ttl_tier="work",
    )
    assert "Your name is Paul Hinton." in result.stdout


@pytest.mark.asyncio
@patch("solana_agent.cli.Live", _FakeLive)
@patch("solana_agent.cli.console.print")
async def test_stream_agent_response_renders_markdown(mock_console_print):
    async def _chunks():
        yield "## Latest News\n\n"
        yield "- item one\n- item two"

    mock_agent = MagicMock()
    mock_agent.process.return_value = _chunks()

    await stream_agent_response(
        mock_agent,
        "What is the latest news?",
        search_enabled=True,
        conversation_id="cli-chat-session12345",
        model="memory",
        memory_ttl_tier="work",
    )

    mock_agent.process.assert_called_once_with(
        message="What is the latest news?",
        output_format="text",
        prompt=None,
        search_enabled=True,
        conversation_id="cli-chat-session12345",
        model="memory",
        memory_ttl_tier="work",
    )

    renderable = mock_console_print.call_args.args[0]

    assert isinstance(renderable, Group)
    assert any(isinstance(item, Markdown) for item in renderable.renderables)


@patch("solana_agent.cli.SolanaAgent")
def test_account_summary_command_passes_privy_runtime_context(
    mock_solana_agent,
):
    mock_agent = MagicMock()
    mock_agent.get_account_summary = AsyncMock(return_value={"wallet": "wallet-123"})
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(
        app,
        [
            "account",
            "summary",
            "--config",
            "config.json",
            "--privy-wallet-id",
            "wallet-123",
        ],
    )

    assert result.exit_code == 0
    mock_solana_agent.assert_called_once_with(config_path="config.json")
    mock_agent.get_account_summary.assert_awaited_once_with(
        privy_wallet_id="wallet-123"
    )
    assert '"wallet": "wallet-123"' in result.stdout


@patch("solana_agent.cli.SolanaAgent")
def test_account_usage_command_passes_query_options(
    mock_solana_agent,
):
    mock_agent = MagicMock()
    mock_agent.get_usage_report = AsyncMock(return_value={"buckets": []})
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(
        app,
        [
            "account",
            "usage",
            "--config",
            "config.json",
            "--granularity",
            "month",
            "--from-date",
            "2026-05-01",
            "--to-date",
            "2026-05-31",
            "--group-by",
            "conversation",
        ],
    )

    assert result.exit_code == 0
    mock_agent.get_usage_report.assert_awaited_once_with(
        "month",
        from_date="2026-05-01",
        to_date="2026-05-31",
        group_by="conversation",
    )
    assert '"buckets": []' in result.stdout


@patch("solana_agent.cli.SolanaAgent")
def test_wallet_create_command_calls_client(mock_solana_agent):
    mock_agent = MagicMock()
    mock_agent.create_wallet = AsyncMock(
        return_value={
            "privy_user_id": "did:privy:user123",
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
        }
    )
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(
        app,
        [
            "wallet",
            "create",
            "--config",
            "config.json",
            "--privy-user-id",
            "did:privy:user123",
        ],
    )

    assert result.exit_code == 0
    mock_solana_agent.assert_called_once_with(config_path="config.json")
    mock_agent.create_wallet.assert_awaited_once_with(
        privy_user_id="did:privy:user123",
        chain_type="solana",
    )
    assert '"wallet_id": "wallet-123"' in result.stdout


@patch("solana_agent.cli.SolanaAgent")
def test_wallet_user_command_calls_client(mock_solana_agent):
    mock_agent = MagicMock()
    mock_agent.create_privy_user = AsyncMock(
        return_value={
            "privy_user_id": "did:privy:user123",
            "created": True,
        }
    )
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(
        app,
        [
            "wallet",
            "user",
            "--config",
            "config.json",
        ],
    )

    assert result.exit_code == 0
    mock_solana_agent.assert_called_once_with(config_path="config.json")
    mock_agent.create_privy_user.assert_awaited_once_with()
    assert '"privy_user_id": "did:privy:user123"' in result.stdout


@patch("solana_agent.cli.SolanaAgent")
def test_wallet_address_command_calls_client(mock_solana_agent):
    mock_agent = MagicMock()
    mock_agent.get_wallet_address = AsyncMock(return_value="WalletPubkey123")
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(
        app,
        [
            "wallet",
            "address",
            "--config",
            "config.json",
            "--wallet-id",
            "wallet-123",
        ],
    )

    assert result.exit_code == 0
    mock_agent.get_wallet_address.assert_awaited_once_with(wallet_id="wallet-123")
    assert '"WalletPubkey123"' in result.stdout


@patch("solana_agent.cli.SolanaAgent")
def test_wallet_address_command_allows_configured_wallet(mock_solana_agent):
    mock_agent = MagicMock()
    mock_agent.get_wallet_address = AsyncMock(return_value="WalletPubkey123")
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(
        app,
        [
            "wallet",
            "address",
            "--config",
            "config.json",
        ],
    )

    assert result.exit_code == 0
    mock_agent.get_wallet_address.assert_awaited_once_with(wallet_id=None)
    assert '"WalletPubkey123"' in result.stdout


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.load_saved_privy_user_id")
@patch("solana_agent.cli.load_saved_wallet_id")
@patch("solana_agent.cli.save_privy_user_id")
@patch("solana_agent.cli.SolanaAgent")
def test_wallet_menu_can_create_privy_user(
    mock_solana_agent,
    mock_save_privy_user_id,
    mock_load_saved_wallet_id,
    mock_load_saved_privy_user_id,
    mock_exists,
):
    del mock_exists
    saved_state = {"privy_user_id": None}

    def _save_privy_user_id(privy_user_id, base_url=None):
        del base_url
        saved_state["privy_user_id"] = privy_user_id

    def _load_privy_user_id(base_url=None):
        del base_url
        return saved_state["privy_user_id"]

    mock_agent = MagicMock()
    mock_agent.create_privy_user = AsyncMock(
        return_value={"privy_user_id": "did:privy:user123", "created": True}
    )
    mock_agent._configured_base_url.return_value = None
    mock_agent._configured_privy_user_id.side_effect = ValueError("missing")
    mock_load_saved_wallet_id.return_value = None
    mock_save_privy_user_id.side_effect = _save_privy_user_id
    mock_load_saved_privy_user_id.side_effect = _load_privy_user_id
    mock_agent.create_wallet = AsyncMock(
        return_value={
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
        }
    )
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(app, ["wallet", "menu"], input="1\n2\n\nq\n")

    assert result.exit_code == 0
    mock_solana_agent.assert_called_once_with()
    mock_agent.create_privy_user.assert_awaited_once_with()
    mock_agent.create_wallet.assert_awaited_once_with(
        privy_user_id="did:privy:user123",
        chain_type="solana",
    )
    mock_save_privy_user_id.assert_any_call(
        "did:privy:user123",
        base_url=None,
    )
    assert '"privy_user_id": "did:privy:user123"' in result.stdout


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.SolanaAgent")
def test_wallet_menu_shows_wallet_address_for_prompted_user(
    mock_solana_agent,
    mock_exists,
):
    del mock_exists
    mock_agent = MagicMock()
    mock_agent._configured_privy_user_id.return_value = None
    mock_agent.create_wallet = AsyncMock(
        return_value={
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
        }
    )
    mock_agent.get_wallet_address = AsyncMock(return_value="WalletPubkey123")
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(
        app,
        ["wallet", "menu"],
        input="3\ndid:privy:user123\nq\n",
    )

    assert result.exit_code == 0
    mock_solana_agent.assert_called_once_with()
    mock_agent.create_wallet.assert_awaited_once_with(
        privy_user_id="did:privy:user123",
        chain_type="solana",
    )
    mock_agent.get_wallet_address.assert_not_awaited()
    assert '"WalletPubkey123"' in result.stdout
    assert '"wallet_id": "wallet-123"' not in result.stdout


def test_wallet_smoke_command_requires_dev():
    result = runner.invoke(app, ["wallet", "smoke"])

    assert result.exit_code == 1
    assert "--dev" in result.stdout


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.run_public_sdk_smoke")
@patch("solana_agent.cli.build_public_sdk_smoke_preview")
@patch("solana_agent.cli.SolanaAgent")
def test_wallet_smoke_command_runs_preview_and_live_smoke(
    mock_solana_agent,
    mock_build_preview,
    mock_run_smoke,
    mock_exists,
):
    del mock_exists
    mock_agent = MagicMock()
    mock_solana_agent.return_value = mock_agent
    mock_build_preview.return_value = {
        "ok": True,
        "preview_only": True,
        "estimate": {"suggested_wallet_funding_usdc": "1.00"},
        "steps": [],
    }
    mock_run_smoke.return_value = {
        "ok": True,
        "preview_only": False,
        "wallet": {
            "wallet_id": "wallet-123",
            "address": "WalletPubkey123",
        },
        "coverage": {
            "includes_search": True,
            "includes_rotate": False,
            "includes_export": False,
        },
        "estimate": {"suggested_wallet_funding_usdc": "1.00"},
        "steps": [{"name": "chat_message", "status": "passed"}],
    }

    result = runner.invoke(app, ["wallet", "smoke", "--dev", "--yes"])

    assert result.exit_code == 0
    mock_solana_agent.assert_called_once_with()
    mock_build_preview.assert_awaited_once_with(
        mock_agent,
        chain_type="solana",
        forecast_window_days=30,
        include_search=True,
        include_rotate=False,
        include_export=False,
        include_priority=False,
        include_memory=False,
        include_jupiter=False,
        include_birdeye=False,
        include_swap=False,
        include_trigger=False,
        include_earn=False,
        include_technical_analysis=False,
        include_token_math=False,
        include_transfer=False,
        transfer_recipient=None,
        transfer_amount_usdc=None,
    )
    mock_run_smoke.assert_awaited_once_with(
        mock_agent,
        chain_type="solana",
        forecast_window_days=30,
        include_search=True,
        include_rotate=False,
        include_export=False,
        include_priority=False,
        include_memory=False,
        include_jupiter=False,
        include_birdeye=False,
        include_swap=False,
        include_trigger=False,
        include_earn=False,
        include_technical_analysis=False,
        include_token_math=False,
        include_transfer=False,
        transfer_recipient=None,
        transfer_amount_usdc=None,
        preview=mock_build_preview.return_value,
    )
    assert "Smoke Result" in result.stdout
    assert "Suggested Funding (USDC)" in result.stdout


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.build_public_sdk_smoke_preview")
@patch("solana_agent.cli.SolanaAgent")
def test_wallet_smoke_command_supports_json_output(
    mock_solana_agent,
    mock_build_preview,
    mock_exists,
):
    del mock_exists
    mock_agent = MagicMock()
    mock_solana_agent.return_value = mock_agent
    mock_build_preview.return_value = {
        "ok": True,
        "preview_only": True,
        "estimate": {"suggested_wallet_funding_usdc": "1.00"},
        "steps": [],
    }

    result = runner.invoke(
        app,
        ["wallet", "smoke", "--dev", "--estimate-only", "--json"],
    )

    assert result.exit_code == 0
    mock_solana_agent.assert_called_once_with()
    mock_build_preview.assert_awaited_once_with(
        mock_agent,
        chain_type="solana",
        forecast_window_days=30,
        include_search=True,
        include_rotate=False,
        include_export=False,
        include_priority=False,
        include_memory=False,
        include_jupiter=False,
        include_birdeye=False,
        include_swap=False,
        include_trigger=False,
        include_earn=False,
        include_technical_analysis=False,
        include_token_math=False,
        include_transfer=False,
        transfer_recipient=None,
        transfer_amount_usdc=None,
    )
    assert '"preview_only": true' in result.stdout
    assert "Smoke Preview" not in result.stdout


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.run_public_sdk_smoke")
@patch("solana_agent.cli.build_public_sdk_smoke_preview")
@patch("solana_agent.cli.SolanaAgent")
def test_wallet_smoke_command_supports_big_profile_and_transfer(
    mock_solana_agent,
    mock_build_preview,
    mock_run_smoke,
    mock_exists,
):
    del mock_exists
    mock_agent = MagicMock()
    mock_solana_agent.return_value = mock_agent
    mock_build_preview.return_value = {
        "ok": True,
        "preview_only": True,
        "estimate": {"suggested_wallet_funding_usdc": "1.60"},
        "steps": [],
    }
    mock_run_smoke.return_value = {
        "ok": True,
        "preview_only": False,
        "estimate": {"suggested_wallet_funding_usdc": "1.60"},
        "steps": [],
    }

    result = runner.invoke(
        app,
        [
            "wallet",
            "smoke",
            "--dev",
            "--yes",
            "--big",
            "--include-transfer",
            "--transfer-recipient",
            "RecipientPubkey123",
            "--transfer-amount-usdc",
            "0.25",
        ],
    )

    assert result.exit_code == 0
    mock_build_preview.assert_awaited_once_with(
        mock_agent,
        chain_type="solana",
        forecast_window_days=30,
        include_search=True,
        include_rotate=False,
        include_export=False,
        include_priority=True,
        include_memory=True,
        include_jupiter=True,
        include_birdeye=True,
        include_swap=True,
        include_trigger=True,
        include_earn=True,
        include_technical_analysis=True,
        include_token_math=True,
        include_transfer=True,
        transfer_recipient="RecipientPubkey123",
        transfer_amount_usdc="0.25",
    )
    mock_run_smoke.assert_awaited_once_with(
        mock_agent,
        chain_type="solana",
        forecast_window_days=30,
        include_search=True,
        include_rotate=False,
        include_export=False,
        include_priority=True,
        include_memory=True,
        include_jupiter=True,
        include_birdeye=True,
        include_swap=True,
        include_trigger=True,
        include_earn=True,
        include_technical_analysis=True,
        include_token_math=True,
        include_transfer=True,
        transfer_recipient="RecipientPubkey123",
        transfer_amount_usdc="0.25",
        preview=mock_build_preview.return_value,
    )


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.run_public_sdk_smoke")
@patch("solana_agent.cli.build_public_sdk_smoke_preview")
@patch("solana_agent.cli.SolanaAgent")
def test_wallet_smoke_command_reports_runtime_failure(
    mock_solana_agent,
    mock_build_preview,
    mock_run_smoke,
    mock_exists,
):
    del mock_exists
    mock_agent = MagicMock()
    mock_solana_agent.return_value = mock_agent
    mock_build_preview.return_value = {
        "ok": True,
        "preview_only": True,
        "estimate": {"suggested_wallet_funding_usdc": "1.00"},
        "steps": [],
    }
    mock_run_smoke.side_effect = ValueError("hosted payment error 402")

    result = runner.invoke(app, ["wallet", "smoke", "--dev", "--yes"])

    assert result.exit_code == 1
    assert "Smoke test failed:" in result.stdout
    assert "hosted payment error 402" in result.stdout


def test_wallet_smoke_command_rejects_transfer_without_recipient():
    result = runner.invoke(
        app,
        ["wallet", "smoke", "--dev", "--include-transfer", "--estimate-only"],
    )

    assert result.exit_code == 2
    assert "Usage:" in result.output


def test_resolve_wallet_smoke_options_requires_transfer_recipient():
    with pytest.raises(typer.BadParameter, match="--transfer-recipient"):
        _resolve_wallet_smoke_options(
            big=False,
            include_search=True,
            include_rotate=False,
            include_export=False,
            include_priority=False,
            include_memory=False,
            include_jupiter=False,
            include_birdeye=False,
            include_swap=False,
            include_trigger=False,
            include_earn=False,
            include_technical_analysis=False,
            include_token_math=False,
            include_transfer=True,
            transfer_recipient=None,
            transfer_amount_usdc=None,
        )


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.wallet_smoke")
@patch("solana_agent.cli.SolanaAgent")
def test_wallet_menu_dev_can_run_smoke(
    mock_solana_agent,
    mock_wallet_smoke,
    mock_exists,
):
    del mock_exists
    mock_solana_agent.return_value = MagicMock()

    result = runner.invoke(
        app,
        ["wallet", "menu", "--dev"],
        input="6\nn\nn\nn\nn\nn\nn\nn\nn\nn\nn\nn\nn\nn\nn\nq\n",
    )

    assert result.exit_code == 0
    mock_solana_agent.assert_called_once_with()
    mock_wallet_smoke.assert_called_once_with(
        config="config.json",
        chain_type="solana",
        forecast_window_days=30,
        include_search=False,
        include_priority=False,
        include_memory=False,
        include_jupiter=False,
        include_birdeye=False,
        include_swap=False,
        include_trigger=False,
        include_earn=False,
        include_technical_analysis=False,
        include_token_math=False,
        include_rotate=False,
        include_export=False,
        include_transfer=False,
        transfer_recipient=None,
        transfer_amount_usdc=None,
        big=False,
        estimate_only=False,
        json_output=False,
        yes=False,
        dev=True,
    )


@patch("solana_agent.cli.SolanaAgent")
def test_wallet_export_command_calls_client(mock_solana_agent):
    mock_agent = MagicMock()
    mock_agent.export_wallet_private_key = AsyncMock(return_value="base58-private-key")
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(
        app,
        [
            "wallet",
            "export",
            "--config",
            "config.json",
            "--wallet-id",
            "wallet-123",
            "--privy-user-id",
            "did:privy:user123",
            "--yes",
        ],
    )

    assert result.exit_code == 0
    mock_agent.export_wallet_private_key.assert_awaited_once_with(
        wallet_id="wallet-123",
        privy_user_id="did:privy:user123",
        chain_type="solana",
    )
    assert '"base58-private-key"' in result.stdout


@patch("solana_agent.cli.SolanaAgent")
def test_wallet_export_command_surfaces_http_error_detail(mock_solana_agent):
    mock_agent = MagicMock()
    request = httpx.Request(
        "POST",
        "https://ai.solana-agent.com/v1/account/wallet/export",
    )
    response = httpx.Response(
        502,
        request=request,
        text='{"error":{"message":"Privy wallet export failed: upstream 403"}}',
    )
    mock_agent.export_wallet_private_key = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "bad gateway",
            request=request,
            response=response,
        )
    )
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(
        app,
        [
            "wallet",
            "export",
            "--privy-user-id",
            "did:privy:user123",
            "--wallet-id",
            "wallet-123",
            "--yes",
        ],
    )

    assert result.exit_code == 1
    assert "Privy wallet export failed" in result.stdout


@patch("solana_agent.cli.SolanaAgent")
def test_wallet_export_command_requires_confirmation(mock_solana_agent):
    mock_agent = MagicMock()
    mock_agent.export_wallet_private_key = AsyncMock(return_value="base58-private-key")
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(app, ["wallet", "export"], input="no\n")

    assert result.exit_code == 1
    mock_agent.export_wallet_private_key.assert_not_awaited()
    assert "Export cancelled" in result.stdout


@patch("solana_agent.cli.Path.exists", return_value=False)
@patch("solana_agent.cli.load_saved_wallet_id", return_value="wallet-saved")
@patch("solana_agent.cli.SolanaAgent")
def test_wallet_menu_export_defaults_to_saved_wallet_id(
    mock_solana_agent,
    mock_load_saved_wallet_id,
    mock_exists,
):
    del mock_exists
    mock_agent = MagicMock()
    mock_agent._configured_privy_user_id.return_value = "did:privy:user123"
    mock_agent._configured_base_url.return_value = None
    mock_agent.export_wallet_private_key = AsyncMock(return_value="base58-private-key")
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(
        app,
        ["wallet", "menu"],
        input="5\nEXPORT\n\n\nq\n",
    )

    assert result.exit_code == 0
    mock_load_saved_wallet_id.assert_called()
    mock_agent.export_wallet_private_key.assert_awaited_once_with(
        wallet_id="wallet-saved",
        privy_user_id="did:privy:user123",
        chain_type="solana",
    )

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from solana_agent.cli import app


runner = CliRunner()


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
        runtime_context={"privy_wallet_id": "wallet-123"}
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
        runtime_context=None,
    )
    assert '"buckets": []' in result.stdout

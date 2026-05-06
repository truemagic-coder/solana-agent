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
@patch("solana_agent.cli.SolanaAgent")
def test_wallet_menu_can_create_privy_user(mock_solana_agent, mock_exists):
    del mock_exists
    mock_agent = MagicMock()
    mock_agent.create_privy_user = AsyncMock(
        return_value={"privy_user_id": "did:privy:user123", "created": True}
    )
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(app, ["wallet", "menu"], input="1\nq\n")

    assert result.exit_code == 0
    mock_solana_agent.assert_called_once_with()
    mock_agent.create_privy_user.assert_awaited_once_with()
    assert '"privy_user_id": "did:privy:user123"' in result.stdout


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
def test_wallet_export_command_requires_confirmation(mock_solana_agent):
    mock_agent = MagicMock()
    mock_agent.export_wallet_private_key = AsyncMock(return_value="base58-private-key")
    mock_solana_agent.return_value = mock_agent

    result = runner.invoke(app, ["wallet", "export"], input="no\n")

    assert result.exit_code == 1
    mock_agent.export_wallet_private_key.assert_not_awaited()
    assert "Export cancelled" in result.stdout

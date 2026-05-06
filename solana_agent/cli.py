import json
from typing import Optional
import typer
import asyncio
import logging
from pathlib import Path
from typing_extensions import Annotated
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.prompt import Prompt

from solana_agent.client.solana_agent import SolanaAgent

# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
# --- End Logging Configuration ---

app = typer.Typer()
account_app = typer.Typer()
wallet_app = typer.Typer()
app.add_typer(account_app, name="account")
app.add_typer(wallet_app, name="wallet")
console = Console()


def _load_agent(config: str) -> SolanaAgent:
    try:
        return SolanaAgent(config_path=config)
    except FileNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] Configuration file not found at '{config}'"
        )
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[bold red]Error loading configuration:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(
            f"[bold red]An unexpected error occurred during initialization:[/bold red] {e}"
        )
        raise typer.Exit(code=1)


def _account_runtime_context(
    privy_wallet_id: Optional[str],
) -> dict[str, str]:
    wallet_id = str(privy_wallet_id or "").strip()
    if not wallet_id:
        return {}
    return {"privy_wallet_id": wallet_id}


def _load_agent_for_menu(config: str) -> SolanaAgent:
    if config and Path(config).exists():
        return _load_agent(config)
    try:
        return SolanaAgent()
    except ValueError as e:
        console.print(f"[bold red]Error loading hosted defaults:[/bold red] {e}")
        raise typer.Exit(code=1)


def _configured_privy_user_id(agent: SolanaAgent) -> Optional[str]:
    try:
        return agent._configured_privy_user_id()
    except Exception:
        return None


def _prompt_privy_user_id(agent: SolanaAgent) -> str:
    configured_user_id = _configured_privy_user_id(agent)
    if configured_user_id:
        return Prompt.ask("Privy user ID", default=configured_user_id)
    return Prompt.ask("Privy user ID")


def _print_json_payload(payload: object) -> None:
    console.print(json.dumps(payload, indent=2, sort_keys=True))


def _run_account_call(coro: object) -> None:
    try:
        payload = asyncio.run(coro)
    except Exception as e:
        console.print(f"[bold red]Account command failed:[/bold red] {e}")
        raise typer.Exit(code=1)
    _print_json_payload(payload)


async def stream_agent_response(
    agent: SolanaAgent,
    message: str,
    prompt: Optional[str] = None,
    search_enabled: bool = False,
):
    """Helper function to stream and display agent response."""
    full_response = ""
    with Live(console=console, refresh_per_second=10, transient=True) as live:
        live.update(Spinner("dots", "Thinking..."))
        try:
            first_chunk = True
            async for chunk in agent.process(
                message=message,
                output_format="text",
                prompt=prompt,  # Pass prompt override if provided
                search_enabled=search_enabled,
            ):
                if first_chunk:
                    live.update("", refresh=True)  # Clear spinner
                    first_chunk = False
                full_response += chunk
                live.update(full_response)

            if first_chunk:  # No response received
                live.update("[yellow]Agent did not produce a response.[/yellow]")

        except Exception as e:
            # Display error within the Live context
            live.update(f"[bold red]\nError during processing:[/bold red] {e}")
            # Keep the error message visible after Live exits by printing it again
            console.print(f"[bold red]Error during processing:[/bold red] {e}")
            full_response = ""  # Ensure error message isn't printed as final response

    # Print the final complete response cleanly after Live context exits
    if full_response:
        console.print(f"[bright_blue]Agent:[/bright_blue] {full_response}")


@app.command()
def chat(
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    prompt: Annotated[  # Allow prompt override via option
        str, typer.Option(help="Optional system prompt override for the session.")
    ] = None,
    search_enabled: Annotated[
        bool,
        typer.Option(
            "--search-enabled",
            help="Enable the hosted search add-on for each request in this chat session.",
        ),
    ] = False,
):
    """
    Start an interactive chat session with the Solana Agent.
    Type 'exit' or 'quit' to end the session.
    """
    with console.status("[bold green]Initializing agent...", spinner="dots"):
        agent = _load_agent(config)
    console.print("[green]Agent initialized. Start chatting![/green]")
    console.print("[dim]Type 'exit' or 'quit' to end.[/dim]")

    # --- Main Interaction Loop ---
    while True:
        try:
            # Use Rich's Prompt for better input handling
            user_message = Prompt.ask("[bold green]You[/bold green]")

            if user_message.lower() in ["exit", "quit"]:
                console.print("[yellow]Exiting chat session.[/yellow]")
                break

            if not user_message.strip():  # Handle empty input
                continue

            # Run the async streaming function for the user's message
            # Pass the optional prompt override from the command line option
            asyncio.run(
                stream_agent_response(
                    agent,
                    user_message,
                    prompt,
                    search_enabled=search_enabled,
                )
            )

        except KeyboardInterrupt:  # Allow Ctrl+C to exit gracefully
            console.print(
                "\n[yellow]Exiting chat session (KeyboardInterrupt).[/yellow]"
            )
            break
        except Exception as loop_error:
            # Catch errors during the input/processing loop without crashing
            console.print(
                f"[bold red]An error occurred in the chat loop:[/bold red] {loop_error}"
            )
            # Optionally add a small delay or specific error handling here
            # Consider if you want to break the loop on certain errors


@account_app.command("summary")
def account_summary(
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    privy_wallet_id: Annotated[
        Optional[str], typer.Option(help="Runtime Privy wallet ID.")
    ] = None,
):
    """Print wallet account summary as JSON."""
    agent = _load_agent(config)
    _run_account_call(
        agent.get_account_summary(**_account_runtime_context(privy_wallet_id))
    )


@account_app.command("usage")
def account_usage(
    granularity: Annotated[
        str, typer.Option(help="Usage granularity: day, month, or year.")
    ],
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    from_date: Annotated[
        Optional[str], typer.Option(help="Inclusive start date or timestamp.")
    ] = None,
    to_date: Annotated[
        Optional[str], typer.Option(help="Exclusive end date or timestamp.")
    ] = None,
    group_by: Annotated[
        Optional[str], typer.Option(help="Optional usage grouping field.")
    ] = None,
    privy_wallet_id: Annotated[
        Optional[str], typer.Option(help="Runtime Privy wallet ID.")
    ] = None,
):
    """Print wallet usage buckets as JSON."""
    agent = _load_agent(config)
    runtime_context = _account_runtime_context(privy_wallet_id)
    _run_account_call(
        agent.get_usage_report(
            granularity,
            from_date=from_date,
            to_date=to_date,
            group_by=group_by,
            **runtime_context,
        )
    )


@account_app.command("forecast")
def account_forecast(
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    window_days: Annotated[int, typer.Option(help="Forecast window in days.")] = 30,
    privy_wallet_id: Annotated[
        Optional[str], typer.Option(help="Runtime Privy wallet ID.")
    ] = None,
):
    """Print wallet usage forecast as JSON."""
    agent = _load_agent(config)
    runtime_context = _account_runtime_context(privy_wallet_id)
    _run_account_call(
        agent.get_usage_forecast(
            window_days=window_days,
            **runtime_context,
        )
    )


@account_app.command("pricing")
def account_pricing(
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    privy_wallet_id: Annotated[
        Optional[str], typer.Option(help="Runtime Privy wallet ID.")
    ] = None,
):
    """Print effective wallet pricing as JSON."""
    agent = _load_agent(config)
    _run_account_call(
        agent.get_pricing_info(**_account_runtime_context(privy_wallet_id))
    )


@wallet_app.command("create")
def wallet_create(
    privy_user_id: Annotated[str, typer.Option(help="Existing hosted Privy DID.")],
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    chain_type: Annotated[
        str,
        typer.Option(help="Wallet chain type. Public SDK defaults to solana."),
    ] = "solana",
):
    """Create or return the hosted wallet for a user."""
    agent = _load_agent(config)
    _run_account_call(
        agent.create_wallet(privy_user_id=privy_user_id, chain_type=chain_type)
    )


@wallet_app.command("user")
def wallet_user(
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
):
    """Create a hosted Privy user."""
    agent = _load_agent(config)
    _run_account_call(agent.create_privy_user())


@wallet_app.command("address")
def wallet_address(
    wallet_id: Annotated[
        Optional[str], typer.Option(help="Existing hosted wallet id.")
    ] = None,
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
):
    """Print the hosted wallet address."""
    agent = _load_agent(config)
    _run_account_call(agent.get_wallet_address(wallet_id=wallet_id))


@wallet_app.command("export")
def wallet_export(
    wallet_id: Annotated[
        Optional[str], typer.Option(help="Optional hosted wallet id to export.")
    ] = None,
    privy_user_id: Annotated[
        Optional[str], typer.Option(help="Existing hosted Privy DID.")
    ] = None,
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    chain_type: Annotated[
        str,
        typer.Option(help="Wallet chain type. Public SDK defaults to solana."),
    ] = "solana",
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Confirm that the private key should be printed to this terminal.",
        ),
    ] = False,
):
    """Export the hosted wallet private key for self-custody."""
    if not yes:
        confirmation = Prompt.ask("Type EXPORT to reveal the private key", default="")
        if confirmation.strip() != "EXPORT":
            console.print("[yellow]Export cancelled.[/yellow]")
            raise typer.Exit(code=1)

    agent = _load_agent(config)
    _run_account_call(
        agent.export_wallet_private_key(
            wallet_id=wallet_id,
            privy_user_id=privy_user_id,
            chain_type=chain_type,
        )
    )


@wallet_app.command("rotate")
def wallet_rotate(
    privy_user_id: Annotated[
        Optional[str], typer.Option(help="Existing hosted Privy DID.")
    ] = None,
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    chain_type: Annotated[
        str,
        typer.Option(help="Wallet chain type. Public SDK defaults to solana."),
    ] = "solana",
):
    """Rotate the hosted wallet for a Privy user."""
    agent = _load_agent(config)
    _run_account_call(
        agent.rotate_wallet(privy_user_id=privy_user_id, chain_type=chain_type)
    )


@wallet_app.command("menu")
def wallet_menu(
    config: Annotated[
        str,
        typer.Option(
            help="Optional configuration JSON file. Defaults are enough to create a Privy user."
        ),
    ] = "config.json",
    chain_type: Annotated[
        str,
        typer.Option(help="Wallet chain type. Public SDK defaults to solana."),
    ] = "solana",
):
    """Open the hosted wallet onboarding menu."""
    agent = _load_agent_for_menu(config)
    while True:
        console.print("\n[bold]Solana Agent Wallet Menu[/bold]")
        console.print("1. Create Privy user")
        console.print("2. Create or fetch wallet")
        console.print("3. Show wallet address")
        console.print("4. Rotate wallet")
        console.print("5. Export private key")
        console.print("q. Quit")
        choice = Prompt.ask("Choose an action", default="1")
        normalized_choice = choice.strip().lower()

        if normalized_choice in {"q", "quit", "exit"}:
            break
        if normalized_choice == "1":
            _run_account_call(agent.create_privy_user())
            continue
        if normalized_choice == "2":
            privy_user_id = _prompt_privy_user_id(agent)
            _run_account_call(
                agent.create_wallet(
                    privy_user_id=privy_user_id,
                    chain_type=chain_type,
                )
            )
            continue
        if normalized_choice == "3":
            configured_user_id = _configured_privy_user_id(agent)
            if configured_user_id:
                _run_account_call(agent.get_wallet_address())
            else:
                privy_user_id = _prompt_privy_user_id(agent)
                _run_account_call(
                    agent.create_wallet(
                        privy_user_id=privy_user_id,
                        chain_type=chain_type,
                    )
                )
            continue
        if normalized_choice == "4":
            privy_user_id = _prompt_privy_user_id(agent)
            _run_account_call(
                agent.rotate_wallet(
                    privy_user_id=privy_user_id,
                    chain_type=chain_type,
                )
            )
            continue
        if normalized_choice == "5":
            confirmation = Prompt.ask(
                "Type EXPORT to reveal the private key",
                default="",
            )
            if confirmation.strip() != "EXPORT":
                console.print("[yellow]Export cancelled.[/yellow]")
                continue
            privy_user_id = _prompt_privy_user_id(agent)
            wallet_id = Prompt.ask(
                "Wallet ID",
                default="",
            ).strip()
            _run_account_call(
                agent.export_wallet_private_key(
                    wallet_id=wallet_id or None,
                    privy_user_id=privy_user_id,
                    chain_type=chain_type,
                )
            )
            continue

        console.print("[yellow]Unknown menu option.[/yellow]")


if __name__ == "__main__":
    app()

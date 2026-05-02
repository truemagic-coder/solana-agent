import json
from typing import Optional
import typer
import asyncio
import logging
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
app.add_typer(account_app, name="account")
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
) -> Optional[dict[str, str]]:
    wallet_id = str(privy_wallet_id or "").strip()
    if not wallet_id:
        return None
    return {"privy_wallet_id": wallet_id}


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
    user_id: str,
    message: str,
    prompt: Optional[str] = None,
):
    """Helper function to stream and display agent response."""
    full_response = ""
    with Live(console=console, refresh_per_second=10, transient=True) as live:
        live.update(Spinner("dots", "Thinking..."))
        try:
            first_chunk = True
            async for chunk in agent.process(
                user_id=user_id,
                message=message,
                output_format="text",
                prompt=prompt,  # Pass prompt override if provided
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
    user_id: Annotated[
        str, typer.Option(help="The user ID for the conversation.")
    ] = "cli_user",
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    prompt: Annotated[  # Allow prompt override via option
        str, typer.Option(help="Optional system prompt override for the session.")
    ] = None,
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
            asyncio.run(stream_agent_response(agent, user_id, user_message, prompt))

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
        Optional[str], typer.Option(help="Runtime Privy wallet ID for x402_privy mode.")
    ] = None,
):
    """Print wallet account summary as JSON."""
    agent = _load_agent(config)
    _run_account_call(
        agent.get_account_summary(
            runtime_context=_account_runtime_context(privy_wallet_id)
        )
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
        Optional[str], typer.Option(help="Runtime Privy wallet ID for x402_privy mode.")
    ] = None,
):
    """Print wallet usage buckets as JSON."""
    agent = _load_agent(config)
    _run_account_call(
        agent.get_usage_report(
            granularity,
            from_date=from_date,
            to_date=to_date,
            group_by=group_by,
            runtime_context=_account_runtime_context(privy_wallet_id),
        )
    )


@account_app.command("forecast")
def account_forecast(
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    window_days: Annotated[int, typer.Option(help="Forecast window in days.")] = 30,
    privy_wallet_id: Annotated[
        Optional[str], typer.Option(help="Runtime Privy wallet ID for x402_privy mode.")
    ] = None,
):
    """Print wallet usage forecast as JSON."""
    agent = _load_agent(config)
    _run_account_call(
        agent.get_usage_forecast(
            window_days=window_days,
            runtime_context=_account_runtime_context(privy_wallet_id),
        )
    )


@account_app.command("pricing")
def account_pricing(
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    privy_wallet_id: Annotated[
        Optional[str], typer.Option(help="Runtime Privy wallet ID for x402_privy mode.")
    ] = None,
):
    """Print effective wallet pricing as JSON."""
    agent = _load_agent(config)
    _run_account_call(
        agent.get_pricing_info(
            runtime_context=_account_runtime_context(privy_wallet_id)
        )
    )


if __name__ == "__main__":
    app()

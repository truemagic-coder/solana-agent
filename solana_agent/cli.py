import json
from typing import Optional
import typer
import asyncio
import logging
from pathlib import Path
import httpx
from typing_extensions import Annotated
from rich import box
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.prompt import Confirm, Prompt
from rich.table import Table

from solana_agent.client.solana_agent import SolanaAgent
from solana_agent.local_state import (
    load_saved_privy_user_id,
    load_saved_wallet_id,
    save_privy_user_id,
)
from solana_agent.smoke import (
    build_public_sdk_smoke_preview,
    run_public_sdk_smoke,
)

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


def _agent_base_url(agent: SolanaAgent) -> Optional[str]:
    resolver = getattr(agent, "_configured_base_url", None)
    if not callable(resolver):
        return None
    try:
        value = str(resolver() or "").strip()
    except Exception:
        return None
    return value or None


def _apply_privy_user_id_to_agent(
    agent: SolanaAgent,
    privy_user_id: Optional[str],
) -> Optional[str]:
    normalized_privy_user_id = str(privy_user_id or "").strip()
    if not normalized_privy_user_id:
        return None

    setter = getattr(agent, "_set_configured_privy_user_id", None)
    if callable(setter):
        try:
            setter(normalized_privy_user_id)
        except Exception:
            pass
    return normalized_privy_user_id


def _remember_privy_user_id(agent: SolanaAgent, privy_user_id: Optional[str]) -> None:
    normalized_privy_user_id = _apply_privy_user_id_to_agent(agent, privy_user_id)
    if not normalized_privy_user_id:
        return

    try:
        save_privy_user_id(
            normalized_privy_user_id,
            base_url=_agent_base_url(agent),
        )
    except OSError as exc:
        console.print(
            f"[yellow]Warning:[/yellow] could not save privy_user_id locally: {exc}"
        )


def _configured_privy_user_id(agent: SolanaAgent) -> Optional[str]:
    try:
        configured_privy_user_id = str(agent._configured_privy_user_id() or "").strip()
    except Exception:
        configured_privy_user_id = ""

    if configured_privy_user_id:
        return configured_privy_user_id

    try:
        saved_privy_user_id = load_saved_privy_user_id(base_url=_agent_base_url(agent))
    except Exception:
        return None
    if saved_privy_user_id:
        _apply_privy_user_id_to_agent(agent, saved_privy_user_id)
    return saved_privy_user_id or None


def _prompt_privy_user_id(agent: SolanaAgent) -> str:
    configured_user_id = _configured_privy_user_id(agent)
    if configured_user_id:
        privy_user_id = Prompt.ask("Privy user ID", default=configured_user_id)
    else:
        privy_user_id = Prompt.ask("Privy user ID")

    _remember_privy_user_id(agent, privy_user_id)
    return str(privy_user_id or "").strip()


def _saved_wallet_id(agent: SolanaAgent) -> Optional[str]:
    try:
        wallet_id = load_saved_wallet_id(base_url=_agent_base_url(agent))
    except Exception:
        return None
    normalized_wallet_id = str(wallet_id or "").strip()
    return normalized_wallet_id or None


def _print_json_payload(payload: object) -> None:
    console.print(json.dumps(payload, indent=2, sort_keys=True))


def _bool_label(value: object) -> str:
    return "yes" if bool(value) else "no"


def _smoke_step_detail(step: dict[str, object]) -> str:
    if "response_excerpt" in step:
        return str(step.get("response_excerpt") or "")
    if step.get("name") == "resolve_privy_user":
        created = "created" if step.get("created") else "existing"
        return f"{step.get('privy_user_id') or ''} ({created})".strip()
    if "wallet_id" in step and "address" in step:
        return (f"{step.get('wallet_id') or ''} {step.get('address') or ''}").strip()
    if "address" in step:
        return str(step.get("address") or "")
    if "projected_month_end_spend_usd" in step:
        return f"month-end spend ${step.get('projected_month_end_spend_usd') or '0'}"
    if "month_spend_usd" in step:
        return f"month spend ${step.get('month_spend_usd') or '0'}"
    if "bucket_count" in step:
        return f"{step.get('bucket_count') or 0} buckets"
    if "private_key_redacted" in step:
        return f"redacted ({step.get('private_key_length') or 0} chars)"
    return ""


def _print_smoke_report(payload: dict[str, object]) -> None:
    preview_only = bool(payload.get("preview_only"))
    estimate = payload.get("estimate")
    wallet = payload.get("wallet")
    coverage = payload.get("coverage")

    summary_table = Table(
        title="Smoke Preview" if preview_only else "Smoke Result",
        box=box.ASCII,
        show_header=True,
        header_style="bold",
    )
    summary_table.add_column("Field")
    summary_table.add_column("Value")
    summary_table.add_row("Status", "passed" if payload.get("ok") else "failed")
    summary_table.add_row(
        "Mode",
        "preview only" if preview_only else "live run",
    )
    summary_table.add_row(
        "Privy User",
        str(payload.get("privy_user_id") or ""),
    )
    if isinstance(wallet, dict):
        summary_table.add_row(
            "Wallet ID",
            str(wallet.get("wallet_id") or ""),
        )
        summary_table.add_row(
            "Wallet Address",
            str(wallet.get("address") or ""),
        )
    if isinstance(coverage, dict):
        summary_table.add_row(
            "Search Check",
            _bool_label(coverage.get("includes_search")),
        )
        summary_table.add_row(
            "Rotate Check",
            _bool_label(coverage.get("includes_rotate")),
        )
        summary_table.add_row(
            "Export Check",
            _bool_label(coverage.get("includes_export")),
        )
    if isinstance(estimate, dict):
        summary_table.add_row(
            "Spend Ceiling (USD)",
            str(estimate.get("estimated_smoke_spend_ceiling_usd") or "0"),
        )
        summary_table.add_row(
            "Suggested Funding (USDC)",
            str(estimate.get("suggested_wallet_funding_usdc") or "0"),
        )
    console.print(summary_table)

    if isinstance(estimate, dict):
        components = estimate.get("components")
        if isinstance(components, dict):
            estimate_table = Table(
                title="Funding Estimate",
                box=box.ASCII,
                show_header=True,
                header_style="bold",
            )
            estimate_table.add_column("Component")
            estimate_table.add_column("Value")
            for key, label in (
                ("standard_chat_request_usd", "Standard Chat"),
                ("search_chat_request_usd", "Search Chat"),
                ("search_surcharge_usd", "Search Surcharge"),
                ("search_provider_cost_ceiling_usd", "Search Provider Ceiling"),
                ("funding_buffer_usdc", "Funding Buffer"),
            ):
                estimate_table.add_row(label, str(components.get(key) or "0"))
            console.print(estimate_table)

    steps = payload.get("steps")
    if isinstance(steps, list) and steps:
        steps_table = Table(
            title="Smoke Steps",
            box=box.ASCII,
            show_header=True,
            header_style="bold",
        )
        steps_table.add_column("Step")
        steps_table.add_column("Status")
        steps_table.add_column("Detail")
        for raw_step in steps:
            if not isinstance(raw_step, dict):
                continue
            steps_table.add_row(
                str(raw_step.get("name") or ""),
                str(raw_step.get("status") or ""),
                _smoke_step_detail(raw_step),
            )
        console.print(steps_table)


def _print_smoke_output(payload: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        _print_json_payload(payload)
        return
    _print_smoke_report(payload)


def _wallet_id_from_payload(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("wallet_id") or payload.get("id") or "").strip()


def _wallet_address_from_payload(payload: object) -> str:
    if isinstance(payload, str):
        address = payload.strip()
    elif isinstance(payload, dict):
        address = str(
            payload.get("address")
            or payload.get("public_address")
            or payload.get("wallet_address")
            or payload.get("public_key")
            or ""
        ).strip()
    else:
        address = ""

    if not address:
        raise ValueError("wallet lookup did not return an address")
    return address


async def _wallet_address_for_user(
    agent: SolanaAgent,
    *,
    privy_user_id: str,
    chain_type: str,
) -> str:
    payload = await agent.create_wallet(
        privy_user_id=privy_user_id,
        chain_type=chain_type,
    )
    return _wallet_address_from_payload(payload)


def _run_account_call(coro: object) -> object:
    try:
        payload = asyncio.run(coro)
    except httpx.HTTPStatusError as e:
        detail = ""
        response = getattr(e, "response", None)
        if response is not None:
            try:
                detail = str(response.text or "").strip()
            except Exception:
                detail = ""
        message = detail or str(e)
        console.print(f"[bold red]Account command failed:[/bold red] {message}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Account command failed:[/bold red] {e}")
        raise typer.Exit(code=1)
    _print_json_payload(payload)
    return payload


def _require_dev_mode(dev: bool) -> None:
    if dev:
        return
    console.print(
        "[bold yellow]Live smoke tests are dev-only.[/bold yellow] Re-run with [bold]--dev[/bold]."
    )
    raise typer.Exit(code=1)


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
    _remember_privy_user_id(agent, privy_user_id)
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
    payload = _run_account_call(agent.create_privy_user())
    if isinstance(payload, dict):
        _remember_privy_user_id(agent, payload.get("privy_user_id"))


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
    _remember_privy_user_id(agent, privy_user_id)
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
    _remember_privy_user_id(agent, privy_user_id)
    _run_account_call(
        agent.rotate_wallet(privy_user_id=privy_user_id, chain_type=chain_type)
    )


@wallet_app.command("smoke")
def wallet_smoke(
    config: Annotated[
        str,
        typer.Option(
            help="Optional configuration JSON file. Defaults are enough to run against the hosted service."
        ),
    ] = "config.json",
    chain_type: Annotated[
        str,
        typer.Option(help="Wallet chain type. Public SDK defaults to solana."),
    ] = "solana",
    forecast_window_days: Annotated[
        int,
        typer.Option(help="Forecast window in days for the funding estimate."),
    ] = 30,
    include_search: Annotated[
        bool,
        typer.Option(
            "--include-search/--skip-search",
            help="Include a search-enabled hosted chat request in the smoke run.",
        ),
    ] = True,
    include_rotate: Annotated[
        bool,
        typer.Option(
            "--include-rotate/--skip-rotate",
            help="Include destructive wallet rotation validation.",
        ),
    ] = False,
    include_export: Annotated[
        bool,
        typer.Option(
            "--include-export/--skip-export",
            help="Include private-key export validation without printing the key.",
        ),
    ] = False,
    estimate_only: Annotated[
        bool,
        typer.Option(
            "--estimate-only",
            help="Only build the preview and funding estimate without executing chat/rotate/export checks.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print machine-readable JSON instead of the default smoke tables.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the preview confirmation prompt and run the live smoke test immediately.",
        ),
    ] = False,
    dev: Annotated[
        bool,
        typer.Option(
            "--dev",
            help="Required for live smoke testing commands.",
        ),
    ] = False,
):
    """Run a live hosted SDK smoke test with a funding estimate."""
    _require_dev_mode(dev)
    agent = _load_agent_for_menu(config)
    preview = asyncio.run(
        build_public_sdk_smoke_preview(
            agent,
            chain_type=chain_type,
            forecast_window_days=forecast_window_days,
            include_search=include_search,
            include_rotate=include_rotate,
            include_export=include_export,
        )
    )

    if estimate_only:
        _print_smoke_output(preview, json_output=json_output)
        return

    if not yes:
        _print_smoke_output(preview, json_output=json_output)
        if not Confirm.ask("Proceed with live smoke test", default=False):
            console.print("[yellow]Smoke test cancelled after preview.[/yellow]")
            raise typer.Exit(code=1)

    result = asyncio.run(
        run_public_sdk_smoke(
            agent,
            chain_type=chain_type,
            forecast_window_days=forecast_window_days,
            include_search=include_search,
            include_rotate=include_rotate,
            include_export=include_export,
            preview=preview,
        )
    )
    _print_smoke_output(result, json_output=json_output)


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
    dev: Annotated[
        bool,
        typer.Option(
            "--dev",
            help="Show development-only menu items, including live smoke tests.",
        ),
    ] = False,
):
    """Open the hosted wallet onboarding menu."""
    agent = _load_agent_for_menu(config)
    session_wallet_id: str | None = _saved_wallet_id(agent)
    while True:
        console.print("\n[bold]Solana Agent Wallet Menu[/bold]")
        console.print("1. Create Privy user")
        console.print("2. Create or fetch wallet")
        console.print("3. Show wallet address")
        console.print("4. Rotate wallet")
        console.print("5. Export private key")
        if dev:
            console.print("6. Run smoke test (dev)")
        console.print("q. Quit")
        choice = Prompt.ask("Choose an action", default="1")
        normalized_choice = choice.strip().lower()

        if normalized_choice in {"q", "quit", "exit"}:
            break
        if normalized_choice == "1":
            payload = _run_account_call(agent.create_privy_user())
            if isinstance(payload, dict):
                _remember_privy_user_id(agent, payload.get("privy_user_id"))
            continue
        if normalized_choice == "2":
            privy_user_id = _prompt_privy_user_id(agent)
            payload = _run_account_call(
                agent.create_wallet(
                    privy_user_id=privy_user_id,
                    chain_type=chain_type,
                )
            )
            session_wallet_id = _wallet_id_from_payload(payload) or session_wallet_id
            continue
        if normalized_choice == "3":
            configured_user_id = _configured_privy_user_id(agent)
            if configured_user_id:
                _run_account_call(agent.get_wallet_address())
            else:
                privy_user_id = _prompt_privy_user_id(agent)
                _run_account_call(
                    _wallet_address_for_user(
                        agent,
                        privy_user_id=privy_user_id,
                        chain_type=chain_type,
                    )
                )
            continue
        if normalized_choice == "4":
            privy_user_id = _prompt_privy_user_id(agent)
            payload = _run_account_call(
                agent.rotate_wallet(
                    privy_user_id=privy_user_id,
                    chain_type=chain_type,
                )
            )
            session_wallet_id = _wallet_id_from_payload(payload) or session_wallet_id
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
                default=session_wallet_id or _saved_wallet_id(agent) or "",
            ).strip()
            _run_account_call(
                agent.export_wallet_private_key(
                    wallet_id=wallet_id or None,
                    privy_user_id=privy_user_id,
                    chain_type=chain_type,
                )
            )
            continue
        if normalized_choice == "6" and dev:
            include_search = Confirm.ask(
                "Include a search-enabled chat check",
                default=True,
            )
            include_rotate = Confirm.ask(
                "Include wallet rotation validation",
                default=False,
            )
            include_export = Confirm.ask(
                "Include private-key export validation without printing the key",
                default=False,
            )
            wallet_smoke(
                config=config,
                chain_type=chain_type,
                forecast_window_days=30,
                include_search=include_search,
                include_rotate=include_rotate,
                include_export=include_export,
                estimate_only=False,
                json_output=False,
                yes=False,
                dev=True,
            )
            continue

        console.print("[yellow]Unknown menu option.[/yellow]")


if __name__ == "__main__":
    app()

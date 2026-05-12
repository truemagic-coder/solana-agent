import json
import re
from typing import Optional
from uuid import uuid4
import typer
import asyncio
import logging
from pathlib import Path
import httpx
from typing_extensions import Annotated
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from solana_agent.client.solana_agent import SolanaAgent
from solana_agent.default_instructions import DEFAULT_PUBLIC_AGENT_INSTRUCTIONS
from solana_agent.local_state import (
    load_saved_privy_user_id,
    load_saved_wallet_id,
    save_privy_user_id,
)
from solana_agent.smoke import (
    DEFAULT_TRANSFER_AMOUNT_USDC,
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

DEFAULT_HOSTED_CHAT_INSTRUCTIONS = DEFAULT_PUBLIC_AGENT_INSTRUCTIONS
_SELF_NAME_STATEMENT_PATTERN = re.compile(r"(?i)^\s*my name is\s+(.+?)\s*[.!?]*\s*$")
_SELF_NAME_QUERY_PATTERN = re.compile(
    r"(?i)^\s*(?:what is my name|what's my name)\s*[.!?]*\s*$"
)


def _agent_response_renderable(response: str) -> Group:
    return Group(
        Text("Agent:", style="bright_blue"),
        Markdown(response),
    )


def _remember_session_name(
    message: str,
    session_state: dict[str, str],
) -> None:
    match = _SELF_NAME_STATEMENT_PATTERN.match(str(message or ""))
    if not match:
        return
    normalized_name = re.sub(r"\s+", " ", match.group(1)).strip(" .!?")
    if normalized_name:
        session_state["user_name"] = normalized_name


def _maybe_fast_path_name_recall(
    message: str,
    session_state: dict[str, str],
) -> str | None:
    if not _SELF_NAME_QUERY_PATTERN.match(str(message or "")):
        return None
    remembered_name = str(session_state.get("user_name") or "").strip()
    if not remembered_name:
        return None
    return f"Your name is {remembered_name}."


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


def _prompt_chat_instructions(instructions: Optional[str]) -> str:
    normalized_instructions = str(instructions or "").strip()
    if normalized_instructions:
        return normalized_instructions

    return str(
        typer.prompt(
            "Agent instructions",
            default=DEFAULT_HOSTED_CHAT_INSTRUCTIONS,
        )
        or DEFAULT_HOSTED_CHAT_INSTRUCTIONS
    ).strip()


def _load_chat_agent(config: str, instructions: Optional[str]) -> SolanaAgent:
    if config and Path(config).exists():
        return _load_agent(config)

    normalized_config = str(config or "").strip()
    if normalized_config:
        console.print(
            f"[yellow]Warning:[/yellow] Configuration file not found at '{normalized_config}'. "
            "Starting hosted chat with interactive instructions instead."
        )

    try:
        return SolanaAgent(
            instructions=_prompt_chat_instructions(instructions),
            model="chat",
        )
    except ValueError as e:
        console.print(f"[bold red]Error loading hosted defaults:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(
            f"[bold red]An unexpected error occurred during initialization:[/bold red] {e}"
        )
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
    if step.get("name") == "transfer_usdc":
        amount = str(step.get("amount_usdc") or "").strip()
        recipient = str(step.get("recipient") or "").strip()
        return f"{amount} USDC -> {recipient}".strip()
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
    transfer = payload.get("transfer")

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
    if isinstance(transfer, dict):
        summary_table.add_row(
            "Transfer Recipient",
            str(transfer.get("recipient") or ""),
        )
        summary_table.add_row(
            "Transfer Amount (USDC)",
            str(transfer.get("amount_usdc") or "0"),
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
        summary_table.add_row(
            "Priority Chat",
            _bool_label(coverage.get("includes_priority_chat")),
        )
        summary_table.add_row(
            "Memory Work (7d)",
            _bool_label(coverage.get("includes_memory_work")),
        )
        summary_table.add_row(
            "Memory Project (30d)",
            _bool_label(coverage.get("includes_memory_project")),
        )
        summary_table.add_row(
            "Priority Memory Work (7d)",
            _bool_label(coverage.get("includes_priority_memory_work")),
        )
        summary_table.add_row(
            "Priority Memory Project (30d)",
            _bool_label(coverage.get("includes_priority_memory_project")),
        )
        summary_table.add_row(
            "Jupiter Quote",
            _bool_label(coverage.get("includes_jupiter_quote")),
        )
        summary_table.add_row(
            "Birdeye Read",
            _bool_label(coverage.get("includes_birdeye_read")),
        )
        summary_table.add_row(
            "Token Math",
            _bool_label(coverage.get("includes_token_math")),
        )
        summary_table.add_row(
            "Technical Analysis",
            _bool_label(coverage.get("includes_technical_analysis")),
        )
        summary_table.add_row(
            "Swap",
            _bool_label(coverage.get("includes_swap")),
        )
        summary_table.add_row(
            "Trigger",
            _bool_label(coverage.get("includes_trigger")),
        )
        summary_table.add_row(
            "Earn",
            _bool_label(coverage.get("includes_earn")),
        )
        summary_table.add_row(
            "USDC Transfer",
            _bool_label(coverage.get("includes_transfer")),
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
                ("memory_work_request_usd", "Memory Work (7d)"),
                ("memory_project_request_usd", "Memory Project (30d)"),
                ("priority_chat_request_usd", "Priority Chat"),
                ("priority_memory_work_request_usd", "Priority Memory Work (7d)"),
                (
                    "priority_memory_project_request_usd",
                    "Priority Memory Project (30d)",
                ),
                ("search_chat_request_usd", "Search Chat"),
                ("tooling_chat_requests_usd", "Tooling Chat"),
                ("transfer_chat_request_usd", "Transfer Chat"),
                ("search_surcharge_usd", "Search Surcharge"),
                ("search_provider_cost_ceiling_usd", "Search Provider Ceiling"),
                ("protocol_tooling_buffer_usdc", "Protocol Tooling Buffer"),
                ("funding_buffer_usdc", "Funding Buffer"),
                ("transfer_amount_usdc", "Transfer Amount"),
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


def _resolve_wallet_smoke_options(
    *,
    big: bool,
    include_search: bool,
    include_rotate: bool,
    include_export: bool,
    include_priority: bool,
    include_memory: bool,
    include_jupiter: bool,
    include_birdeye: bool,
    include_swap: bool,
    include_trigger: bool,
    include_earn: bool,
    include_technical_analysis: bool,
    include_token_math: bool,
    include_transfer: bool,
    transfer_recipient: Optional[str],
    transfer_amount_usdc: Optional[str],
) -> dict[str, object]:
    resolved_big = bool(big)
    resolved_include_priority = bool(include_priority or resolved_big)
    resolved_include_memory = bool(include_memory or resolved_big)
    resolved_include_jupiter = bool(include_jupiter or resolved_big)
    resolved_include_birdeye = bool(include_birdeye or resolved_big)
    resolved_include_swap = bool(include_swap or resolved_big)
    resolved_include_trigger = bool(include_trigger or resolved_big)
    resolved_include_earn = bool(include_earn or resolved_big)
    resolved_include_technical_analysis = bool(
        include_technical_analysis or resolved_big
    )
    resolved_include_token_math = bool(include_token_math or resolved_big)
    resolved_transfer_recipient = str(transfer_recipient or "").strip() or None
    resolved_transfer_amount = str(transfer_amount_usdc or "").strip() or None

    if include_transfer and not resolved_transfer_recipient:
        raise typer.BadParameter(
            "--transfer-recipient is required when --include-transfer is enabled"
        )

    if include_transfer and not resolved_transfer_amount:
        resolved_transfer_amount = str(DEFAULT_TRANSFER_AMOUNT_USDC)

    return {
        "include_search": include_search,
        "include_rotate": include_rotate,
        "include_export": include_export,
        "include_priority": resolved_include_priority,
        "include_memory": resolved_include_memory,
        "include_jupiter": resolved_include_jupiter,
        "include_birdeye": resolved_include_birdeye,
        "include_swap": resolved_include_swap,
        "include_trigger": resolved_include_trigger,
        "include_earn": resolved_include_earn,
        "include_technical_analysis": resolved_include_technical_analysis,
        "include_token_math": resolved_include_token_math,
        "include_transfer": include_transfer,
        "transfer_recipient": resolved_transfer_recipient,
        "transfer_amount_usdc": resolved_transfer_amount,
    }


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
    **runtime_context: object,
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
                prompt=prompt,
                search_enabled=search_enabled,
                **runtime_context,
            ):
                if first_chunk:
                    live.update("", refresh=True)  # Clear spinner
                    first_chunk = False
                full_response += chunk
                live.update(_agent_response_renderable(full_response))

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
        console.print(_agent_response_renderable(full_response))


async def _chat_session(
    agent: SolanaAgent,
    *,
    prompt: Optional[str] = None,
    search_enabled: bool = False,
) -> None:
    chat_runtime_context = {
        "conversation_id": f"cli-chat-{uuid4().hex[:12]}",
        "model": "memory",
        "memory_ttl_tier": "work",
    }
    session_state: dict[str, str] = {}

    while True:
        try:
            user_message = Prompt.ask("[bold green]You[/bold green]")

            if user_message.lower() in ["exit", "quit"]:
                console.print("[yellow]Exiting chat session.[/yellow]")
                break

            if not user_message.strip():
                continue

            _remember_session_name(user_message, session_state)

            fast_path_response = _maybe_fast_path_name_recall(
                user_message,
                session_state,
            )
            if fast_path_response:
                console.print(_agent_response_renderable(fast_path_response))
                continue

            await stream_agent_response(
                agent,
                user_message,
                prompt,
                search_enabled=search_enabled,
                **chat_runtime_context,
            )

        except KeyboardInterrupt:
            console.print(
                "\n[yellow]Exiting chat session (KeyboardInterrupt).[/yellow]"
            )
            break
        except Exception as loop_error:
            console.print(
                f"[bold red]An error occurred in the chat loop:[/bold red] {loop_error}"
            )


def _resolve_chat_search_enabled(
    *,
    config_exists: bool,
    search_enabled: Optional[bool],
) -> bool:
    if search_enabled is not None:
        return bool(search_enabled)
    if config_exists:
        return False
    return bool(
        Confirm.ask(
            "Enable hosted search add-on for live web and X results?",
            default=False,
        )
    )


@app.command()
def chat(
    config: Annotated[
        str, typer.Option(help="Path to the configuration JSON file.")
    ] = "config.json",
    instructions: Annotated[
        Optional[str],
        typer.Option(
            help="Agent instructions to use when no config file is available."
        ),
    ] = None,
    prompt: Annotated[  # Allow prompt override via option
        str, typer.Option(help="Optional system prompt override for the session.")
    ] = None,
    search_enabled: Annotated[
        Optional[bool],
        typer.Option(
            "--search-enabled/--no-search-enabled",
            help=(
                "Enable the hosted search add-on for each request in this chat "
                "session. When omitted in hosted no-config chat, the CLI asks once "
                "at startup."
            ),
        ),
    ] = None,
):
    """
    Start an interactive chat session with the Solana Agent.
    Type 'exit' or 'quit' to end the session.
    """
    config_exists = Path(config).exists()
    needs_interactive_instructions = (
        not config_exists and not str(instructions or "").strip()
    )

    resolved_instructions = instructions
    if needs_interactive_instructions:
        resolved_instructions = _prompt_chat_instructions(instructions)
    resolved_search_enabled = _resolve_chat_search_enabled(
        config_exists=config_exists,
        search_enabled=search_enabled,
    )

    if needs_interactive_instructions:
        agent = _load_chat_agent(config, resolved_instructions)
    else:
        with console.status("[bold green]Initializing agent...", spinner="dots"):
            agent = _load_chat_agent(config, resolved_instructions)
    console.print("[green]Agent initialized. Start chatting![/green]")
    console.print("[dim]Type 'exit' or 'quit' to end.[/dim]")
    asyncio.run(
        _chat_session(
            agent,
            prompt=prompt,
            search_enabled=resolved_search_enabled,
        )
    )


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
    include_priority: Annotated[
        bool,
        typer.Option(
            "--include-priority/--skip-priority",
            help="Include hosted priority-tier validation. When memory checks are enabled, this also runs priority memory coverage.",
        ),
    ] = False,
    include_memory: Annotated[
        bool,
        typer.Option(
            "--include-memory/--skip-memory",
            help="Include hosted memory recall checks for both the 7-day work tier and the 30-day project tier.",
        ),
    ] = False,
    include_jupiter: Annotated[
        bool,
        typer.Option(
            "--include-jupiter/--skip-jupiter",
            help="Include a read-only Jupiter swap quote check.",
        ),
    ] = False,
    include_birdeye: Annotated[
        bool,
        typer.Option(
            "--include-birdeye/--skip-birdeye",
            help="Include a read-only Birdeye market-data check.",
        ),
    ] = False,
    include_swap: Annotated[
        bool,
        typer.Option(
            "--include-swap/--skip-swap",
            help="Include a tiny live privy_swap execution check.",
        ),
    ] = False,
    include_trigger: Annotated[
        bool,
        typer.Option(
            "--include-trigger/--skip-trigger",
            help="Include a create-plus-cancel Jupiter Trigger check.",
        ),
    ] = False,
    include_earn: Annotated[
        bool,
        typer.Option(
            "--include-earn/--skip-earn",
            help="Include a reversible Jupiter Earn deposit-plus-withdraw check.",
        ),
    ] = False,
    include_technical_analysis: Annotated[
        bool,
        typer.Option(
            "--include-technical-analysis/--skip-technical-analysis",
            help="Include a technical_analysis check.",
        ),
    ] = False,
    include_token_math: Annotated[
        bool,
        typer.Option(
            "--include-token-math/--skip-token-math",
            help="Include a deterministic token_math round-trip check.",
        ),
    ] = False,
    include_transfer: Annotated[
        bool,
        typer.Option(
            "--include-transfer/--skip-transfer",
            help="Include a live USDC transfer check.",
        ),
    ] = False,
    transfer_recipient: Annotated[
        Optional[str],
        typer.Option(
            help="Recipient wallet address for the live USDC transfer check.",
        ),
    ] = None,
    transfer_amount_usdc: Annotated[
        Optional[str],
        typer.Option(
            help="USDC amount for the live transfer check. Defaults to 0.10 when transfer is enabled.",
        ),
    ] = None,
    big: Annotated[
        bool,
        typer.Option(
            "--big",
            help="Enable the full expanded smoke profile: memory (7d and 30d), priority chat and memory, Jupiter, Birdeye, token math, technical analysis, swap, trigger, and earn.",
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
    smoke_options = _resolve_wallet_smoke_options(
        big=big,
        include_search=include_search,
        include_rotate=include_rotate,
        include_export=include_export,
        include_priority=include_priority,
        include_memory=include_memory,
        include_jupiter=include_jupiter,
        include_birdeye=include_birdeye,
        include_swap=include_swap,
        include_trigger=include_trigger,
        include_earn=include_earn,
        include_technical_analysis=include_technical_analysis,
        include_token_math=include_token_math,
        include_transfer=include_transfer,
        transfer_recipient=transfer_recipient,
        transfer_amount_usdc=transfer_amount_usdc,
    )
    try:
        preview = asyncio.run(
            build_public_sdk_smoke_preview(
                agent,
                chain_type=chain_type,
                forecast_window_days=forecast_window_days,
                **smoke_options,
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
                **smoke_options,
                preview=preview,
            )
        )
        _print_smoke_output(result, json_output=json_output)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]Smoke test failed:[/bold red] {exc}")
        raise typer.Exit(code=1)


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
            include_big = Confirm.ask(
                "Include the full expanded smoke profile (memory 7d/30d + priority chat/memory + Jupiter + Birdeye + token math + technical analysis + swap + trigger + earn)",
                default=False,
            )
            if include_big:
                include_priority = False
                include_memory = False
                include_jupiter = False
                include_birdeye = False
                include_swap = False
                include_trigger = False
                include_earn = False
                include_technical_analysis = False
                include_token_math = False
            else:
                include_priority = Confirm.ask(
                    "Include priority-tier validation for chat and any enabled memory checks",
                    default=False,
                )
                include_memory = Confirm.ask(
                    "Include hosted memory recall checks for both 7-day and 30-day tiers",
                    default=False,
                )
                include_jupiter = Confirm.ask(
                    "Include a Jupiter quote check",
                    default=False,
                )
                include_birdeye = Confirm.ask(
                    "Include a Birdeye market-data check",
                    default=False,
                )
                include_token_math = Confirm.ask(
                    "Include a token-math check",
                    default=False,
                )
                include_technical_analysis = Confirm.ask(
                    "Include a technical-analysis check",
                    default=False,
                )
                include_swap = Confirm.ask(
                    "Include a tiny live swap check",
                    default=False,
                )
                include_trigger = Confirm.ask(
                    "Include a create-plus-cancel trigger check",
                    default=False,
                )
                include_earn = Confirm.ask(
                    "Include a reversible earn check",
                    default=False,
                )
            include_rotate = Confirm.ask(
                "Include wallet rotation validation",
                default=False,
            )
            include_export = Confirm.ask(
                "Include private-key export validation without printing the key",
                default=False,
            )
            include_transfer = Confirm.ask(
                "Include a live USDC transfer check",
                default=False,
            )
            transfer_recipient = None
            transfer_amount_usdc = None
            if include_transfer:
                transfer_recipient = Prompt.ask("Transfer recipient wallet address")
                transfer_amount_usdc = Prompt.ask(
                    "Transfer amount in USDC",
                    default=str(DEFAULT_TRANSFER_AMOUNT_USDC),
                )
            wallet_smoke(
                config=config,
                chain_type=chain_type,
                forecast_window_days=30,
                include_search=include_search,
                include_priority=include_priority,
                include_memory=include_memory,
                include_jupiter=include_jupiter,
                include_birdeye=include_birdeye,
                include_swap=include_swap,
                include_trigger=include_trigger,
                include_earn=include_earn,
                include_technical_analysis=include_technical_analysis,
                include_token_math=include_token_math,
                include_rotate=include_rotate,
                include_export=include_export,
                include_transfer=include_transfer,
                transfer_recipient=transfer_recipient,
                transfer_amount_usdc=transfer_amount_usdc,
                big=include_big,
                estimate_only=False,
                json_output=False,
                yes=False,
                dev=True,
            )
            continue

        console.print("[yellow]Unknown menu option.[/yellow]")


if __name__ == "__main__":
    app()

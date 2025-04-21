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
console = Console()


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
    try:
        with console.status("[bold green]Initializing agent...", spinner="dots"):
            agent = SolanaAgent(config_path=config)
        console.print("[green]Agent initialized. Start chatting![/green]")
        console.print("[dim]Type 'exit' or 'quit' to end.[/dim]")

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


if __name__ == "__main__":
    app()

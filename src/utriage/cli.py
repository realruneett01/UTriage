"""Command-line interface for UTriage. Subcommands are added phase by phase."""

import typer

from utriage import __version__

app = typer.Typer(
    help="UTriage: advisory, offline UT indication triage. Decisions belong to the inspector.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """UTriage: advisory and offline. It never accepts or passes a part."""


@app.command()
def version() -> None:
    """Print the UTriage version."""
    typer.echo(f"utriage {__version__}")

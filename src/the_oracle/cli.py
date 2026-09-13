"""The ``the-oracle`` command line.

Phase 0 implements ``domain list``, ``domain show``, ``rebuild``, and
``version``. Every other command is a loud, honest stub.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from the_oracle import __version__
from the_oracle.config import get_settings
from the_oracle.context import LearnerContext

console = Console()

app = typer.Typer(
    name="the-oracle",
    help="A tutor that learns you, plans the path, finds the material, and keeps you moving.",
    no_args_is_help=True,
    add_completion=False,
)
domain_app = typer.Typer(name="domain", help="Work with domain packs.", no_args_is_help=True)
app.add_typer(domain_app)

# Phase 1 commands own their own modules; the CLI only wires them in.
from the_oracle.commands import assess as _assess_cmd  # noqa: E402
from the_oracle.commands import learner as _learner_cmd  # noqa: E402

app.add_typer(_assess_cmd.app)
app.add_typer(_learner_cmd.app, name="learner")

def _not_built(command: str, phase: str) -> None:
    console.print(
        Panel(
            f"[bold]{command}[/bold] is not built yet.\nIt lands in {phase}.",
            title="not built yet",
            border_style="yellow",
        )
    )
    raise typer.Exit(code=1)


def _registry():  # type: ignore[no-untyped-def]
    """Import the domains registry lazily, so the engine loads without it."""
    from the_oracle.domains import registry

    return registry


@app.command()
def version() -> None:
    """Print the version and the resolved data directory."""
    settings = get_settings()
    console.print(f"the-oracle [bold]{__version__}[/bold]")
    console.print(f"home: {settings.home}")
    console.print(f"database: {settings.sqlalchemy_url}")


@domain_app.command("list")
def domain_list() -> None:
    """List the domain packs found in the data directory."""
    settings = get_settings()
    try:
        ids = _registry().list_domains()
    except ModuleNotFoundError:
        console.print("[yellow]The domains package is not installed yet.[/yellow]")
        raise typer.Exit(code=1) from None

    if not ids:
        console.print(
            Panel(
                f"No domain packs in [bold]{settings.domains_dir}[/bold].\n"
                "Run [bold]the-oracle domain add[/bold] to generate one.",
                title="blank oracle",
                border_style="cyan",
            )
        )
        return

    table = Table(title="domains", show_lines=False)
    table.add_column("id", style="bold")
    table.add_column("title")
    table.add_column("objectives", justify="right")
    table.add_column("modules", justify="right")

    registry = _registry()
    for domain_id in sorted(ids):
        try:
            domain = registry.load_domain(domain_id)
        except Exception as exc:  # fail loud, keep listing the rest
            table.add_row(domain_id, f"[red]invalid: {exc}[/red]", "-", "-")
            continue
        table.add_row(
            domain.id,
            getattr(domain, "title", ""),
            str(len(getattr(domain, "objectives", []))),
            str(len(getattr(domain, "modules", []))),
        )
    console.print(table)


@domain_app.command("show")
def domain_show(
    domain_id: Annotated[str, typer.Argument(help="Domain pack id.")],
) -> None:
    """Show one pack: modules, objectives in teaching order, prerequisites."""
    registry = _registry()
    try:
        domain = registry.load_domain(domain_id)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print(Panel(getattr(domain, "description", "") or "", title=getattr(domain, "title", domain_id)))

    table = Table(title="modules")
    table.add_column("#", justify="right")
    table.add_column("module", style="bold")
    table.add_column("objectives")
    for position, module in enumerate(getattr(domain, "modules", []), start=1):
        table.add_row(str(position), module.title, ", ".join(module.objectives))
    console.print(table)

    try:
        order = domain.teaching_order()
    except Exception:
        order = []
    if order:
        console.print("teaching order: " + " -> ".join(order))


@domain_app.command("add")
def domain_add(
    brief: Annotated[str, typer.Argument(help="What you want to learn.")] = "",
) -> None:
    """Generate a new domain pack from an interview."""
    _not_built("domain add", "Phase 2 (Plan me)")


@app.command()
def rebuild(
    learner: Annotated[str | None, typer.Option(help="Learner id. Defaults to the current one.")] = None,
) -> None:
    """Drop derived state and replay the event log."""
    from the_oracle.store.db import create_all, get_engine
    from the_oracle.store.rebuild import rebuild as run_rebuild

    settings = get_settings()
    settings.ensure_home()
    engine = get_engine(settings)
    create_all(engine)

    ctx = (
        LearnerContext.for_learner(learner, settings)
        if learner
        else LearnerContext.resolve(settings)
    )
    report = run_rebuild(ctx.learner_id, engine)

    table = Table(title=f"rebuild: {report.learner_id}")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("events read", str(report.events_read))
    table.add_row("events handled", str(report.events_handled))
    table.add_row("events unhandled", str(report.events_skipped))
    table.add_row("derived rows", str(report.rows_written))
    table.add_row("tables cleared", ", ".join(report.tables_cleared) or "-")
    console.print(table)


@app.command()
def plan(domain_id: Annotated[str, typer.Argument()] = "") -> None:
    """Build or refresh the syllabus."""
    _not_built("plan", "Phase 2 (Plan me)")


@app.command()
def study(domain_id: Annotated[str, typer.Argument()] = "") -> None:
    """Run a study session."""
    _not_built("study", "Phase 4 (Study loop)")


@app.command()
def review() -> None:
    """Work through everything that is due."""
    _not_built("review", "Phase 4 (Study loop)")


@app.command()
def report() -> None:
    """Print the weekly report."""
    _not_built("report", "Phase 5 (Push me)")


if __name__ == "__main__":  # pragma: no cover
    app()
